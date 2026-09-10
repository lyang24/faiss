/* Reproducible paired HNSW benchmark for IndexQINCo and native Faiss SQ8. */

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexQINCo.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/index_io.h>

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Matrix {
    size_t n = 0;
    int d = 0;
    std::vector<float> data;
};

struct Options {
    std::string base;
    std::string query;
    std::string graph;
    std::string groundtruth;
    std::string output;
    std::string filter = "none";
    std::string construction_storage = "fp32";
    size_t nb = 1000000;
    size_t nq = 1000;
    int threads = 16;
    int M = 32;
    int efc = 100;
    size_t block = 64;
    float exact_norm_weight = 0.0f;
    int doc_bits = 8;
    int query_bits = 8;
    int rerank = 0;
    int pca_dim = 0;
    size_t pca_train = 50000;
    size_t int8_head_dims = 0;
    bool normalize = false;
    bool rebuild_graph = false;
    bool validate_only = false;
    std::vector<int> efs =
            {16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512};
};

std::unordered_map<std::string, std::string> parse_args(int argc, char** argv) {
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) != 0) {
            throw std::runtime_error("expected --key, got " + key);
        }
        if (key == "--normalize" || key == "--rebuild-graph" ||
            key == "--validate-only") {
            args[key] = "1";
        } else {
            if (++i == argc) {
                throw std::runtime_error("missing value for " + key);
            }
            args[key] = argv[i];
        }
    }
    return args;
}

std::vector<int> parse_ints(const std::string& value) {
    std::vector<int> out;
    size_t begin = 0;
    while (begin < value.size()) {
        size_t end = value.find(',', begin);
        out.push_back(std::stoi(value.substr(begin, end - begin)));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return out;
}

Options options_from(int argc, char** argv) {
    const auto a = parse_args(argc, argv);
    Options o;
    auto get = [&](const char* k, const std::string& fallback) {
        auto it = a.find(k);
        return it == a.end() ? fallback : it->second;
    };
    o.base = get("--base", "");
    o.query = get("--query", "");
    o.graph = get("--graph", "hnsw_fp32.faissindex");
    o.groundtruth = get("--groundtruth", "");
    o.output = get("--output", "-");
    o.filter = get("--filter", "none");
    o.construction_storage = get("--construction-storage", "fp32");
    o.nb = std::stoull(get("--nb", "1000000"));
    o.nq = std::stoull(get("--nq", "1000"));
    o.threads = std::stoi(get("--threads", "16"));
    o.M = std::stoi(get("--M", "32"));
    o.efc = std::stoi(get("--efConstruction", "100"));
    o.block = std::stoull(get("--block", "64"));
    o.exact_norm_weight = std::stof(get("--exact-norm-weight", "0"));
    o.doc_bits = std::stoi(get("--doc-bits", "8"));
    o.query_bits = std::stoi(get("--query-bits", "8"));
    o.rerank = std::stoi(get("--rerank", "0"));
    o.pca_dim = std::stoi(get("--pca-dim", "0"));
    o.pca_train = std::stoull(get("--pca-train", "50000"));
    o.int8_head_dims = std::stoull(get("--int8-head-dims", "0"));
    o.normalize = a.count("--normalize") != 0;
    o.rebuild_graph = a.count("--rebuild-graph") != 0;
    o.validate_only = a.count("--validate-only") != 0;
    if (a.count("--efs")) {
        o.efs = parse_ints(a.at("--efs"));
    }
    if (o.base.empty() || o.query.empty()) {
        throw std::runtime_error("--base and --query are required");
    }
    if (o.pca_dim < 0 ||
        (o.pca_dim > 0 && o.int8_head_dims > size_t(o.pca_dim))) {
        throw std::runtime_error("invalid PCA/head dimensions");
    }
    return o;
}

Matrix read_fvecs(const std::string& path, size_t limit) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    int32_t d = 0;
    f.read(reinterpret_cast<char*>(&d), sizeof(d));
    if (!f || d <= 0) {
        throw std::runtime_error("bad fvecs header in " + path);
    }
    f.seekg(0, std::ios::end);
    const size_t bytes = size_t(f.tellg());
    const size_t row_bytes = sizeof(int32_t) + size_t(d) * sizeof(float);
    if (bytes % row_bytes != 0) {
        throw std::runtime_error("fvecs size is not a whole number of rows");
    }
    Matrix m;
    m.n = std::min(limit, bytes / row_bytes);
    m.d = d;
    m.data.resize(m.n * d);
    f.seekg(0);
    for (size_t i = 0; i < m.n; ++i) {
        int32_t row_d;
        f.read(reinterpret_cast<char*>(&row_d), sizeof(row_d));
        if (row_d != d) {
            throw std::runtime_error("inconsistent fvecs dimension");
        }
        f.read(reinterpret_cast<char*>(m.data.data() + i * d),
               d * sizeof(float));
    }
    return m;
}

Matrix read_npy_f32(const std::string& path, size_t limit) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    char magic[6];
    uint8_t major = 0, minor = 0;
    f.read(magic, sizeof(magic));
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);
    if (!f || std::string(magic, sizeof(magic)) != "\x93NUMPY" || major < 1 ||
        major > 3) {
        throw std::runtime_error("bad npy header in " + path);
    }
    uint32_t header_bytes = 0;
    if (major == 1) {
        uint16_t n = 0;
        f.read(reinterpret_cast<char*>(&n), sizeof(n));
        header_bytes = n;
    } else {
        f.read(reinterpret_cast<char*>(&header_bytes), sizeof(header_bytes));
    }
    std::string header(header_bytes, '\0');
    f.read(header.data(), header.size());
    if (!f ||
        (header.find("'<f4'") == std::string::npos &&
         header.find("'|f4'") == std::string::npos) ||
        header.find("False") == std::string::npos) {
        throw std::runtime_error(
                "npy must be little-endian C-contiguous float32");
    }
    const size_t shape = header.find("shape");
    const size_t left = header.find('(', shape);
    const size_t comma = header.find(',', left);
    const size_t right = header.find(')', comma);
    if (shape == std::string::npos || left == std::string::npos ||
        comma == std::string::npos || right == std::string::npos) {
        throw std::runtime_error("npy must have a two-dimensional shape");
    }
    const size_t rows = std::stoull(header.substr(left + 1, comma - left - 1));
    const size_t cols =
            std::stoull(header.substr(comma + 1, right - comma - 1));
    if (rows == 0 || cols == 0 ||
        cols > size_t(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("invalid npy shape");
    }
    Matrix m;
    m.n = std::min(limit, rows);
    m.d = int(cols);
    m.data.resize(m.n * cols);
    f.read(reinterpret_cast<char*>(m.data.data()),
           m.data.size() * sizeof(float));
    if (!f) {
        throw std::runtime_error("truncated npy payload in " + path);
    }
    return m;
}

Matrix read_matrix(const std::string& path, size_t limit) {
    return path.size() >= 4 && path.substr(path.size() - 4) == ".npy"
            ? read_npy_f32(path, limit)
            : read_fvecs(path, limit);
}

std::vector<faiss::idx_t> read_ivecs_top10(
        const std::string& path,
        size_t expected_n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    std::vector<faiss::idx_t> result(expected_n * 10);
    for (size_t i = 0; i < expected_n; ++i) {
        int32_t k;
        f.read(reinterpret_cast<char*>(&k), sizeof(k));
        if (!f || k < 10) {
            throw std::runtime_error("invalid ivecs row");
        }
        std::vector<int32_t> row(k);
        f.read(reinterpret_cast<char*>(row.data()),
               size_t(k) * sizeof(int32_t));
        for (int j = 0; j < 10; ++j) {
            result[i * 10 + j] = row[j];
        }
    }
    return result;
}

void normalize_rows(Matrix& m) {
#pragma omp parallel for
    for (size_t i = 0; i < m.n; ++i) {
        float norm2 = 0.0f;
        for (int j = 0; j < m.d; ++j) {
            norm2 += m.data[i * m.d + j] * m.data[i * m.d + j];
        }
        if (norm2 > 0.0f) {
            const float inv = 1.0f / std::sqrt(norm2);
            for (int j = 0; j < m.d; ++j) {
                m.data[i * m.d + j] *= inv;
            }
        }
    }
}

std::unique_ptr<faiss::IndexHNSW> clone_with_sq8(
        const faiss::IndexHNSW& graph,
        const Matrix& base) {
    auto storage = std::make_unique<faiss::IndexScalarQuantizer>(
            base.d, faiss::ScalarQuantizer::QT_8bit, faiss::METRIC_L2);
    storage->train(base.n, base.data.data());
    storage->add(base.n, base.data.data());
    auto out = std::make_unique<faiss::IndexHNSW>(
            storage.release(), graph.hnsw.nb_neighbors(0) / 2);
    out->own_fields = true;
    out->hnsw = graph.hnsw;
    out->ntotal = graph.ntotal;
    out->is_trained = true;
    return out;
}

std::vector<faiss::idx_t> exact_ground_truth(
        const Matrix& base,
        const Matrix& query,
        const std::vector<uint8_t>* eligible = nullptr) {
    std::vector<float> selected;
    std::vector<faiss::idx_t> original_ids;
    const float* xb = base.data.data();
    size_t n = base.n;
    if (eligible) {
        original_ids.reserve(base.n);
        for (size_t i = 0; i < base.n; ++i) {
            if ((*eligible)[i]) {
                original_ids.push_back(i);
                selected.insert(
                        selected.end(),
                        base.data.begin() + i * base.d,
                        base.data.begin() + (i + 1) * base.d);
            }
        }
        xb = selected.data();
        n = original_ids.size();
    }
    faiss::IndexFlatL2 flat(base.d);
    flat.add(n, xb);
    std::vector<float> distances(query.n * 10);
    std::vector<faiss::idx_t> gt(query.n * 10);
    flat.search(query.n, query.data.data(), 10, distances.data(), gt.data());
    if (eligible) {
        for (auto& id : gt) {
            if (id >= 0) {
                id = original_ids[id];
            }
        }
    }
    return gt;
}

double recall_at_10(
        const std::vector<faiss::idx_t>& got,
        const std::vector<faiss::idx_t>& gt) {
    size_t found = 0;
    const size_t nq = gt.size() / 10;
    for (size_t i = 0; i < nq; ++i) {
        for (int j = 0; j < 10; ++j) {
            const auto id = got[i * 10 + j];
            for (int p = 0; p < 10; ++p) {
                found += id >= 0 && id == gt[i * 10 + p];
            }
        }
    }
    return double(found) / double(nq * 10);
}

struct Run {
    double ms;
    double recall;
    size_t incomplete;
};

Run run_search(
        const faiss::IndexHNSW& index,
        const Matrix& query,
        const std::vector<faiss::idx_t>& gt,
        int ef,
        int batch,
        bool normalize_query,
        int rerank = 0,
        const faiss::IDSelector* selector = nullptr,
        const faiss::VectorTransform* transform = nullptr) {
    faiss::SearchParametersHNSW p;
    p.efSearch = ef;
    p.sel = const_cast<faiss::IDSelector*>(selector);
    const auto start = Clock::now();
    const int search_k = std::max(10, rerank);
    std::vector<float> distances(query.n * search_k);
    std::vector<faiss::idx_t> labels(query.n * search_k);
    std::vector<float> scoring_queries;
    if (rerank > 10) {
        scoring_queries.resize(query.n * index.d);
    }
    if (batch == 1) {
        for (size_t i = 0; i < query.n; ++i) {
            std::vector<float> one(
                    query.data.begin() + i * query.d,
                    query.data.begin() + (i + 1) * query.d);
            if (normalize_query) {
                float norm2 = 0.0f;
                for (float v : one) {
                    norm2 += v * v;
                }
                if (norm2 > 0.0f) {
                    const float inv = 1.0f / std::sqrt(norm2);
                    for (float& v : one) {
                        v *= inv;
                    }
                }
            }
            std::vector<float> projected;
            const float* search_query = one.data();
            if (transform) {
                projected.resize(transform->d_out);
                transform->apply_noalloc(1, one.data(), projected.data());
                search_query = projected.data();
            }
            if (rerank > 10) {
                std::copy(
                        search_query,
                        search_query + index.d,
                        scoring_queries.begin() + i * index.d);
            }
            index.search(
                    1,
                    search_query,
                    search_k,
                    distances.data() + i * search_k,
                    labels.data() + i * search_k,
                    &p);
        }
    } else {
        std::vector<float> prepared = query.data;
        if (normalize_query) {
            Matrix view{query.n, query.d, {}};
            view.data.swap(prepared);
            normalize_rows(view);
            prepared.swap(view.data);
        }
        std::vector<float> projected;
        const float* search_queries = prepared.data();
        if (transform) {
            projected.resize(query.n * transform->d_out);
            transform->apply_noalloc(
                    query.n, prepared.data(), projected.data());
            search_queries = projected.data();
        }
        if (rerank > 10) {
            std::copy(
                    search_queries,
                    search_queries + query.n * index.d,
                    scoring_queries.begin());
        }
        index.search(
                query.n,
                search_queries,
                search_k,
                distances.data(),
                labels.data(),
                &p);
    }
    if (rerank > 10) {
        const auto* storage =
                dynamic_cast<const faiss::IndexQINCo*>(index.storage);
        if (!storage) {
            throw std::runtime_error("rerank requires IndexQINCo storage");
        }
        std::vector<faiss::idx_t> reranked(query.n * 10, -1);
#pragma omp parallel
        {
            std::vector<float> decoded(index.d);
            std::vector<std::pair<float, faiss::idx_t>> candidates(search_k);
#pragma omp for
            for (size_t i = 0; i < query.n; ++i) {
                const float* q = scoring_queries.data() + i * index.d;
                for (int p = 0; p < search_k; ++p) {
                    const faiss::idx_t id = labels[i * search_k + p];
                    float dis = std::numeric_limits<float>::infinity();
                    if (id >= 0) {
                        storage->reconstruct(id, decoded.data());
                        dis = 0.0f;
                        for (int j = 0; j < index.d; ++j) {
                            const float delta = q[j] - decoded[j];
                            dis += delta * delta;
                        }
                    }
                    candidates[p] = {dis, id};
                }
                std::partial_sort(
                        candidates.begin(),
                        candidates.begin() + 10,
                        candidates.end());
                for (int p = 0; p < 10; ++p) {
                    reranked[i * 10 + p] = candidates[p].second;
                }
            }
        }
        labels.swap(reranked);
    }
    const double ms =
            std::chrono::duration<double, std::milli>(Clock::now() - start)
                    .count();
    const size_t incomplete =
            std::count(labels.begin(), labels.end(), faiss::idx_t(-1));
    return {ms, recall_at_10(labels, gt), incomplete};
}

void emit(
        std::ostream& out,
        const std::string& method,
        const std::string& phase,
        int ef,
        int round,
        int batch,
        size_t nq,
        size_t bytes_per_vector,
        double retained,
        const std::string& filter,
        const Run& r) {
    out << "{\"method\":\"" << method << "\",\"phase\":\"" << phase
        << "\",\"ef\":" << ef << ",\"round\":" << round
        << ",\"batch\":" << batch << ",\"nq\":" << nq << ",\"ms\":" << r.ms
        << ",\"qps\":" << (1000.0 * nq / r.ms)
        << ",\"recall_at_10\":" << r.recall
        << ",\"incomplete\":" << r.incomplete
        << ",\"bytes_per_vector\":" << bytes_per_vector
        << ",\"retained\":" << retained << ",\"filter\":\"" << filter
        << "\"}\n";
    out.flush();
}

int first_ef_at(const std::vector<std::pair<int, Run>>& sweep, double floor) {
    for (const auto& [ef, r] : sweep) {
        if (r.recall >= floor) {
            return ef;
        }
    }
    return -1;
}

struct BitmapSelector final : faiss::IDSelector {
    const std::vector<uint8_t>& bits;
    explicit BitmapSelector(const std::vector<uint8_t>& b) : bits(b) {}
    bool is_member(faiss::idx_t id) const override {
        return id >= 0 && size_t(id) < bits.size() && bits[id];
    }
};

std::vector<uint8_t> make_filter(
        const Matrix& base,
        double fraction,
        const std::string& type) {
    const size_t keep = std::max<size_t>(1, std::floor(base.n * fraction));
    std::vector<size_t> ids(base.n);
    std::iota(ids.begin(), ids.end(), 0);
    if (type == "random") {
        std::mt19937_64 rng(0x51494e434fULL);
        std::shuffle(ids.begin(), ids.end(), rng);
    } else if (type == "feature0") {
        if (keep < base.n) {
            std::nth_element(
                    ids.begin(),
                    ids.begin() + keep,
                    ids.end(),
                    [&](size_t a, size_t b) {
                        return base.data[a * base.d] < base.data[b * base.d];
                    });
        }
    } else {
        throw std::runtime_error("filter must be none, random, or feature0");
    }
    std::vector<uint8_t> bits(base.n, 0);
    for (size_t i = 0; i < keep; ++i) {
        bits[ids[i]] = 1;
    }
    return bits;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options o = options_from(argc, argv);
        omp_set_num_threads(o.threads);
        Matrix base = read_matrix(o.base, o.nb);
        Matrix query = read_matrix(o.query, o.nq);
        if (base.d != query.d || base.n != o.nb || query.n != o.nq) {
            throw std::runtime_error(
                    "dataset shape does not match requested shape");
        }
        if (o.normalize) {
            normalize_rows(base);
        }
        Matrix ground_truth_queries = query;
        if (o.normalize) {
            normalize_rows(ground_truth_queries);
        }

        std::unique_ptr<faiss::IndexHNSW> fp;
        const auto graph_start = Clock::now();
        bool graph_loaded = false;
        std::ifstream graph_test(o.graph, std::ios::binary);
        if (graph_test.good() && !o.rebuild_graph) {
            graph_loaded = true;
            auto loaded = faiss::read_index_up(o.graph.c_str());
            auto* cast = dynamic_cast<faiss::IndexHNSW*>(loaded.release());
            if (!cast) {
                throw std::runtime_error("cached graph is not HNSW");
            }
            fp.reset(cast);
            if (fp->d != base.d || size_t(fp->ntotal) != base.n) {
                throw std::runtime_error("cached graph shape mismatch");
            }
        } else {
            if (o.construction_storage == "fp32") {
                auto built =
                        std::make_unique<faiss::IndexHNSWFlat>(base.d, o.M);
                built->hnsw.efConstruction = o.efc;
                built->add(base.n, base.data.data());
                faiss::write_index(built.get(), o.graph.c_str());
                fp = std::move(built);
            } else if (o.construction_storage == "sq8") {
                auto built = std::make_unique<faiss::IndexHNSWSQ>(
                        base.d,
                        faiss::ScalarQuantizer::QT_8bit,
                        o.M,
                        faiss::METRIC_L2);
                built->hnsw.efConstruction = o.efc;
                built->train(base.n, base.data.data());
                built->add(base.n, base.data.data());
                faiss::write_index(built.get(), o.graph.c_str());
                fp = std::move(built);
            } else {
                throw std::runtime_error(
                        "construction storage must be fp32 or sq8");
            }
        }
        const double graph_ms = std::chrono::duration<double, std::milli>(
                                        Clock::now() - graph_start)
                                        .count();

        const auto representation_start = Clock::now();
        auto sq8 = clone_with_sq8(*fp, base);
        std::unique_ptr<faiss::PCAMatrix> pca;
        Matrix scoring_base;
        const Matrix* compressed_base = &base;
        if (o.pca_dim > 0) {
            if (o.pca_dim > base.d) {
                throw std::runtime_error("PCA dimension exceeds input");
            }
            pca = std::make_unique<faiss::PCAMatrix>(
                    base.d, o.pca_dim, 0.0f, false);
            pca->train(std::min(o.pca_train, base.n), base.data.data());
            scoring_base.n = base.n;
            scoring_base.d = o.pca_dim;
            scoring_base.data.resize(base.n * size_t(o.pca_dim));
            pca->apply_noalloc(
                    base.n, base.data.data(), scoring_base.data.data());
            compressed_base = &scoring_base;
        }
        std::unique_ptr<faiss::IndexHNSW> qinco;
        if (!pca) {
            qinco.reset(
                    faiss::clone_hnsw_with_qinco_storage(
                            *fp,
                            base.n,
                            base.data.data(),
                            o.block,
                            o.exact_norm_weight,
                            o.doc_bits,
                            o.query_bits,
                            o.int8_head_dims));
        } else {
            auto storage = std::make_unique<faiss::IndexQINCo>(
                    compressed_base->d,
                    o.block,
                    o.exact_norm_weight,
                    o.doc_bits,
                    o.query_bits,
                    o.int8_head_dims);
            storage->train(base.n, compressed_base->data.data());
            storage->add(base.n, compressed_base->data.data());
            qinco = std::make_unique<faiss::IndexHNSW>(
                    storage.release(), fp->hnsw.nb_neighbors(0) / 2);
            qinco->own_fields = true;
            qinco->hnsw = fp->hnsw;
            qinco->ntotal = base.n;
            qinco->is_trained = true;
            qinco->metric_arg = fp->metric_arg;
            qinco->use_visited_hashset = fp->use_visited_hashset;
        }
        const double representation_ms =
                std::chrono::duration<double, std::milli>(
                        Clock::now() - representation_start)
                        .count();
        auto* qstorage = dynamic_cast<faiss::IndexQINCo*>(qinco->storage);
        const size_t graph_bytes =
                fp->hnsw.neighbors.size() * sizeof(faiss::HNSW::storage_idx_t) +
                fp->hnsw.offsets.size() * sizeof(size_t) +
                fp->hnsw.levels.size() * sizeof(int);

        std::ofstream file;
        std::ostream* output = &std::cout;
        if (o.output != "-") {
            file.open(o.output, std::ios::app);
            output = &file;
        }
        *output << "{\"phase\":\"metadata\",\"d\":" << base.d
                << ",\"nb\":" << base.n << ",\"nq\":" << query.n
                << ",\"threads\":" << o.threads << ",\"M\":" << o.M
                << ",\"efConstruction\":" << o.efc
                << ",\"construction_storage\":\"" << o.construction_storage
                << "\""
                << ",\"graph_loaded\":" << (graph_loaded ? "true" : "false")
                << ",\"graph_ms\":" << graph_ms
                << ",\"normalize\":" << (o.normalize ? "true" : "false")
                << ",\"qinco_block\":" << o.block
                << ",\"exact_norm_weight\":" << o.exact_norm_weight
                << ",\"doc_bits\":" << o.doc_bits
                << ",\"query_bits\":" << o.query_bits
                << ",\"pca_dim\":" << o.pca_dim
                << ",\"pca_train\":" << o.pca_train
                << ",\"int8_head_dims\":" << o.int8_head_dims
                << ",\"representation_ms\":" << representation_ms
                << ",\"rerank\":" << o.rerank
                << ",\"graph_bytes\":" << graph_bytes << "}\n";

        const std::string qmethod =
                o.rerank > 10 ? "qinco_r" + std::to_string(o.rerank) : "qinco";
        if (o.filter != "none") {
            const std::vector<double> fractions = {
                    1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001};
            for (double fraction : fractions) {
                auto bits = make_filter(base, fraction, o.filter);
                BitmapSelector selector(bits);
                auto gt = exact_ground_truth(base, ground_truth_queries, &bits);
                for (int ef : o.efs) {
                    const Run r = run_search(
                            *qinco,
                            query,
                            gt,
                            ef,
                            1000,
                            o.normalize,
                            o.rerank,
                            &selector,
                            pca.get());
                    emit(*output,
                         qmethod,
                         "filter_sweep",
                         ef,
                         0,
                         1000,
                         query.n,
                         qstorage->bytes_per_vector(),
                         fraction,
                         o.filter,
                         r);
                }
            }
            return 0;
        }

        const auto gt = exact_ground_truth(base, ground_truth_queries);
        if (!o.groundtruth.empty()) {
            const auto source_gt = read_ivecs_top10(o.groundtruth, query.n);
            *output << "{\"phase\":\"groundtruth_validation\","
                    << "\"source_vs_recomputed_recall_at_10\":"
                    << recall_at_10(source_gt, gt) << "}\n";
        }
        if (o.validate_only) {
            return 0;
        }
        std::vector<std::pair<int, Run>> sq_sweep, qi_sweep;
        for (int ef : o.efs) {
            Run sq = run_search(*sq8, query, gt, ef, 1000, o.normalize);
            emit(*output,
                 "sq8",
                 "tune",
                 ef,
                 0,
                 1000,
                 query.n,
                 size_t(base.d),
                 1.0,
                 "none",
                 sq);
            sq_sweep.push_back({ef, sq});
            Run qi = run_search(
                    *qinco,
                    query,
                    gt,
                    ef,
                    1000,
                    o.normalize,
                    o.rerank,
                    nullptr,
                    pca.get());
            emit(*output,
                 qmethod,
                 "tune",
                 ef,
                 0,
                 1000,
                 query.n,
                 qstorage->bytes_per_vector(),
                 1.0,
                 "none",
                 qi);
            qi_sweep.push_back({ef, qi});
            Run sq_bracket = run_search(*sq8, query, gt, ef, 1000, o.normalize);
            emit(*output,
                 "sq8",
                 "bracket",
                 ef,
                 0,
                 1000,
                 query.n,
                 size_t(base.d),
                 1.0,
                 "none",
                 sq_bracket);
        }

        for (double floor : {0.90, 0.95}) {
            const int sqef = first_ef_at(sq_sweep, floor);
            const int qief = first_ef_at(qi_sweep, floor);
            *output << "{\"phase\":\"freeze\",\"recall_floor\":" << floor
                    << ",\"sq8_ef\":" << sqef << ",\"qinco_ef\":" << qief
                    << "}\n";
            if (sqef < 0 || qief < 0) {
                continue;
            }
            for (int round = 1; round <= 3; ++round) {
                if (round % 2) {
                    emit(*output,
                         "sq8",
                         "confirm",
                         sqef,
                         round,
                         1000,
                         query.n,
                         size_t(base.d),
                         1.0,
                         "none",
                         run_search(*sq8, query, gt, sqef, 1000, o.normalize));
                    emit(*output,
                         qmethod,
                         "confirm",
                         qief,
                         round,
                         1000,
                         query.n,
                         qstorage->bytes_per_vector(),
                         1.0,
                         "none",
                         run_search(
                                 *qinco,
                                 query,
                                 gt,
                                 qief,
                                 1000,
                                 o.normalize,
                                 o.rerank,
                                 nullptr,
                                 pca.get()));
                } else {
                    emit(*output,
                         qmethod,
                         "confirm",
                         qief,
                         round,
                         1000,
                         query.n,
                         qstorage->bytes_per_vector(),
                         1.0,
                         "none",
                         run_search(
                                 *qinco,
                                 query,
                                 gt,
                                 qief,
                                 1000,
                                 o.normalize,
                                 o.rerank,
                                 nullptr,
                                 pca.get()));
                    emit(*output,
                         "sq8",
                         "confirm",
                         sqef,
                         round,
                         1000,
                         query.n,
                         size_t(base.d),
                         1.0,
                         "none",
                         run_search(*sq8, query, gt, sqef, 1000, o.normalize));
                }
            }
            emit(*output,
                 "sq8",
                 "batch1",
                 sqef,
                 1,
                 1,
                 query.n,
                 size_t(base.d),
                 1.0,
                 "none",
                 run_search(*sq8, query, gt, sqef, 1, o.normalize));
            emit(*output,
                 qmethod,
                 "batch1",
                 qief,
                 1,
                 1,
                 query.n,
                 qstorage->bytes_per_vector(),
                 1.0,
                 "none",
                 run_search(
                         *qinco,
                         query,
                         gt,
                         qief,
                         1,
                         o.normalize,
                         o.rerank,
                         nullptr,
                         pca.get()));
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 2;
    }
}
