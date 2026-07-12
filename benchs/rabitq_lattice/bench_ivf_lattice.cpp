/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/IndexIVFRaBitQLattice.h>

#include <omp.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

struct Dataset {
    int64_t nb = 0;
    int64_t nq = 0;
    int64_t d = 0;
    int64_t kgt = 0;
    std::vector<float> xb;
    std::vector<float> xq;
    std::vector<faiss::idx_t> gt;
};

template <typename T>
void read_exact(std::ifstream& f, T* ptr, size_t n) {
    f.read(reinterpret_cast<char*>(ptr), sizeof(T) * n);
    if (!f) {
        throw std::runtime_error("short read");
    }
}

Dataset read_dataset(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("cannot open dataset");
    }
    Dataset ds;
    read_exact(f, &ds.nb, 1);
    read_exact(f, &ds.nq, 1);
    read_exact(f, &ds.d, 1);
    read_exact(f, &ds.kgt, 1);
    ds.xb.resize(ds.nb * ds.d);
    ds.xq.resize(ds.nq * ds.d);
    ds.gt.resize(ds.nq * ds.kgt);
    read_exact(f, ds.xb.data(), ds.xb.size());
    read_exact(f, ds.xq.data(), ds.xq.size());
    read_exact(f, ds.gt.data(), ds.gt.size());
    return ds;
}

struct Args {
    std::string data;
    int nlist = 1024;
    int nprobe = 64;
    int k = 100;
    int ntrain = 50000;
    int niter = 10;
    int rounds = 3;
    int threads = 1;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(
                        std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (a == "--data") {
            args.data = need("--data");
        } else if (a == "--nlist") {
            args.nlist = std::atoi(need("--nlist"));
        } else if (a == "--nprobe") {
            args.nprobe = std::atoi(need("--nprobe"));
        } else if (a == "--k") {
            args.k = std::atoi(need("--k"));
        } else if (a == "--ntrain") {
            args.ntrain = std::atoi(need("--ntrain"));
        } else if (a == "--niter") {
            args.niter = std::atoi(need("--niter"));
        } else if (a == "--rounds") {
            args.rounds = std::atoi(need("--rounds"));
        } else if (a == "--threads") {
            args.threads = std::atoi(need("--threads"));
        } else {
            throw std::runtime_error("unknown arg " + a);
        }
    }
    if (args.data.empty()) {
        throw std::runtime_error("--data is required");
    }
    return args;
}

std::vector<float> train_centroids(
        const Dataset& ds,
        int nlist,
        int ntrain,
        int niter) {
    ntrain = std::min<int64_t>(ntrain, ds.nb);
    faiss::ClusteringParameters cp;
    cp.niter = niter;
    cp.seed = 12345;
    faiss::Clustering clus(ds.d, nlist, cp);
    faiss::IndexFlatL2 assigner(ds.d);
    clus.train(ntrain, ds.xb.data(), assigner);
    return clus.centroids;
}

std::unique_ptr<faiss::IndexFlatL2> make_quantizer(
        int64_t d,
        const std::vector<float>& centroids) {
    auto q = std::make_unique<faiss::IndexFlatL2>(d);
    q->add(centroids.size() / d, centroids.data());
    return q;
}

double recall_at(
        const std::vector<faiss::idx_t>& labels,
        const std::vector<faiss::idx_t>& gt,
        int64_t nq,
        int k,
        int kgt) {
    size_t hit = 0;
    for (int64_t qi = 0; qi < nq; qi++) {
        for (int i = 0; i < k; i++) {
            const faiss::idx_t id = labels[qi * k + i];
            for (int j = 0; j < k; j++) {
                if (id == gt[qi * kgt + j]) {
                    hit++;
                    break;
                }
            }
        }
    }
    return double(hit) / double(nq * k);
}

template <typename IndexT>
void bench_one(
        const char* name,
        IndexT& index,
        const Dataset& ds,
        int k,
        int rounds) {
    std::vector<float> distances(ds.nq * k);
    std::vector<faiss::idx_t> labels(ds.nq * k);

    index.search(ds.nq, ds.xq.data(), k, distances.data(), labels.data());
    const double recall = recall_at(labels, ds.gt, ds.nq, k, ds.kgt);

    double best_s = 1e100;
    for (int r = 0; r < rounds; r++) {
        auto t0 = std::chrono::steady_clock::now();
        index.search(ds.nq, ds.xq.data(), k, distances.data(), labels.data());
        auto t1 = std::chrono::steady_clock::now();
        const double s = std::chrono::duration<double>(t1 - t0).count();
        best_s = std::min(best_s, s);
    }

    std::printf(
            "%s recall@%d=%.6f best_s=%.6f qps=%.3f\n",
            name,
            k,
            recall,
            best_s,
            double(ds.nq) / best_s);
}

} // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        if (args.threads > 0) {
            omp_set_num_threads(args.threads);
        }
        Dataset ds = read_dataset(args.data);
        if (ds.d % 8 != 0) {
            throw std::runtime_error("dimension must be divisible by 8");
        }
        if (args.k > ds.kgt) {
            throw std::runtime_error("k must be <= exported ground-truth k");
        }

        std::printf(
                "loaded nb=%ld nq=%ld d=%ld kgt=%ld nlist=%d nprobe=%d\n",
                long(ds.nb),
                long(ds.nq),
                long(ds.d),
                long(ds.kgt),
                args.nlist,
                args.nprobe);

        auto centroids =
                train_centroids(ds, args.nlist, args.ntrain, args.niter);

        auto q_base = make_quantizer(ds.d, centroids);
        faiss::IndexIVFRaBitQ base(
                q_base.release(),
                ds.d,
                args.nlist,
                faiss::METRIC_INNER_PRODUCT,
                true,
                1);
        base.qb = 0;
        base.nprobe = args.nprobe;
        const int train_n = std::min<int64_t>(args.ntrain, ds.nb);

        base.train(train_n, ds.xb.data());
        base.add(ds.nb, ds.xb.data());

        auto q_lat = make_quantizer(ds.d, centroids);
        faiss::IndexIVFRaBitQLattice lattice(
                q_lat.release(),
                ds.d,
                args.nlist,
                faiss::METRIC_INNER_PRODUCT,
                faiss::rabitq_lattice::LatticeBook::E8_SYM256,
                true);
        lattice.nprobe = args.nprobe;
        lattice.train(train_n, ds.xb.data());
        lattice.add(ds.nb, ds.xb.data());

        bench_one("ivf_rabitq_sign", base, ds, args.k, args.rounds);
        bench_one(
                "ivf_rabitq_lattice_sym256", lattice, ds, args.k, args.rounds);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
