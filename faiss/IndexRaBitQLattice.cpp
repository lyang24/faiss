/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexRaBitQLattice.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/RaBitQLatticeCodebook.h>
#include <faiss/impl/ResultHandler.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace faiss {

namespace {

using rabitq_lattice::decode_e8_finite_256_wide;
using rabitq_lattice::decode_e8_lattice_book;
using rabitq_lattice::encode_e8_lattice_book_direction;
using rabitq_lattice::kE8ChunkDim;
using rabitq_lattice::LatticeBook;

struct LatticeFactors {
    float norm = 0.0f;
    float dp_multiplier = 0.0f;
};

size_t e8_num_chunks(idx_t d) {
    FAISS_THROW_IF_NOT_MSG(
            d > 0 && d % kE8ChunkDim == 0,
            "IndexRaBitQLattice requires d to be a positive multiple of 8");
    return static_cast<size_t>(d) / kE8ChunkDim;
}

size_t e8_code_size(idx_t d) {
    return e8_num_chunks(d) + sizeof(LatticeFactors);
}

LatticeFactors load_factors(const uint8_t* code, idx_t d) {
    LatticeFactors factors;
    std::memcpy(&factors, code + e8_num_chunks(d), sizeof(factors));
    return factors;
}

void store_factors(uint8_t* code, idx_t d, const LatticeFactors& factors) {
    std::memcpy(code + e8_num_chunks(d), &factors, sizeof(factors));
}

float fvec_norm_L2sqr_local(const float* x, size_t d) {
    float s = 0.0f;
    for (size_t i = 0; i < d; i++) {
        s += x[i] * x[i];
    }
    return s;
}

float decoded_norm_from_code(const uint8_t* code, idx_t d, LatticeBook book) {
    float norm2 = 0.0f;
    float chunk[kE8ChunkDim];
    const size_t n_chunks = e8_num_chunks(d);
    for (size_t chunk_id = 0; chunk_id < n_chunks; chunk_id++) {
        decode_e8_lattice_book(code[chunk_id], book, chunk);
        for (size_t k = 0; k < kE8ChunkDim; k++) {
            norm2 += chunk[k] * chunk[k];
        }
    }
    return std::sqrt(norm2);
}

float dot_query_decoded_unit(
        const float* q,
        const uint8_t* code,
        idx_t d,
        LatticeBook book) {
    const float decoded_norm = decoded_norm_from_code(code, d, book);
    if (decoded_norm == 0.0f) {
        return 0.0f;
    }

    float dot = 0.0f;
    float chunk[kE8ChunkDim];
    const size_t n_chunks = e8_num_chunks(d);
    for (size_t chunk_id = 0; chunk_id < n_chunks; chunk_id++) {
        decode_e8_lattice_book(code[chunk_id], book, chunk);
        const size_t offset = chunk_id * kE8ChunkDim;
        for (size_t k = 0; k < kE8ChunkDim; k++) {
            dot += q[offset + k] * chunk[k];
        }
    }
    return dot / decoded_norm;
}

void decode_lattice_vector(
        const uint8_t* code,
        idx_t d,
        LatticeBook book,
        float* x) {
    const LatticeFactors factors = load_factors(code, d);
    const float decoded_norm = decoded_norm_from_code(code, d, book);
    if (factors.norm == 0.0f || decoded_norm == 0.0f) {
        std::fill(x, x + d, 0.0f);
        return;
    }

    const float scale = factors.norm / decoded_norm;
    float chunk[kE8ChunkDim];
    const size_t n_chunks = e8_num_chunks(d);
    for (size_t chunk_id = 0; chunk_id < n_chunks; chunk_id++) {
        decode_e8_lattice_book(code[chunk_id], book, chunk);
        const size_t offset = chunk_id * kE8ChunkDim;
        for (size_t k = 0; k < kE8ChunkDim; k++) {
            x[offset + k] = chunk[k] * scale;
        }
    }
}

struct LatticeDistanceComputer : FlatCodesDistanceComputer {
    idx_t d = 0;
    MetricType metric_type = METRIC_INNER_PRODUCT;
    LatticeBook book = LatticeBook::E8_LEX15;
    float query_norm2 = 0.0f;

    LatticeDistanceComputer(idx_t d, MetricType metric_type, LatticeBook book)
            : d(d), metric_type(metric_type), book(book) {}

    void set_query(const float* x) override {
        q = x;
        query_norm2 = fvec_norm_L2sqr_local(q, d);
    }

    float estimated_inner_product(const uint8_t* code) const {
        const LatticeFactors factors = load_factors(code, d);
        if (factors.norm == 0.0f || factors.dp_multiplier == 0.0f) {
            return 0.0f;
        }
        return factors.dp_multiplier * dot_query_decoded_unit(q, code, d, book);
    }

    float distance_to_code(const uint8_t* code) override {
        const float ip = estimated_inner_product(code);
        if (metric_type == METRIC_INNER_PRODUCT) {
            return ip;
        }
        const LatticeFactors factors = load_factors(code, d);
        return std::max(
                0.0f, query_norm2 + factors.norm * factors.norm - 2.0f * ip);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        std::vector<float> xi(d);
        std::vector<float> xj(d);
        decode_lattice_vector(codes + i * code_size, d, book, xi.data());
        decode_lattice_vector(codes + j * code_size, d, book, xj.data());

        float ip = 0.0f;
        float l2 = 0.0f;
        for (idx_t k = 0; k < d; k++) {
            ip += xi[k] * xj[k];
            const float diff = xi[k] - xj[k];
            l2 += diff * diff;
        }
        return metric_type == METRIC_INNER_PRODUCT ? ip : l2;
    }
};

} // namespace

namespace {

struct Run_search_lattice_dc {
    using T = void;

    template <class BlockResultHandler>
    void f(BlockResultHandler& res,
           const IndexRaBitQLattice* index,
           const float* xq) {
        using SingleResultHandler =
                typename BlockResultHandler::SingleResultHandler;
        const size_t ntotal = index->ntotal;
        const idx_t d = index->d;
        std::exception_ptr ex;
        std::atomic<bool> interrupted{false};

#pragma omp parallel
        {
            std::unique_ptr<FlatCodesDistanceComputer> dc;
            std::unique_ptr<SingleResultHandler> resi;
            try {
                dc.reset(index->get_FlatCodesDistanceComputer());
                resi = std::make_unique<SingleResultHandler>(res);
            } catch (...) {
#pragma omp critical
                {
                    if (!ex) {
                        ex = std::current_exception();
                    }
                    interrupted.store(true, std::memory_order_relaxed);
                }
            }

#pragma omp for
            for (int64_t q = 0; q < static_cast<int64_t>(res.nq); q++) {
                if (interrupted.load(std::memory_order_relaxed)) {
                    continue;
                }
                try {
                    resi->begin(q);
                    dc->set_query(xq + d * q);
                    for (size_t i = 0; i < ntotal; i++) {
                        if (res.is_in_selection(i)) {
                            resi->add_result((*dc)(i), i);
                        }
                    }
                    resi->end();
                } catch (...) {
#pragma omp critical
                    {
                        if (!ex) {
                            ex = std::current_exception();
                        }
                        interrupted.store(true, std::memory_order_relaxed);
                    }
                }
            }
        }
        if (ex) {
            std::rethrow_exception(ex);
        }
    }
};

} // namespace

IndexRaBitQLattice::IndexRaBitQLattice() = default;

IndexRaBitQLattice::IndexRaBitQLattice(
        idx_t d_in,
        MetricType metric,
        LatticeBook book_in)
        : IndexFlatCodes(e8_code_size(d_in), d_in, metric), book(book_in) {
    is_trained = true;
}

void IndexRaBitQLattice::train(idx_t /*n*/, const float* /*x*/) {
    FAISS_THROW_IF_NOT(d % kE8ChunkDim == 0);
    is_trained = true;
}

void IndexRaBitQLattice::sa_encode(idx_t n, const float* x, uint8_t* bytes)
        const {
    FAISS_THROW_IF_NOT(is_trained);
    const size_t n_chunks = e8_num_chunks(d);

    for (idx_t i = 0; i < n; i++) {
        uint8_t* code = bytes + i * code_size;
        const auto stats = encode_e8_lattice_book_direction(
                x + i * d, static_cast<size_t>(d), book, code);
        LatticeFactors factors;
        factors.norm = stats.norm;
        factors.dp_multiplier = stats.norm == 0.0f ? 0.0f : stats.dp_multiplier;
        store_factors(code, d, factors);

        FAISS_ASSERT(n_chunks + sizeof(LatticeFactors) == code_size);
    }
}

void IndexRaBitQLattice::sa_decode(idx_t n, const uint8_t* bytes, float* x)
        const {
    FAISS_THROW_IF_NOT(is_trained);
    for (idx_t i = 0; i < n; i++) {
        decode_lattice_vector(bytes + i * code_size, d, book, x + i * d);
    }
}

FlatCodesDistanceComputer* IndexRaBitQLattice::get_FlatCodesDistanceComputer()
        const {
    auto dc = std::make_unique<LatticeDistanceComputer>(d, metric_type, book);
    dc->code_size = code_size;
    dc->codes = codes.data();
    return dc.release();
}

void IndexRaBitQLattice::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(is_trained);
    const IDSelector* sel = params ? params->sel : nullptr;
    Run_search_lattice_dc r;
    dispatch_knn_ResultHandler(
            n, distances, labels, k, metric_type, sel, r, this, x);
}

} // namespace faiss
