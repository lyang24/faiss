/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIVFRaBitQLattice.h>

#include <faiss/impl/FaissAssert.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace faiss {

namespace {

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
            "IndexIVFRaBitQLattice requires d to be a positive multiple of 8");
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

float fvec_norm2_local(const float* x, idx_t d) {
    float s = 0.0f;
    for (idx_t i = 0; i < d; i++) {
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

float dot_decoded_unit(
        const float* x,
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
            dot += x[offset + k] * chunk[k];
        }
    }
    return dot / decoded_norm;
}

void decode_lattice_residual(
        const uint8_t* code,
        idx_t d,
        LatticeBook book,
        float* residual) {
    const LatticeFactors factors = load_factors(code, d);
    const float decoded_norm = decoded_norm_from_code(code, d, book);
    if (factors.norm == 0.0f || decoded_norm == 0.0f) {
        std::fill(residual, residual + d, 0.0f);
        return;
    }

    const float scale = factors.norm / decoded_norm;
    float chunk[kE8ChunkDim];
    const size_t n_chunks = e8_num_chunks(d);
    for (size_t chunk_id = 0; chunk_id < n_chunks; chunk_id++) {
        decode_e8_lattice_book(code[chunk_id], book, chunk);
        const size_t offset = chunk_id * kE8ChunkDim;
        for (size_t k = 0; k < kE8ChunkDim; k++) {
            residual[offset + k] = chunk[k] * scale;
        }
    }
}

struct IVFRaBitQLatticeScanner : InvertedListScanner {
    const IndexIVFRaBitQLattice* index = nullptr;
    LatticeBook book = LatticeBook::E8_SYM256;
    MetricType metric_type = METRIC_INNER_PRODUCT;
    idx_t d = 0;
    const float* query = nullptr;
    std::vector<float> centroid;
    std::vector<float> query_minus_centroid;
    float query_dot_centroid = 0.0f;
    float query_minus_centroid_norm2 = 0.0f;

    IVFRaBitQLatticeScanner(
            const IndexIVFRaBitQLattice* index_in,
            bool store_pairs,
            const IDSelector* sel)
            : InvertedListScanner(store_pairs, sel),
              index(index_in),
              book(index_in->book),
              metric_type(index_in->metric_type),
              d(index_in->d),
              centroid(d),
              query_minus_centroid(d) {
        keep_max = is_similarity_metric(metric_type);
        code_size = index->code_size;
    }

    void set_query(const float* query_vector) override {
        query = query_vector;
    }

    void set_list(idx_t list_no_in, float /*coarse_dis*/) override {
        list_no = list_no_in;
        index->quantizer->reconstruct(list_no, centroid.data());

        query_dot_centroid = 0.0f;
        query_minus_centroid_norm2 = 0.0f;
        for (idx_t i = 0; i < d; i++) {
            query_dot_centroid += query[i] * centroid[i];
            const float diff = query[i] - centroid[i];
            query_minus_centroid[i] = diff;
            query_minus_centroid_norm2 += diff * diff;
        }
    }

    float distance_to_code(const uint8_t* code) const final {
        const LatticeFactors factors = load_factors(code, d);
        if (factors.norm == 0.0f || factors.dp_multiplier == 0.0f) {
            if (metric_type == METRIC_INNER_PRODUCT) {
                return query_dot_centroid;
            }
            return query_minus_centroid_norm2;
        }

        if (metric_type == METRIC_INNER_PRODUCT) {
            return query_dot_centroid +
                    factors.dp_multiplier *
                    dot_decoded_unit(query, code, d, book);
        }

        const float residual_ip = factors.dp_multiplier *
                dot_decoded_unit(query_minus_centroid.data(), code, d, book);
        return std::max(
                0.0f,
                query_minus_centroid_norm2 + factors.norm * factors.norm -
                        2.0f * residual_ip);
    }
};

} // namespace

IndexIVFRaBitQLattice::IndexIVFRaBitQLattice() = default;

IndexIVFRaBitQLattice::IndexIVFRaBitQLattice(
        Index* quantizer,
        size_t d,
        size_t nlist,
        MetricType metric,
        LatticeBook book_in,
        bool own_invlists)
        : IndexIVF(quantizer, d, nlist, e8_code_size(d), metric, own_invlists),
          book(book_in) {
    by_residual = true;
}

void IndexIVFRaBitQLattice::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(list_nos != nullptr);

    const size_t coarse_size = include_listnos ? coarse_code_size() : 0;
    std::vector<float> centroid(d);
    std::vector<float> residual(d);

    for (idx_t i = 0; i < n; i++) {
        const idx_t list_no = list_nos[i];
        uint8_t* code = codes + i * (code_size + coarse_size);
        if (list_no < 0) {
            std::memset(code, 0, code_size + coarse_size);
            continue;
        }

        if (include_listnos) {
            encode_listno(list_no, code);
            code += coarse_size;
        }

        quantizer->reconstruct(list_no, centroid.data());
        const float* xi = x + i * d;
        for (idx_t j = 0; j < d; j++) {
            residual[j] = xi[j] - centroid[j];
        }

        const auto stats = encode_e8_lattice_book_direction(
                residual.data(), static_cast<size_t>(d), book, code);
        LatticeFactors factors;
        factors.norm = stats.norm;
        factors.dp_multiplier = stats.norm == 0.0f ? 0.0f : stats.dp_multiplier;
        store_factors(code, d, factors);
    }
}

void IndexIVFRaBitQLattice::decode_vectors(
        idx_t n,
        const uint8_t* codes,
        const idx_t* list_nos,
        float* x) const {
    FAISS_THROW_IF_NOT(list_nos != nullptr);
    std::vector<float> centroid(d);
    std::vector<float> residual(d);

    for (idx_t i = 0; i < n; i++) {
        const uint8_t* code = codes + i * code_size;
        float* xi = x + i * d;
        decode_lattice_residual(code, d, book, residual.data());
        quantizer->reconstruct(list_nos[i], centroid.data());
        for (idx_t j = 0; j < d; j++) {
            xi[j] = centroid[j] + residual[j];
        }
    }
}

InvertedListScanner* IndexIVFRaBitQLattice::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters* /*params*/) const {
    return new IVFRaBitQLatticeScanner(this, store_pairs, sel);
}

void IndexIVFRaBitQLattice::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    InvertedLists::ScopedCodes scode(invlists, list_no, offset);
    decode_vectors(1, scode.get(), &list_no, recons);
}

} // namespace faiss
