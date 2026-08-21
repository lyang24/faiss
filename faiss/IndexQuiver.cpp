/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexQuiver.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/popcount.h>

namespace faiss {

namespace {

size_t quiver_plane_size(idx_t d) {
    FAISS_THROW_IF_NOT_MSG(d >= 0, "QuIVer dimension must be non-negative");
    return (static_cast<size_t>(d) + 7) / 8;
}

void encode_one(idx_t d, const float* x, uint8_t* code) {
    const size_t plane_bytes = quiver_plane_size(d);
    uint8_t* positive = code;
    uint8_t* strong = code + plane_bytes;
    std::memset(code, 0, 2 * plane_bytes);

    float abs_sum = 0.0f;
    for (idx_t i = 0; i < d; ++i) {
        abs_sum += std::abs(x[i]);
    }
    const float threshold = d > 0 ? abs_sum / static_cast<float>(d) : 0.0f;

    for (idx_t i = 0; i < d; ++i) {
        const uint8_t bit = static_cast<uint8_t>(1U << (i & 7));
        if (x[i] > 0.0f) {
            positive[i >> 3] |= bit;
        }
        if (std::abs(x[i]) > threshold) {
            strong[i >> 3] |= bit;
        }
    }
}

float code_similarity(const uint8_t* a, const uint8_t* b, size_t plane_bytes) {
    const uint8_t* a_strong = a + plane_bytes;
    const uint8_t* b_strong = b + plane_bytes;
    uint64_t penalty = 0;

    size_t i = 0;
    for (; i + sizeof(uint64_t) <= plane_bytes; i += sizeof(uint64_t)) {
        uint64_t a_pos_word;
        uint64_t b_pos_word;
        uint64_t a_strong_word;
        uint64_t b_strong_word;
        std::memcpy(&a_pos_word, a + i, sizeof(uint64_t));
        std::memcpy(&b_pos_word, b + i, sizeof(uint64_t));
        std::memcpy(&a_strong_word, a_strong + i, sizeof(uint64_t));
        std::memcpy(&b_strong_word, b_strong + i, sizeof(uint64_t));

        const uint64_t sign_diff = a_pos_word ^ b_pos_word;
        const uint64_t any_strong = a_strong_word | b_strong_word;
        const uint64_t both_strong = a_strong_word & b_strong_word;
        penalty += static_cast<uint64_t>(popcount64(sign_diff));
        penalty += static_cast<uint64_t>(popcount64(sign_diff & any_strong));
        penalty += 2ULL *
                static_cast<uint64_t>(popcount64(sign_diff & both_strong));
    }

    for (; i < plane_bytes; ++i) {
        const uint8_t sign_diff = a[i] ^ b[i];
        const uint8_t any_strong = a_strong[i] | b_strong[i];
        const uint8_t both_strong = a_strong[i] & b_strong[i];
        penalty += static_cast<uint64_t>(popcount64(sign_diff));
        penalty += static_cast<uint64_t>(popcount64(sign_diff & any_strong));
        penalty += 2ULL *
                static_cast<uint64_t>(popcount64(sign_diff & both_strong));
    }

    // IndexQuiver uses METRIC_INNER_PRODUCT, whose HNSW ordering expects
    // larger values to be better. Negating the non-negative penalty preserves
    // the QuIVer nearest-neighbor ordering.
    return -static_cast<float>(penalty);
}

struct QuiverDistanceComputer final : FlatCodesDistanceComputer {
    const idx_t d;
    const size_t plane_bytes;
    std::vector<uint8_t> query_code;

    QuiverDistanceComputer(const uint8_t* codes, size_t code_size, idx_t d)
            : FlatCodesDistanceComputer(codes, code_size),
              d(d),
              plane_bytes(quiver_plane_size(d)),
              query_code(code_size) {}

    void set_query(const float* x) override {
        encode_one(d, x, query_code.data());
    }

    float distance_to_code(const uint8_t* code) override {
        return code_similarity(query_code.data(), code, plane_bytes);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return code_similarity(
                codes + static_cast<size_t>(i) * code_size,
                codes + static_cast<size_t>(j) * code_size,
                plane_bytes);
    }
};

struct RunQuiverSearch {
    using T = void;

    template <class BlockResultHandler>
    void f(BlockResultHandler& res,
           const IndexQuiver* index,
           const float* queries) {
        using SingleResultHandler =
                typename BlockResultHandler::SingleResultHandler;

#pragma omp parallel
        {
            std::unique_ptr<FlatCodesDistanceComputer> dc(
                    index->get_FlatCodesDistanceComputer());
            SingleResultHandler resi(res);
#pragma omp for
            for (int64_t q = 0; q < static_cast<int64_t>(res.nq); ++q) {
                resi.begin(q);
                dc->set_query(queries + q * index->d);
                for (idx_t i = 0; i < index->ntotal; ++i) {
                    if (res.is_in_selection(i)) {
                        resi.add_result((*dc)(i), i);
                    }
                }
                resi.end();
            }
        }
    }
};

} // namespace

IndexQuiver::IndexQuiver() = default;

IndexQuiver::IndexQuiver(idx_t d_in, MetricType metric)
        : IndexFlatCodes(2 * quiver_plane_size(d_in), d_in, metric) {
    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_INNER_PRODUCT,
            "IndexQuiver supports only METRIC_INNER_PRODUCT over normalized "
            "cosine embeddings");
    is_trained = true;
}

void IndexQuiver::train(idx_t, const float*) {
    FAISS_THROW_IF_NOT_MSG(
            metric_type == METRIC_INNER_PRODUCT,
            "IndexQuiver supports only METRIC_INNER_PRODUCT");
    is_trained = true;
}

void IndexQuiver::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    FAISS_THROW_IF_NOT(is_trained);
#pragma omp parallel for if (n > 1000)
    for (idx_t i = 0; i < n; ++i) {
        encode_one(d, x + i * d, bytes + static_cast<size_t>(i) * code_size);
    }
}

void IndexQuiver::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    const size_t plane_bytes = plane_size();
    for (idx_t row = 0; row < n; ++row) {
        const uint8_t* code = bytes + static_cast<size_t>(row) * code_size;
        const uint8_t* positive = code;
        const uint8_t* strong = code + plane_bytes;
        float norm2 = 0.0f;
        for (idx_t i = 0; i < d; ++i) {
            const uint8_t bit = static_cast<uint8_t>(1U << (i & 7));
            const float magnitude = (strong[i >> 3] & bit) ? 2.0f : 1.0f;
            x[row * d + i] = (positive[i >> 3] & bit) ? magnitude : -magnitude;
            norm2 += magnitude * magnitude;
        }
        if (norm2 > 0.0f) {
            const float inv_norm = 1.0f / std::sqrt(norm2);
            for (idx_t i = 0; i < d; ++i) {
                x[row * d + i] *= inv_norm;
            }
        }
    }
}

FlatCodesDistanceComputer* IndexQuiver::get_FlatCodesDistanceComputer() const {
    return new QuiverDistanceComputer(codes.data(), code_size, d);
}

uint32_t IndexQuiver::code_distance(idx_t i, idx_t j) const {
    FAISS_THROW_IF_NOT(i >= 0 && i < ntotal && j >= 0 && j < ntotal);
    return static_cast<uint32_t>(-code_similarity(
            codes.data() + static_cast<size_t>(i) * code_size,
            codes.data() + static_cast<size_t>(j) * code_size,
            plane_size()));
}

void IndexQuiver::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(is_trained);
    const IDSelector* sel = params ? params->sel : nullptr;
    RunQuiverSearch run;
    dispatch_knn_ResultHandler(
            n, distances, labels, k, metric_type, sel, run, this, x);
}

void IndexQuiver::range_search(
        idx_t,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(is_trained);
    const IDSelector* sel = params ? params->sel : nullptr;
    RunQuiverSearch run;
    dispatch_range_ResultHandler(
            result, radius, metric_type, sel, run, this, x);
}

size_t IndexQuiver::plane_size() const {
    return quiver_plane_size(d);
}

} // namespace faiss
