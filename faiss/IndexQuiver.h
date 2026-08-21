/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexFlatCodes.h>

namespace faiss {

/** Training-free 2-bit Sign-Magnitude quantization from QuIVer.
 *
 * Each vector is encoded as two bit planes:
 *
 *  - positive[i] = x[i] > 0
 *  - strong[i] = abs(x[i]) > mean(abs(x))
 *
 * The symmetric distance between two codes assigns a penalty only when their
 * signs differ: 1 when both coordinates are weak, 2 when exactly one is
 * strong, and 4 when both are strong. This makes both query-to-code and
 * code-to-code comparisons available using only bitwise operations and
 * popcount. In particular, an IndexHNSW using IndexQuiver as storage builds
 * and searches its graph in the quantized metric space.
 *
 * QuIVer is intended for normalized, cosine-native embeddings. Faiss exposes
 * cosine search as inner product over normalized vectors, so this index only
 * accepts METRIC_INNER_PRODUCT. Distances returned by the quantized index are
 * negative integer penalties (larger is better), not approximate inner
 * products. Wrap it in IndexRefineFlat when true inner-product scores and
 * final full-precision reranking are required.
 *
 * Reference: https://arxiv.org/abs/2605.02171
 */
struct IndexQuiver : IndexFlatCodes {
    IndexQuiver();

    explicit IndexQuiver(idx_t d, MetricType metric = METRIC_INNER_PRODUCT);

    void train(idx_t n, const float* x) override;

    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    /// Number of bytes occupied by one of the two bit planes.
    size_t plane_size() const;
};

} // namespace faiss
