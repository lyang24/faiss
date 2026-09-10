/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <faiss/IndexFlatCodes.h>

namespace faiss {

struct IndexHNSW;

/** Query-int8, independently addressable scalar-quantized storage.
 *
 * Documents use one trained affine int8 code per dimension. Queries are
 * transformed to int8 in contiguous blocks. For L2, exact FP32 document norms
 * are retained and only the dot product is approximated. This is intended as
 * random-access storage for an HNSW graph built with FP32 distances.
 */
struct IndexQINCo : IndexFlatCodes {
    size_t block_size = 64;
    size_t padded_d = 0;
    float exact_norm_weight = 0.0f;
    int doc_bits = 8; // supported: 4, 7, 8
    int query_bits = 8;
    // With doc_bits=4, store this aligned PCA head as signed int8 and pack the
    // remaining tail as uint4. Zero retains the uniform packed-int4 format.
    size_t int8_head_dims = 0;

    // x[j] ~= bias[j] + scale[j] * signed_code[j]
    std::vector<float> scale;
    std::vector<float> bias;
    // Weighted blend of reconstruction and exact FP32 norms.
    std::vector<float> l2_norms;

    IndexQINCo();
    explicit IndexQINCo(
            int d,
            size_t block_size = 64,
            float exact_norm_weight = 0.0f,
            int doc_bits = 8,
            int query_bits = 8,
            size_t int8_head_dims = 0);

    void train(idx_t n, const float* x) override;
    void add(idx_t n, const float* x) override;
    void reset() override;

    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

    /** Score arbitrary IDs, preserving the order and repetitions in labels. */
    void compute_distance_subset(
            idx_t n,
            const float* queries,
            idx_t k,
            float* distances,
            const idx_t* labels) const;

    size_t bytes_per_vector() const;
};

/** Copy an existing HNSW graph exactly, replacing only its vector storage.
 * The returned index owns its IndexQINCo storage. The source graph is not
 * modified, so it can be used for paired scorer comparisons.
 */
IndexHNSW* clone_hnsw_with_qinco_storage(
        const IndexHNSW& fp32_graph,
        idx_t n,
        const float* x,
        size_t block_size = 64,
        float exact_norm_weight = 0.0f,
        int doc_bits = 8,
        int query_bits = 8,
        size_t int8_head_dims = 0);

} // namespace faiss
