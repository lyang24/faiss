/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexFlatCodes.h>
#include <faiss/impl/RaBitQuantizer.h>

namespace faiss {

enum RaBitQFullCodeMode : uint8_t {
    RABITQ_FULL_CODE_PACKED = 0,
    RABITQ_FULL_CODE_EXPANDED = 1,
    RABITQ_FULL_CODE_INT8 = 2,
    /// Integer query ADC directly over native packed codes; no expanded cache.
    RABITQ_FULL_CODE_PACKED_INT8 = 3,
};

struct RaBitQSearchParameters : SearchParameters {
    uint8_t qb = 4;
    bool centered = false;
};

struct IndexRaBitQ : IndexFlatCodes {
    RaBitQuantizer rabitq;

    // center of all points
    std::vector<float> center;

    // the default number of bits to quantize a query with.
    // use '0' to disable quantization and use raw fp32 values.
    // Note: qb=0 is NOT supported by FastScan variants, which require
    // quantized queries for SIMD lookup table construction.
    uint8_t qb = 4;

    // quantize the query with a zero-centered scalar quantizer.
    bool centered = false;

    /** Optional runtime cache of existing RaBitQ full levels.
     *
     * Each row contains d signed level bytes followed by the original
     * ExtraBitsFactors. Packed codes remain authoritative and unchanged.
     * The cache and selected mode are deliberately not serialized.
     */
    std::vector<uint8_t> expanded_codes;
    RaBitQFullCodeMode full_code_mode = RABITQ_FULL_CODE_PACKED;

    IndexRaBitQ();

    explicit IndexRaBitQ(
            idx_t d,
            MetricType metric = METRIC_L2,
            uint8_t nb_bits = 1);

    void train(idx_t n, const float* x) override;

    void add(idx_t n, const float* x) override;
    void add_sa_codes(idx_t n, const uint8_t* x, const idx_t* xids) override;
    void reset() override;
    size_t remove_ids(const IDSelector& sel) override;
    void merge_from(Index& other_index, idx_t add_id = 0) override;
    void permute_entries(const idx_t* perm);

    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    // returns a quantized-to-qb bits DC if qb > 0
    // returns a default fp32-based DC if qb == 0
    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

    // returns a quantized-to-qb bits DC if qb_in > 0
    // returns a default fp32-based DC if qb_in == 0
    FlatCodesDistanceComputer* get_quantized_distance_computer(
            const uint8_t qb_in,
            bool centered) const;

    /** Select the full-code scorer. Optional modes support L2 and 2..8 total
     * RaBitQ bits. EXPANDED and INT8 derive a d+8 byte cache per vector;
     * PACKED_INT8 uses integer queries without a document cache. Existing
     * modes are not changed automatically based on dimension or bit width.
     * Packed SIMD kernels support 2/4/6/8 bits on dot-product ARM and
     * SPR-dispatched AVX512 CPUs with VBMI; other bit widths/CPUs use a scalar
     * fallback. INT8 remains available when expanded scoring is faster.
     * Calling this method again rebuilds a potentially stale expanded cache.
     */
    void set_full_code_mode(uint8_t mode);

    size_t expanded_code_size() const;
    void rebuild_expanded_codes();
    bool expanded_integer_uses_native_dotprod() const;
    bool packed_integer_uses_native_dotprod() const;

    // Don't rely on sa_decode(), bcz it is good for IP, but not for L2.
    //   As a result, use get_FlatCodesDistanceComputer() for the search.
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
};

} // namespace faiss
