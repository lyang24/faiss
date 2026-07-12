/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexFlatCodes.h>
#include <faiss/impl/RaBitQLatticeCodebook.h>

namespace faiss {

/**
 * Experimental flat RaBitQ variant backed by an 8D finite-E8 codebook.
 *
 * This index is intentionally separate from IndexRaBitQ. It is a research
 * vehicle for evaluating whether an 8D lattice codebook can improve the
 * matched-rate 1-bit RaBitQ estimator before touching IVF/FastScan storage.
 */
struct IndexRaBitQLattice : IndexFlatCodes {
    IndexRaBitQLattice();

    explicit IndexRaBitQLattice(
            idx_t d,
            MetricType metric = METRIC_INNER_PRODUCT,
            rabitq_lattice::LatticeBook book =
                    rabitq_lattice::LatticeBook::E8_LEX15);

    rabitq_lattice::LatticeBook book = rabitq_lattice::LatticeBook::E8_LEX15;

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
};

} // namespace faiss
