/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexIVF.h>
#include <faiss/impl/RaBitQLatticeCodebook.h>

namespace faiss {

/**
 * Experimental IVF RaBitQ variant backed by an 8D E8-style lattice codebook.
 *
 * This is a research prototype for residual-domain recall experiments. It is
 * intentionally not wired into index_factory or serialization yet.
 */
struct IndexIVFRaBitQLattice : IndexIVF {
    IndexIVFRaBitQLattice();

    IndexIVFRaBitQLattice(
            Index* quantizer,
            size_t d,
            size_t nlist,
            MetricType metric = METRIC_INNER_PRODUCT,
            rabitq_lattice::LatticeBook book =
                    rabitq_lattice::LatticeBook::E8_SYM256,
            bool own_invlists = true);

    rabitq_lattice::LatticeBook book = rabitq_lattice::LatticeBook::E8_SYM256;

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const override;

    void decode_vectors(
            idx_t n,
            const uint8_t* codes,
            const idx_t* list_nos,
            float* x) const override;

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs = false,
            const IDSelector* sel = nullptr,
            const IVFSearchParameters* params = nullptr) const override;

    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;
};

} // namespace faiss
