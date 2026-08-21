/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexNSG.h>
#include <faiss/IndexQuiver.h>

namespace faiss {

struct SearchParametersQuiverVamana : SearchParameters {
    /// Beam width. Values <= 0 use IndexQuiverVamana::search_ef.
    int search_ef = 0;
};

/** Single-level Vamana graph built and searched entirely with QuIVer's
 *  symmetric 2-bit distance.
 *
 * `m` follows the QuIVer paper's convention: the stored maximum degree is
 * `2 * m`. Construction uses a deterministic random graph, BQ beam-search
 * candidates, alpha-diversity RobustPrune, bidirectional links, and a final
 * degree-convergence pass. The index returns negative integer BQ penalties.
 * Wrap it in IndexRefineFlat and request `ef / k` times as many candidates to
 * reproduce QuIVer's full-precision final reranking stage.
 *
 * The first implementation intentionally supports one-shot construction only.
 */
struct IndexQuiverVamana : IndexNSG {
    int m = 32;
    int construction_ef = 128;
    int search_ef = 64;
    float alpha = 1.2f;
    uint64_t random_seed = 12345;

    IndexQuiverVamana();

    explicit IndexQuiverVamana(int d, int m = 32);

    void add(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void reset() override;

    /// Number of outgoing neighbors currently stored for a node.
    int get_degree(idx_t i) const;
};

} // namespace faiss
