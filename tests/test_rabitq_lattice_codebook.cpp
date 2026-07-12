/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/RaBitQLatticeCodebook.h>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

float dot(const float* a, const float* b, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

float norm2(const float* x, size_t n) {
    return dot(x, x, n);
}

} // namespace

TEST(RaBitQLatticeCodebook, E8Finite256WideShellCounts) {
    const auto& cb = faiss::rabitq_lattice::e8_finite_256_wide_codebook();
    ASSERT_EQ(cb.size(), faiss::rabitq_lattice::kE8Finite256Size);

    size_t n_zero = 0;
    size_t n_root8 = 0;
    size_t n_shell16 = 0;
    for (const auto& c : cb) {
        const float n2 = norm2(c.data(), c.size());
        if (n2 < 0.5f) {
            n_zero++;
        } else if (std::abs(n2 - 8.0f) < 1e-5f) {
            n_root8++;
        } else if (std::abs(n2 - 16.0f) < 1e-5f) {
            n_shell16++;
        }
    }

    EXPECT_EQ(n_zero, 1);
    EXPECT_EQ(n_root8, 240);
    EXPECT_EQ(n_shell16, 15);
}

TEST(RaBitQLatticeCodebook, EncodeDecodeCodebookPoints) {
    const auto& cb = faiss::rabitq_lattice::e8_finite_256_wide_codebook();

    for (size_t i = 0; i < cb.size(); i++) {
        const uint8_t code =
                faiss::rabitq_lattice::encode_e8_finite_256_wide(cb[i].data());
        EXPECT_EQ(code, i);

        float decoded[faiss::rabitq_lattice::kE8ChunkDim];
        faiss::rabitq_lattice::decode_e8_finite_256_wide(code, decoded);
        for (size_t j = 0; j < faiss::rabitq_lattice::kE8ChunkDim; j++) {
            EXPECT_EQ(decoded[j], cb[i][j]);
        }
    }
}

TEST(RaBitQLatticeCodebook, DirectionEncodingProducesUnitDirection) {
    constexpr size_t d = 16;
    const float residual[d] = {
            1.25f,
            -0.75f,
            0.10f,
            2.00f,
            -1.50f,
            0.25f,
            0.90f,
            -0.40f,
            -0.30f,
            1.70f,
            -2.20f,
            0.60f,
            0.35f,
            -1.10f,
            1.45f,
            -0.95f};

    std::vector<uint8_t> codes(d / faiss::rabitq_lattice::kE8ChunkDim);
    std::vector<float> decoded_unit(d);
    const auto stats =
            faiss::rabitq_lattice::encode_e8_finite_256_wide_direction(
                    residual, d, codes.data(), decoded_unit.data());

    EXPECT_GT(stats.norm, 0.0f);
    EXPECT_GT(stats.decoded_norm, 0.0f);
    EXPECT_GT(stats.correlation, 0.0f);
    EXPECT_LE(stats.correlation, 1.0f + 1e-6f);
    EXPECT_NEAR(norm2(decoded_unit.data(), d), 1.0f, 1e-5f);

    // For q == residual, the RaBitQ-style scaling norm/rho makes the
    // codebook direction reproduce ||residual||^2 exactly along that direction.
    const float projected =
            stats.dp_multiplier * dot(residual, decoded_unit.data(), d);
    EXPECT_NEAR(projected, norm2(residual, d), 1e-4f);
}

TEST(RaBitQLatticeCodebook, SignCorrelationMatchesManualFormula) {
    const float residual[8] = {
            1.0f, -2.0f, 0.5f, -0.25f, 3.0f, -4.0f, 0.75f, 2.5f};

    float abs_sum = 0.0f;
    float n2 = 0.0f;
    for (float v : residual) {
        abs_sum += std::abs(v);
        n2 += v * v;
    }
    const float expected = abs_sum / (std::sqrt(n2) * std::sqrt(8.0f));
    EXPECT_NEAR(
            faiss::rabitq_lattice::sign_direction_correlation(residual, 8),
            expected,
            1e-6f);
}
