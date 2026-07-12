/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexRaBitQLattice.h>
#include <faiss/impl/RaBitQLatticeCodebook.h>

#include <gtest/gtest.h>

#include <cmath>
#include <random>
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

float codeword_dist(
        const float* x,
        uint8_t code,
        faiss::rabitq_lattice::LatticeBook book) {
    return faiss::rabitq_lattice::e8_lattice_codeword_l2sqr(x, code, book);
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

TEST(RaBitQLatticeCodebook, SelectableBookShellCounts) {
    struct Case {
        faiss::rabitq_lattice::LatticeBook book;
        size_t n_zero;
        size_t n_root8;
        size_t n_shell16;
    };
    const Case cases[] = {
            {faiss::rabitq_lattice::LatticeBook::E8_LEX15, 1, 240, 15},
            {faiss::rabitq_lattice::LatticeBook::E8_SYM256, 0, 240, 16},
            {faiss::rabitq_lattice::LatticeBook::E8_ZERO_AXIS15, 1, 240, 15},
    };

    for (const Case& tc : cases) {
        const auto& cb = faiss::rabitq_lattice::e8_lattice_codebook(tc.book);
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
        EXPECT_EQ(n_zero, tc.n_zero);
        EXPECT_EQ(n_root8, tc.n_root8);
        EXPECT_EQ(n_shell16, tc.n_shell16);
    }
}

TEST(RaBitQLatticeCodebook, SelectableBookChecksumsMatchPythonHarness) {
    struct Case {
        faiss::rabitq_lattice::LatticeBook book;
        uint64_t checksum;
    };
    const Case cases[] = {
            {faiss::rabitq_lattice::LatticeBook::E8_LEX15,
             0x675b798167b452c3ULL},
            {faiss::rabitq_lattice::LatticeBook::E8_SYM256,
             0x30b66a6c34714383ULL},
            {faiss::rabitq_lattice::LatticeBook::E8_ZERO_AXIS15,
             0x54bf4954a1863b43ULL},
    };

    for (const Case& tc : cases) {
        EXPECT_EQ(
                faiss::rabitq_lattice::e8_lattice_codebook_checksum(tc.book),
                tc.checksum);
    }
}

TEST(RaBitQLatticeCodebook, Sym256IsNegationClosed) {
    const auto& cb = faiss::rabitq_lattice::e8_lattice_codebook(
            faiss::rabitq_lattice::LatticeBook::E8_SYM256);
    for (const auto& c : cb) {
        bool found = false;
        for (const auto& other : cb) {
            bool same = true;
            for (size_t i = 0; i < c.size(); i++) {
                same = same && other[i] == -c[i];
            }
            found = found || same;
        }
        EXPECT_TRUE(found);
    }
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

TEST(RaBitQLatticeCodebook, FastEncodeMatchesBruteForceDistance) {
    std::mt19937 rng(123);
    std::normal_distribution<float> normal;
    const faiss::rabitq_lattice::LatticeBook books[] = {
            faiss::rabitq_lattice::LatticeBook::E8_LEX15,
            faiss::rabitq_lattice::LatticeBook::E8_SYM256,
            faiss::rabitq_lattice::LatticeBook::E8_ZERO_AXIS15,
    };

    for (auto book : books) {
        size_t fast_count = 0;
        for (size_t i = 0; i < 10000; i++) {
            float x[faiss::rabitq_lattice::kE8ChunkDim];
            for (float& v : x) {
                v = normal(rng);
            }
            bool used_fast = false;
            const uint8_t fast_code =
                    faiss::rabitq_lattice::encode_e8_lattice_book_fast(
                            x, book, &used_fast);
            const uint8_t brute_code =
                    faiss::rabitq_lattice::encode_e8_lattice_book(x, book);
            fast_count += used_fast ? 1 : 0;
            EXPECT_LE(
                    codeword_dist(x, fast_code, book),
                    codeword_dist(x, brute_code, book) + 1e-5f);
        }
        EXPECT_EQ(fast_count, 10000);
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

TEST(IndexRaBitQLattice, SearchFindsSelfForInnerProductAndL2) {
    constexpr size_t d = 16;
    const std::vector<float> xb = {
            1.25f,  -0.75f, 0.10f,  2.00f,  -1.50f, 0.25f,  0.90f,  -0.40f,
            -0.30f, 1.70f,  -2.20f, 0.60f,  0.35f,  -1.10f, 1.45f,  -0.95f,

            -1.25f, 0.75f,  -0.10f, -2.00f, 1.50f,  -0.25f, -0.90f, 0.40f,
            0.30f,  -1.70f, 2.20f,  -0.60f, -0.35f, 1.10f,  -1.45f, 0.95f,

            0.20f,  0.30f,  -0.10f, 0.50f,  -0.70f, 0.40f,  0.10f,  -0.20f,
            0.05f,  0.15f,  -0.25f, 0.35f,  0.45f,  -0.55f, 0.65f,  -0.75f};
    const float* query = xb.data();

    {
        faiss::IndexRaBitQLattice index(
                d,
                faiss::METRIC_INNER_PRODUCT,
                faiss::rabitq_lattice::LatticeBook::E8_SYM256);
        index.add(3, xb.data());

        float dis = 0.0f;
        faiss::idx_t label = -1;
        index.search(1, query, 1, &dis, &label);
        EXPECT_EQ(label, 0);
        EXPECT_NEAR(dis, norm2(query, d), 1e-4f);
    }

    {
        faiss::IndexRaBitQLattice index(
                d,
                faiss::METRIC_L2,
                faiss::rabitq_lattice::LatticeBook::E8_SYM256);
        index.add(3, xb.data());

        float dis = -1.0f;
        faiss::idx_t label = -1;
        index.search(1, query, 1, &dis, &label);
        EXPECT_EQ(label, 0);
        EXPECT_NEAR(dis, 0.0f, 1e-4f);
    }
}
