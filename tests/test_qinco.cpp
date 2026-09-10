/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <faiss/IndexHNSW.h>
#include <faiss/IndexQINCo.h>
#include <faiss/impl/DistanceComputer.h>

namespace {

std::vector<float> random_matrix(int n, int d, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal;
    std::vector<float> x(size_t(n) * d);
    for (float& v : x) {
        v = normal(rng);
    }
    return x;
}

float float_query_oracle(
        const faiss::IndexQINCo& index,
        const float* query,
        faiss::idx_t id) {
    std::vector<float> reconstructed(index.d);
    index.reconstruct(id, reconstructed.data());
    float qnorm = 0.0f;
    const float xnorm = index.l2_norms[id];
    float dot = 0.0f;
    for (int j = 0; j < index.d; ++j) {
        qnorm += query[j] * query[j];
        dot += query[j] * reconstructed[j];
    }
    return xnorm + qnorm - 2.0f * dot;
}

float integer_equation_oracle(
        const faiss::IndexQINCo& index,
        const float* query,
        faiss::idx_t id) {
    float qnorm = 0.0f;
    float dot = 0.0f;
    for (int j = 0; j < index.d; ++j) {
        qnorm += query[j] * query[j];
        dot += index.bias[j] * query[j];
    }
    const auto* code = index.codes.data() + size_t(id) * index.code_size;
    for (size_t begin = 0; begin < index.padded_d; begin += index.block_size) {
        const size_t end = std::min(begin + index.block_size, size_t(index.d));
        float max_abs = 0.0f;
        for (size_t j = begin; j < end; ++j) {
            max_abs = std::max(max_abs, std::abs(index.scale[j] * query[j]));
        }
        const int qmax = index.query_bits == 6 ? 63 : 127;
        const float qs = max_abs > 0.0f ? max_abs / float(qmax) : 0.0f;
        int32_t idot = 0;
        if (qs > 0.0f) {
            for (size_t j = begin; j < end; ++j) {
                int qcode = int(std::lrint(index.scale[j] * query[j] / qs));
                qcode = std::max(-qmax, std::min(qmax, qcode));
                int doc_code;
                if (index.doc_bits == 4 && j < index.int8_head_dims) {
                    doc_code = int(reinterpret_cast<const int8_t*>(code)[j]);
                } else if (index.doc_bits == 4) {
                    const size_t tail = j - index.int8_head_dims;
                    doc_code =
                            int((code[index.int8_head_dims + tail / 2] >>
                                 (4 * (tail & 1))) &
                                0x0f);
                } else if (index.doc_bits == 7 || index.query_bits == 6) {
                    doc_code = int(code[j]);
                } else {
                    doc_code = int(reinterpret_cast<const int8_t*>(code)[j]);
                }
                idot += doc_code * qcode;
            }
        }
        dot += qs * float(idot);
    }
    return index.l2_norms[id] + qnorm - 2.0f * dot;
}

} // namespace

TEST(QINCo, EncodeDecodeAndEdgeCases) {
    constexpr int d = 65;
    std::vector<float> xb(4 * d, 0.0f);
    for (int j = 0; j < d; ++j) {
        xb[d + j] = float(j - 30);
        xb[2 * d + j] = float(2 * j + 1);
        xb[3 * d + j] = -float(j);
    }
    faiss::IndexQINCo index(d, 32);
    index.train(4, xb.data());
    index.add(4, xb.data());
    EXPECT_EQ(index.code_size, 96);
    EXPECT_EQ(index.bytes_per_vector(), 100);

    std::vector<float> decoded(4 * d);
    index.reconstruct_n(0, 4, decoded.data());
    for (float v : decoded) {
        EXPECT_TRUE(std::isfinite(v));
    }

    std::vector<float> zero(d, 0.0f);
    std::unique_ptr<faiss::DistanceComputer> dc(index.get_distance_computer());
    dc->set_query(zero.data());
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR((*dc)(i), index.l2_norms[i], 1e-5);
    }
}

TEST(QINCo, SimdMatchesIndependentFloatReconstructionOracle) {
    constexpr int n = 117;
    constexpr int d = 173;
    auto xb = random_matrix(n, d, 123);
    auto xq = random_matrix(7, d, 456);
    faiss::IndexQINCo index(d, 64);
    index.train(n, xb.data());
    index.add(n, xb.data());

    const std::vector<faiss::idx_t> ids = {116, 3, 88, 3, 0, 52, 101};
    std::vector<float> got(7 * ids.size());
    std::vector<faiss::idx_t> labels(got.size());
    for (size_t i = 0; i < labels.size(); ++i) {
        labels[i] = ids[(i * 5 + i / ids.size()) % ids.size()];
    }
    index.compute_distance_subset(
            7, xq.data(), ids.size(), got.data(), labels.data());

    float max_query_quantization_error = 0.0f;
    for (int qi = 0; qi < 7; ++qi) {
        for (size_t j = 0; j < ids.size(); ++j) {
            const auto id = labels[qi * ids.size() + j];
            const float expected =
                    integer_equation_oracle(index, xq.data() + qi * d, id);
            EXPECT_NEAR(got[qi * ids.size() + j], expected, 1e-4f);
            const float float_query =
                    float_query_oracle(index, xq.data() + qi * d, id);
            max_query_quantization_error = std::max(
                    max_query_quantization_error,
                    std::abs(got[qi * ids.size() + j] - float_query));
        }
    }
    EXPECT_LT(max_query_quantization_error, 0.5f);

    std::unique_ptr<faiss::FlatCodesDistanceComputer> dc(
            index.get_FlatCodesDistanceComputer());
    dc->set_query(xq.data());
    float d0, d1, d2, d3;
    dc->distances_batch_4(116, 3, 88, 0, d0, d1, d2, d3);
    EXPECT_FLOAT_EQ(d0, (*dc)(116));
    EXPECT_FLOAT_EQ(d1, (*dc)(3));
    EXPECT_FLOAT_EQ(d2, (*dc)(88));
    EXPECT_FLOAT_EQ(d3, (*dc)(0));
}

TEST(QINCo, ClonesExactFp32Graph) {
    constexpr int n = 200;
    constexpr int d = 32;
    auto xb = random_matrix(n, d, 789);
    faiss::IndexHNSWFlat fp(d, 8);
    fp.hnsw.efConstruction = 40;
    fp.add(n, xb.data());

    std::unique_ptr<faiss::IndexHNSW> compressed(
            faiss::clone_hnsw_with_qinco_storage(fp, n, xb.data(), 32));
    EXPECT_EQ(compressed->hnsw.entry_point, fp.hnsw.entry_point);
    EXPECT_EQ(compressed->hnsw.levels, fp.hnsw.levels);
    ASSERT_EQ(compressed->hnsw.neighbors.size(), fp.hnsw.neighbors.size());
    EXPECT_TRUE(
            std::equal(
                    compressed->hnsw.neighbors.begin(),
                    compressed->hnsw.neighbors.end(),
                    fp.hnsw.neighbors.begin()));
}

TEST(QINCo, SevenBitMaddubsMatchesScalarOracle) {
    constexpr int n = 71;
    constexpr int d = 96;
    auto xb = random_matrix(n, d, 321);
    auto xq = random_matrix(3, d, 654);
    faiss::IndexQINCo index(d, 32, 0.0f, 7);
    index.train(n, xb.data());
    index.add(n, xb.data());
    std::unique_ptr<faiss::DistanceComputer> dc(index.get_distance_computer());
    for (int qi = 0; qi < 3; ++qi) {
        dc->set_query(xq.data() + qi * d);
        for (faiss::idx_t id :
             {faiss::idx_t(70),
              faiss::idx_t(0),
              faiss::idx_t(33),
              faiss::idx_t(1)}) {
            EXPECT_NEAR(
                    (*dc)(id),
                    integer_equation_oracle(index, xq.data() + qi * d, id),
                    1e-4f);
        }
    }
}

TEST(QINCo, UnsignedEightBitSixBitQueryMatchesScalarOracle) {
    constexpr int n = 73;
    constexpr int d = 128;
    auto xb = random_matrix(n, d, 987);
    auto xq = random_matrix(2, d, 876);
    faiss::IndexQINCo index(d, 64, 0.0f, 8, 6);
    index.train(n, xb.data());
    index.add(n, xb.data());
    std::unique_ptr<faiss::FlatCodesDistanceComputer> dc(
            index.get_FlatCodesDistanceComputer());
    for (int qi = 0; qi < 2; ++qi) {
        dc->set_query(xq.data() + qi * d);
        float d0, d1, d2, d3;
        dc->distances_batch_4(72, 2, 55, 0, d0, d1, d2, d3);
        EXPECT_NEAR(
                d0,
                integer_equation_oracle(index, xq.data() + qi * d, 72),
                1e-4f);
        EXPECT_NEAR(
                d1,
                integer_equation_oracle(index, xq.data() + qi * d, 2),
                1e-4f);
        EXPECT_NEAR(
                d2,
                integer_equation_oracle(index, xq.data() + qi * d, 55),
                1e-4f);
        EXPECT_NEAR(
                d3,
                integer_equation_oracle(index, xq.data() + qi * d, 0),
                1e-4f);
    }
}

TEST(QINCo, PackedFourBitMatchesScalarOracle) {
    constexpr int n = 79;
    constexpr int d = 97;
    auto xb = random_matrix(n, d, 135);
    auto xq = random_matrix(2, d, 246);
    faiss::IndexQINCo index(d, 32, 0.0f, 4, 8);
    index.train(n, xb.data());
    index.add(n, xb.data());
    EXPECT_EQ(index.padded_d, 128);
    EXPECT_EQ(index.code_size, 64);
    std::unique_ptr<faiss::DistanceComputer> dc(index.get_distance_computer());
    for (int qi = 0; qi < 2; ++qi) {
        dc->set_query(xq.data() + qi * d);
        for (faiss::idx_t id :
             {faiss::idx_t(78),
              faiss::idx_t(4),
              faiss::idx_t(0),
              faiss::idx_t(61)}) {
            EXPECT_NEAR(
                    (*dc)(id),
                    integer_equation_oracle(index, xq.data() + qi * d, id),
                    1e-4f);
        }
    }
}

TEST(QINCo, MixedInt8HeadInt4TailMatchesScalarOracle) {
    constexpr int n = 83;
    constexpr int d = 160;
    auto xb = random_matrix(n, d, 975);
    auto xq = random_matrix(2, d, 864);
    faiss::IndexQINCo index(d, 64, 0.0f, 4, 8, 64);
    index.train(n, xb.data());
    index.add(n, xb.data());
    EXPECT_EQ(index.padded_d, 192);
    EXPECT_EQ(index.code_size, 128);
    std::unique_ptr<faiss::FlatCodesDistanceComputer> dc(
            index.get_FlatCodesDistanceComputer());
    for (int qi = 0; qi < 2; ++qi) {
        dc->set_query(xq.data() + qi * d);
        float d0, d1, d2, d3;
        dc->distances_batch_4(82, 7, 41, 0, d0, d1, d2, d3);
        const float got[] = {d0, d1, d2, d3};
        const faiss::idx_t ids[] = {82, 7, 41, 0};
        for (int i = 0; i < 4; ++i) {
            EXPECT_NEAR(
                    got[i],
                    integer_equation_oracle(index, xq.data() + qi * d, ids[i]),
                    1e-4f);
        }
    }
}
