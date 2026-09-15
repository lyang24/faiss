/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/rabitq_packed_adc.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <random>
#include <vector>

TEST(RaBitQPackedADC, IndependentIntegerOracle) {
    std::mt19937 rng(5341);
    const std::array<size_t, 17> dimensions = {
            0,
            1,
            7,
            8,
            9,
            15,
            16,
            17,
            31,
            32,
            33,
            65,
            768,
            960,
            1536,
            4097,
            131137};
    const std::array<int, 8> ids = {8, 0, 7, 2, 8, 3, 5, 1};
    for (int bits = 2; bits <= 8; ++bits) {
        auto selected = faiss::rabitq_packed_adc::get_dot_product(bits);
        auto scalar = faiss::rabitq_packed_adc::get_dot_product_scalar(bits);
        for (size_t d : dimensions) {
            const size_t offset = (d + 7) / 8 + 12;
            const size_t bytes = offset + (d * (bits - 1) + 7) / 8 + 8;
            std::vector<int8_t> query(d);
            // Separate allocations make arbitrary-ID addressing and short
            // final-row reads visible to AddressSanitizer.
            std::array<std::vector<uint8_t>, 9> documents;
            for (auto& doc : documents)
                doc.resize(bytes);
            for (int pattern = 0; pattern < 3; ++pattern) {
                for (auto& q : query)
                    q = pattern == 0 ? int8_t(rng() % 256 - 128) : -128;
                for (auto& doc : documents)
                    for (auto& v : doc)
                        v = pattern == 0 ? uint8_t(rng())
                                         : (pattern == 1 ? 0 : 255);
                const uint8_t* rows[8];
                int64_t expected[8] = {};
                for (int i = 0; i < 8; ++i) {
                    rows[i] = documents[ids[i]].data();
                    for (size_t j = 0; j < d; ++j) {
                        int low = 0;
                        for (int b = 0; b < bits - 1; ++b) {
                            const size_t pos = j * (bits - 1) + b;
                            low |= ((rows[i][offset + pos / 8] >> (pos % 8)) &
                                    1)
                                    << b;
                        }
                        const int sign = (rows[i][j / 8] >> (j % 8)) & 1;
                        const int z =
                                (sign << (bits - 1)) + low - (1 << (bits - 1));
                        expected[i] += int64_t(query[j]) * z;
                    }
                }
                for (int count : {1, 3, 4, 7, 8}) {
                    int64_t actual[8], control[8];
                    selected(query.data(), rows, d, count, actual);
                    scalar(query.data(), rows, d, count, control);
                    for (int i = 0; i < count; ++i) {
                        ASSERT_EQ(actual[i], expected[i])
                                << bits << "/" << d << "/" << pattern << "/"
                                << count;
                        ASSERT_EQ(control[i], expected[i]);
                    }
                }
            }
        }
    }
}

TEST(RaBitQPackedADC, WidenedAccumulationBeyondResearchCaps) {
    for (int bits : {2, 4, 6, 8}) {
        const size_t d = 1048577;
        const size_t bytes = (d + 7) / 8 + 12 + (d * (bits - 1) + 7) / 8 + 8;
        std::vector<int8_t> query(d, -128);
        std::vector<uint8_t> negative(bytes, 0), positive(bytes, 255);
        const uint8_t* rows[4] = {
                negative.data(),
                positive.data(),
                negative.data(),
                positive.data()};
        int64_t actual[4];
        faiss::rabitq_packed_adc::get_dot_product(bits)(
                query.data(), rows, d, 4, actual);
        for (int i = 0; i < 4; ++i) {
            const int level =
                    i % 2 ? (1 << (bits - 1)) - 1 : -(1 << (bits - 1));
            ASSERT_EQ(actual[i], int64_t(-128) * level * int64_t(d));
        }
    }
}
