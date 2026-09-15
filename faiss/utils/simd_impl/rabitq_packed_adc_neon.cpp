/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_neon.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/utils/rabitq_packed_adc.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <type_traits>
#include <utility>

namespace faiss::rabitq_packed_adc {
namespace {

template <int Bits>
int scalar_level(const uint8_t* sign, const uint8_t* low, size_t j) {
    const int s = (sign[j / 8] >> (j % 8)) & 1;
    const int l = rabitq_utils::extract_code_inline(low, j, Bits - 1);
    return (s << (Bits - 1)) + l - (1 << (Bits - 1));
}

// Compile-time document lanes avoid loop/index overhead without relying on
// compiler-specific unroll pragmas.
template <typename F, size_t... I>
inline void lanes(F&& f, std::index_sequence<I...>) {
    (f(std::integral_constant<size_t, I>{}), ...);
}
template <int N, typename F>
inline void lanes(F&& f) {
    lanes(std::forward<F>(f), std::make_index_sequence<N>{});
}
template <int Bits>
inline int8x16_t unpack16(const uint8_t* sign, const uint8_t* low) {
    const uint8x16_t masks = {
            1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    auto sign_bytes = vcombine_u8(vdup_n_u8(sign[0]), vdup_n_u8(sign[1]));
    auto negative_sign_mask = vreinterpretq_s8_u8(vtstq_u8(sign_bytes, masks));
    int8x16_t levels;
    if constexpr (Bits == 2) {
        auto low_bytes = vcombine_u8(vdup_n_u8(low[0]), vdup_n_u8(low[1]));
        auto negative_low_mask =
                vreinterpretq_s8_u8(vtstq_u8(low_bytes, masks));
        levels = vsubq_s8(
                vdupq_n_s8(-2),
                vaddq_s8(
                        vaddq_s8(negative_sign_mask, negative_sign_mask),
                        negative_low_mask));
    } else {
        uint8x16_t table;
        if constexpr (Bits == 4) {
            uint32_t first;
            uint16_t last;
            std::memcpy(&first, low, 4);
            std::memcpy(&last, low + 4, 2);
            uint64_t word = uint64_t(first) | (uint64_t(last) << 32);
            table = vreinterpretq_u8_u64(
                    vcombine_u64(vcreate_u64(word), vdup_n_u64(0)));
        } else if constexpr (Bits == 6) {
            uint64_t first;
            uint16_t last;
            std::memcpy(&first, low, 8);
            std::memcpy(&last, low + 8, 2);
            table = vreinterpretq_u8_u64(
                    vcombine_u64(vcreate_u64(first), vcreate_u64(last)));
        } else {
            static_assert(Bits == 8);
            uint64_t first;
            uint32_t middle;
            uint16_t last;
            std::memcpy(&first, low, 8);
            std::memcpy(&middle, low + 8, 4);
            std::memcpy(&last, low + 12, 2);
            uint64_t upper = uint64_t(middle) | (uint64_t(last) << 32);
            table = vreinterpretq_u8_u64(
                    vcombine_u64(vcreate_u64(first), vcreate_u64(upper)));
        }
        static constexpr auto byte_indices = [] {
            std::array<uint8_t, 16> result{};
            for (int j = 0; j < 16; ++j)
                result[j] = j * (Bits - 1) / 8;
            return result;
        }();
        static constexpr auto shifts = [] {
            std::array<int8_t, 16> result{};
            for (int j = 0; j < 16; ++j)
                result[j] = -(j * (Bits - 1) % 8);
            return result;
        }();
        const auto byte_ids = vld1q_u8(byte_indices.data());
        const auto right = vld1q_s8(shifts.data());
        auto first = vshlq_u8(vqtbl1q_u8(table, byte_ids), right);
        auto second = vshlq_u8(
                vqtbl1q_u8(table, vaddq_u8(byte_ids, vdupq_n_u8(1))),
                vaddq_s8(right, vdupq_n_s8(8)));
        auto lo = vreinterpretq_s8_u8(vandq_u8(
                vorrq_u8(first, second), vdupq_n_u8((1 << (Bits - 1)) - 1)));
        levels = vsubq_s8(
                vaddq_s8(lo, vdupq_n_s8(-(1 << (Bits - 1)))),
                vshlq_n_s8(negative_sign_mask, Bits - 1));
    }
    return levels;
}
template <int Bits, int N>
void packed_dots(
        const int8_t* query,
        const uint8_t* const* codes,
        size_t d,
        int64_t* dots) {
    const size_t offset =
            (d + 7) / 8 + sizeof(rabitq_utils::SignBitFactorsWithError);
    lanes<N>([&](auto n) { dots[n] = 0; });
    size_t j = 0;
    while (j + 16 <= d) {
        // At most 4096 signed-byte products per document before widening.
        const size_t end = std::min(d - (d - j) % 16, j + size_t(4096));
        int32x4_t sums[N];
        lanes<N>([&](auto n) { sums[n] = vdupq_n_s32(0); });
        for (; j + 16 <= end; j += 16) {
            const auto q = vld1q_s8(query + j);
            lanes<N>([&](auto n) {
                const auto z = unpack16<Bits>(
                        codes[n] + j / 8,
                        codes[n] + offset + j * (Bits - 1) / 8);
                sums[n] = vdotq_s32(sums[n], q, z);
            });
        }
        lanes<N>([&](auto n) { dots[n] += vaddlvq_s32(sums[n]); });
    }
    for (; j < d; ++j)
        lanes<N>([&](auto n) {
            dots[n] += int64_t(query[j]) *
                    scalar_level<Bits>(codes[n], codes[n] + offset, j);
        });
}

template <int Bits>
void dispatch_count(
        const int8_t* query,
        const uint8_t* const* codes,
        size_t d,
        int count,
        int64_t* dots) {
    switch (count) {
        case 1:
            packed_dots<Bits, 1>(query, codes, d, dots);
            return;
        case 4:
            packed_dots<Bits, 4>(query, codes, d, dots);
            return;
        case 8:
            packed_dots<Bits, 8>(query, codes, d, dots);
            return;
        default:
            for (int i = 0; i < count; ++i)
                packed_dots<Bits, 1>(query, codes + i, d, dots + i);
    }
}
} // namespace

DotProduct get_dot_product_arm(size_t bits) {
    switch (bits) {
        case 2:
            return dispatch_count<2>;
        case 4:
            return dispatch_count<4>;
        case 6:
            return dispatch_count<6>;
        case 8:
            return dispatch_count<8>;
        default:
            return get_dot_product_scalar(bits);
    }
}
} // namespace faiss::rabitq_packed_adc
