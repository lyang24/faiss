/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/RaBitQUtils.h>
#include <faiss/utils/rabitq_packed_adc.h>
#include <immintrin.h>
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
inline __m256i unpack32(const uint8_t* sign, const uint8_t* low) {
    uint32_t signs;
    std::memcpy(&signs, sign, sizeof(signs));
    const auto sign_mask = _mm256_movm_epi8(static_cast<__mmask32>(signs));
    __m256i lo;
    if constexpr (Bits == 2) {
        uint32_t lows;
        std::memcpy(&lows, low, sizeof(lows));
        lo = _mm256_and_si256(
                _mm256_movm_epi8(static_cast<__mmask32>(lows)),
                _mm256_set1_epi8(1));
    } else {
        static_assert(Bits == 4 || Bits == 6 || Bits == 8);
        constexpr int K = Bits - 1;
        constexpr __mmask32 valid_bytes = (uint32_t(1) << (4 * K)) - 1;
        const auto packed = _mm256_maskz_loadu_epi8(valid_bytes, low);
        // Each group of eight coordinates occupies exactly K bytes. Place
        // those bytes into one 64-bit lane; pad with masked-off byte 31.
        static constexpr auto offsets = [] {
            std::array<uint8_t, 32> a{};
            for (int j = 0; j < 32; ++j)
                a[j] = (j % 8 < K) ? (j / 8) * K + j % 8 : 31;
            return a;
        }();
        static constexpr auto shifts = [] {
            std::array<uint8_t, 32> a{};
            for (int j = 0; j < 32; ++j)
                a[j] = (j % 8) * K;
            return a;
        }();
        const auto lanes = _mm256_permutexvar_epi8(
                _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(offsets.data())),
                packed);
        lo = _mm256_and_si256(
                _mm256_multishift_epi64_epi8(
                        _mm256_loadu_si256(
                                reinterpret_cast<const __m256i*>(
                                        shifts.data())),
                        lanes),
                _mm256_set1_epi8((1 << K) - 1));
    }
    // z = low - 2^(Bits-1) when sign=0, otherwise z=low.
    // Byte wraparound gives the exact int8 level, including -128 at 8 bits.
    const auto offset = _mm256_andnot_si256(
            sign_mask, _mm256_set1_epi8(static_cast<char>(1 << (Bits - 1))));
    return _mm256_sub_epi8(lo, offset);
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
    while (j + 32 <= d) {
        const size_t end = std::min(d - (d - j) % 32, j + size_t(4096));
        alignas(64) __m512i sums[N];
        lanes<N>([&](auto n) { sums[n] = _mm512_setzero_si512(); });
        for (; j + 32 <= end; j += 32) {
            const auto q = _mm512_cvtepi8_epi16(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(query + j)));
            lanes<N>([&](auto n) {
                const auto z = _mm512_cvtepi8_epi16(
                        unpack32<Bits>(
                                codes[n] + j / 8,
                                codes[n] + offset + j * (Bits - 1) / 8));
                sums[n] = _mm512_dpwssd_epi32(sums[n], q, z);
            });
        }
        lanes<N>([&](auto n) {
            const auto lo =
                    _mm512_cvtepi32_epi64(_mm512_castsi512_si256(sums[n]));
            const auto hi = _mm512_cvtepi32_epi64(
                    _mm512_extracti64x4_epi64(sums[n], 1));
            dots[n] +=
                    _mm512_reduce_add_epi64(lo) + _mm512_reduce_add_epi64(hi);
        });
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

DotProduct get_dot_product_avx512(size_t bits) {
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
