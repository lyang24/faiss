/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexQINCo.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>

#ifdef __AVX2__
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

#include <faiss/IndexHNSW.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {
namespace {

size_t qinco_code_size(
        int d,
        size_t block_size,
        int doc_bits,
        size_t int8_head_dims) {
    const size_t padded =
            ((size_t(d) + block_size - 1) / block_size) * block_size;
    if (doc_bits == 4 && int8_head_dims > 0) {
        return int8_head_dims + (padded - int8_head_dims) / 2;
    }
    return padded * size_t(doc_bits == 4 ? 1 : 2) / 2;
}

inline int32_t dot_i8_scalar(const int8_t* a, const int8_t* b, size_t n) {
    int32_t sum = 0;
    for (size_t j = 0; j < n; ++j) {
        sum += int32_t(a[j]) * int32_t(b[j]);
    }
    return sum;
}

#ifdef __AVX2__
inline int32_t hsum256_epi32(__m256i x) {
    __m128i s = _mm_add_epi32(
            _mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

inline int32_t dot_i8_avx2(const int8_t* a, const int8_t* b, size_t n) {
    __m256i acc = _mm256_setzero_si256();
    size_t j = 0;
    for (; j + 32 <= n; j += 32) {
        const __m256i va =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
        const __m256i vb =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));
        const __m256i alo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
        const __m256i ahi =
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        const __m256i blo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
        const __m256i bhi =
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(alo, blo));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(ahi, bhi));
    }
    int32_t sum = hsum256_epi32(acc);
    return sum + dot_i8_scalar(a + j, b + j, n - j);
}
#endif

#if defined(__aarch64__)
inline int32_t dot_i8_neon(const int8_t* a, const int8_t* b, size_t n) {
    int32x4_t acc = vdupq_n_s32(0);
    size_t j = 0;
    for (; j + 16 <= n; j += 16) {
        const int8x16_t va = vld1q_s8(a + j);
        const int8x16_t vb = vld1q_s8(b + j);
#if defined(__ARM_FEATURE_DOTPROD)
        acc = vdotq_s32(acc, va, vb);
#else
        const int16x8_t lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
        const int16x8_t hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
        acc = vaddq_s32(acc, vpaddlq_s16(lo));
        acc = vaddq_s32(acc, vpaddlq_s16(hi));
#endif
    }
    return vaddvq_s32(acc) + dot_i8_scalar(a + j, b + j, n - j);
}
#endif

inline int32_t dot_i8(const int8_t* a, const int8_t* b, size_t n) {
#ifdef __AVX2__
    return dot_i8_avx2(a, b, n);
#elif defined(__aarch64__)
    return dot_i8_neon(a, b, n);
#else
    return dot_i8_scalar(a, b, n);
#endif
}

inline int32_t dot_unsigned_signed_bounded(
        const uint8_t* a,
        const int8_t* b,
        size_t n) {
#ifdef __AVX2__
    __m256i acc = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    for (size_t j = 0; j < n; j += 32) {
        const __m256i va =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + j));
        const __m256i vb =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));
        // Callers ensure 2 * max(a) * max(abs(b)) <= INT16_MAX.
        const __m256i pair = _mm256_maddubs_epi16(va, vb);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(pair, ones));
    }
    return hsum256_epi32(acc);
#elif defined(__aarch64__)
    int32x4_t acc = vdupq_n_s32(0);
    for (size_t j = 0; j < n; j += 16) {
        const uint8x16_t va = vld1q_u8(a + j);
        const int8x16_t vb = vld1q_s8(b + j);
#if defined(__ARM_FEATURE_MATMUL_INT8)
        acc = vusdotq_s32(acc, va, vb);
#else
        const int16x8_t alo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(va)));
        const int16x8_t ahi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(va)));
        const int16x8_t blo = vmovl_s8(vget_low_s8(vb));
        const int16x8_t bhi = vmovl_s8(vget_high_s8(vb));
        acc = vaddq_s32(acc, vpaddlq_s16(vmulq_s16(alo, blo)));
        acc = vaddq_s32(acc, vpaddlq_s16(vmulq_s16(ahi, bhi)));
#endif
    }
    return vaddvq_s32(acc);
#else
    int32_t sum = 0;
    for (size_t j = 0; j < n; ++j) {
        sum += int32_t(a[j]) * int32_t(b[j]);
    }
    return sum;
#endif
}

inline int32_t dot_u4_i8(const uint8_t* packed, const int8_t* b, size_t n) {
#ifdef __AVX2__
    __m256i acc = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    const __m128i mask = _mm_set1_epi8(0x0f);
    for (size_t j = 0; j < n; j += 32) {
        const __m128i p = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(packed + j / 2));
        const __m128i lo = _mm_and_si128(p, mask);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(p, 4), mask);
        __m256i codes = _mm256_castsi128_si256(_mm_unpacklo_epi8(lo, hi));
        codes = _mm256_inserti128_si256(codes, _mm_unpackhi_epi8(lo, hi), 1);
        const __m256i vb =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));
        const __m256i pair = _mm256_maddubs_epi16(codes, vb);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(pair, ones));
    }
    return hsum256_epi32(acc);
#elif defined(__aarch64__)
    int32x4_t acc = vdupq_n_s32(0);
    const uint8x16_t mask = vdupq_n_u8(0x0f);
    for (size_t j = 0; j < n; j += 32) {
        const uint8x16_t p = vld1q_u8(packed + j / 2);
        const uint8x16_t lo = vandq_u8(p, mask);
        const uint8x16_t hi = vshrq_n_u8(p, 4);
        const int8x16_t c0 = vreinterpretq_s8_u8(vzip1q_u8(lo, hi));
        const int8x16_t c1 = vreinterpretq_s8_u8(vzip2q_u8(lo, hi));
        const int8x16_t q0 = vld1q_s8(b + j);
        const int8x16_t q1 = vld1q_s8(b + j + 16);
#if defined(__ARM_FEATURE_DOTPROD)
        acc = vdotq_s32(acc, c0, q0);
        acc = vdotq_s32(acc, c1, q1);
#else
        const int16x8_t p0 = vmull_s8(vget_low_s8(c0), vget_low_s8(q0));
        const int16x8_t p1 = vmull_s8(vget_high_s8(c0), vget_high_s8(q0));
        const int16x8_t p2 = vmull_s8(vget_low_s8(c1), vget_low_s8(q1));
        const int16x8_t p3 = vmull_s8(vget_high_s8(c1), vget_high_s8(q1));
        acc = vaddq_s32(acc, vpaddlq_s16(p0));
        acc = vaddq_s32(acc, vpaddlq_s16(p1));
        acc = vaddq_s32(acc, vpaddlq_s16(p2));
        acc = vaddq_s32(acc, vpaddlq_s16(p3));
#endif
    }
    return vaddvq_s32(acc);
#else
    int32_t sum = 0;
    for (size_t j = 0; j < n; ++j) {
        const uint8_t code = (packed[j / 2] >> (4 * (j & 1))) & 0x0f;
        sum += int32_t(code) * int32_t(b[j]);
    }
    return sum;
#endif
}

inline void dot_i8_batch4(
        const int8_t* a0,
        const int8_t* a1,
        const int8_t* a2,
        const int8_t* a3,
        const int8_t* b,
        size_t n,
        int32_t& s0,
        int32_t& s1,
        int32_t& s2,
        int32_t& s3) {
#ifdef __AVX2__
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    for (size_t j = 0; j < n; j += 32) {
        const __m256i vb =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));
        const __m256i blo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
        const __m256i bhi =
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
#define FAISS_QINCO_ACCUMULATE(OUT, PTR)                                      \
    do {                                                                      \
        const __m256i va = _mm256_loadu_si256(                                \
                reinterpret_cast<const __m256i*>((PTR) + j));                 \
        const __m256i alo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va)); \
        const __m256i ahi =                                                   \
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));        \
        (OUT) = _mm256_add_epi32((OUT), _mm256_madd_epi16(alo, blo));         \
        (OUT) = _mm256_add_epi32((OUT), _mm256_madd_epi16(ahi, bhi));         \
    } while (false)
        FAISS_QINCO_ACCUMULATE(acc0, a0);
        FAISS_QINCO_ACCUMULATE(acc1, a1);
        FAISS_QINCO_ACCUMULATE(acc2, a2);
        FAISS_QINCO_ACCUMULATE(acc3, a3);
#undef FAISS_QINCO_ACCUMULATE
    }
    s0 = hsum256_epi32(acc0);
    s1 = hsum256_epi32(acc1);
    s2 = hsum256_epi32(acc2);
    s3 = hsum256_epi32(acc3);
#elif defined(__aarch64__)
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);
    for (size_t j = 0; j < n; j += 16) {
        const int8x16_t vb = vld1q_s8(b + j);
#define FAISS_QINCO_NEON_ACCUMULATE(OUT, PTR)                              \
    do {                                                                   \
        const int8x16_t va = vld1q_s8((PTR) + j);                          \
        /* Graviton 2 lacks dotprod; Graviton 3+ uses the first branch. */ \
        FAISS_QINCO_NEON_DOT((OUT), va, vb);                               \
    } while (false)
#if defined(__ARM_FEATURE_DOTPROD)
#define FAISS_QINCO_NEON_DOT(OUT, VA, VB) (OUT) = vdotq_s32((OUT), (VA), (VB))
#else
#define FAISS_QINCO_NEON_DOT(OUT, VA, VB)                                      \
    do {                                                                       \
        const int16x8_t lo = vmull_s8(vget_low_s8((VA)), vget_low_s8((VB)));   \
        const int16x8_t hi = vmull_s8(vget_high_s8((VA)), vget_high_s8((VB))); \
        (OUT) = vaddq_s32((OUT), vpaddlq_s16(lo));                             \
        (OUT) = vaddq_s32((OUT), vpaddlq_s16(hi));                             \
    } while (false)
#endif
        FAISS_QINCO_NEON_ACCUMULATE(acc0, a0);
        FAISS_QINCO_NEON_ACCUMULATE(acc1, a1);
        FAISS_QINCO_NEON_ACCUMULATE(acc2, a2);
        FAISS_QINCO_NEON_ACCUMULATE(acc3, a3);
#undef FAISS_QINCO_NEON_ACCUMULATE
#undef FAISS_QINCO_NEON_DOT
    }
    s0 = vaddvq_s32(acc0);
    s1 = vaddvq_s32(acc1);
    s2 = vaddvq_s32(acc2);
    s3 = vaddvq_s32(acc3);
#else
    s0 = dot_i8_scalar(a0, b, n);
    s1 = dot_i8_scalar(a1, b, n);
    s2 = dot_i8_scalar(a2, b, n);
    s3 = dot_i8_scalar(a3, b, n);
#endif
}

inline void dot_unsigned_signed_bounded_batch4(
        const uint8_t* a0,
        const uint8_t* a1,
        const uint8_t* a2,
        const uint8_t* a3,
        const int8_t* b,
        size_t n,
        int32_t& s0,
        int32_t& s1,
        int32_t& s2,
        int32_t& s3) {
#ifdef __AVX2__
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    for (size_t j = 0; j < n; j += 32) {
        const __m256i vb =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + j));
#define FAISS_QINCO_U7_ACCUMULATE(OUT, PTR)                             \
    do {                                                                \
        const __m256i va = _mm256_loadu_si256(                          \
                reinterpret_cast<const __m256i*>((PTR) + j));           \
        const __m256i pair = _mm256_maddubs_epi16(va, vb);              \
        (OUT) = _mm256_add_epi32((OUT), _mm256_madd_epi16(pair, ones)); \
    } while (false)
        FAISS_QINCO_U7_ACCUMULATE(acc0, a0);
        FAISS_QINCO_U7_ACCUMULATE(acc1, a1);
        FAISS_QINCO_U7_ACCUMULATE(acc2, a2);
        FAISS_QINCO_U7_ACCUMULATE(acc3, a3);
#undef FAISS_QINCO_U7_ACCUMULATE
    }
    s0 = hsum256_epi32(acc0);
    s1 = hsum256_epi32(acc1);
    s2 = hsum256_epi32(acc2);
    s3 = hsum256_epi32(acc3);
#elif defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    int32x4_t acc2 = vdupq_n_s32(0);
    int32x4_t acc3 = vdupq_n_s32(0);
    for (size_t j = 0; j < n; j += 16) {
        const int8x16_t vb = vld1q_s8(b + j);
        acc0 = vusdotq_s32(acc0, vld1q_u8(a0 + j), vb);
        acc1 = vusdotq_s32(acc1, vld1q_u8(a1 + j), vb);
        acc2 = vusdotq_s32(acc2, vld1q_u8(a2 + j), vb);
        acc3 = vusdotq_s32(acc3, vld1q_u8(a3 + j), vb);
    }
    s0 = vaddvq_s32(acc0);
    s1 = vaddvq_s32(acc1);
    s2 = vaddvq_s32(acc2);
    s3 = vaddvq_s32(acc3);
#else
    s0 = dot_unsigned_signed_bounded(a0, b, n);
    s1 = dot_unsigned_signed_bounded(a1, b, n);
    s2 = dot_unsigned_signed_bounded(a2, b, n);
    s3 = dot_unsigned_signed_bounded(a3, b, n);
#endif
}

inline void dot_u4_i8_batch4(
        const uint8_t* a0,
        const uint8_t* a1,
        const uint8_t* a2,
        const uint8_t* a3,
        const int8_t* b,
        size_t n,
        int32_t& s0,
        int32_t& s1,
        int32_t& s2,
        int32_t& s3) {
    // The unpack is document-specific; keeping one routine makes the arbitrary
    // ID semantics explicit and lets each architecture inline its native path.
    s0 = dot_u4_i8(a0, b, n);
    s1 = dot_u4_i8(a1, b, n);
    s2 = dot_u4_i8(a2, b, n);
    s3 = dot_u4_i8(a3, b, n);
}

inline int document_code_at(
        const IndexQINCo& index,
        const uint8_t* code,
        int j) {
    if (index.doc_bits == 4) {
        if (size_t(j) < index.int8_head_dims) {
            return int(reinterpret_cast<const int8_t*>(code)[j]);
        }
        const size_t tail = size_t(j) - index.int8_head_dims;
        return (code[index.int8_head_dims + tail / 2] >> (4 * (tail & 1))) &
                0x0f;
    }
    if (index.doc_bits == 7 || index.query_bits == 6) {
        return int(code[j]);
    }
    return int(reinterpret_cast<const int8_t*>(code)[j]);
}

struct QINCoDistanceComputer final : FlatCodesDistanceComputer {
    const IndexQINCo& index;
    std::vector<int8_t> query_code;
    std::vector<float> query_scale;
    float query_norm = 0.0f;
    float bias_dot_query = 0.0f;

    explicit QINCoDistanceComputer(const IndexQINCo& index_in)
            : FlatCodesDistanceComputer(
                      index_in.codes.data(),
                      index_in.code_size),
              index(index_in),
              query_code(index_in.padded_d),
              query_scale(index_in.padded_d / index_in.block_size) {}

    void set_query(const float* x) override {
        q = x;
        query_norm = 0.0f;
        bias_dot_query = 0.0f;
        for (int j = 0; j < index.d; ++j) {
            query_norm += x[j] * x[j];
            bias_dot_query += index.bias[j] * x[j];
        }

        for (size_t b = 0; b < query_scale.size(); ++b) {
            const size_t begin = b * index.block_size;
            const size_t end =
                    std::min(begin + index.block_size, size_t(index.d));
            float max_abs = 0.0f;
            for (size_t j = begin; j < end; ++j) {
                max_abs = std::max(max_abs, std::abs(index.scale[j] * x[j]));
            }
            const int qmax = index.query_bits == 6 ? 63 : 127;
            const float qs = max_abs > 0.0f ? max_abs / float(qmax) : 0.0f;
            query_scale[b] = qs;
            if (qs > 0.0f) {
                const float inv = 1.0f / qs;
                for (size_t j = begin; j < end; ++j) {
                    const int v = int(std::lrint(index.scale[j] * x[j] * inv));
                    query_code[j] = int8_t(std::max(-qmax, std::min(qmax, v)));
                }
            } else {
                std::fill(
                        query_code.begin() + begin,
                        query_code.begin() + end,
                        0);
            }
            std::fill(
                    query_code.begin() + end,
                    query_code.begin() + begin + index.block_size,
                    0);
        }
    }

    float approximate_dot(const uint8_t* code) const {
        float dot = bias_dot_query;
        for (size_t b = 0; b < query_scale.size(); ++b) {
            const size_t begin = b * index.block_size;
            const bool unsigned_code =
                    index.doc_bits == 7 || index.query_bits == 6;
            const bool int8_head =
                    index.doc_bits == 4 && begin < index.int8_head_dims;
            const int32_t idot = int8_head
                    ? dot_i8(reinterpret_cast<const int8_t*>(code) + begin,
                             query_code.data() + begin,
                             index.block_size)
                    : index.doc_bits == 4
                    ? dot_u4_i8(
                              code + index.int8_head_dims +
                                      (begin - index.int8_head_dims) / 2,
                              query_code.data() + begin,
                              index.block_size)
                    : unsigned_code
                    ? dot_unsigned_signed_bounded(
                              code + begin,
                              query_code.data() + begin,
                              index.block_size)
                    : dot_i8(reinterpret_cast<const int8_t*>(code) + begin,
                             query_code.data() + begin,
                             index.block_size);
            dot += query_scale[b] * float(idot);
        }
        return dot;
    }

    float distance_to_code(const uint8_t* code) override {
        FAISS_THROW_IF_NOT(code >= codes);
        const ptrdiff_t delta = code - codes;
        FAISS_THROW_IF_NOT(delta % ptrdiff_t(code_size) == 0);
        const idx_t i = delta / ptrdiff_t(code_size);
        FAISS_THROW_IF_NOT(i >= 0 && i < index.ntotal);
        return index.l2_norms[i] + query_norm - 2.0f * approximate_dot(code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        FAISS_THROW_IF_NOT(i >= 0 && i < index.ntotal);
        FAISS_THROW_IF_NOT(j >= 0 && j < index.ntotal);
        const uint8_t* a = codes + size_t(i) * code_size;
        const uint8_t* b = codes + size_t(j) * code_size;
        float dis = 0.0f;
        for (int p = 0; p < index.d; ++p) {
            const int ai = document_code_at(index, a, p);
            const int bi = document_code_at(index, b, p);
            const float delta = index.scale[p] * float(ai - bi);
            dis += delta * delta;
        }
        return dis;
    }

    void distance_to_code_batch_4(
            const uint8_t* c0,
            const uint8_t* c1,
            const uint8_t* c2,
            const uint8_t* c3,
            float& d0,
            float& d1,
            float& d2,
            float& d3) override {
        float dot0 = bias_dot_query;
        float dot1 = bias_dot_query;
        float dot2 = bias_dot_query;
        float dot3 = bias_dot_query;
        for (size_t b = 0; b < query_scale.size(); ++b) {
            const size_t begin = b * index.block_size;
            int32_t s0, s1, s2, s3;
            if (index.doc_bits == 4 && begin < index.int8_head_dims) {
                dot_i8_batch4(
                        reinterpret_cast<const int8_t*>(c0) + begin,
                        reinterpret_cast<const int8_t*>(c1) + begin,
                        reinterpret_cast<const int8_t*>(c2) + begin,
                        reinterpret_cast<const int8_t*>(c3) + begin,
                        query_code.data() + begin,
                        index.block_size,
                        s0,
                        s1,
                        s2,
                        s3);
            } else if (index.doc_bits == 4) {
                const size_t tail_offset = index.int8_head_dims +
                        (begin - index.int8_head_dims) / 2;
                dot_u4_i8_batch4(
                        c0 + tail_offset,
                        c1 + tail_offset,
                        c2 + tail_offset,
                        c3 + tail_offset,
                        query_code.data() + begin,
                        index.block_size,
                        s0,
                        s1,
                        s2,
                        s3);
            } else if (index.doc_bits == 7 || index.query_bits == 6) {
                dot_unsigned_signed_bounded_batch4(
                        c0 + begin,
                        c1 + begin,
                        c2 + begin,
                        c3 + begin,
                        query_code.data() + begin,
                        index.block_size,
                        s0,
                        s1,
                        s2,
                        s3);
            } else {
                dot_i8_batch4(
                        reinterpret_cast<const int8_t*>(c0) + begin,
                        reinterpret_cast<const int8_t*>(c1) + begin,
                        reinterpret_cast<const int8_t*>(c2) + begin,
                        reinterpret_cast<const int8_t*>(c3) + begin,
                        query_code.data() + begin,
                        index.block_size,
                        s0,
                        s1,
                        s2,
                        s3);
            }
            const float qs = query_scale[b];
            dot0 += qs * float(s0);
            dot1 += qs * float(s1);
            dot2 += qs * float(s2);
            dot3 += qs * float(s3);
        }
        const idx_t i0 = (c0 - codes) / ptrdiff_t(code_size);
        const idx_t i1 = (c1 - codes) / ptrdiff_t(code_size);
        const idx_t i2 = (c2 - codes) / ptrdiff_t(code_size);
        const idx_t i3 = (c3 - codes) / ptrdiff_t(code_size);
        d0 = index.l2_norms[i0] + query_norm - 2.0f * dot0;
        d1 = index.l2_norms[i1] + query_norm - 2.0f * dot1;
        d2 = index.l2_norms[i2] + query_norm - 2.0f * dot2;
        d3 = index.l2_norms[i3] + query_norm - 2.0f * dot3;
    }
};

} // namespace

IndexQINCo::IndexQINCo() : IndexFlatCodes() {
    is_trained = false;
}

IndexQINCo::IndexQINCo(
        int d_in,
        size_t block_size_in,
        float exact_norm_weight_in,
        int doc_bits_in,
        int query_bits_in,
        size_t int8_head_dims_in)
        : IndexFlatCodes(
                  qinco_code_size(
                          d_in,
                          block_size_in,
                          doc_bits_in,
                          int8_head_dims_in),
                  d_in,
                  METRIC_L2),
          block_size(block_size_in),
          padded_d(
                  ((size_t(d_in) + block_size_in - 1) / block_size_in) *
                  block_size_in),
          exact_norm_weight(exact_norm_weight_in),
          doc_bits(doc_bits_in),
          query_bits(query_bits_in),
          int8_head_dims(int8_head_dims_in),
          scale(d_in),
          bias(d_in) {
    FAISS_THROW_IF_NOT(d_in > 0);
    FAISS_THROW_IF_NOT(block_size_in >= 32 && block_size_in % 32 == 0);
    FAISS_THROW_IF_NOT(
            exact_norm_weight_in >= 0.0f && exact_norm_weight_in <= 1.0f);
    FAISS_THROW_IF_NOT(
            doc_bits_in == 4 || doc_bits_in == 7 || doc_bits_in == 8);
    FAISS_THROW_IF_NOT(query_bits_in == 6 || query_bits_in == 8);
    FAISS_THROW_IF_NOT(doc_bits_in != 7 || query_bits_in == 8);
    FAISS_THROW_IF_NOT(doc_bits_in != 4 || query_bits_in == 8);
    FAISS_THROW_IF_NOT(int8_head_dims_in <= padded_d);
    FAISS_THROW_IF_NOT(
            int8_head_dims_in == 0 ||
            (doc_bits_in == 4 && int8_head_dims_in % block_size_in == 0));
    is_trained = false;
}

void IndexQINCo::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(n > 0);
    std::vector<float> vmin(d, std::numeric_limits<float>::infinity());
    std::vector<float> vmax(d, -std::numeric_limits<float>::infinity());
    for (idx_t i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            const float v = x[i * d + j];
            FAISS_THROW_IF_NOT_MSG(
                    std::isfinite(v), "training data is not finite");
            vmin[j] = std::min(vmin[j], v);
            vmax[j] = std::max(vmax[j], v);
        }
    }
    for (int j = 0; j < d; ++j) {
        const float range = vmax[j] - vmin[j];
        if (range > 0.0f && std::isfinite(range)) {
            const bool int8_head = doc_bits == 4 && size_t(j) < int8_head_dims;
            const float levels = int8_head ? 255.0f
                    : doc_bits == 4        ? 15.0f
                    : doc_bits == 7        ? 127.0f
                                           : 255.0f;
            scale[j] = range / levels;
            const bool unsigned_code =
                    !int8_head && (doc_bits != 8 || query_bits == 6);
            bias[j] = unsigned_code ? vmin[j] : vmin[j] + 128.0f * scale[j];
        } else {
            scale[j] = 0.0f;
            bias[j] = vmin[j];
        }
    }
    is_trained = true;
}

void IndexQINCo::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    const idx_t old_ntotal = ntotal;
    IndexFlatCodes::add(n, x);
    l2_norms.resize(ntotal);
#pragma omp parallel for if (n > 1000)
    for (idx_t i = 0; i < n; ++i) {
        float exact_norm = 0.0f;
        float reconstructed_norm = 0.0f;
        const uint8_t* code = codes.data() + size_t(old_ntotal + i) * code_size;
        for (int j = 0; j < d; ++j) {
            exact_norm += x[i * d + j] * x[i * d + j];
            const int value = document_code_at(*this, code, j);
            const float reconstructed = bias[j] + scale[j] * float(value);
            reconstructed_norm += reconstructed * reconstructed;
        }
        l2_norms[old_ntotal + i] = exact_norm_weight * exact_norm +
                (1.0f - exact_norm_weight) * reconstructed_norm;
    }
}

void IndexQINCo::reset() {
    IndexFlatCodes::reset();
    l2_norms.clear();
}

void IndexQINCo::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    FAISS_THROW_IF_NOT(is_trained);
#pragma omp parallel for if (n > 1000)
    for (idx_t i = 0; i < n; ++i) {
        uint8_t* out = bytes + i * code_size;
        std::memset(out, 0, code_size);
        for (int j = 0; j < d; ++j) {
            const bool int8_head = doc_bits == 4 && size_t(j) < int8_head_dims;
            int code = 0;
            if (scale[j] > 0.0f) {
                code = int(std::lrint((x[i * d + j] - bias[j]) / scale[j]));
                if (!int8_head && (doc_bits != 8 || query_bits == 6)) {
                    const int max_code =
                            doc_bits == 4 ? 15 : (doc_bits == 7 ? 127 : 255);
                    code = std::max(0, std::min(max_code, code));
                } else {
                    code = std::max(-128, std::min(127, code));
                }
            }
            if (int8_head) {
                out[j] = uint8_t(int8_t(code));
            } else if (doc_bits == 4) {
                const size_t tail = size_t(j) - int8_head_dims;
                out[int8_head_dims + tail / 2] |=
                        uint8_t(code << (4 * (tail & 1)));
            } else {
                out[j] = (doc_bits == 7 || query_bits == 6)
                        ? uint8_t(code)
                        : uint8_t(int8_t(code));
            }
        }
    }
}

void IndexQINCo::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
#pragma omp parallel for if (n > 1000)
    for (idx_t i = 0; i < n; ++i) {
        const uint8_t* in = bytes + i * code_size;
        for (int j = 0; j < d; ++j) {
            const int value = document_code_at(*this, in, j);
            x[i * d + j] = bias[j] + scale[j] * float(value);
        }
    }
}

FlatCodesDistanceComputer* IndexQINCo::get_FlatCodesDistanceComputer() const {
    FAISS_THROW_IF_NOT(is_trained);
    return new QINCoDistanceComputer(*this);
}

void IndexQINCo::compute_distance_subset(
        idx_t n,
        const float* queries,
        idx_t k,
        float* distances,
        const idx_t* labels) const {
#pragma omp parallel
    {
        std::unique_ptr<DistanceComputer> dc(get_distance_computer());
#pragma omp for
        for (idx_t i = 0; i < n; ++i) {
            dc->set_query(queries + i * d);
            for (idx_t j = 0; j < k; ++j) {
                const idx_t id = labels[i * k + j];
                distances[i * k + j] = id < 0
                        ? std::numeric_limits<float>::infinity()
                        : (*dc)(id);
            }
        }
    }
}

size_t IndexQINCo::bytes_per_vector() const {
    return code_size + sizeof(float);
}

IndexHNSW* clone_hnsw_with_qinco_storage(
        const IndexHNSW& fp32_graph,
        idx_t n,
        const float* x,
        size_t block_size,
        float exact_norm_weight,
        int doc_bits,
        int query_bits,
        size_t int8_head_dims) {
    FAISS_THROW_IF_NOT(fp32_graph.metric_type == METRIC_L2);
    FAISS_THROW_IF_NOT(fp32_graph.ntotal == n);
    auto storage = std::make_unique<IndexQINCo>(
            fp32_graph.d,
            block_size,
            exact_norm_weight,
            doc_bits,
            query_bits,
            int8_head_dims);
    storage->train(n, x);
    storage->add(n, x);

    const int M = fp32_graph.hnsw.nb_neighbors(0) / 2;
    auto result = std::make_unique<IndexHNSW>(storage.release(), M);
    result->own_fields = true;
    result->hnsw = fp32_graph.hnsw;
    result->ntotal = n;
    result->is_trained = true;
    result->metric_arg = fp32_graph.metric_arg;
    result->use_visited_hashset = fp32_graph.use_visited_hashset;
    return result.release();
}

} // namespace faiss
