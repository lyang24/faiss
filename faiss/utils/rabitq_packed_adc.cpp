/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/rabitq_packed_adc.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/utils/rabitq_integer_adc.h>
#include <faiss/utils/simd_levels.h>

#ifdef COMPILE_SIMD_AVX512_SPR
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace faiss::rabitq_packed_adc {
namespace {

template <int Bits>
void scalar_dots(
        const int8_t* query,
        const uint8_t* const* codes,
        size_t d,
        int count,
        int64_t* dots) {
    const size_t offset =
            (d + 7) / 8 + sizeof(rabitq_utils::SignBitFactorsWithError);
    for (int n = 0; n < count; ++n) {
        dots[n] = 0;
        for (size_t j = 0; j < d; ++j) {
            const int sign = (codes[n][j / 8] >> (j % 8)) & 1;
            const int low = rabitq_utils::extract_code_inline(
                    codes[n] + offset, j, Bits - 1);
            const int level = (sign << (Bits - 1)) + low - (1 << (Bits - 1));
            dots[n] += int64_t(query[j]) * level;
        }
    }
}

// Deliberately separate from RaBitQExpandedDistanceComputer: enabling this
// memory-saving path must not change existing expanded ADC code generation,
// default mode selection, query arithmetic or HNSW batch scheduling.
struct PackedDistanceComputer final : FlatCodesDistanceComputer,
                                      DistanceComputerBatch {
    const size_t d;
    const float* center;
    DotProduct dot_product;
    bool native;
    std::vector<float> residual;
    std::vector<int8_t> quantized;
    float norm = 0, half_sum = 0, scale = 1;

    PackedDistanceComputer(
            const uint8_t* codes,
            size_t stride,
            size_t d,
            size_t bits,
            const float* center)
            : FlatCodesDistanceComputer(codes, stride),
              d(d),
              center(center),
              dot_product(get_dot_product(bits)),
              native(uses_native_dotprod(bits)),
              residual(d),
              quantized(d) {}

    void set_query(const float* x) override {
        FAISS_THROW_IF_NOT_MSG(x, "null RaBitQ query");
        q = x;
        norm = 0;
        float sum = 0, maximum = 0;
        for (size_t j = 0; j < d; ++j) {
            const float v = x[j] - (center ? center[j] : 0.0f);
            residual[j] = v;
            norm += v * v;
            sum += v;
            maximum = std::max(maximum, std::abs(v));
        }
        half_sum = 0.5f * sum;
        scale = maximum > 0 ? maximum / 127.0f : 1.0f;
        for (size_t j = 0; j < d; ++j) {
            quantized[j] = static_cast<int8_t>(std::clamp(
                    std::nearbyint(residual[j] / scale), -127.0f, 127.0f));
        }
    }

    float score(const uint8_t* code, int64_t dot) const {
        rabitq_utils::ExtraBitsFactors f;
        std::memcpy(&f, code + code_size - sizeof(f), sizeof(f));
        // Native full levels represent z + 1/2. Keep the norm and half-sum
        // from the unquantized centered query, as in expanded integer ADC.
        return std::max(
                0.0f,
                norm + f.f_add_ex +
                        f.f_rescale_ex *
                                (scale * static_cast<float>(dot) + half_sum));
    }

    float distance_to_code(const uint8_t* code) override {
        int64_t dot;
        dot_product(quantized.data(), &code, d, 1, &dot);
        return score(code, dot);
    }

    void distance_to_code_batch_4(
            const uint8_t* a,
            const uint8_t* b,
            const uint8_t* c,
            const uint8_t* e,
            float& w,
            float& x,
            float& y,
            float& z) override {
        const uint8_t* rows[4] = {a, b, c, e};
        int64_t dots[4];
        dot_product(quantized.data(), rows, d, 4, dots);
        w = score(a, dots[0]);
        x = score(b, dots[1]);
        y = score(c, dots[2]);
        z = score(e, dots[3]);
    }

    void batch8(const int32_t* ids, float* distances) {
        const uint8_t* rows[8];
        int64_t dots[8];
        for (int i = 0; i < 8; ++i)
            rows[i] = codes + size_t(ids[i]) * code_size;
        dot_product(quantized.data(), rows, d, 8, dots);
        for (int i = 0; i < 8; ++i)
            distances[i] = score(rows[i], dots[i]);
    }
    int preferred_batch_size() const override {
        return native ? 8 : 4;
    }
    int max_tail_batch_size() const override {
        return 0;
    }
    void distances_batch_8(const int32_t* ids, float* distances) override {
        batch8(ids, distances);
    }
    void distances_batch_16(const int32_t* ids, float* distances) override {
        batch8(ids, distances);
        batch8(ids + 8, distances + 8);
    }
    void distances_batch_tail(const int32_t* ids, int count, float* distances)
            override {
        for (int i = 0; i < count; ++i)
            distances[i] = (*this)(ids[i]);
    }
    float symmetric_dis(idx_t, idx_t) override {
        FAISS_THROW_MSG(
                "packed integer RaBitQ ADC does not support graph construction");
    }
};

} // namespace

DotProduct get_dot_product_scalar(size_t bits) {
    switch (bits) {
        case 2:
            return scalar_dots<2>;
        case 3:
            return scalar_dots<3>;
        case 4:
            return scalar_dots<4>;
        case 5:
            return scalar_dots<5>;
        case 6:
            return scalar_dots<6>;
        case 7:
            return scalar_dots<7>;
        case 8:
            return scalar_dots<8>;
        default:
            FAISS_THROW_MSG(
                    "packed integer RaBitQ ADC requires 2..8 total bits");
    }
}

bool uses_native_dotprod(size_t bits) {
    if (bits != 2 && bits != 4 && bits != 6 && bits != 8)
        return false;
#ifdef COMPILE_SIMD_ARM_NEON
    const auto level = SIMDConfig::get_dispatched_level();
    return (level == SIMDLevel::ARM_NEON || level == SIMDLevel::ARM_SVE) &&
            rabitq_integer_adc::arm_dotprod_supported();
#elif defined(COMPILE_SIMD_AVX512_SPR)
    if (SIMDConfig::get_dispatched_level() != SIMDLevel::AVX512_SPR)
        return false;
    // The existing SPR dispatch contract does not require VBMI. Gate the
    // new unpacker separately, without changing dispatch for expanded ADC.
#ifdef _MSC_VER
    int regs[4];
    __cpuidex(regs, 7, 0);
    return (regs[2] & (1 << 1)) != 0;
#else
    unsigned int eax, ebx, ecx, edx;
    return __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) && (ecx & (1u << 1));
#endif
#else
    return false;
#endif
}

DotProduct get_dot_product(size_t bits) {
    if (uses_native_dotprod(bits)) {
#ifdef COMPILE_SIMD_ARM_NEON
        return get_dot_product_arm(bits);
#endif
#ifdef COMPILE_SIMD_AVX512_SPR
        return get_dot_product_avx512(bits);
#endif
    }
    return get_dot_product_scalar(bits);
}

FlatCodesDistanceComputer* get_distance_computer(
        const uint8_t* codes,
        size_t code_size,
        size_t d,
        size_t bits,
        const float* center) {
    FAISS_THROW_IF_NOT_MSG(
            bits >= 2 && bits <= 8,
            "packed integer RaBitQ ADC requires 2..8 total bits");
    return new PackedDistanceComputer(codes, code_size, d, bits, center);
}

} // namespace faiss::rabitq_packed_adc
