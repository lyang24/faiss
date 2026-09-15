/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/platform_macros.h>

namespace faiss::rabitq_packed_adc {

// All code pointers may address unrelated documents. Each row retains the
// native sign field, SignBitFactorsWithError, extra bits and ExtraBitsFactors.
using DotProduct = void (*)(
        const int8_t* query,
        const uint8_t* const* codes,
        size_t d,
        int count,
        int64_t* dots);

FAISS_API DotProduct get_dot_product_scalar(size_t bits);
FAISS_API DotProduct get_dot_product(size_t bits);
#ifdef COMPILE_SIMD_ARM_NEON
FAISS_API DotProduct get_dot_product_arm(size_t bits);
#endif
#ifdef COMPILE_SIMD_AVX512_SPR
FAISS_API DotProduct get_dot_product_avx512(size_t bits);
#endif

FAISS_API bool uses_native_dotprod(size_t bits);
FAISS_API FlatCodesDistanceComputer* get_distance_computer(
        const uint8_t* codes,
        size_t code_size,
        size_t d,
        size_t bits,
        const float* center);

} // namespace faiss::rabitq_packed_adc
