/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace faiss::rabitq_lattice {

constexpr size_t kE8ChunkDim = 8;
constexpr size_t kE8Finite256Size = 256;

using E8Chunk = std::array<float, kE8ChunkDim>;

enum class LatticeBook {
    E8_LEX15 = 0,
    E8_SYM256 = 1,
    E8_ZERO_AXIS15 = 2,
};

struct DirectionEncodingStats {
    float norm = 0.0f;
    float decoded_norm = 0.0f;
    float correlation = 0.0f;
    float dp_multiplier = 1.0f;
};

/**
 * Experimental finite-E8-wide block codebook from the SAQ-lattice evidence
 * branch: enumerate the first 256 E8 points by (norm^2, lexicographic order),
 * then scale every point by 2. This gives 1 byte per 8D chunk, i.e. the same
 * body rate as 1-bit-per-dimension RaBitQ sign codes, but with a different
 * 8D spherical codebook.
 *
 * This is a research primitive, not a production RaBitQ storage format.
 */
const std::array<E8Chunk, kE8Finite256Size>& e8_finite_256_wide_codebook();

const std::array<E8Chunk, kE8Finite256Size>& e8_lattice_codebook(
        LatticeBook book);

uint64_t e8_lattice_codebook_checksum(LatticeBook book);

uint8_t encode_e8_finite_256_wide(const float* chunk8);

uint8_t encode_e8_lattice_book(const float* chunk8, LatticeBook book);

uint8_t encode_e8_lattice_book_fast(
        const float* chunk8,
        LatticeBook book,
        bool* used_fast_path = nullptr);

uint8_t encode_e8_lattice_book_shell(const float* chunk8, LatticeBook book);

float e8_lattice_codeword_l2sqr(
        const float* chunk8,
        uint8_t code,
        LatticeBook book);

void decode_e8_finite_256_wide(uint8_t code, float* chunk8);

void decode_e8_lattice_book(uint8_t code, LatticeBook book, float* chunk8);

/**
 * Encode a residual vector as finite-E8-wide 8D chunks and optionally write the
 * globally normalized decoded direction.
 *
 * The input is first normalized to unit length and then multiplied by sqrt(d)
 * before per-chunk codebook lookup. This matches the RaBitQ convention where a
 * random unit vector has per-coordinate RMS 1/sqrt(d), while the E8-wide
 * codebook is calibrated to chunk RMS around 1.
 *
 * @param residual       input residual vector
 * @param d              dimensionality, must be a multiple of 8
 * @param codes          output codes, d / 8 bytes
 * @param decoded_unit   optional output direction, length d, unit norm unless
 *                       the residual is zero
 */
DirectionEncodingStats encode_e8_finite_256_wide_direction(
        const float* residual,
        size_t d,
        uint8_t* codes,
        float* decoded_unit = nullptr);

DirectionEncodingStats encode_e8_lattice_book_direction(
        const float* residual,
        size_t d,
        LatticeBook book,
        uint8_t* codes,
        float* decoded_unit = nullptr);

/**
 * Correlation between a residual direction and the classic RaBitQ sign
 * direction sign(residual) / sqrt(d). Useful as the matched-rate baseline when
 * evaluating block codebooks.
 */
float sign_direction_correlation(const float* residual, size_t d);

} // namespace faiss::rabitq_lattice
