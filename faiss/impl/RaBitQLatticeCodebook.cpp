/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/RaBitQLatticeCodebook.h>

#include <faiss/impl/FaissAssert.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace faiss::rabitq_lattice {

namespace {

bool is_d8_point(const E8Chunk& c) {
    long sum = 0;
    for (float v : c) {
        sum += static_cast<long>(std::lround(v));
    }
    return (sum & 1L) == 0L;
}

bool is_d8_plus_h_point(const E8Chunk& c) {
    long sum = 0;
    for (float v : c) {
        sum += static_cast<long>(std::lround(v - 0.5f));
    }
    return (sum & 1L) == 0L;
}

float chunk_norm2(const E8Chunk& c) {
    float s = 0.0f;
    for (float v : c) {
        s += v * v;
    }
    return s;
}

bool lex_less(const E8Chunk& a, const E8Chunk& b) {
    for (size_t i = 0; i < kE8ChunkDim; i++) {
        if (a[i] < b[i]) {
            return true;
        }
        if (a[i] > b[i]) {
            return false;
        }
    }
    return false;
}

struct CodebookEntry {
    float norm2 = 0.0f;
    E8Chunk point{};
};

bool entry_less(const CodebookEntry& a, const CodebookEntry& b) {
    if (a.norm2 != b.norm2) {
        return a.norm2 < b.norm2;
    }
    return lex_less(a.point, b.point);
}

template <typename KeepFn>
void enumerate_tuples(
        const float* coords,
        int num_vals,
        float norm_cap,
        KeepFn keep,
        std::vector<CodebookEntry>& out) {
    int idx[kE8ChunkDim] = {0};
    while (true) {
        E8Chunk c{};
        for (size_t i = 0; i < kE8ChunkDim; i++) {
            c[i] = coords[idx[i]];
        }

        const float n2 = chunk_norm2(c);
        if (n2 <= norm_cap && keep(c)) {
            out.push_back({n2, c});
        }

        int k = static_cast<int>(kE8ChunkDim) - 1;
        while (k >= 0) {
            if (++idx[k] < num_vals) {
                break;
            }
            idx[k] = 0;
            --k;
        }
        if (k < 0) {
            break;
        }
    }
}

std::array<E8Chunk, kE8Finite256Size> build_e8_finite_256_wide() {
    constexpr float kNormCap = 4.0f + 1e-4f;

    std::vector<CodebookEntry> all;
    all.reserve(3000);

    static constexpr float kD8Coords[5] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    enumerate_tuples(kD8Coords, 5, kNormCap, is_d8_point, all);

    static constexpr float kD8hCoords[4] = {-1.5f, -0.5f, 0.5f, 1.5f};
    enumerate_tuples(kD8hCoords, 4, kNormCap, is_d8_plus_h_point, all);

    std::sort(all.begin(), all.end(), entry_less);
    FAISS_THROW_IF_NOT_FMT(
            all.size() >= kE8Finite256Size,
            "Enumerated %zu finite-E8 candidates, need %zu",
            all.size(),
            kE8Finite256Size);

    std::array<E8Chunk, kE8Finite256Size> cb{};
    for (size_t i = 0; i < kE8Finite256Size; i++) {
        cb[i] = all[i].point;
        for (size_t k = 0; k < kE8ChunkDim; k++) {
            cb[i][k] *= 2.0f;
        }
    }

    return cb;
}

float l2sqr_8(const float* x, const E8Chunk& y) {
    float s = 0.0f;
    for (size_t i = 0; i < kE8ChunkDim; i++) {
        const float diff = x[i] - y[i];
        s += diff * diff;
    }
    return s;
}

} // namespace

const std::array<E8Chunk, kE8Finite256Size>& e8_finite_256_wide_codebook() {
    static const std::array<E8Chunk, kE8Finite256Size> cb =
            build_e8_finite_256_wide();
    return cb;
}

uint8_t encode_e8_finite_256_wide(const float* chunk8) {
    FAISS_THROW_IF_NOT(chunk8 != nullptr);

    const auto& cb = e8_finite_256_wide_codebook();
    float best = std::numeric_limits<float>::infinity();
    size_t best_idx = 0;
    for (size_t i = 0; i < kE8Finite256Size; i++) {
        const float dist = l2sqr_8(chunk8, cb[i]);
        if (dist < best) {
            best = dist;
            best_idx = i;
        }
    }
    return static_cast<uint8_t>(best_idx);
}

void decode_e8_finite_256_wide(uint8_t code, float* chunk8) {
    FAISS_THROW_IF_NOT(chunk8 != nullptr);
    const E8Chunk& c = e8_finite_256_wide_codebook()[code];
    std::copy(c.begin(), c.end(), chunk8);
}

DirectionEncodingStats encode_e8_finite_256_wide_direction(
        const float* residual,
        size_t d,
        uint8_t* codes,
        float* decoded_unit) {
    FAISS_THROW_IF_NOT(residual != nullptr);
    FAISS_THROW_IF_NOT(codes != nullptr);
    FAISS_THROW_IF_NOT(d % kE8ChunkDim == 0);
    FAISS_THROW_IF_NOT(d > 0);

    float norm2 = 0.0f;
    for (size_t i = 0; i < d; i++) {
        norm2 += residual[i] * residual[i];
    }

    DirectionEncodingStats stats;
    stats.norm = std::sqrt(norm2);
    if (stats.norm == 0.0f) {
        std::fill(codes, codes + d / kE8ChunkDim, uint8_t{0});
        if (decoded_unit != nullptr) {
            std::fill(decoded_unit, decoded_unit + d, 0.0f);
        }
        return stats;
    }

    const float scale_to_codebook =
            std::sqrt(static_cast<float>(d)) / stats.norm;

    float decoded_norm2 = 0.0f;
    float scaled_dot = 0.0f;
    E8Chunk scaled_chunk{};

    for (size_t chunk = 0; chunk < d / kE8ChunkDim; chunk++) {
        const size_t offset = chunk * kE8ChunkDim;
        for (size_t k = 0; k < kE8ChunkDim; k++) {
            scaled_chunk[k] = residual[offset + k] * scale_to_codebook;
        }

        const uint8_t code = encode_e8_finite_256_wide(scaled_chunk.data());
        codes[chunk] = code;
        const E8Chunk& decoded = e8_finite_256_wide_codebook()[code];
        for (size_t k = 0; k < kE8ChunkDim; k++) {
            decoded_norm2 += decoded[k] * decoded[k];
            scaled_dot += scaled_chunk[k] * decoded[k];
        }
    }

    stats.decoded_norm = std::sqrt(decoded_norm2);
    if (stats.decoded_norm > 0.0f) {
        stats.correlation = scaled_dot /
                (std::sqrt(static_cast<float>(d)) * stats.decoded_norm);
        constexpr float kMinCorrelation = 1e-20f;
        stats.dp_multiplier =
                stats.norm / std::max(stats.correlation, kMinCorrelation);
    }

    if (decoded_unit != nullptr) {
        for (size_t chunk = 0; chunk < d / kE8ChunkDim; chunk++) {
            const E8Chunk& decoded =
                    e8_finite_256_wide_codebook()[codes[chunk]];
            const size_t offset = chunk * kE8ChunkDim;
            for (size_t k = 0; k < kE8ChunkDim; k++) {
                decoded_unit[offset + k] = decoded[k] / stats.decoded_norm;
            }
        }
    }

    return stats;
}

float sign_direction_correlation(const float* residual, size_t d) {
    FAISS_THROW_IF_NOT(residual != nullptr);
    FAISS_THROW_IF_NOT(d > 0);

    float norm2 = 0.0f;
    float abs_sum = 0.0f;
    for (size_t i = 0; i < d; i++) {
        norm2 += residual[i] * residual[i];
        abs_sum += std::abs(residual[i]);
    }

    if (norm2 == 0.0f) {
        return 0.0f;
    }
    return abs_sum / (std::sqrt(norm2) * std::sqrt(static_cast<float>(d)));
}

} // namespace faiss::rabitq_lattice
