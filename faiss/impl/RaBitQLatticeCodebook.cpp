/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/RaBitQLatticeCodebook.h>

#include <faiss/impl/FaissAssert.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>
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

void check_shell_count(
        const std::array<E8Chunk, kE8Finite256Size>& cb,
        float norm2,
        size_t expected,
        const char* name) {
    size_t got = 0;
    for (const E8Chunk& c : cb) {
        if (std::abs(chunk_norm2(c) - norm2) < 1e-5f) {
            got++;
        }
    }
    FAISS_THROW_IF_NOT_FMT(
            got == expected,
            "%s shell %.1f count %zu, expected %zu",
            name,
            norm2,
            got,
            expected);
}

bool chunk_equal(const E8Chunk& a, const E8Chunk& b) {
    for (size_t i = 0; i < kE8ChunkDim; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

int64_t chunk_key(const E8Chunk& c) {
    int64_t key = 0;
    for (float v : c) {
        const int64_t coord = static_cast<int64_t>(std::lround(v)) + 8;
        key = key * 17 + coord;
    }
    return key;
}

size_t popcount_size_t(size_t x) {
    size_t count = 0;
    while (x != 0) {
        count += x & 1U;
        x >>= 1U;
    }
    return count;
}

void check_negation_closed(
        const std::array<E8Chunk, kE8Finite256Size>& cb,
        const char* name) {
    for (const E8Chunk& c : cb) {
        E8Chunk neg{};
        for (size_t i = 0; i < kE8ChunkDim; i++) {
            neg[i] = -c[i];
        }
        bool found = false;
        for (const E8Chunk& other : cb) {
            if (chunk_equal(neg, other)) {
                found = true;
                break;
            }
        }
        FAISS_THROW_IF_NOT_FMT(found, "%s is not closed under negation", name);
    }
}

std::vector<E8Chunk> build_e8_roots_wide() {
    std::vector<E8Chunk> roots;
    roots.reserve(240);

    for (size_t i = 0; i < kE8ChunkDim; i++) {
        for (size_t j = i + 1; j < kE8ChunkDim; j++) {
            for (float si : {-1.0f, 1.0f}) {
                for (float sj : {-1.0f, 1.0f}) {
                    E8Chunk c{};
                    c[i] = 2.0f * si;
                    c[j] = 2.0f * sj;
                    roots.push_back(c);
                }
            }
        }
    }

    for (size_t mask = 0; mask < (size_t{1} << kE8ChunkDim); mask++) {
        if (popcount_size_t(mask) % 2 != 0) {
            continue;
        }
        E8Chunk c{};
        for (size_t i = 0; i < kE8ChunkDim; i++) {
            c[i] = (mask & (size_t{1} << i)) ? -1.0f : 1.0f;
        }
        roots.push_back(c);
    }

    std::sort(roots.begin(), roots.end(), lex_less);
    FAISS_THROW_IF_NOT_FMT(
            roots.size() == 240,
            "E8 root count %zu, expected 240",
            roots.size());
    return roots;
}

std::vector<E8Chunk> build_axis16_wide() {
    std::vector<E8Chunk> axis;
    axis.reserve(16);
    for (size_t i = 0; i < kE8ChunkDim; i++) {
        for (float sign : {-1.0f, 1.0f}) {
            E8Chunk c{};
            c[i] = 4.0f * sign;
            axis.push_back(c);
        }
    }
    std::sort(axis.begin(), axis.end(), lex_less);
    return axis;
}

std::array<E8Chunk, kE8Finite256Size> build_e8_sym256() {
    std::array<E8Chunk, kE8Finite256Size> cb{};
    const auto roots = build_e8_roots_wide();
    const auto axis = build_axis16_wide();
    std::copy(roots.begin(), roots.end(), cb.begin());
    std::copy(axis.begin(), axis.end(), cb.begin() + roots.size());
    check_shell_count(cb, 8.0f, 240, "sym256");
    check_shell_count(cb, 16.0f, 16, "sym256");
    check_negation_closed(cb, "sym256");
    return cb;
}

std::array<E8Chunk, kE8Finite256Size> build_e8_zero_axis15() {
    std::array<E8Chunk, kE8Finite256Size> cb{};
    size_t out = 0;
    cb[out++] = E8Chunk{};
    const auto roots = build_e8_roots_wide();
    for (const E8Chunk& c : roots) {
        cb[out++] = c;
    }
    const auto axis = build_axis16_wide();
    E8Chunk dropped{};
    dropped[0] = -4.0f;
    for (const E8Chunk& c : axis) {
        if (chunk_equal(c, dropped)) {
            continue;
        }
        cb[out++] = c;
    }
    FAISS_THROW_IF_NOT_FMT(
            out == kE8Finite256Size,
            "zero_axis15 built %zu entries, expected %zu",
            out,
            kE8Finite256Size);
    check_shell_count(cb, 0.0f, 1, "zero_axis15");
    check_shell_count(cb, 8.0f, 240, "zero_axis15");
    check_shell_count(cb, 16.0f, 15, "zero_axis15");
    return cb;
}

bool find_codeword(
        const E8Chunk& c,
        const std::array<E8Chunk, kE8Finite256Size>& cb,
        uint8_t* code) {
    static const std::array<std::unordered_map<int64_t, uint8_t>, 3> maps = [] {
        std::array<std::unordered_map<int64_t, uint8_t>, 3> out;
        const LatticeBook books[] = {
                LatticeBook::E8_LEX15,
                LatticeBook::E8_SYM256,
                LatticeBook::E8_ZERO_AXIS15,
        };
        for (LatticeBook book : books) {
            const auto& book_cb = e8_lattice_codebook(book);
            auto& map = out[static_cast<size_t>(book)];
            map.reserve(book_cb.size());
            for (size_t i = 0; i < book_cb.size(); i++) {
                map.emplace(chunk_key(book_cb[i]), static_cast<uint8_t>(i));
            }
        }
        return out;
    }();

    const auto& cb0 = e8_lattice_codebook(LatticeBook::E8_LEX15);
    const auto& cb1 = e8_lattice_codebook(LatticeBook::E8_SYM256);
    const auto& cb2 = e8_lattice_codebook(LatticeBook::E8_ZERO_AXIS15);
    size_t book_index = 0;
    if (&cb == &cb1) {
        book_index = static_cast<size_t>(LatticeBook::E8_SYM256);
    } else if (&cb == &cb2) {
        book_index = static_cast<size_t>(LatticeBook::E8_ZERO_AXIS15);
    } else {
        FAISS_THROW_IF_NOT(&cb == &cb0);
        book_index = static_cast<size_t>(LatticeBook::E8_LEX15);
    }

    const auto it = maps[book_index].find(chunk_key(c));
    if (it != maps[book_index].end()) {
        *code = it->second;
        return true;
    }
    return false;
}

void maybe_keep_candidate(
        const float* chunk8,
        LatticeBook book,
        const E8Chunk& candidate,
        bool& has_best,
        float& best_dist,
        uint8_t& best_code) {
    uint8_t code = 0;
    if (!find_codeword(candidate, e8_lattice_codebook(book), &code)) {
        return;
    }
    const float dist = l2sqr_8(chunk8, candidate);
    if (!has_best || dist < best_dist) {
        has_best = true;
        best_dist = dist;
        best_code = code;
    }
}

E8Chunk best_a_root_candidate(const float* x) {
    size_t i0 = 0;
    size_t i1 = 1;
    if (std::abs(x[i1]) > std::abs(x[i0])) {
        std::swap(i0, i1);
    }
    for (size_t i = 2; i < kE8ChunkDim; i++) {
        const float ai = std::abs(x[i]);
        if (ai > std::abs(x[i0])) {
            i1 = i0;
            i0 = i;
        } else if (ai > std::abs(x[i1])) {
            i1 = i;
        }
    }
    E8Chunk c{};
    c[i0] = x[i0] < 0.0f ? -2.0f : 2.0f;
    c[i1] = x[i1] < 0.0f ? -2.0f : 2.0f;
    return c;
}

E8Chunk best_b_root_candidate(const float* x) {
    E8Chunk c{};
    size_t neg_count = 0;
    size_t min_abs = 0;
    for (size_t i = 0; i < kE8ChunkDim; i++) {
        c[i] = x[i] < 0.0f ? -1.0f : 1.0f;
        neg_count += c[i] < 0.0f ? 1 : 0;
        if (std::abs(x[i]) < std::abs(x[min_abs])) {
            min_abs = i;
        }
    }
    if (neg_count % 2 != 0) {
        c[min_abs] = -c[min_abs];
    }
    return c;
}

} // namespace

const std::array<E8Chunk, kE8Finite256Size>& e8_finite_256_wide_codebook() {
    static const std::array<E8Chunk, kE8Finite256Size> cb =
            build_e8_finite_256_wide();
    return cb;
}

const std::array<E8Chunk, kE8Finite256Size>& e8_lattice_codebook(
        LatticeBook book) {
    switch (book) {
        case LatticeBook::E8_LEX15:
            return e8_finite_256_wide_codebook();
        case LatticeBook::E8_SYM256: {
            static const std::array<E8Chunk, kE8Finite256Size> cb =
                    build_e8_sym256();
            return cb;
        }
        case LatticeBook::E8_ZERO_AXIS15: {
            static const std::array<E8Chunk, kE8Finite256Size> cb =
                    build_e8_zero_axis15();
            return cb;
        }
    }
    FAISS_THROW_MSG("Unknown LatticeBook");
}

uint64_t e8_lattice_codebook_checksum(LatticeBook book) {
    constexpr uint64_t kOffset = 1469598103934665603ULL;
    constexpr uint64_t kPrime = 1099511628211ULL;
    uint64_t hash = kOffset;
    const auto& cb = e8_lattice_codebook(book);
    for (const E8Chunk& c : cb) {
        for (float v : c) {
            uint32_t bits = 0;
            static_assert(sizeof(bits) == sizeof(v));
            std::memcpy(&bits, &v, sizeof(bits));
            for (size_t i = 0; i < sizeof(bits); i++) {
                const uint8_t byte = (bits >> (8 * i)) & 0xff;
                hash ^= byte;
                hash *= kPrime;
            }
        }
    }
    return hash;
}

uint8_t encode_e8_finite_256_wide(const float* chunk8) {
    return encode_e8_lattice_book(chunk8, LatticeBook::E8_LEX15);
}

uint8_t encode_e8_lattice_book(const float* chunk8, LatticeBook book) {
    FAISS_THROW_IF_NOT(chunk8 != nullptr);

    const auto& cb = e8_lattice_codebook(book);
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

uint8_t encode_e8_lattice_book_fast(
        const float* chunk8,
        LatticeBook book,
        bool* used_fast_path) {
    FAISS_THROW_IF_NOT(chunk8 != nullptr);
    if (used_fast_path != nullptr) {
        *used_fast_path = true;
    }
    return encode_e8_lattice_book_shell(chunk8, book);
}

uint8_t encode_e8_lattice_book_shell(const float* chunk8, LatticeBook book) {
    FAISS_THROW_IF_NOT(chunk8 != nullptr);

    bool has_best = false;
    float best_dist = std::numeric_limits<float>::infinity();
    uint8_t best_code = 0;

    if (book == LatticeBook::E8_LEX15 || book == LatticeBook::E8_ZERO_AXIS15) {
        maybe_keep_candidate(
                chunk8, book, E8Chunk{}, has_best, best_dist, best_code);
    }
    maybe_keep_candidate(
            chunk8,
            book,
            best_a_root_candidate(chunk8),
            has_best,
            best_dist,
            best_code);
    maybe_keep_candidate(
            chunk8,
            book,
            best_b_root_candidate(chunk8),
            has_best,
            best_dist,
            best_code);
    const auto& cb = e8_lattice_codebook(book);
    for (size_t i = 0; i < cb.size(); i++) {
        if (std::abs(chunk_norm2(cb[i]) - 16.0f) < 1e-5f) {
            const float dist = l2sqr_8(chunk8, cb[i]);
            if (!has_best || dist < best_dist) {
                has_best = true;
                best_dist = dist;
                best_code = static_cast<uint8_t>(i);
            }
        }
    }

    FAISS_THROW_IF_NOT(has_best);
    return best_code;
}

float e8_lattice_codeword_l2sqr(
        const float* chunk8,
        uint8_t code,
        LatticeBook book) {
    FAISS_THROW_IF_NOT(chunk8 != nullptr);
    return l2sqr_8(chunk8, e8_lattice_codebook(book)[code]);
}

void decode_e8_finite_256_wide(uint8_t code, float* chunk8) {
    decode_e8_lattice_book(code, LatticeBook::E8_LEX15, chunk8);
}

void decode_e8_lattice_book(uint8_t code, LatticeBook book, float* chunk8) {
    FAISS_THROW_IF_NOT(chunk8 != nullptr);
    const E8Chunk& c = e8_lattice_codebook(book)[code];
    std::copy(c.begin(), c.end(), chunk8);
}

DirectionEncodingStats encode_e8_finite_256_wide_direction(
        const float* residual,
        size_t d,
        uint8_t* codes,
        float* decoded_unit) {
    return encode_e8_lattice_book_direction(
            residual, d, LatticeBook::E8_LEX15, codes, decoded_unit);
}

DirectionEncodingStats encode_e8_lattice_book_direction(
        const float* residual,
        size_t d,
        LatticeBook book,
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

        const uint8_t code =
                encode_e8_lattice_book_fast(scaled_chunk.data(), book);
        codes[chunk] = code;
        const E8Chunk& decoded = e8_lattice_codebook(book)[code];
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
            const E8Chunk& decoded = e8_lattice_codebook(book)[codes[chunk]];
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
