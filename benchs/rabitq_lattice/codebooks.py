#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np


D8 = 8
BOOK_NAMES = ("lex15", "sym256", "zero_axis15", "voronoi_off")


@dataclass
class CodebookResult:
    name: str
    book: np.ndarray | None
    status: str = "ok"
    diagnostics: str = ""


def _is_d8(point: np.ndarray) -> bool:
    return int(np.rint(point).sum()) % 2 == 0


def _is_d8_plus_h(point: np.ndarray) -> bool:
    return int(np.rint(point - 0.5).sum()) % 2 == 0


def _sort_rows_by_norm_lex(points: list[tuple[float, ...]]) -> list[tuple[float, ...]]:
    return sorted(points, key=lambda p: (sum(x * x for x in p), p))


def _unique_rows(points: list[tuple[float, ...]]) -> list[tuple[float, ...]]:
    return sorted(set(points))


def _assert_shape(book: np.ndarray, name: str) -> None:
    if book.shape != (256, D8):
        raise AssertionError(f"{name}: expected shape (256, 8), got {book.shape}")


def _assert_negation_closed(book: np.ndarray, name: str) -> None:
    rows = {tuple(row.tolist()) for row in book}
    missing = [tuple((-row).tolist()) for row in book if tuple((-row).tolist()) not in rows]
    if missing:
        raise AssertionError(f"{name}: not closed under negation, first missing={missing[0]}")


def _assert_shell_counts(book: np.ndarray, expected: dict[float, int], name: str) -> None:
    norms = np.sum(book * book, axis=1)
    for norm2, count in expected.items():
        got = int(np.sum(np.isclose(norms, norm2, atol=1e-5)))
        if got != count:
            raise AssertionError(f"{name}: shell norm2={norm2} count {got}, expected {count}")


def e8_roots_wide() -> np.ndarray:
    roots: list[tuple[float, ...]] = []

    # D8 roots: all permutations of (+/-1, +/-1, 0^6), then wide-scale by 2.
    for i, j in combinations(range(D8), 2):
        for si, sj in product((-1.0, 1.0), repeat=2):
            p = [0.0] * D8
            p[i] = 2.0 * si
            p[j] = 2.0 * sj
            roots.append(tuple(p))

    # Half-integer E8 roots: (+/-1/2)^8 with even parity, then wide-scale by 2.
    for signs in product((-1.0, 1.0), repeat=D8):
        nminus = sum(1 for s in signs if s < 0)
        if nminus % 2 == 0:
            roots.append(tuple(signs))

    roots = _unique_rows(roots)
    if len(roots) != 240:
        raise AssertionError(f"E8 root count {len(roots)}, expected 240")
    book = np.asarray(roots, dtype=np.float32)
    _assert_shell_counts(book, {8.0: 240}, "e8_roots_wide")
    _assert_negation_closed(book, "e8_roots_wide")
    return book


def axis16_wide() -> np.ndarray:
    pts = []
    for i in range(D8):
        for sign in (-1.0, 1.0):
            p = [0.0] * D8
            p[i] = 4.0 * sign
            pts.append(tuple(p))
    book = np.asarray(_unique_rows(pts), dtype=np.float32)
    if book.shape != (16, D8):
        raise AssertionError(f"axis orbit shape {book.shape}, expected (16, 8)")
    _assert_shell_counts(book, {16.0: 16}, "axis16_wide")
    _assert_negation_closed(book, "axis16_wide")
    return book


def lex15() -> np.ndarray:
    pts: list[tuple[float, ...]] = []

    for coords, keep in (
        ((-2.0, -1.0, 0.0, 1.0, 2.0), _is_d8),
        ((-1.5, -0.5, 0.5, 1.5), _is_d8_plus_h),
    ):
        for values in product(coords, repeat=D8):
            p = np.asarray(values, dtype=np.float32)
            if float(np.dot(p, p)) <= 4.0001 and keep(p):
                pts.append(tuple(float(x) for x in p))

    pts = _sort_rows_by_norm_lex(pts)[:256]
    book = (np.asarray(pts, dtype=np.float32) * 2.0).astype(np.float32)
    _assert_shape(book, "lex15")
    _assert_shell_counts(book, {0.0: 1, 8.0: 240, 16.0: 15}, "lex15")
    return book


def sym256() -> np.ndarray:
    book = np.vstack([e8_roots_wide(), axis16_wide()]).astype(np.float32)
    _assert_shape(book, "sym256")
    _assert_shell_counts(book, {8.0: 240, 16.0: 16}, "sym256")
    _assert_negation_closed(book, "sym256")
    return book


def zero_axis15() -> np.ndarray:
    axis = axis16_wide()
    drop = np.array([-4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    kept_axis = np.asarray(
        [row for row in axis if not np.array_equal(row, drop)], dtype=np.float32
    )
    book = np.vstack([np.zeros((1, D8), dtype=np.float32), e8_roots_wide(), kept_axis])
    _assert_shape(book, "zero_axis15")
    _assert_shell_counts(book, {0.0: 1, 8.0: 240, 16.0: 15}, "zero_axis15")
    return book.astype(np.float32)


def voronoi_off() -> CodebookResult:
    # This is intentionally conservative. We enumerate local E8 candidates and
    # try root/2 offsets, then reject any offset that produces boundary ties.
    base = []
    for coords, keep in (
        ((-2.0, -1.0, 0.0, 1.0, 2.0), _is_d8),
        ((-1.5, -0.5, 0.5, 1.5), _is_d8_plus_h),
    ):
        for values in product(coords, repeat=D8):
            p = np.asarray(values, dtype=np.float32)
            if float(np.dot(p, p)) <= 8.0001 and keep(p):
                base.append(tuple(float(x) for x in p))
    lattice = np.asarray(_unique_rows(base), dtype=np.float32)
    roots = e8_roots_wide() / 2.0

    failures = []
    for root in roots:
        s = root / 2.0
        reps_by_key: dict[tuple[int, ...], tuple[float, np.ndarray, int]] = {}
        # Approximate coset classification by coordinates mod 2 on the
        # enumerated local window. This is a diagnostic builder, not production.
        for p in lattice + s[None, :]:
            key = tuple(np.mod(np.rint(2.0 * p).astype(np.int32), 4).tolist())
            norm2 = float(np.dot(p, p))
            if key not in reps_by_key:
                reps_by_key[key] = (norm2, p, 1)
                continue
            best_norm2, best_p, ties = reps_by_key[key]
            if norm2 < best_norm2 - 1e-7:
                reps_by_key[key] = (norm2, p, 1)
            elif abs(norm2 - best_norm2) <= 1e-7:
                reps_by_key[key] = (best_norm2, best_p, ties + 1)
        tie_count = sum(1 for _, _, ties in reps_by_key.values() if ties != 1)
        reps = [tuple((2.0 * p).tolist()) for _, p, _ in reps_by_key.values()]
        reps = _sort_rows_by_norm_lex(reps)
        if len(reps) >= 256 and tie_count == 0:
            book = np.asarray(reps[:256], dtype=np.float32)
            try:
                _assert_shape(book, "voronoi_off")
                _assert_negation_closed(book, "voronoi_off")
            except AssertionError as exc:
                failures.append(str(exc))
                continue
            return CodebookResult("voronoi_off", book)
        failures.append(f"offset={root.tolist()} reps={len(reps)} ties={tie_count}")

    return CodebookResult(
        "voronoi_off",
        None,
        status="failed",
        diagnostics="; ".join(failures[:8]),
    )


def make_codebook(name: str) -> CodebookResult:
    if name == "lex15":
        return CodebookResult(name, lex15())
    if name == "sym256":
        return CodebookResult(name, sym256())
    if name == "zero_axis15":
        return CodebookResult(name, zero_axis15())
    if name == "voronoi_off":
        return voronoi_off()
    raise ValueError(f"unknown codebook {name!r}; expected one of {BOOK_NAMES}")


def dump_book(path: str, name: str) -> None:
    result = make_codebook(name)
    if result.book is None:
        raise RuntimeError(f"{name} failed: {result.diagnostics}")
    np.save(path, result.book)


def fnv1a_float32_le(book: np.ndarray) -> int:
    h = 1469598103934665603
    prime = 1099511628211
    data = np.asarray(book, dtype="<f4").tobytes(order="C")
    for b in data:
        h ^= b
        h = (h * prime) & 0xFFFFFFFFFFFFFFFF
    return h


if __name__ == "__main__":
    for book_name in BOOK_NAMES:
        result = make_codebook(book_name)
        if result.book is None:
            print(f"{book_name}: {result.status} {result.diagnostics}")
            continue
        norms = np.sum(result.book * result.book, axis=1)
        shells = {float(k): int(np.sum(np.isclose(norms, k))) for k in sorted(set(norms))}
        print(
            f"{book_name}: shape={result.book.shape} shells={shells} "
            f"checksum=0x{fnv1a_float32_le(result.book):016x}"
        )
