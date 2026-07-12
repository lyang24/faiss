#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from codebooks import BOOK_NAMES, make_codebook

try:
    import h5py
except ImportError:  # pragma: no cover - exercised on minimal local envs.
    h5py = None


class ArrayDataset:
    def __init__(self, path: str) -> None:
        self.path = path
        self.kind = "npz" if path.endswith(".npz") else "hdf5"
        self._npz = None
        if self.kind == "npz":
            self._npz = np.load(path)
        elif h5py is None:
            raise RuntimeError("h5py is required for HDF5 input; use .npz for local smoke")

    def train_shape(self) -> tuple[int, int]:
        if self.kind == "npz":
            return tuple(self._npz["train"].shape)
        with h5py.File(self.path, "r") as f:
            return tuple(f["train"].shape)

    def test_slice(self, stop: int) -> np.ndarray:
        if self.kind == "npz":
            return np.asarray(self._npz["test"][:stop], dtype=np.float32)
        with h5py.File(self.path, "r") as f:
            return np.asarray(f["test"][:stop], dtype=np.float32)

    def train_slice(self, start: int, stop: int) -> np.ndarray:
        if self.kind == "npz":
            return np.asarray(self._npz["train"][start:stop], dtype=np.float32)
        with h5py.File(self.path, "r") as f:
            return np.asarray(f["train"][start:stop], dtype=np.float32)

    def train_rows(self, ids: np.ndarray) -> np.ndarray:
        if self.kind == "npz":
            return np.asarray(self._npz["train"][ids], dtype=np.float32)
        with h5py.File(self.path, "r") as f:
            return np.asarray(f["train"][ids], dtype=np.float32)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1)
    return (x / np.where(norms > 0, norms, 1.0)[:, None]).astype(np.float32)


def random_rotation(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((d, d), dtype=np.float32)
    q, r = np.linalg.qr(a.astype(np.float64))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    q = q * signs[None, :]
    return q.astype(np.float32)


def maybe_transform(x: np.ndarray, mean: np.ndarray | None, rot: np.ndarray | None) -> np.ndarray:
    y = x.astype(np.float32, copy=True)
    if mean is not None:
        y -= mean[None, :]
    if rot is not None:
        y = y @ rot
    return np.ascontiguousarray(y, dtype=np.float32)


def dataset_mean(ds: ArrayDataset, nb: int, chunk: int) -> np.ndarray:
    d = ds.train_shape()[1]
    acc = np.zeros(d, dtype=np.float64)
    count = 0
    for i0 in range(0, nb, chunk):
        x = ds.train_slice(i0, min(nb, i0 + chunk))
        acc += x.sum(axis=0, dtype=np.float64)
        count += x.shape[0]
    return (acc / count).astype(np.float32)


def merge_topk(
    old_vals: np.ndarray,
    old_ids: np.ndarray,
    new_vals: np.ndarray,
    new_ids: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    vals = np.concatenate([old_vals, new_vals], axis=1)
    ids = np.concatenate([old_ids, new_ids], axis=1)
    kk = min(k, vals.shape[1])
    idx = np.argpartition(vals, -kk, axis=1)[:, -kk:]
    vals2 = np.take_along_axis(vals, idx, axis=1)
    ids2 = np.take_along_axis(ids, idx, axis=1)
    order = np.argsort(-vals2, axis=1)
    return np.take_along_axis(vals2, order, axis=1), np.take_along_axis(ids2, order, axis=1)


def topk_block(scores: np.ndarray, base0: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    kk = min(k, scores.shape[1])
    idx = np.argpartition(scores, -kk, axis=1)[:, -kk:]
    vals = np.take_along_axis(scores, idx, axis=1)
    ids = idx + base0
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(vals, order, axis=1), np.take_along_axis(ids, order, axis=1)


class RunningErrors:
    def __init__(self) -> None:
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.sum = 0.0
        self.count = 0

    def add(self, err: np.ndarray) -> None:
        self.abs_sum += float(np.sum(np.abs(err)))
        self.sq_sum += float(np.sum(err * err))
        self.sum += float(np.sum(err))
        self.count += int(err.size)

    def row(self, prefix: str) -> dict[str, float]:
        if self.count == 0:
            return {
                f"{prefix}_mae": float("nan"),
                f"{prefix}_rmse": float("nan"),
                f"{prefix}_bias": float("nan"),
                f"{prefix}_err_std": float("nan"),
            }
        mean = self.sum / self.count
        mse = self.sq_sum / self.count
        return {
            f"{prefix}_mae": self.abs_sum / self.count,
            f"{prefix}_rmse": float(np.sqrt(mse)),
            f"{prefix}_bias": mean,
            f"{prefix}_err_std": float(np.sqrt(max(0.0, mse - mean * mean))),
        }


class RhoStats:
    def __init__(self) -> None:
        self.values: list[np.ndarray] = []

    def add(self, rho: np.ndarray) -> None:
        self.values.append(np.asarray(rho, dtype=np.float32))

    def row(self) -> dict[str, float]:
        x = np.concatenate(self.values) if self.values else np.asarray([], dtype=np.float32)
        return {
            "rho_mean": float(np.mean(x)),
            "rho_std": float(np.std(x)),
            "rho_p1": float(np.percentile(x, 1)),
            "rho_p5": float(np.percentile(x, 5)),
            "rho_p50": float(np.percentile(x, 50)),
        }


def encode_sign_dirs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d = x.shape
    norms = np.linalg.norm(x, axis=1).astype(np.float32)
    nonzero = norms > 0
    safe = np.where(nonzero, norms, 1.0).astype(np.float32)
    u = (np.where(x >= 0, 1.0, -1.0).astype(np.float32) / np.sqrt(np.float32(d))).astype(
        np.float32
    )
    u = np.where(nonzero[:, None], u, 0.0).astype(np.float32)
    rho = np.sum((x / safe[:, None]) * u, axis=1).astype(np.float32)
    rho = np.where(nonzero, rho, 0.0).astype(np.float32)
    dp = np.where(nonzero, safe / np.maximum(rho, np.float32(1e-20)), 0.0).astype(
        np.float32
    )
    return u, rho, dp


def encode_book_dirs(x: np.ndarray, book: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d = x.shape
    norms = np.linalg.norm(x, axis=1).astype(np.float32)
    nonzero = norms > 0
    safe = np.where(nonzero, norms, 1.0).astype(np.float32)
    scaled = x * (np.sqrt(np.float32(d)) / safe)[:, None]
    decoded = np.empty_like(x, dtype=np.float32)
    book_norm = np.sum(book * book, axis=1).astype(np.float32)

    for j in range(d // 8):
        chunk = scaled[:, j * 8 : (j + 1) * 8]
        dist = (
            np.sum(chunk * chunk, axis=1)[:, None]
            + book_norm[None, :]
            - 2.0 * (chunk @ book.T)
        )
        code = np.argmin(dist, axis=1)
        decoded[:, j * 8 : (j + 1) * 8] = book[code]

    dec_norm = np.linalg.norm(decoded, axis=1).astype(np.float32)
    dec_nonzero = dec_norm > 0
    u = decoded / np.where(dec_nonzero, dec_norm, 1.0)[:, None]
    valid = nonzero & dec_nonzero
    rho = np.sum((x / safe[:, None]) * u, axis=1).astype(np.float32)
    rho = np.where(valid, rho, 0.0).astype(np.float32)
    u = np.where(valid[:, None], u, 0.0).astype(np.float32)
    dp = np.where(valid, safe / np.maximum(rho, np.float32(1e-20)), 0.0).astype(
        np.float32
    )
    return u, rho, dp


def estimated_scores(q: np.ndarray, x_norm2: np.ndarray, u: np.ndarray, dp: np.ndarray, metric: str) -> np.ndarray:
    ip = (q @ u.T) * dp[None, :]
    if metric == "ip":
        return ip
    q_norm2 = np.sum(q * q, axis=1).astype(np.float32)
    return -(q_norm2[:, None] + x_norm2[None, :] - 2.0 * ip)


def recall_at(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    return sum(len(set(p[:k]).intersection(t[:k])) for p, t in zip(pred, truth)) / (
        pred.shape[0] * k
    )


def gather_rows(ds: ArrayDataset, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = ids.reshape(-1)
    uniq, inverse = np.unique(flat, return_inverse=True)
    rows = ds.train_rows(uniq)
    return rows, inverse.reshape(ids.shape)


def exact_scores(q: np.ndarray, x: np.ndarray, metric: str) -> np.ndarray:
    ip = q @ x.T
    if metric == "ip":
        return ip
    q_norm2 = np.sum(q * q, axis=1).astype(np.float32)
    x_norm2 = np.sum(x * x, axis=1).astype(np.float32)
    return -(q_norm2[:, None] + x_norm2[None, :] - 2.0 * ip)


def evaluate_top_pool(
    ds: ArrayDataset,
    q: np.ndarray,
    ids: np.ndarray,
    metric: str,
    mean: np.ndarray | None,
    rot: np.ndarray | None,
    normalize: bool,
    books: dict[str, np.ndarray | None],
    query_batch: int,
) -> dict[str, RunningErrors]:
    rows, inverse = gather_rows(ds, ids)
    x = maybe_transform(rows, mean, rot)
    if normalize:
        x = normalize_rows(x)
    x_norm2 = np.sum(x * x, axis=1).astype(np.float32)
    enc_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, book in books.items():
        if name == "sign":
            enc_cache[name] = encode_sign_dirs(x)
        elif book is not None:
            enc_cache[name] = encode_book_dirs(x, book)

    out = {name: RunningErrors() for name in books}
    for q0 in range(0, q.shape[0], query_batch):
        q1 = min(q.shape[0], q0 + query_batch)
        exact_by_unique = exact_scores(q[q0:q1], x, metric)
        for local_qi in range(q1 - q0):
            unique_cols = inverse[q0 + local_qi]
            exact = exact_by_unique[local_qi, unique_cols]
            for name, (u, _, dp) in enc_cache.items():
                scores = estimated_scores(
                    q[q0 + local_qi : q0 + local_qi + 1],
                    x_norm2,
                    u,
                    dp,
                    metric,
                )[0, unique_cols]
                out[name].add(scores - exact)
    return out


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    ds = ArrayDataset(args.dataset)
    nb = min(args.nb, ds.train_shape()[0])
    q = ds.test_slice(args.nq)
    d = q.shape[1]
    if d % 8 != 0:
        raise ValueError(f"E8 codebooks require d % 8 == 0, got d={d}")

    mean = dataset_mean(ds, nb, args.chunk) if args.center else None
    rot = random_rotation(d, args.seed) if args.rotate == "qr" else None
    q = maybe_transform(q, mean, rot)
    if args.normalize:
        q = normalize_rows(q)

    requested_books = ["sign"] + list(args.books)
    books: dict[str, np.ndarray | None] = {"sign": None}
    book_status: dict[str, str] = {"sign": "ok"}
    for name in args.books:
        result = make_codebook(name)
        books[name] = result.book
        book_status[name] = result.status if result.book is not None else result.diagnostics

    nq = q.shape[0]
    exact_vals = np.full((nq, 0), -np.inf, dtype=np.float32)
    exact_ids = np.zeros((nq, 0), dtype=np.int64)
    exact_pool_vals = np.full((nq, 0), -np.inf, dtype=np.float32)
    exact_pool_ids = np.zeros((nq, 0), dtype=np.int64)

    pred_vals = {
        name: np.full((nq, 0), -np.inf, dtype=np.float32) for name in requested_books
    }
    pred_ids = {name: np.zeros((nq, 0), dtype=np.int64) for name in requested_books}
    global_errors = {name: RunningErrors() for name in requested_books}
    rhos = {name: RhoStats() for name in requested_books}

    t0 = time.perf_counter()
    for base0 in range(0, nb, args.chunk):
        raw = ds.train_slice(base0, min(nb, base0 + args.chunk))
        x = maybe_transform(raw, mean, rot)
        if args.normalize:
            x = normalize_rows(x)
        x_norm2 = np.sum(x * x, axis=1).astype(np.float32)
        exact = exact_scores(q, x, args.metric)
        vals, ids = topk_block(exact, base0, args.k)
        exact_vals, exact_ids = merge_topk(exact_vals, exact_ids, vals, ids, args.k)
        vals, ids = topk_block(exact, base0, args.top_pool)
        exact_pool_vals, exact_pool_ids = merge_topk(
            exact_pool_vals, exact_pool_ids, vals, ids, args.top_pool
        )

        enc = {"sign": encode_sign_dirs(x)}
        for name, book in books.items():
            if name == "sign" or book is None:
                continue
            enc[name] = encode_book_dirs(x, book)

        for name, (u, rho, dp) in enc.items():
            scores = estimated_scores(q, x_norm2, u, dp, args.metric)
            vals, ids = topk_block(scores, base0, args.k)
            pred_vals[name], pred_ids[name] = merge_topk(
                pred_vals[name], pred_ids[name], vals, ids, args.k
            )
            global_errors[name].add(scores - exact)
            rhos[name].add(rho)

        if args.progress and (base0 // args.chunk) % args.progress == 0:
            print(
                json.dumps(
                    {
                        "progress": min(base0 + args.chunk, nb),
                        "elapsed_s": time.perf_counter() - t0,
                    }
                ),
                flush=True,
            )

    top_pool_errors = evaluate_top_pool(
        ds,
        q,
        exact_pool_ids,
        args.metric,
        mean,
        rot,
        args.normalize,
        books,
        args.pool_query_batch,
    )

    rows: list[dict[str, object]] = []
    for name in requested_books:
        row: dict[str, object] = {
            "dataset": args.dataset,
            "nb": nb,
            "nq": nq,
            "dim": d,
            "metric": args.metric,
            "normalize": args.normalize,
            "center": args.center,
            "rotate": args.rotate,
            "seed": args.seed,
            "book": name,
            "book_status": book_status[name],
            "recall@1": recall_at(pred_ids[name], exact_ids, 1),
            "recall@10": recall_at(pred_ids[name], exact_ids, min(10, args.k)),
            "recall@100": recall_at(pred_ids[name], exact_ids, min(100, args.k)),
            "top1": float(np.mean(pred_ids[name][:, 0] == exact_ids[:, 0])),
        }
        row.update(rhos[name].row())
        row.update(global_errors[name].row("global"))
        row.update(top_pool_errors[name].row("top1000"))
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--metric", choices=("ip", "l2"), required=True)
    ap.add_argument("--nb", type=int, default=100000)
    ap.add_argument("--nq", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--top-pool", type=int, default=1000)
    ap.add_argument("--pool-query-batch", type=int, default=50)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--rotate", choices=("none", "qr"), default="none")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--books", nargs="+", default=["lex15", "sym256", "zero_axis15"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress", type=int, default=10)
    args = ap.parse_args()

    for name in args.books:
        if name not in BOOK_NAMES:
            raise ValueError(f"unknown book {name!r}")

    rows = run(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
