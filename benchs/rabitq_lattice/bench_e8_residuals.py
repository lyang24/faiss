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

from bench_e8_books import (
    ArrayDataset,
    RhoStats,
    RunningErrors,
    dataset_mean,
    encode_book_dirs,
    encode_sign_dirs,
    exact_scores,
    maybe_transform,
    merge_topk,
    normalize_rows,
    random_rotation,
    recall_at,
    topk_block,
)
from codebooks import BOOK_NAMES, make_codebook


def require_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local env.
        raise RuntimeError(
            "bench_e8_residuals.py requires the FAISS Python module for "
            "kmeans and coarse assignment"
        ) from exc
    return faiss


def train_coarse_quantizer(
    xt: np.ndarray,
    nlist: int,
    niter: int,
    seed: int,
    verbose: bool,
) -> np.ndarray:
    faiss = require_faiss()
    km = faiss.Kmeans(
        xt.shape[1],
        nlist,
        niter=niter,
        seed=seed,
        verbose=verbose,
        spherical=False,
        gpu=False,
    )
    km.train(np.ascontiguousarray(xt, dtype=np.float32))
    return np.ascontiguousarray(km.centroids, dtype=np.float32)


def make_l2_index(x: np.ndarray):
    faiss = require_faiss()
    index = faiss.IndexFlatL2(x.shape[1])
    index.add(np.ascontiguousarray(x, dtype=np.float32))
    return index


def assign_centroids(index, x: np.ndarray, k: int) -> np.ndarray:
    _, ids = index.search(np.ascontiguousarray(x, dtype=np.float32), k)
    return np.asarray(ids, dtype=np.int64)


def masked_merge_topk(
    old_vals: np.ndarray,
    old_ids: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray,
    base0: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    masked = np.where(mask, scores, -np.inf).astype(np.float32)
    vals, ids = topk_block(masked, base0, k)
    valid = np.take_along_axis(mask, ids - base0, axis=1)
    vals = np.where(valid, vals, -np.inf).astype(np.float32)
    ids = np.where(valid, ids, -1).astype(np.int64)
    return merge_topk(old_vals, old_ids, vals, ids, k)


def row_membership(labels: np.ndarray, probes: np.ndarray) -> np.ndarray:
    # nq and nprobe are small in this harness; the dense comparison keeps the
    # implementation simple and avoids building Python sets in the hot loop.
    return np.any(labels[None, :, None] == probes[:, None, :], axis=2)


def residual_estimated_scores(
    q: np.ndarray,
    centroids_for_x: np.ndarray,
    residual_norm2: np.ndarray,
    u: np.ndarray,
    dp: np.ndarray,
    metric: str,
) -> np.ndarray:
    if metric == "ip":
        return q @ centroids_for_x.T + (q @ u.T) * dp[None, :]

    q_minus_c = q[:, None, :] - centroids_for_x[None, :, :]
    cross = np.einsum("nmd,md->nm", q_minus_c, u, optimize=True) * dp[None, :]
    qmc_norm2 = np.einsum("nmd,nmd->nm", q_minus_c, q_minus_c, optimize=True)
    return -(qmc_norm2 + residual_norm2[None, :] - 2.0 * cross)


def residual_exact_scores(
    q: np.ndarray,
    x: np.ndarray,
    metric: str,
) -> np.ndarray:
    return exact_scores(q, x, metric)


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    ds = ArrayDataset(args.dataset)
    nb = min(args.nb, ds.train_shape()[0])
    q_raw = ds.test_slice(args.nq)
    d = q_raw.shape[1]
    if d % 8 != 0:
        raise ValueError(f"E8 codebooks require d % 8 == 0, got d={d}")
    if args.nprobe > args.nlist:
        raise ValueError("--nprobe must be <= --nlist")

    mean = dataset_mean(ds, nb, args.chunk) if args.center else None
    rot = random_rotation(d, args.seed) if args.rotate == "qr" else None
    q = maybe_transform(q_raw, mean, rot)
    if args.normalize:
        q = normalize_rows(q)

    train_n = min(args.kmeans_train, nb)
    xt = ds.train_slice(0, train_n)
    xt = maybe_transform(xt, mean, rot)
    if args.normalize:
        xt = normalize_rows(xt)

    t0 = time.perf_counter()
    centroids = train_coarse_quantizer(
        xt, args.nlist, args.kmeans_iter, args.seed, args.verbose_kmeans
    )
    coarse_index = make_l2_index(centroids)
    q_probes = assign_centroids(coarse_index, q, args.nprobe)
    train_s = time.perf_counter() - t0

    requested_books = ["ivf_exact", "sign"] + list(args.books)
    books: dict[str, np.ndarray | None] = {"sign": None}
    book_status: dict[str, str] = {"ivf_exact": "ok", "sign": "ok"}
    for name in args.books:
        result = make_codebook(name)
        books[name] = result.book
        book_status[name] = result.status if result.book is not None else result.diagnostics

    nq = q.shape[0]
    exact_vals = np.full((nq, 0), -np.inf, dtype=np.float32)
    exact_ids = np.zeros((nq, 0), dtype=np.int64)
    pred_vals = {
        name: np.full((nq, 0), -np.inf, dtype=np.float32) for name in requested_books
    }
    pred_ids = {name: np.zeros((nq, 0), dtype=np.int64) for name in requested_books}
    scanned_errors = {name: RunningErrors() for name in requested_books if name != "ivf_exact"}
    rhos = {name: RhoStats() for name in requested_books if name != "ivf_exact"}

    n_scanned = 0
    list_counts = np.zeros(args.nlist, dtype=np.int64)
    scan_t0 = time.perf_counter()
    for base0 in range(0, nb, args.chunk):
        raw = ds.train_slice(base0, min(nb, base0 + args.chunk))
        x = maybe_transform(raw, mean, rot)
        if args.normalize:
            x = normalize_rows(x)

        labels = assign_centroids(coarse_index, x, 1).reshape(-1)
        list_counts += np.bincount(labels, minlength=args.nlist)
        mask = row_membership(labels, q_probes)
        n_scanned += int(mask.sum())

        exact = residual_exact_scores(q, x, args.metric)
        vals, ids = topk_block(exact, base0, args.k)
        exact_vals, exact_ids = merge_topk(exact_vals, exact_ids, vals, ids, args.k)
        pred_vals["ivf_exact"], pred_ids["ivf_exact"] = masked_merge_topk(
            pred_vals["ivf_exact"],
            pred_ids["ivf_exact"],
            exact,
            mask,
            base0,
            args.k,
        )

        centroids_for_x = centroids[labels]
        residuals = np.ascontiguousarray(x - centroids_for_x, dtype=np.float32)
        residual_norm2 = np.sum(residuals * residuals, axis=1).astype(np.float32)

        enc = {"sign": encode_sign_dirs(residuals)}
        for name, book in books.items():
            if name == "sign" or book is None:
                continue
            enc[name] = encode_book_dirs(residuals, book)

        for name, (u, rho, dp) in enc.items():
            scores = residual_estimated_scores(
                q, centroids_for_x, residual_norm2, u, dp, args.metric
            )
            pred_vals[name], pred_ids[name] = masked_merge_topk(
                pred_vals[name], pred_ids[name], scores, mask, base0, args.k
            )
            scanned_errors[name].add((scores - exact)[mask])
            rhos[name].add(rho)

        if args.progress and (base0 // args.chunk) % args.progress == 0:
            print(
                json.dumps(
                    {
                        "progress": min(base0 + args.chunk, nb),
                        "elapsed_s": time.perf_counter() - scan_t0,
                        "avg_scanned_per_query": n_scanned / max(1, nq),
                    }
                ),
                flush=True,
            )

    rows: list[dict[str, object]] = []
    avg_scanned = n_scanned / max(1, nq)
    imbalance = float(np.mean(list_counts * list_counts) / (np.mean(list_counts) ** 2))
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
            "nlist": args.nlist,
            "nprobe": args.nprobe,
            "kmeans_train": train_n,
            "kmeans_iter": args.kmeans_iter,
            "book": name,
            "book_status": book_status.get(name, "ok"),
            "recall@1": recall_at(pred_ids[name], exact_ids, 1),
            "recall@10": recall_at(pred_ids[name], exact_ids, min(10, args.k)),
            "recall@100": recall_at(pred_ids[name], exact_ids, min(100, args.k)),
            "top1": float(np.mean(pred_ids[name][:, 0] == exact_ids[:, 0])),
            "avg_scanned_per_query": avg_scanned,
            "scan_fraction": avg_scanned / nb,
            "list_imbalance": imbalance,
            "kmeans_train_s": train_s,
            "scan_s": time.perf_counter() - scan_t0,
        }
        if name == "ivf_exact":
            row.update(
                {
                    "rho_mean": float("nan"),
                    "rho_std": float("nan"),
                    "rho_p1": float("nan"),
                    "rho_p5": float("nan"),
                    "rho_p50": float("nan"),
                }
            )
            row.update(RunningErrors().row("scanned"))
        else:
            row.update(rhos[name].row())
            row.update(scanned_errors[name].row("scanned"))
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
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--rotate", choices=("none", "qr"), default="qr")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--nprobe", type=int, default=64)
    ap.add_argument("--kmeans-train", type=int, default=100000)
    ap.add_argument("--kmeans-iter", type=int, default=20)
    ap.add_argument("--verbose-kmeans", action="store_true")
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
