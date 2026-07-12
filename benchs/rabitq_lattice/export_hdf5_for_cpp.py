#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import h5py
import numpy as np


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1)
    return (x / np.where(norms > 0, norms, 1.0)[:, None]).astype(np.float32)


def random_rotation(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((d, d), dtype=np.float32)
    q, r = np.linalg.qr(a.astype(np.float64))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return (q * signs[None, :]).astype(np.float32)


def exact_topk_ip(xq: np.ndarray, xb: np.ndarray, k: int, chunk: int) -> np.ndarray:
    nq = xq.shape[0]
    vals = np.full((nq, 0), -np.inf, dtype=np.float32)
    ids = np.zeros((nq, 0), dtype=np.int64)
    for i0 in range(0, xb.shape[0], chunk):
        x = xb[i0 : i0 + chunk]
        scores = xq @ x.T
        kk = min(k, scores.shape[1])
        idx = np.argpartition(scores, -kk, axis=1)[:, -kk:]
        v = np.take_along_axis(scores, idx, axis=1)
        ii = idx + i0
        vals = np.concatenate([vals, v], axis=1)
        ids = np.concatenate([ids, ii], axis=1)
        keep = np.argpartition(vals, -k, axis=1)[:, -k:]
        vals = np.take_along_axis(vals, keep, axis=1)
        ids = np.take_along_axis(ids, keep, axis=1)
        order = np.argsort(-vals, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)
        ids = np.take_along_axis(ids, order, axis=1)
    return ids.astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nb", type=int, required=True)
    ap.add_argument("--nq", type=int, required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--rotate", choices=("none", "qr"), default="none")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--gt-chunk", type=int, default=10000)
    args = ap.parse_args()

    with h5py.File(args.dataset, "r") as f:
        xb = np.asarray(f["train"][: args.nb], dtype=np.float32)
        xq = np.asarray(f["test"][: args.nq], dtype=np.float32)

    if args.normalize:
        xb = normalize_rows(xb)
        xq = normalize_rows(xq)
    if args.rotate == "qr":
        rot = random_rotation(xb.shape[1], args.seed)
        xb = np.ascontiguousarray(xb @ rot, dtype=np.float32)
        xq = np.ascontiguousarray(xq @ rot, dtype=np.float32)

    gt = exact_topk_ip(xq, xb, args.k, args.gt_chunk)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(struct.pack("<qqqq", xb.shape[0], xq.shape[0], xb.shape[1], args.k))
        f.write(np.ascontiguousarray(xb, dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(xq, dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(gt, dtype="<i8").tobytes())
    print(
        {
            "out": str(out),
            "nb": xb.shape[0],
            "nq": xq.shape[0],
            "d": xb.shape[1],
            "k": args.k,
        }
    )


if __name__ == "__main__":
    main()
