#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Compare native and FP32 construction for the same HNSW RaBitQ codec."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import faiss
import numpy as np


def digest(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def recall_at_10(actual, expected):
    return sum(
        len(set(row[row >= 0]).intersection(truth))
        for row, truth in zip(actual, expected)
    ) / actual.size


def graph_bytes(index):
    return sum(
        faiss.vector_to_array(getattr(index.hnsw, field)).nbytes
        for field in ("levels", "offsets", "neighbors")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--nb", type=int, default=100000)
    parser.add_argument("--nq", type=int, default=1000)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--efc", type=int, default=100)
    parser.add_argument("--bits", type=int, default=7)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--ef", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rotation", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output = args.output.open("w", encoding="utf-8")

    def emit(row):
        line = json.dumps(row, sort_keys=True)
        print(line, flush=True)
        if output:
            print(line, file=output, flush=True)

    xb = np.ascontiguousarray(np.load(args.base, mmap_mode="r")[: args.nb], dtype="float32")
    xq = np.ascontiguousarray(np.load(args.query, mmap_mode="r")[: args.nq], dtype="float32")
    if xb.ndim != 2 or xq.shape[1] != xb.shape[1]:
        raise ValueError("base/query dimensions do not match")
    norms = np.linalg.norm(xb, axis=1)
    if not np.allclose(norms, 1.0, rtol=2e-5, atol=2e-5):
        raise ValueError("this benchmark expects normalized base vectors")
    if not np.allclose(np.linalg.norm(xq, axis=1), 1.0, rtol=2e-5, atol=2e-5):
        raise ValueError("this benchmark expects normalized query vectors")

    faiss.omp_set_num_threads(args.threads)
    exact = faiss.IndexFlatL2(xb.shape[1])
    exact.add(xb)
    _, truth = exact.search(xq, 10)
    del exact

    rotation = None
    encoded = xb
    if args.rotation:
        rotation = faiss.RandomRotationMatrix(xb.shape[1], xb.shape[1])
        rotation.init(1234)
        encoded = rotation.apply_py(xb)
    train = np.ascontiguousarray(encoded[: min(len(encoded), 50000)])

    emit(
        {
            "kind": "metadata",
            "args": vars(args) | {"base": str(args.base), "query": str(args.query), "output": str(args.output)},
            "platform": platform.platform(),
            "compile": faiss.get_compile_options(),
            "omp_env": os.environ.get("OMP_NUM_THREADS"),
            "blas_env": os.environ.get("OPENBLAS_NUM_THREADS"),
            "base_sha256": digest(xb),
            "query_sha256": digest(xq),
            "ground_truth_sha256": digest(truth),
        }
    )

    indexes = {}
    code_digests = {}
    for name in ("native", "fp32"):
        index = faiss.IndexHNSWRaBitQ(xb.shape[1], args.m, args.bits, faiss.METRIC_L2)
        index.hnsw.efConstruction = args.efc
        train_start = time.perf_counter()
        index.train(train)
        train_seconds = time.perf_counter() - train_start
        build_start = time.perf_counter()
        if name == "fp32":
            index.add_with_fp32_graph(encoded)
        else:
            index.add(encoded)
        build_seconds = time.perf_counter() - build_start
        storage = faiss.downcast_index(index.storage)
        codes = faiss.vector_to_array(storage.codes)
        code_digests[name] = digest(codes)
        indexes[name] = index
        emit(
            {
                "kind": "build",
                "variant": name,
                "train_seconds": train_seconds,
                "build_seconds": build_seconds,
                "ntotal": index.ntotal,
                "code_size": storage.code_size,
                "codes_sha256": code_digests[name],
                "graph_sha256": digest(faiss.vector_to_array(index.hnsw.neighbors)),
                "graph_bytes": graph_bytes(index),
                "serving_bytes": graph_bytes(index) + codes.nbytes,
                "temporary_fp32_construction_bytes": encoded.nbytes if name == "fp32" else 0,
                "entry_point": index.hnsw.entry_point,
                "max_level": index.hnsw.max_level,
                "fp32_graph_built": index.fp32_graph_built,
            }
        )

    if code_digests["native"] != code_digests["fp32"]:
        raise RuntimeError("construction policy changed RaBitQ document codes")

    for name, index in indexes.items():
        searcher = faiss.IndexPreTransform(rotation, index) if rotation else index
        for ef in args.ef:
            params = faiss.SearchParametersHNSW(efSearch=ef)
            searcher.search(xq, 10, params=params)
            samples = []
            ids = None
            for _ in range(args.repeats):
                start = time.perf_counter()
                _, ids = searcher.search(xq, 10, params=params)
                samples.append(time.perf_counter() - start)
            emit(
                {
                    "kind": "search",
                    "variant": name,
                    "ef": ef,
                    "recall_at_10": recall_at_10(ids, truth),
                    "qps": len(xq) / float(np.median(samples)),
                    "seconds_samples": samples,
                }
            )

    emit({"kind": "validation", "same_codes": True})
    if output:
        output.close()


if __name__ == "__main__":
    main()
