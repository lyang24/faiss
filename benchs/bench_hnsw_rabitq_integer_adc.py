#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Compare RaBitQ scorers on one FP32-constructed HNSW graph."""

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


def current_rss_bytes():
    with open("/proc/self/statm", encoding="ascii") as statm:
        resident_pages = int(statm.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


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
    parser.add_argument(
        "--ef", type=int, nargs="+", default=[16, 24, 32, 48, 64, 96, 128]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch1-iterations", type=int, default=200)
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
    if not np.allclose(np.linalg.norm(xb, axis=1), 1.0, rtol=2e-5, atol=2e-5):
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
            "args": vars(args)
            | {
                "base": str(args.base),
                "query": str(args.query),
                "output": str(args.output),
            },
            "platform": platform.platform(),
            "compile": faiss.get_compile_options(),
            "omp_env": os.environ.get("OMP_NUM_THREADS"),
            "blas_env": os.environ.get("OPENBLAS_NUM_THREADS"),
            "base_sha256": digest(xb),
            "query_sha256": digest(xq),
            "ground_truth_sha256": digest(truth),
        }
    )

    index = faiss.IndexHNSWRaBitQ(xb.shape[1], args.m, args.bits, faiss.METRIC_L2)
    index.hnsw.efConstruction = args.efc
    start = time.perf_counter()
    index.train(train)
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    index.add_with_fp32_graph(encoded)
    build_seconds = time.perf_counter() - start
    storage = faiss.downcast_index(index.storage)
    graph_digest = digest(faiss.vector_to_array(index.hnsw.neighbors))
    packed_bytes = faiss.vector_to_array(storage.codes).nbytes
    graph_size = graph_bytes(index)
    emit(
        {
            "kind": "build",
            "train_seconds": train_seconds,
            "build_seconds": build_seconds,
            "ntotal": index.ntotal,
            "packed_code_size": storage.code_size,
            "packed_bytes": packed_bytes,
            "graph_bytes": graph_size,
            "graph_sha256": graph_digest,
            "native_dotprod": bool(storage.expanded_integer_uses_native_dotprod()),
        }
    )

    modes = (
        ("packed_staged", faiss.RABITQ_FULL_CODE_PACKED),
        ("expanded_float", faiss.RABITQ_FULL_CODE_EXPANDED),
        ("expanded_int8", faiss.RABITQ_FULL_CODE_INT8),
    )
    for name, mode in modes:
        rss_before = current_rss_bytes()
        start = time.perf_counter()
        index.set_full_code_mode(mode)
        mode_setup_seconds = time.perf_counter() - start
        rss_after = current_rss_bytes()
        storage = faiss.downcast_index(index.storage)
        expanded_bytes = faiss.vector_to_array(storage.expanded_codes).nbytes
        if digest(faiss.vector_to_array(index.hnsw.neighbors)) != graph_digest:
            raise RuntimeError("scorer selection changed the HNSW graph")
        emit(
            {
                "kind": "mode",
                "variant": name,
                "mode_setup_seconds": mode_setup_seconds,
                "packed_bytes": packed_bytes,
                "expanded_bytes": expanded_bytes,
                "serving_bytes": graph_size + packed_bytes + expanded_bytes,
                "rss_before": rss_before,
                "rss_after": rss_after,
                "hnsw_search_method": index.hnsw.search_method,
            }
        )

        searcher = faiss.IndexPreTransform(rotation, index) if rotation else index
        for batch in (1, args.nq):
            queries = xq[:batch]
            expected = truth[:batch]
            iterations = args.batch1_iterations if batch == 1 else 1
            for ef in args.ef:
                params = faiss.SearchParametersHNSW(efSearch=ef)
                searcher.search(queries, 10, params=params)
                samples = []
                ids = None
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    for _ in range(iterations):
                        _, ids = searcher.search(queries, 10, params=params)
                    samples.append(time.perf_counter() - start)
                emit(
                    {
                        "kind": "search",
                        "variant": name,
                        "batch": batch,
                        "iterations_per_sample": iterations,
                        "ef": ef,
                        "recall_at_10": recall_at_10(ids, expected),
                        "qps": batch * iterations / float(np.median(samples)),
                        "seconds_samples": samples,
                    }
                )

    emit({"kind": "validation", "same_graph": True})
    if output:
        output.close()


if __name__ == "__main__":
    main()
