#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Sweep serial and block-parallel rotation with RaBitQ integer ADC."""

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


def search_parameters(ef, transform_threads, block_size):
    inner = faiss.SearchParametersHNSW(efSearch=ef)
    if transform_threads == 0:
        return inner, inner
    outer = faiss.SearchParametersPreTransform()
    outer.index_params = inner
    outer.transform_threads = transform_threads
    outer.transform_block_size = block_size
    # Keep both Python proxies alive through the C++ call.
    return outer, (outer, inner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--nb", type=int, default=100000)
    parser.add_argument("--nq", type=int, default=1000)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--efc", type=int, default=100)
    parser.add_argument("--bits", type=int, default=7)
    parser.add_argument("--search-threads", type=int, default=16)
    parser.add_argument("--transform-threads", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--block-size", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--ef", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch1-iterations", type=int, default=200)
    parser.add_argument(
        "--confirmation",
        action="store_true",
        help="run one frozen candidate bracketed by serial controls",
    )
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

    faiss.omp_set_num_threads(args.search_threads)
    exact = faiss.IndexFlatL2(xb.shape[1])
    exact.add(xb)
    _, truth = exact.search(xq, 10)
    del exact

    rotation = faiss.RandomRotationMatrix(xb.shape[1], xb.shape[1])
    rotation.init(1234)
    encoded = rotation.apply_py(xb)
    train = np.ascontiguousarray(encoded[: min(len(encoded), 50000)])
    index = faiss.IndexHNSWRaBitQ(xb.shape[1], args.m, args.bits, faiss.METRIC_L2)
    index.hnsw.efConstruction = args.efc
    index.train(train)
    index.add_with_fp32_graph(encoded)
    index.set_full_code_mode(faiss.RABITQ_FULL_CODE_INT8)
    searcher = faiss.IndexPreTransform(rotation, index)

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
            "graph_sha256": digest(faiss.vector_to_array(index.hnsw.neighbors)),
            "native_dotprod": bool(
                faiss.downcast_index(index.storage).expanded_integer_uses_native_dotprod()
            ),
        }
    )

    if args.confirmation:
        if len(args.transform_threads) != 1 or len(args.block_size) != 1:
            raise ValueError("confirmation requires one thread and block candidate")
        threads = args.transform_threads[0]
        block = args.block_size[0]
        strategies = [
            ("serial_before", 0, 0),
            (f"parallel_t{threads}_b{block}", threads, block),
            ("serial_after", 0, 0),
        ]
    else:
        strategies = [("serial", 0, 0)]
        strategies.extend(
            (f"parallel_t{threads}_b{block}", threads, block)
            for threads in args.transform_threads
            for block in args.block_size
        )
    for name, transform_threads, block_size in strategies:
        for batch in (1, args.nq):
            queries = xq[:batch]
            expected = truth[:batch]
            iterations = args.batch1_iterations if batch == 1 else 1
            for ef in args.ef:
                params, keepalive = search_parameters(
                    ef, transform_threads, block_size
                )
                searcher.search(queries, 10, params=params)
                samples = []
                ids = None
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    for _ in range(iterations):
                        _, ids = searcher.search(queries, 10, params=params)
                    samples.append(time.perf_counter() - start)
                if keepalive is None:
                    raise RuntimeError("unreachable")
                emit(
                    {
                        "kind": "search",
                        "variant": name,
                        "transform_threads": transform_threads,
                        "block_size": block_size,
                        "search_threads": args.search_threads,
                        "batch": batch,
                        "iterations_per_sample": iterations,
                        "ef": ef,
                        "recall_at_10": recall_at_10(ids, expected),
                        "qps": batch * iterations / float(np.median(samples)),
                        "seconds_samples": samples,
                    }
                )

    if output:
        output.close()


if __name__ == "__main__":
    main()
