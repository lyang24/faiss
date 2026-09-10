#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Matched-recall HNSW SQ8 versus combined RaBitQ on one FP32 graph."""

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


def load_vectors(path, limit):
    if path.suffix == ".npy":
        return np.ascontiguousarray(
            np.load(path, mmap_mode="r")[:limit], dtype="float32"
        )
    if path.suffix == ".fvecs":
        words = np.memmap(path, dtype="int32", mode="r")
        d = int(words[0])
        if d <= 0 or words.size % (d + 1) != 0:
            raise ValueError(f"invalid fvecs file: {path}")
        rows = words.reshape(-1, d + 1)
        if len(rows) < limit or not np.all(rows[:limit, 0] == d):
            raise ValueError(f"inconsistent fvecs dimensions: {path}")
        values = words.view("float32").reshape(-1, d + 1)[:limit, 1:]
        return np.ascontiguousarray(values, dtype="float32")
    raise ValueError(f"unsupported vector file: {path}")


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


def graph_digest(index):
    arrays = [
        faiss.vector_to_array(getattr(index.hnsw, field))
        for field in ("levels", "offsets", "neighbors")
    ]
    sha = hashlib.sha256()
    for array in arrays:
        sha.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return sha.hexdigest()


def params_for(variant, ef, transform_threads, block_size):
    inner = faiss.SearchParametersHNSW(efSearch=ef)
    if variant == "sq8":
        return inner, inner
    outer = faiss.SearchParametersPreTransform()
    outer.index_params = inner
    outer.transform_threads = transform_threads
    outer.transform_block_size = block_size
    return outer, (outer, inner)


def timed_search(searcher, queries, params, iterations=1):
    start = time.perf_counter()
    ids = None
    for _ in range(iterations):
        _, ids = searcher.search(queries, 10, params=params)
    return time.perf_counter() - start, ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--nt", type=int, default=100000)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--nb", type=int, default=1_000_000)
    parser.add_argument("--nq", type=int, default=1000)
    parser.add_argument("--bits", type=int, default=7)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--transform-threads", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument(
        "--ef", type=int, nargs="+", default=[16, 24, 32, 48, 64, 96, 128, 192, 256]
    )
    parser.add_argument("--floors", type=float, nargs="+", default=[0.90, 0.95])
    parser.add_argument("--batch1-iterations", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as output:
        def emit(row):
            line = json.dumps(row, sort_keys=True)
            print(line, flush=True)
            print(line, file=output, flush=True)

        xb = load_vectors(args.base, args.nb)
        xq = load_vectors(args.query, args.nq)
        xtrain = load_vectors(args.train, args.nt) if args.train else None
        if xb.ndim != 2 or xq.shape != (args.nq, xb.shape[1]):
            raise ValueError("invalid base/query shape")
        if args.normalize:
            # The mmap is read-only and may already be contiguous.
            xb = np.array(xb, dtype="float32", order="C", copy=True)
            xq = np.array(xq, dtype="float32", order="C", copy=True)
            xb /= np.linalg.norm(xb, axis=1, keepdims=True)
            xq /= np.linalg.norm(xq, axis=1, keepdims=True)

        faiss.omp_set_num_threads(args.threads)
        start = time.perf_counter()
        exact = faiss.IndexFlatL2(xb.shape[1])
        exact.add(xb)
        _, truth = exact.search(xq, 10)
        ground_truth_seconds = time.perf_counter() - start
        del exact

        start = time.perf_counter()
        source_graph = faiss.read_index(str(args.graph))
        graph_load_seconds = time.perf_counter() - start
        if source_graph.ntotal != args.nb or source_graph.d != xb.shape[1]:
            raise ValueError("graph and dataset do not match")
        m = source_graph.hnsw.nb_neighbors(0) // 2
        topology_sha = graph_digest(source_graph)
        topology_bytes = graph_bytes(source_graph)

        emit(
            {
                "kind": "metadata",
                "dataset": args.dataset,
                "args": vars(args)
                | {
                    "base": str(args.base),
                    "query": str(args.query),
                    "train": str(args.train) if args.train else None,
                    "graph": str(args.graph),
                    "output": str(args.output),
                },
                "platform": platform.platform(),
                "compile": faiss.get_compile_options(),
                "omp_env": os.environ.get("OMP_NUM_THREADS"),
                "blas_env": os.environ.get("OPENBLAS_NUM_THREADS"),
                "base_sha256": digest(xb),
                "query_sha256": digest(xq),
                "train_sha256": digest(xtrain) if xtrain is not None else None,
                "train_shape": list(xtrain.shape) if xtrain is not None else None,
                "ground_truth_sha256": digest(truth),
                "ground_truth_seconds": ground_truth_seconds,
                "graph_load_seconds": graph_load_seconds,
                "graph_sha256": topology_sha,
                "graph_bytes": topology_bytes,
                "m": m,
            }
        )

        start = time.perf_counter()
        sq_storage = faiss.IndexScalarQuantizer(
            xb.shape[1], faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2
        )
        sq_storage.train(xtrain if xtrain is not None else xb)
        sq_storage.add(xb)
        sq8 = faiss.IndexHNSW(sq_storage, m)
        sq8.own_fields = False
        sq8.hnsw = source_graph.hnsw
        sq8.ntotal = args.nb
        sq8.is_trained = True
        sq_seconds = time.perf_counter() - start

        start = time.perf_counter()
        rotation = faiss.RandomRotationMatrix(xb.shape[1], xb.shape[1])
        rotation.init(1234)
        rotated_base = rotation.apply_py(xb)
        rotated_train = (
            rotation.apply_py(xtrain) if xtrain is not None else rotated_base
        )
        rq = faiss.IndexHNSWRaBitQ(
            xb.shape[1], m, args.bits, faiss.METRIC_L2
        )
        rq.train(
            np.ascontiguousarray(
                rotated_train[: min(len(rotated_train), 50000)]
            )
        )
        rq_storage = faiss.downcast_index(rq.storage)
        rq_storage.add(rotated_base)
        rq.ntotal = args.nb
        rq.hnsw = source_graph.hnsw
        rq.fp32_graph_built = True
        rq.set_full_code_mode(faiss.RABITQ_FULL_CODE_INT8)
        rabitq = faiss.IndexPreTransform(rotation, rq)
        rq_seconds = time.perf_counter() - start
        del rotated_base, rotated_train

        if graph_digest(sq8) != topology_sha or graph_digest(rq) != topology_sha:
            raise RuntimeError("scorers do not share the loaded FP32 graph")
        del source_graph
        emit(
            {
                "kind": "build",
                "variant": "sq8",
                "seconds": sq_seconds,
                "code_bytes": faiss.vector_to_array(sq_storage.codes).nbytes,
                "serving_bytes": topology_bytes
                + faiss.vector_to_array(sq_storage.codes).nbytes,
                "code_size": sq_storage.code_size,
            }
        )
        rq_packed_bytes = faiss.vector_to_array(rq_storage.codes).nbytes
        rq_expanded_bytes = faiss.vector_to_array(rq_storage.expanded_codes).nbytes
        emit(
            {
                "kind": "build",
                "variant": "rabitq_combined",
                "seconds": rq_seconds,
                "packed_bytes": rq_packed_bytes,
                "expanded_bytes": rq_expanded_bytes,
                "serving_bytes": topology_bytes
                + rq_packed_bytes
                + rq_expanded_bytes,
                "packed_code_size": rq_storage.code_size,
                "expanded_code_size": rq_storage.expanded_code_size(),
                "native_dotprod": bool(
                    rq_storage.expanded_integer_uses_native_dotprod()
                ),
            }
        )

        searchers = {"sq8": sq8, "rabitq_combined": rabitq}
        sweep = {}
        for variant, searcher in searchers.items():
            for ef in args.ef:
                params, keepalive = params_for(
                    "sq8" if variant == "sq8" else "rabitq",
                    ef,
                    args.transform_threads,
                    args.block_size,
                )
                searcher.search(xq, 10, params=params)
                seconds, ids = timed_search(searcher, xq, params)
                recall = recall_at_10(ids, truth)
                sweep[(variant, ef)] = recall
                emit(
                    {
                        "kind": "sweep",
                        "variant": variant,
                        "ef": ef,
                        "recall_at_10": recall,
                        "seconds": seconds,
                        "qps": args.nq / seconds,
                    }
                )
                if keepalive is None:
                    raise RuntimeError("unreachable")

        selected = {}
        for floor in args.floors:
            for variant in searchers:
                candidates = [
                    ef for ef in args.ef if sweep[(variant, ef)] >= floor
                ]
                if not candidates:
                    emit(
                        {
                            "kind": "selection",
                            "floor": floor,
                            "variant": variant,
                            "ef": None,
                        }
                    )
                    continue
                ef = min(candidates)
                selected[(floor, variant)] = ef
                emit(
                    {
                        "kind": "selection",
                        "floor": floor,
                        "variant": variant,
                        "ef": ef,
                        "recall_at_10": sweep[(variant, ef)],
                    }
                )

        for floor in args.floors:
            if (floor, "sq8") not in selected or (
                floor,
                "rabitq_combined",
            ) not in selected:
                continue
            order = ("sq8", "rabitq_combined", "sq8")
            for round_number in range(3):
                for position, variant in enumerate(order):
                    ef = selected[(floor, variant)]
                    params, keepalive = params_for(
                        "sq8" if variant == "sq8" else "rabitq",
                        ef,
                        args.transform_threads,
                        args.block_size,
                    )
                    seconds, ids = timed_search(searchers[variant], xq, params)
                    emit(
                        {
                            "kind": "confirmation",
                            "floor": floor,
                            "round": round_number,
                            "position": position,
                            "variant": variant,
                            "ef": ef,
                            "recall_at_10": recall_at_10(ids, truth),
                            "seconds": seconds,
                            "qps": args.nq / seconds,
                        }
                    )
                    if keepalive is None:
                        raise RuntimeError("unreachable")

            for variant in searchers:
                ef = selected[(floor, variant)]
                params, keepalive = params_for(
                    "sq8" if variant == "sq8" else "rabitq",
                    ef,
                    args.transform_threads,
                    args.block_size,
                )
                searchers[variant].search(xq[:1], 10, params=params)
                samples = []
                ids = None
                for _ in range(3):
                    seconds, ids = timed_search(
                        searchers[variant],
                        xq[:1],
                        params,
                        args.batch1_iterations,
                    )
                    samples.append(seconds)
                emit(
                    {
                        "kind": "batch1",
                        "floor": floor,
                        "variant": variant,
                        "ef": ef,
                        "recall_at_10": recall_at_10(ids, truth[:1]),
                        "iterations_per_sample": args.batch1_iterations,
                        "seconds_samples": samples,
                        "qps": args.batch1_iterations / float(np.median(samples)),
                    }
                )
                if keepalive is None:
                    raise RuntimeError("unreachable")


if __name__ == "__main__":
    main()
