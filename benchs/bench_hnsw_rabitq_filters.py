#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate GIST HNSW filtering with random and feature-correlated metadata."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import faiss
import numpy as np

from bench_hnsw_rabitq_vs_sq8 import digest, graph_bytes, graph_digest, recall_at_10


def exact_filtered_ground_truth(base, query, eligible_ids):
    selected = np.ascontiguousarray(base[eligible_ids], dtype="float32")
    exact = faiss.IndexFlatL2(base.shape[1])
    exact.add(selected)
    _, local_ids = exact.search(query, 10)
    truth = eligible_ids[local_ids]
    return truth


def make_bitmap(mask):
    bitmap = np.packbits(mask, bitorder="little")
    selector = faiss.IDSelectorBitmap(len(bitmap), faiss.swig_ptr(bitmap))
    return selector, bitmap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--nb", type=int, default=1_000_000)
    parser.add_argument("--nq", type=int, default=1000)
    parser.add_argument("--bits", type=int, default=7)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--transform-threads", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--ef", type=int, nargs="+", default=[512, 4096])
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as output:
        def emit(row):
            line = json.dumps(row, sort_keys=True)
            print(line, flush=True)
            print(line, file=output, flush=True)

        xb = np.ascontiguousarray(
            np.load(args.base, mmap_mode="r")[: args.nb], dtype="float32"
        )
        xq = np.ascontiguousarray(
            np.load(args.query, mmap_mode="r")[: args.nq], dtype="float32"
        )
        if xb.shape != (args.nb, 960) or xq.shape != (args.nq, 960):
            raise ValueError("this filter benchmark expects GIST 1M/960D")
        faiss.omp_set_num_threads(args.threads)

        source_graph = faiss.read_index(str(args.graph))
        if source_graph.ntotal != args.nb or source_graph.d != xb.shape[1]:
            raise ValueError("graph and dataset do not match")
        m = source_graph.hnsw.nb_neighbors(0) // 2
        topology_sha = graph_digest(source_graph)
        topology_bytes = graph_bytes(source_graph)

        sq_storage = faiss.IndexScalarQuantizer(
            xb.shape[1], faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2
        )
        sq_storage.train(xb)
        sq_storage.add(xb)
        sq8 = faiss.IndexHNSW(sq_storage, m)
        sq8.own_fields = False
        sq8.hnsw = source_graph.hnsw
        sq8.ntotal = args.nb
        sq8.is_trained = True

        rotation = faiss.RandomRotationMatrix(xb.shape[1], xb.shape[1])
        rotation.init(1234)
        rotated_base = rotation.apply_py(xb)
        rq = faiss.IndexHNSWRaBitQ(
            xb.shape[1], m, args.bits, faiss.METRIC_L2
        )
        rq.train(np.ascontiguousarray(rotated_base[:50000]))
        rq_storage = faiss.downcast_index(rq.storage)
        rq_storage.add(rotated_base)
        rq.ntotal = args.nb
        rq.hnsw = source_graph.hnsw
        rq.fp32_graph_built = True
        rq.set_full_code_mode(faiss.RABITQ_FULL_CODE_INT8)
        rabitq = faiss.IndexPreTransform(rotation, rq)
        del rotated_base, source_graph

        emit(
            {
                "kind": "metadata",
                "args": vars(args)
                | {
                    "base": str(args.base),
                    "query": str(args.query),
                    "graph": str(args.graph),
                    "output": str(args.output),
                },
                "platform": platform.platform(),
                "compile": faiss.get_compile_options(),
                "omp_env": os.environ.get("OMP_NUM_THREADS"),
                "blas_env": os.environ.get("OPENBLAS_NUM_THREADS"),
                "base_sha256": digest(xb),
                "query_sha256": digest(xq),
                "graph_sha256": topology_sha,
                "graph_bytes": topology_bytes,
                "random_filter": "nested permutation, seed 777",
                "correlated_filter": "nested ascending base-vector feature 0",
            }
        )

        random_order = np.random.default_rng(777).permutation(args.nb)
        correlated_order = np.argsort(xb[:, 0], kind="stable")
        searchers = {"sq8": sq8, "rabitq_combined": rabitq}
        for fraction in args.fractions:
            count = max(10, int(round(args.nb * fraction)))
            filter_orders = (("all", np.arange(args.nb)),) if fraction == 1.0 else (
                ("random", random_order),
                ("correlated_feature0", correlated_order),
            )
            for filter_kind, order in filter_orders:
                eligible_ids = np.sort(order[:count]).astype(np.int64, copy=False)
                mask = np.zeros(args.nb, dtype=bool)
                mask[eligible_ids] = True
                selector, bitmap = make_bitmap(mask)
                start = time.perf_counter()
                truth = exact_filtered_ground_truth(xb, xq, eligible_ids)
                gt_seconds = time.perf_counter() - start
                emit(
                    {
                        "kind": "filter",
                        "filter": filter_kind,
                        "fraction": fraction,
                        "eligible": count,
                        "mask_sha256": hashlib.sha256(bitmap).hexdigest(),
                        "ground_truth_sha256": digest(truth),
                        "ground_truth_seconds": gt_seconds,
                    }
                )

                for ef in args.ef:
                    for variant, searcher in searchers.items():
                        inner = faiss.SearchParametersHNSW(efSearch=ef)
                        inner.sel = selector
                        if variant == "sq8":
                            params = inner
                            keepalive = (inner, selector, bitmap)
                        else:
                            outer = faiss.SearchParametersPreTransform()
                            outer.index_params = inner
                            outer.transform_threads = args.transform_threads
                            outer.transform_block_size = args.block_size
                            params = outer
                            keepalive = (outer, inner, selector, bitmap)
                        searcher.search(xq, 10, params=params)
                        samples = []
                        ids = None
                        for _ in range(args.repeats):
                            start = time.perf_counter()
                            _, ids = searcher.search(xq, 10, params=params)
                            samples.append(time.perf_counter() - start)
                        invalid_returned = int(
                            np.count_nonzero((ids >= 0) & ~mask[np.maximum(ids, 0)])
                        )
                        emit(
                            {
                                "kind": "search",
                                "filter": filter_kind,
                                "fraction": fraction,
                                "eligible": count,
                                "variant": variant,
                                "ef": ef,
                                "recall_at_10": recall_at_10(ids, truth),
                                "returned": int(np.count_nonzero(ids >= 0)),
                                "missing": int(np.count_nonzero(ids < 0)),
                                "incomplete_queries": int(
                                    np.count_nonzero(np.any(ids < 0, axis=1))
                                ),
                                "invalid_returned": invalid_returned,
                                "seconds_samples": samples,
                                "qps": args.nq / float(np.median(samples)),
                            }
                        )
                        if not keepalive:
                            raise RuntimeError("unreachable")


if __name__ == "__main__":
    main()
