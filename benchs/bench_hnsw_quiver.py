#!/usr/bin/env python3

"""Build and benchmark a Faiss HNSW index backed by QuIVer codes.

The input arrays must already be L2-normalized. Ground truth is expected to
contain nearest-neighbor ids ordered by decreasing inner product.
"""

import argparse
import json
import os
import time

import faiss
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="float32 .npy base vectors")
    parser.add_argument("--query", required=True, help="float32 .npy queries")
    parser.add_argument("--gt", required=True, help="integer .npy ground truth")
    parser.add_argument("--index", help="read/write the serialized index here")
    parser.add_argument("--factory", default="HNSW32,Quiver,RFlat")
    parser.add_argument(
        "--refine",
        choices=["none", "sq8"],
        default="none",
        help="optionally add a separately encoded refinement index",
    )
    parser.add_argument("--ef-construction", type=int, default=40)
    parser.add_argument("--ef-search", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--k-factor", type=float, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser.parse_args()


def mmap_array(path):
    array = np.load(path, mmap_mode="r")
    if array.ndim != 2:
        raise ValueError(f"expected a matrix in {path}, got {array.shape}")
    return array


def hnsw_of(index):
    current = faiss.downcast_index(index)
    while hasattr(current, "base_index"):
        current = faiss.downcast_index(current.base_index)
    if not hasattr(current, "hnsw"):
        raise TypeError(f"index does not contain HNSW: {type(current).__name__}")
    return current.hnsw


def build_index(args, xb):
    index = faiss.index_factory(
        xb.shape[1], args.factory, faiss.METRIC_INNER_PRODUCT
    )
    hnsw_of(index).efConstruction = args.ef_construction
    start = time.perf_counter()
    for begin in range(0, xb.shape[0], args.batch_size):
        end = min(begin + args.batch_size, xb.shape[0])
        index.add(np.asarray(xb[begin:end], dtype="float32"))
        print(
            json.dumps(
                {
                    "event": "build_progress",
                    "vectors": end,
                    "seconds": time.perf_counter() - start,
                }
            ),
            flush=True,
        )
    return index, time.perf_counter() - start


def add_sq8_refinement(base_index, xb, batch_size):
    refine = faiss.IndexScalarQuantizer(
        xb.shape[1], faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
    )
    train_size = min(100_000, xb.shape[0])
    refine.train(np.asarray(xb[:train_size], dtype="float32"))
    for begin in range(0, xb.shape[0], batch_size):
        end = min(begin + batch_size, xb.shape[0])
        refine.add(np.asarray(xb[begin:end], dtype="float32"))
    wrapped = faiss.IndexRefine(base_index, refine)
    # IndexRefine does not own constructor arguments. Keep both Python
    # references reachable for the lifetime of the benchmark.
    wrapped._quiver_base_ref = base_index
    wrapped._quiver_refine_ref = refine
    return wrapped


def recall_at_k(found, truth, k):
    total = 0
    for result, expected in zip(found[:, :k], truth[:, :k]):
        total += len(set(result.tolist()) & set(expected.tolist()))
    return total / (found.shape[0] * k)


def main():
    args = parse_args()
    faiss.omp_set_num_threads(args.threads)
    xb = mmap_array(args.base)
    xq = mmap_array(args.query)[: args.queries]
    gt = mmap_array(args.gt)[: args.queries]
    if xb.dtype != np.float32 or xq.dtype != np.float32:
        raise TypeError("base and query arrays must be float32")
    if xb.shape[1] != xq.shape[1]:
        raise ValueError("base/query dimensions differ")

    if args.index and os.path.exists(args.index):
        started = time.perf_counter()
        index = faiss.read_index(args.index)
        build_seconds = None
        load_seconds = time.perf_counter() - started
    else:
        index, build_seconds = build_index(args, xb)
        load_seconds = None

        if args.refine == "sq8":
            refine_started = time.perf_counter()
            index = add_sq8_refinement(index, xb, args.batch_size)
            print(
                json.dumps(
                    {
                        "event": "refine_ready",
                        "kind": args.refine,
                        "seconds": time.perf_counter() - refine_started,
                    }
                ),
                flush=True,
            )
        if args.index:
            faiss.write_index(index, args.index)

    print(
        json.dumps(
            {
                "event": "index_ready",
                "factory": args.factory,
                "ntotal": index.ntotal,
                "d": index.d,
                "build_seconds": build_seconds,
                "load_seconds": load_seconds,
                "serialized_bytes": os.path.getsize(args.index)
                if args.index and os.path.exists(args.index)
                else None,
            }
        ),
        flush=True,
    )

    refine = faiss.downcast_index(index)
    factors = args.k_factor if hasattr(refine, "k_factor") else [None]
    query = np.asarray(xq, dtype="float32")
    truth = np.asarray(gt)
    for factor in factors:
        if factor is not None:
            refine.k_factor = factor
        for ef_search in args.ef_search:
            hnsw_of(index).efSearch = ef_search
            # Discard one complete search so the index pages and code paths are
            # warm before collecting interleaved repeat samples.
            distances, labels = index.search(query, args.k)
            samples = []
            for _ in range(args.repeat):
                start = time.perf_counter()
                distances, labels = index.search(query, args.k)
                samples.append(time.perf_counter() - start)
            median = float(np.median(samples))
            print(
                json.dumps(
                    {
                        "event": "search",
                        "ef_search": ef_search,
                        "k_factor": factor,
                        "qps": query.shape[0] / median,
                        "seconds": samples,
                        "recall_at_k": recall_at_k(labels, truth, args.k),
                        "id_checksum": int(labels.sum()),
                        "finite_distances": bool(np.isfinite(distances).all()),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
