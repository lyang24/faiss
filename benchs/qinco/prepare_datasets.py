#!/usr/bin/env python3
"""Download/convert benchmark datasets without changing row order.

Cohere output is deliberately *not* normalized here: query normalization is
part of the timed benchmark path, and base normalization is performed before
FP32 graph construction by bench_hnsw_qinco.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import struct
import urllib.request
import tarfile

import numpy as np
import pyarrow.parquet as pq


COHERE = {
    "base": "https://assets.zilliz.com/benchmark/cohere_medium_1m/shuffle_train.parquet",
    "query": "https://assets.zilliz.com/benchmark/cohere_medium_1m/test.parquet",
    "neighbors": "https://assets.zilliz.com/benchmark/cohere_medium_1m/neighbors.parquet",
}

DBPEDIA_URL = (
    "https://storage.googleapis.com/ann-filtered-benchmark/datasets/"
    "dbpedia_openai_1M.tgz"
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        return
    temporary = output.with_suffix(output.suffix + ".partial")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(output)


def list_to_numpy(column, expected_d: int) -> np.ndarray:
    offsets = np.asarray(column.offsets)
    lengths = np.diff(offsets)
    if not np.all(lengths == expected_d):
        raise ValueError(f"vector dimension mismatch: {np.unique(lengths)}")
    values = np.asarray(column.values).astype("<f4", copy=False)
    begin = int(offsets[0])
    end = int(offsets[-1])
    return values[begin:end].reshape(len(column), expected_d)


def parquet_to_fvecs(
    source: pathlib.Path,
    output: pathlib.Path,
    ids_output: pathlib.Path,
    expected_rows: int,
    expected_d: int,
) -> None:
    parquet = pq.ParquetFile(source)
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(
            f"{source}: expected {expected_rows} rows, got {parquet.metadata.num_rows}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output.open("wb") as vectors, ids_output.open("wb") as ids:
        for batch in parquet.iter_batches(
            batch_size=4096, columns=["id", "emb"], use_threads=True
        ):
            matrix = list_to_numpy(batch.column("emb"), expected_d)
            if not np.isfinite(matrix).all():
                raise ValueError(f"{source}: non-finite embedding at row {rows}")
            row_block = np.empty((len(matrix), expected_d + 1), dtype="<f4")
            row_block[:, 0].view("<i4")[:] = expected_d
            row_block[:, 1:] = matrix
            row_block.tofile(vectors)
            np.asarray(batch.column("id")).astype("<i8", copy=False).tofile(ids)
            rows += len(matrix)
    if rows != expected_rows:
        raise AssertionError((rows, expected_rows))


def inspect_fvecs(path: pathlib.Path) -> dict:
    with path.open("rb") as stream:
        d = struct.unpack("<i", stream.read(4))[0]
    row_bytes = 4 + 4 * d
    size = path.stat().st_size
    if size % row_bytes:
        raise ValueError(f"{path}: truncated fvecs")
    return {"path": str(path), "rows": size // row_bytes, "d": d, "sha256": sha256(path)}


def write_cohere_source_gt(root: pathlib.Path) -> pathlib.Path:
    base_ids = np.fromfile(root / "cohere_base_ids.i64", dtype="<i8")
    if len(base_ids) != 1_000_000 or base_ids.min() != 0 or base_ids.max() != 999_999:
        raise ValueError("Cohere base ids are not a permutation of [0, 1M)")
    row_for_id = np.empty(1_000_000, dtype="<i8")
    row_for_id[base_ids] = np.arange(1_000_000, dtype="<i8")
    table = pq.read_table(root / "raw" / "neighbors.parquet")
    query_ids = np.asarray(table.column("id")).astype("<i8", copy=False)
    if not np.array_equal(query_ids, np.arange(1000)):
        raise ValueError("Cohere neighbor rows do not match query row ids")
    neighbors = list_to_numpy(table.column("neighbors_id").combine_chunks(), 1000).astype("<i8")
    row_neighbors = row_for_id[neighbors[:, :100]]
    output = root / "cohere_source_groundtruth.ivecs"
    block = np.empty((1000, 101), dtype="<i4")
    block[:, 0] = 100
    block[:, 1:] = row_neighbors.astype("<i4")
    block.tofile(output)
    return output


def prepare_cohere(root: pathlib.Path) -> None:
    raw = root / "raw"
    for name, url in COHERE.items():
        download(url, raw / f"{name}.parquet")
    parquet_to_fvecs(
        raw / "base.parquet",
        root / "cohere_base.fvecs",
        root / "cohere_base_ids.i64",
        1_000_000,
        768,
    )
    source_gt = write_cohere_source_gt(root)
    parquet_to_fvecs(
        raw / "query.parquet",
        root / "cohere_query.fvecs",
        root / "cohere_query_ids.i64",
        1_000,
        768,
    )
    manifest = {
        "dataset": "cohere_medium_1m",
        "source_urls": COHERE,
        "raw_sha256": {name: sha256(raw / f"{name}.parquet") for name in COHERE},
        "base": inspect_fvecs(root / "cohere_base.fvecs"),
        "query": inspect_fvecs(root / "cohere_query.fvecs"),
        "source_groundtruth": {
            "path": str(source_gt),
            "rows": 1000,
            "k": 100,
            "sha256": sha256(source_gt),
            "ids": "mapped from original embedding ids to preserved fvecs row ids",
        },
        "row_order": "exact Parquet row order; original id stored in parallel .i64",
        "normalization": "none in files; benchmark normalizes base offline and each query inside timing",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def write_fvecs(output: pathlib.Path, batches, d: int) -> int:
    rows = 0
    with output.open("wb") as stream:
        for matrix in batches:
            matrix = np.asarray(matrix, dtype="<f4")
            if matrix.ndim != 2 or matrix.shape[1] != d:
                raise ValueError(f"expected (*, {d}), got {matrix.shape}")
            if not np.isfinite(matrix).all():
                raise ValueError(f"non-finite vector near row {rows}")
            block = np.empty((len(matrix), d + 1), dtype="<f4")
            block[:, 0].view("<i4")[:] = d
            block[:, 1:] = matrix
            block.tofile(stream)
            rows += len(matrix)
    return rows


def prepare_dbpedia(root: pathlib.Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "dbpedia_openai_1M.tgz"
    download(DBPEDIA_URL, archive)
    vectors_npy = root / "vectors.npy"
    tests_jsonl = root / "tests.jsonl"
    if not vectors_npy.exists() or not tests_jsonl.exists():
        with tarfile.open(archive, "r:gz") as tar:
            for name in ("vectors.npy", "tests.jsonl"):
                source = tar.extractfile(tar.getmember(name))
                if source is None:
                    raise ValueError(f"missing {name} in {archive}")
                with (root / name).open("wb") as output:
                    shutil.copyfileobj(source, output)

    vectors = np.load(vectors_npy, mmap_mode="r")
    if vectors.shape != (975_000, 1536) or vectors.dtype != np.float32:
        raise ValueError(f"unexpected DBpedia base: {vectors.shape} {vectors.dtype}")
    base_rows = write_fvecs(
        root / "dbpedia_base.fvecs",
        (vectors[i : i + 4096] for i in range(0, len(vectors), 4096)),
        1536,
    )

    def query_batches():
        batch = []
        with tests_jsonl.open() as stream:
            for line in stream:
                item = json.loads(line)
                if item.get("conditions"):
                    raise ValueError("expected unfiltered DBpedia query split")
                batch.append(item["query"])
                if len(batch) == 1000:
                    yield np.asarray(batch, dtype="<f4")
                    batch.clear()
            if batch:
                yield np.asarray(batch, dtype="<f4")

    query_rows = write_fvecs(root / "dbpedia_query.fvecs", query_batches(), 1536)
    sample_norms = np.linalg.norm(np.asarray(vectors[::1000]), axis=1)
    manifest = {
        "dataset": "dbpedia-openai-1M-1536-angular",
        "source_url": DBPEDIA_URL,
        "archive_sha256": sha256(archive),
        "base": inspect_fvecs(root / "dbpedia_base.fvecs"),
        "query": inspect_fvecs(root / "dbpedia_query.fvecs"),
        "verified_shape": {"base_rows": base_rows, "query_rows": query_rows, "d": 1536},
        "note": "the file named 1M contains 975,000 base rows and 5,000 unfiltered queries",
        "normalization": {
            "stored": "already unit length",
            "sample_min_norm": float(sample_norms.min()),
            "sample_max_norm": float(sample_norms.max()),
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["cohere", "dbpedia"])
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.dataset == "cohere":
        prepare_cohere(args.output)
    elif args.dataset == "dbpedia":
        prepare_dbpedia(args.output)


if __name__ == "__main__":
    main()
