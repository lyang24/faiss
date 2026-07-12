# RaBitQ E8 Lattice Codebook Harness

This directory contains the quantizer-level A/B harness for the experimental
RaBitQ E8 codebook work. It does not call FAISS search code and does not modify
the production RaBitQ, IVF, or FastScan paths.

## Codebooks

`codebooks.py` builds and validates:

- `lex15`: current finite-E8-wide book, byte-for-byte intended to match
  `e8_finite_256_wide_codebook()`: zero + 240 roots + first 15 axis-shell points.
- `sym256`: 240 E8 roots + all 16 axis-shell points, closed under negation.
- `zero_axis15`: zero + 240 E8 roots + deterministic 15 of 16 axis-shell points.
- `voronoi_off`: exploratory offset Voronoi diagnostic. It is not included in
  default runs. Current local diagnostic reports strict-minimum ties for tested
  root/2 offsets, so it should be treated as failed until the offset construction
  is studied offline.

Run:

```bash
python3 benchs/rabitq_lattice/codebooks.py
```

## Harness

`bench_e8_books.py` supports:

- seeded QR random rotation: `--rotate none|qr --seed N`
- optional train-mean centering: `--center`
- optional row normalization for angular/cosine datasets: `--normalize`
- pluggable books: `--books lex15 sym256 zero_axis15`
- CSV output with rho mean/std/p1/p5/p50, MAE/RMSE/bias/std globally, and the
  same error diagnostics restricted to the exact true-top1000 pool.

The script supports HDF5 files with `train` and `test` datasets and local `.npz`
files for smoke tests.

## Local Smoke

```bash
python3 - <<'PY'
import numpy as np
rng=np.random.default_rng(123)
d=16; nb=2000; nq=20
xb=rng.standard_normal((nb,d), dtype=np.float32)
xq=rng.standard_normal((nq,d), dtype=np.float32)
scores=xq@xb.T
idx=np.argsort(-scores, axis=1)[:,:100].astype('int32')
vals=np.take_along_axis(scores, idx, axis=1)
np.savez('/tmp/rabitq_lattice_synth_ip.npz', train=xb, test=xq, neighbors=idx, distances=(1-vals).astype('float32'))
PY

python3 benchs/rabitq_lattice/bench_e8_books.py \
  --dataset /tmp/rabitq_lattice_synth_ip.npz --metric ip \
  --nb 2000 --nq 20 --chunk 500 --k 100 --top-pool 200 \
  --books lex15 sym256 zero_axis15 --rotate none \
  --out /tmp/rabitq_lattice_synth_none.csv

python3 benchs/rabitq_lattice/bench_e8_books.py \
  --dataset /tmp/rabitq_lattice_synth_ip.npz --metric ip \
  --nb 2000 --nq 20 --chunk 500 --k 100 --top-pool 200 \
  --books lex15 sym256 zero_axis15 --rotate qr --seed 7 \
  --out /tmp/rabitq_lattice_synth_qr.csv
```

## Remote 1M Command Templates

Remote machine used previously:

```bash
ssh -i ~/Downloads/aws.pem ubuntu@54.177.244.215
```

The remote has the real HDF5 datasets under `/home/ubuntu/data`. Copy this
directory or run these commands on that machine after syncing this harness.

```bash
mkdir -p /home/ubuntu/e8_results

# Cohere, 1024D IP
python3 benchs/rabitq_lattice/bench_e8_books.py \
  --dataset /home/ubuntu/data/cohere-wiki-1024-ip.hdf5 --metric ip \
  --nb 1000000 --nq 100 --chunk 10000 --k 100 --top-pool 1000 \
  --books lex15 sym256 zero_axis15 --rotate none \
  --out /home/ubuntu/e8_results/cohere_none.csv
python3 benchs/rabitq_lattice/bench_e8_books.py \
  --dataset /home/ubuntu/data/cohere-wiki-1024-ip.hdf5 --metric ip \
  --nb 1000000 --nq 100 --chunk 10000 --k 100 --top-pool 1000 \
  --books lex15 sym256 zero_axis15 --rotate qr --seed 12345 \
  --out /home/ubuntu/e8_results/cohere_qr.csv

# Deep image angular/cosine, 96D IP
python3 benchs/rabitq_lattice/bench_e8_books.py \
  --dataset /home/ubuntu/data/deep-image-96-angular.hdf5 --metric ip --normalize \
  --nb 1000000 --nq 100 --chunk 10000 --k 100 --top-pool 1000 \
  --books lex15 sym256 zero_axis15 --rotate none \
  --out /home/ubuntu/e8_results/deep_none.csv
python3 benchs/rabitq_lattice/bench_e8_books.py \
  --dataset /home/ubuntu/data/deep-image-96-angular.hdf5 --metric ip --normalize \
  --nb 1000000 --nq 100 --chunk 10000 --k 100 --top-pool 1000 \
  --books lex15 sym256 zero_axis15 --rotate qr --seed 12345 \
  --out /home/ubuntu/e8_results/deep_qr.csv

# OpenAI DBpedia, 1536D IP, 99k train / 1k query
python3 benchs/rabitq_lattice/bench_e8_books.py \
  --dataset /home/ubuntu/data/dbpedia-openai3-small-1536-angular-100k.hdf5 \
  --metric ip --nb 99000 --nq 1000 --chunk 3000 --k 100 --top-pool 1000 \
  --books lex15 sym256 zero_axis15 --rotate none \
  --out /home/ubuntu/e8_results/openai_none.csv
python3 benchs/rabitq_lattice/bench_e8_books.py \
  --dataset /home/ubuntu/data/dbpedia-openai3-small-1536-angular-100k.hdf5 \
  --metric ip --nb 99000 --nq 1000 --chunk 3000 --k 100 --top-pool 1000 \
  --books lex15 sym256 zero_axis15 --rotate qr --seed 12345 \
  --out /home/ubuntu/e8_results/openai_qr.csv

# SIFT/GIST raw L2. These are useful quantizer-error stress tests, not a
# production RaBitQ recall proxy unless the production residual/rotation path is
# also modeled.
python3 benchs/rabitq_lattice/bench_e8_books.py \
  --dataset /home/ubuntu/data/sift-128-euclidean.hdf5 --metric l2 --center \
  --nb 1000000 --nq 100 --chunk 10000 --k 100 --top-pool 1000 \
  --books lex15 sym256 zero_axis15 --rotate qr --seed 12345 \
  --out /home/ubuntu/e8_results/sift_center_qr.csv
python3 benchs/rabitq_lattice/bench_e8_books.py \
  --dataset /home/ubuntu/data/gist-960-euclidean.hdf5 --metric l2 --center \
  --nb 1000000 --nq 100 --chunk 10000 --k 100 --top-pool 1000 \
  --books lex15 sym256 zero_axis15 --rotate qr --seed 12345 \
  --out /home/ubuntu/e8_results/gist_center_qr.csv
```
