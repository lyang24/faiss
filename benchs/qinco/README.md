# QINCo HNSW experiment

QINCo (temporary shorthand for "query-integer, norm-corrected") is an
independently implemented random-access integer scorer for an HNSW graph
constructed with FP32 distances. It is not a RaBitQ implementation and the
name is not a novelty claim.

For dimension `j`, document training produces `a[j]` and `b[j]`, and stores a
signed byte `c[j]` such that `x[j] ~= b[j] + a[j] c[j]`. At query time the
transformed values `a[j] q[j]` are quantized to signed bytes `v[j]` with one
scale `s[B]` per contiguous block. The estimated dot product is

```
b dot q + sum_B s[B] * sum_(j in B) c[j] v[j]
```

and L2 is the reconstruction norm (or an explicitly selected blend with the
exact norm), plus the exact query norm, minus twice this dot product. The
default reconstruction norm is intentional: using the exact norm with an
approximate dot product caused score bias and worse recall in the first
ablation.

The separable encoding modes are:

- `--doc-bits 8 --query-bits 8`: full-range signed documents; AVX2 widens
  signed bytes, while AArch64 uses native `sdot`.
- `--doc-bits 7 --query-bits 8`: 0..127 documents; AVX2 uses saturating
  `maddubs`, with a proven product bound that prevents saturation. ARMv9 I8MM
  builds use native unsigned-by-signed `usdot`.
- `--doc-bits 8 --query-bits 6`: full-range unsigned documents and query
  magnitude at most 63, giving the same safe `maddubs` bound.
- `--doc-bits 4`: packed full-dimensional int4 documents. This is an ablation,
  not the recommended high-recall mode.
- `--pca-dim P --int8-head-dims H --doc-bits 4`: train PCA offline, retain
  `P` dimensions, store the aligned head `[0,H)` as int8 and the tail as
  packed int4. Query normalization, PCA, allocation and search remain timed.
- `--rerank N`: request `N` candidates without changing navigation, then rank
  those arbitrary IDs with the float query and stored reconstruction.

## Build

```bash
cmake -S . -B build-qinco -G Ninja \
  -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF \
  -DFAISS_ENABLE_C_API=OFF -DFAISS_ENABLE_MKL=OFF \
  -DFAISS_OPT_LEVEL=avx2 -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-qinco --target bench_hnsw_qinco -j "$(nproc)"
```

The `FAISS_ENABLE_MKL=OFF` command records the OpenBLAS configuration used in
the local study. Use the same optimized BLAS for both scorers on another host.
The reader accepts either standard `fvecs` files or C-contiguous little-endian
2-D float32 `.npy` files.

For the AWS Neoverse-V2 confirmation build:

```bash
cmake -S . -B build-qinco-arm -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=sve \
  -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF \
  -DFAISS_ENABLE_C_API=OFF -DFAISS_ENABLE_MKL=OFF -DBUILD_TESTING=OFF
cmake --build build-qinco-arm --target bench_hnsw_qinco -j 16
```

The benchmark CMake target appends `-march=armv9-a+sve2+i8mm
-mtune=neoverse-v2` to both the Faiss SVE library (including SQ8) and the
benchmark. Do not use that target on an older ARM CPU.

## Cohere 1M

```bash
python3 benchs/qinco/prepare_datasets.py cohere --output /data/cohere-qinco
OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 \
  build-qinco/benchs/bench_hnsw_qinco \
  --base /data/cohere-qinco/cohere_base.fvecs \
  --query /data/cohere-qinco/cohere_query.fvecs \
  --graph /data/cohere-qinco/hnsw_M32_efc100.faissindex \
  --output /data/cohere-qinco/results.jsonl \
  --normalize --threads 16 --M 32 --efConstruction 100 --block 128
```

The first run builds the FP32 graph and caches it. All later scorer variants
copy exactly that graph. Add `--rebuild-graph` only for the separate graph
construction ablation. Query normalization, allocations, QINCo query
quantization, traversal, and top-k are within the reported interval.

To measure the construction hypothesis separately, use a distinct graph path
and `--construction-storage sq8 --rebuild-graph`. The JSON metadata records
construction time and storage type. Do not compare scorers across those graph
files; within each invocation SQ8 and QINCo still receive identical graph
links.

## GIST/SIFT and filters

Use the unmodified TexMex `gist_base.fvecs` / `gist_query.fvecs` or SIFT files;
do not pass `--normalize`. For the GIST retained-fraction sweep, run both:

```bash
.../bench_hnsw_qinco --base gist_base.fvecs --query gist_query.fvecs \
  --graph gist_M32_efc100.faissindex --filter random --output gist_random.jsonl
.../bench_hnsw_qinco --base gist_base.fvecs --query gist_query.fvecs \
  --graph gist_M32_efc100.faissindex --filter feature0 --output gist_feature0.jsonl
```

`random` uses a fixed seeded subset. `feature0` is a vector-correlated derived
attribute, not genuine GIST metadata; results label it accordingly. Ground
truth is recomputed over eligible rows for every fraction. The JSONL includes
the number of `-1` results so incomplete top-k is not silently treated as low
recall.
