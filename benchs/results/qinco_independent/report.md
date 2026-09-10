# Independent quantized HNSW scorer: AWS ARM results

## Outcome

The full-dimensional implementation beats native Faiss SQ8 at both requested
recall floors on Cohere 1M and GIST 1M. These are medians of three frozen-ef
confirmation rounds; times include output allocation, query copying and
normalization, query quantization, traversal, and top-k. Each pair uses an
exact copy of the same FP32-built HNSW links.

| dataset / selected representation | floor | SQ8 ef / recall | new ef / recall | SQ8 ms | new ms | QPS speedup | total SQ8 / new memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cohere 1M, signed doc8/query8 | 90% | 32 / 91.22% | 32 / 90.93% | 24.910 | 9.987 | **2.494x** | 1040.1 / 1044.1 MB |
| Cohere 1M, signed doc8/query8 | 95% | 64 / 95.58% | 64 / 95.14% | 42.258 | 17.306 | **2.442x** | 1040.1 / 1044.1 MB |
| GIST 1M, signed doc8/query8 + r64 | 90% | 224 / 91.03% | 224 / 91.08% | 153.557 | 69.774 | **2.201x** | 1232.1 / 1236.1 MB |
| GIST 1M, signed doc8/query8 + r64 | 95% | 512 / 95.14% | 512 / 95.22% | 308.246 | 135.532 | **2.274x** | 1232.1 / 1236.1 MB |
| DBpedia OpenAI 975k, doc7/query8 + r20 | 90% | 24 / 90.37% | 24 / 90.14% | 39.873 | 15.294 | **2.607x** | 1762.9 / 1766.8 MB |
| DBpedia OpenAI 975k, doc7/query8 + r20 | 95% | 56 / 95.07% | 64 / 95.50% | 70.017 | 26.990 | **2.594x** | 1762.9 / 1766.8 MB |
| SIFT 1M, doc7/query8 | 90% | 32 / 90.51% | 32 / 90.19% | 7.273 | 4.252 | **1.710x** | 400.1 / 404.1 MB |
| SIFT 1M, doc7/query8 | 95% | 64 / 95.48% | 80 / 96.02% | 12.617 | 9.190 | **1.373x** | 400.1 / 404.1 MB |

Memory includes codes, the four-byte per-vector norm, and measured HNSW graph
arrays. It excludes small per-index scale/bias arrays and temporary query
buffers. The main design is four bytes/vector larger than SQ8: its benefit is
throughput, not memory reduction.

Batch=1 runs reallocate and transform every request:

| dataset | floor | SQ8 ms / 1000 requests | new ms / 1000 requests | speedup |
|---|---:|---:|---:|---:|
| Cohere | 90% | 403.224 | 123.014 | 3.278x |
| Cohere | 95% | 675.011 | 212.113 | 3.182x |
| GIST | 90% | 2525.950 | 804.341 | 3.140x |
| GIST | 95% | 5076.200 | 1766.090 | 2.874x |
| DBpedia | 90% | 631.807 | 143.672 | 4.398x |
| DBpedia | 95% | 1107.960 | 290.074 | 3.820x |
| SIFT | 90% | 123.172 | 64.881 | 1.898x |
| SIFT | 95% | 207.462 | 141.808 | 1.463x |

Raw data: [Cohere](cohere_i8_i8mm_no_rerank.jsonl),
[GIST](gist_i8_i8mm_r64.jsonl), [DBpedia](dbpedia_u7_i8mm_r20.jsonl),
and [SIFT](sift_u7_i8mm_no_rerank.jsonl).

## Machine and build

- AWS host `54.67.30.55`, AArch64 Neoverse-V2, 16 physical cores (one thread
  per core), 64 KiB L1d/core, 2 MiB L2/core, 36 MiB shared L3.
- 66,190,868,480 bytes RAM, no swap; Linux `7.0.0-1011-aws`, Ubuntu 24.04.
- GCC 13.3.0, CMake 3.28.3, GNU Make 4.3, OpenBLAS pthread 0.3.26.
- Faiss `00928e3f7628b8288c17cfece5ee9600a8287f56` (reported by `describe` as
  `v1.14.3-188-g00928e3f-dirty` only because this patch is applied).
- Release C++20, `-O3 -DNDEBUG -fPIC -march=armv8-a+sve
  -march=armv9-a+sve2+i8mm -mtune=neoverse-v2 -fopenmp`.
- `FAISS_OPT_LEVEL=sve`; GPU, Python, C API, MKL, and test-suite aggregation
  disabled. The benchmark links `libopenblas.so.0` and `libgomp.so.1`.
- Main thread budget: `OMP_NUM_THREADS=16`, `OMP_PROC_BIND=close`,
  `OMP_PLACES=cores`, `OPENBLAS_NUM_THREADS=1`. The same Faiss SVE library and
  CPU flags compile both native SQ8 and the new scorer. Object disassembly was
  checked for `sdot` and `usdot` instructions.

## Representation and distance equation

For signed doc8, dimension `j` learns

```
a_j = (max_j - min_j) / 255
b_j = min_j + 128 a_j
c_ij = clip(round((x_ij - b_j) / a_j), -128, 127)
```

For unsigned doc7, the divisor is 127, `b_j=min_j`, and the code range is
0..127. For query `q` and each contiguous block `B`, define

```
t_j = a_j q_j
alpha_B = max(j in B) |t_j| / 127
z_j = clip(round(t_j / alpha_B), -127, 127)
dot_hat(x_i,q) = b dot q + sum_B alpha_B sum_(j in B) c_ij z_j
d_hat(i,q) = N_i + ||q||^2 - 2 dot_hat(x_i,q)
```

`N_i` defaults to the squared norm of the reconstructed document. An explicit
blend with the exact FP32 norm exists, but exact-norm weight 1 reduced Cohere
recall and speed at matched recall. The code also supports unsigned doc8 with
six-bit queries and packed int4. On AVX2, doc7 uses `maddubs`; the pair bound
`2*127*127=32258 < 32767` prevents saturation. On this ARM CPU, doc7 uses
unsigned-by-signed `usdot`; signed doc8 uses `sdot`. Accumulators are int32,
and every document remains directly addressable by arbitrary ID.

## Experimental protocol

1. Build HNSW with FP32 distances, `M=32`, `efConstruction=100`.
2. Copy entry point, levels, offsets and links exactly; swap only storage for
   `IndexHNSWSQ(QT_8bit)` or the new storage.
3. Recompute exact L2 top-10 over all eligible documents with `IndexFlatL2`.
4. Sweep method-specific ef candidates. Run SQ8 immediately before and after
   each candidate to bracket drift.
5. Freeze the first ef meeting each recall floor independently, then run three
   alternating confirmation rounds. The table reports the median.
6. Separately issue 1000 single-query calls for batch=1.

The same 1000 queries are used for tuning and confirmation; there is no held-out
ef-selection set, so cross-dataset validation is more meaningful than the last
decimal of a single dataset.

## Dataset verification

- Cohere: 1,000,000 base rows and 1,000 queries, 768D. Base is normalized
  offline and queries are normalized inside each timed run. Recomputed exact
  top-10 agrees 100% with the ID-remapped published neighbor file. Prepared
  fvecs SHA256: base
  `11eb9c3826d73ad94b77beca19e6737c3f1b0248ecd670fab8eab8c3e7447b95`,
  query
  `84d8ff9d4087b3de0181626eceac7b2d506845bd9759701c8e41c2ebaeeca3de`.
- GIST: original TexMex 1,000,000 base, 1,000 query, 960D, unnormalized L2.
  Archive SHA256
  `01469a7f1c3768853525e543d537e2dfa1adece927616405e360952e3f67df73`.
  Published-vs-recomputed top-10 overlap is 99.91%.
- SIFT: original TexMex 1,000,000 base, first 1,000 of 10,000 queries, 128D,
  unnormalized L2. Archive SHA256
  `92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a`;
  published-vs-recomputed overlap is 99.94%.
- DBpedia OpenAI archive SHA256
  `b54909da6e2fc646174d51b2087a083f6800997979ce55aa29f7b54580f2033d`.
  Despite `1M` in the archive name, the verified split is 975,000 base rows,
  5,000 queries, 1536D; the main run uses the first 1,000 queries. Norms range
  from 0.99999988 to 1.00000012, and explicit normalization is still applied.

## Ablations

### Native ARM integer dot

Before enabling I8MM `usdot`, Cohere doc7 medians were 24.808/13.048 ms at
90% and 42.331/24.703 ms at 95% (1.90x/1.71x). With `usdot`, they became
24.886/10.117 and 42.245/19.428 ms (2.46x/2.17x). The raw pre-I8MM run is
[here](cohere_u7_no_rerank.jsonl); the I8MM run is
[here](cohere_u7_i8mm_no_rerank.jsonl).

### Encoding and correction choices on Cohere

| mode | >=90% speedup | >=95% speedup | bytes/vector | observation |
|---|---:|---:|---:|---|
| signed doc8/query8 | **2.494x** | **2.442x** | 772 | selected |
| unsigned doc7/query8 | 2.460x | 2.174x | 772 | 95% needs ef 72 |
| unsigned doc8/query6 | 2.461x | 1.970x | 772 | query error costs ef |
| packed doc4/query8 | 1.039x | not reached | 388 | unpack + ef=128 erase byte savings |
| signed 8/8, exact norm | 2.432x | 2.141x | 772 | worse than reconstruction norm |

Raw files are colocated with this report.

### Construction storage

FP32 construction took 70.889 s. SQ8 construction took 70.961 s. At ef=32,
the QINCo recall was 90.74% on the FP32 graph and 90.98% on the SQ8-built
graph; near the 95% floor it was 95.31% vs 95.30%. This experiment does not
support the initial hypothesis that FP32 construction improves this Cohere
configuration. See [raw construction ablation](cohere_sq8_construction_ablation.jsonl).

### PCA and int8-head/int4-tail extension

PCA is trained offline on the first 50,000 rows. Base projection and encoding
are not query-time work; query allocation, normalization, PCA, integer
quantization, traversal and top-k are timed.

- Cohere 768->384, int8 head 192/int4 tail 192: 292 B/vector; 1.358x at 90%,
  but ef=384 is needed at 95%, producing a **0.606x regression**. Batch=1 at
  95% is 0.691x. [Raw](cohere_pca384_head192_tail4.jsonl).
- DBpedia 1536->768, int8 head 384/int4 tail 384: 580 B/vector; 0.795x at
  90% and never reaches 95% through ef=512. Raising OpenBLAS from one to 16
  threads changes 0.795x to only 0.798x. [Raw single-thread BLAS](dbpedia_pca768_head384_tail4.jsonl),
  [raw 16-thread BLAS](dbpedia_pca768_head384_tail4_blas16.jsonl).

The extension saves memory but loses the throughput-at-recall objective. It is
not selected.

## GIST retained-fraction sweep

Ground truth is recomputed over each eligible set. `random` is a fixed seeded
random subset. `feature0` keeps rows with the lowest first coordinate and is a
vector-correlated proxy, not genuine metadata. Results below use ef=512 and
report missing result slots out of 10,000.

| retained | random recall / missing | feature0 recall / missing |
|---:|---:|---:|
| 100% | 95.22% / 0 | 95.22% / 0 |
| 50% | 94.72% / 0 | 92.73% / 0 |
| 20% | 93.49% / 0 | 80.76% / 4 |
| 10% | 92.17% / 0 | 68.34% / 305 |
| 5% | 90.68% / 0 | 56.48% / 1,344 |
| 1% | 80.62% / 0 | 33.79% / 4,699 |
| 0.1% | 44.19% / 1,751 | 9.40% / 8,690 |

Raw: [random](gist_filter_random_i8_r64.jsonl),
[correlated proxy](gist_filter_feature0_i8_r64.jsonl). Filtering was not given
a navigation redesign; the collapse at selective/correlated filters is a real
limitation of post-filtered HNSW traversal.

## Correctness evidence

- Seven unit tests compare SIMD distances against an independently written
  scalar equation oracle for signed 8/8, unsigned 7/8, unsigned 8/6, packed
  int4, and mixed int8-head/int4-tail.
- Edge cases cover constant dimensions, zero query, dimension 65 padding,
  repeated/end-point arbitrary IDs, and batch-of-four versus scalar scoring.
- Graph clone tests compare entry point, levels and every neighbor link.
- All seven tests pass on both x86 AVX2 and AWS ARM I8MM. The final seven-test
  x86 run also passes with the new scorer source compiled under ASan+UBSan and
  leak detection enabled.

## Reproduction

Build commands are in `benchs/qinco/README.md`. A representative AWS main run:

```bash
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
OPENBLAS_NUM_THREADS=1 build-qinco-arm/benchs/bench_hnsw_qinco \
  --base data/cohere_medium_1m/train_f32_norm.npy \
  --query data/cohere_medium_1m/test_f32_norm.npy \
  --graph qinco_graphs/cohere_hnsw_M32_efc100.faissindex \
  --output cohere_i8_i8mm_no_rerank.jsonl \
  --nb 1000000 --nq 1000 --threads 16 --M 32 --efConstruction 100 \
  --block 128 --doc-bits 8 --query-bits 8 \
  --efs 24,32,40,48,56,64,72,80,96 --normalize
```

Use `--rebuild-graph` only for the first FP32 construction. The benchmark
appends JSONL, so use a new output path for each run.

## Limitations and prior work

The temporary QINCo name is engineering shorthand, not a novelty claim. This
prototype keeps standard HNSW navigation and changes random-access storage and
scoring. HNSW itself is due to Malkov and Yashunin
([paper](https://arxiv.org/abs/1603.09320)); Faiss already provides per-dimension
SQ8 and HNSW+SQ8 ([ScalarQuantizer API](https://faiss.ai/cpp_api/struct/structfaiss_1_1ScalarQuantizer.html),
[index factory](https://github.com/facebookresearch/faiss/wiki/The-index-factory)).
Faiss also documents PCA+SQ codecs
([vector codecs](https://github.com/facebookresearch/faiss/wiki/Vector-codecs)).
RaBitQ is a distinct randomized quantizer with a theoretical error bound
([paper](https://arxiv.org/abs/2405.12497)). Recent work also combines HNSW,
quantization and reranking, for example AQR-HNSW
([preprint](https://arxiv.org/abs/2602.21600)). No claim of novelty is made.

The implementation was developed without reading or copying the hinted
`CLiqing:feature/rabitq-dense-multibit` prototype. The AWS checkout is a fresh
official Faiss clone at the commit above with only the files in this patch
applied.

The compressed storage is rebuilt from base vectors when the benchmark starts;
custom `index_io` serialization is not implemented in this prototype. FP32
graphs are cached and reusable. Production integration would need a stable
format and backward-compatible reader/writer tags.
