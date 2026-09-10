# Integrated Faiss RaBitQ ARM evaluation

## Outcome

The integrated full-dimensional RaBitQ path beats the native Faiss SQ8
baseline at both requested recall floors on Cohere 1M and GIST 1M. It also
wins on DBpedia OpenAI and SIFT, although SIFT's margin is much smaller because
RQ7 needs a higher `efSearch` than SQ8.

All rows use k=10, 1,000 queries, batch=1,000, M=32, efConstruction=100, one
byte-identical FP32 graph per dataset, 16 search/transform threads, and
single-thread OpenBLAS. Each method independently selects the first swept ef
meeting the floor. The table reports the median candidate time from three
confirmation rounds; SQ8 is run immediately before and after every candidate,
so its time is the median of six bracketing samples.

| Dataset | Floor | SQ8 ef / recall | RaBitQ ef / recall | SQ8 ms | RaBitQ ms | RaBitQ QPS | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cohere 1M/768D | 90% | 32 / 0.9122 | 32 / 0.9134 | 25.643 | 11.427 | 87,510 | **2.244x** |
| Cohere 1M/768D | 95% | 64 / 0.9558 | 64 / 0.9550 | 43.345 | 18.347 | 54,506 | **2.363x** |
| GIST 1M/960D | 90% | 256 / 0.9187 | 192 / 0.9045 | 201.912 | 57.600 | 17,361 | **3.505x** |
| GIST 1M/960D | 95% | 512 / 0.9514 | 448 / 0.9558 | 414.832 | 126.073 | 7,932 | **3.290x** |
| DBpedia OpenAI 975K/1536D | 90% | 24 / 0.9038 | 24 / 0.9031 | 42.377 | 19.519 | 51,231 | **2.171x** |
| DBpedia OpenAI 975K/1536D | 95% | 64 / 0.9589 | 64 / 0.9591 | 84.896 | 32.802 | 30,485 | **2.588x** |
| SIFT 1M/128D | 90% | 32 / 0.9067 | 48 / 0.9293 | 7.589 | 6.405 | 156,139 | **1.185x** |
| SIFT 1M/128D | 95% | 64 / 0.9570 | 96 / 0.9620 | 13.063 | 11.720 | 85,328 | **1.115x** |

The GIST 90% selection is deliberately conservative: SQ8 ef=192 measured
0.8980 and therefore does not qualify; RaBitQ ef=192 measured 0.9045. GIST 95%
required a second high-ef sweep because neither method reached the floor by
ef=256. No recall value was extrapolated.

## Batch=1 and memory

Every batch=1 sample contains 200 complete calls and includes allocation,
rotation where applicable, query quantization, traversal, and top-k.

| Dataset | Floor | SQ8 QPS | RaBitQ QPS | Speedup |
|---|---:|---:|---:|---:|
| Cohere | 90% | 3,824 | 9,908 | 2.591x |
| Cohere | 95% | 1,817 | 6,946 | 3.823x |
| GIST | 90% | 457 | 1,941 | 4.250x |
| GIST | 95% | 252 | 773 | 3.064x |
| DBpedia | 90% | 2,353 | 2,760 | 1.173x |
| DBpedia | 95% | 1,133 | 2,118 | 1.869x |
| SIFT | 90% | 14,413 | 34,322 | 2.381x |
| SIFT | 95% | 8,226 | 18,745 | 2.279x |

Logical serving memory includes measured HNSW arrays and all retained code
buffers. RaBitQ keeps its packed codes authoritative and adds a `d+8` byte
cache, so this implementation is faster but larger than SQ8.

| Dataset | SQ8 bytes | RaBitQ bytes | SQ8 bytes/vector | RaBitQ packed + expanded bytes/vector |
|---|---:|---:|---:|---:|
| Cohere | 1,040,129,288 | 1,740,129,288 | 768 | 692 + 776 |
| GIST | 1,232,129,288 | 2,100,129,288 | 960 | 860 + 968 |
| DBpedia | 1,762,923,432 | 3,100,623,432 | 1536 | 1364 + 1544 |
| SIFT | 400,129,288 | 540,129,288 | 128 | 132 + 136 |

## What was integrated

1. `IndexHNSWRaBitQ::add_with_fp32_graph` constructs links with exact FP32 L2
   while the unchanged existing RaBitQ trainer produces serving codes. It is an
   empty-index batch operation and prevents later mixed-distance append.
2. Existing packed RaBitQ levels are expanded to signed bytes while retaining
   the original center and `ExtraBitsFactors`. For residual `r=q-center`,
   `s=max(abs(r))/127` (or one for zero residual), and
   `qi=clamp(nearbyint(r/s),-127,127)`, the full score is

   ```text
   max(0, sum(r*r) + f_add_ex[id]
          + f_rescale_ex[id] * (s * dot(qi,z[id]) + 0.5 * sum(r)))
   ```

   The half sum is FP32 residual data, not reconstructed int8 data. Packed
   codes and factors are never refit. Expanded float is retained as a control;
   the ordinary packed staged scorer remains the default/backward-compatible
   path.
3. ARM single-ID and arbitrary-ID batch-4 dots use signed `sdot` behind the
   Linux `HWCAP_ASIMDDP` gate, with scalar fallback. Query transform parameters
   optionally schedule the existing transform chain in independent blocks;
   default behavior remains serial and batch=1 avoids an OpenMP region.

## Mechanism ablations

- **FP32 construction, Cohere 100K:** native RaBitQ construction required ef=48
  for 0.9089 recall and ef=128 for 0.9586. FP32 construction required ef=32 for
  0.9098 and ef=64 for 0.9503, improving serving QPS by 1.115x and 1.295x.
  Codes were byte-identical. This is a screening ablation, not a claimed 1M
  construction result.
- **Scorer only, same Cohere 100K FP32 graph and serial rotation:** integer
  SDOT beats packed/staged by 1.533x at >=90% and 1.441x at >=95%.
  Expanded-float regresses from 24,541 to 16,943 QPS at the 90% point and from
  21,207 to 11,473 at 95%. Expansion alone is not an optimization.
- **Rotation only, same integer scorer/topology:** after sweeping 2/4/8/16
  threads and blocks 16/32/64/128, 16 threads/block 64 was frozen. An isolated
  bracketed confirmation improved Cohere 100K by 2.750x at 0.9094 recall and
  2.090x at 0.9514. Batch=1 was unchanged. Block 128 underutilized 16 cores.

These results also show why reconstruction MSE or compression ratio is not a
sufficient objective: expanded float has the same document levels as integer
ADC and is much slower, while SIFT's inexpensive dot still needs extra graph
work to recover recall.

## GIST retained-fraction sweep

Ground truth is recomputed over each eligible set. `random` uses a nested fixed
permutation (seed 777). `feature0` is a nested set selected by ascending original
GIST dimension 0; it is explicitly a synthetic correlated-metadata proxy, not
real metadata. The selector is a bitmap, so arbitrary IDs remain independently
scoreable. `missing` counts unfilled slots out of 10,000. No returned ID ever
violated its selector.

| Retained | Filter | ef=512 SQ8/RQ recall; missing | ef=4096 SQ8/RQ recall; missing | RQ/SQ QPS at 512 / 4096 |
|---:|---|---|---|---:|
| 100% | all | .9514/.9594; 0/0 | .9716/.9822; 0/0 | 2.740x / 1.882x |
| 50% | random | .9490/.9557; 0/0 | .9751/.9841; 0/0 | 2.731x / 1.874x |
| 50% | feature0 | .9265/.9333; 0/0 | .9720/.9810; 0/0 | 2.739x / 1.874x |
| 20% | random | .9387/.9458; 0/0 | .9753/.9856; 0/0 | 2.718x / 1.875x |
| 20% | feature0 | .8081/.8108; 4/4 | .9504/.9575; 0/0 | 2.739x / 1.878x |
| 10% | random | .9254/.9301; 0/0 | .9749/.9835; 0/0 | 2.751x / 1.872x |
| 10% | feature0 | .6845/.6858; 305/298 | .8988/.9038; 0/0 | 2.741x / 1.881x |
| 5% | random | .9049/.9079; 0/0 | .9734/.9814; 0/0 | 2.704x / 1.878x |
| 5% | feature0 | .5628/.5657; 1351/1348 | .8239/.8273; 97/93 | 2.750x / 1.881x |
| 1% | random | .8070/.8082; 0/0 | .9693/.9753; 0/0 | 2.765x / 1.882x |
| 1% | feature0 | .3373/.3387; 4697/4699 | .5830/.5855; 1976/1973 | 2.766x / 1.879x |
| 0.1% | random | .4232/.4239; 1894/1891 | .8764/.8780; 0/0 | 2.737x / 1.878x |
| 0.1% | feature0 | .0938/.0942; 8683/8683 | .2285/.2280; 6983/6985 | 2.754x / 1.881x |

At fixed ef, throughput is nearly independent of retained fraction. The
selector filters results but does not redesign navigation, so sparse filtering
does not reduce traversal work. Correlation is dramatically harder than random
filtering and can remain incomplete even at ef=4096. This is a navigation
limitation, deliberately outside the current scope.

## Correctness and dataset verification

- The unchanged handoff script verified all 18 source files, passed 896 integer
  oracle comparisons, and rebuilt its three ARM shared libraries against the
  matching Faiss build. The production test independently passed another 896
  comparisons over actual trained codes, total bits 2/4/7/8, zero residuals,
  tails through 4096D, and arbitrary IDs `{3,0,2,1}`.
- Object disassembly contains `sdot`; runtime reports `native_dotprod=true`.
  Focused kernel ASan+UBSan passed 112 unaligned/tail/100,000D comparisons. A
  fully instrumented SVE Faiss library passed 15,840 end-to-end serial/parallel
  HNSW results plus selector checks. C++ PR3 tests pass 3/3 and Python RaBitQ
  tests pass 15/15.
- Cohere files are exactly 1,000,000 base, 1,000 query, 768D. Maximum norm
  error is `1.79e-7` for base and `1.19e-7` for query. Recomputed top-10 equals
  the supplied IDs exactly.
- GIST is original unnormalized 1,000,000 base, 1,000 query, 960D L2.
  Recomputed versus supplied top-10 has set recall 0.9993. All seven
  substitutions are byte-identical duplicate vectors at exactly equal L2;
  positional differences are tie ordering, not split mismatch.
- SIFT uses original 1,000,000 base, first 1,000 of 10,000 query, 100,000 learn,
  128D unnormalized L2. SQ8 trains on the learn split. Recomputed versus
  supplied top-10 has set recall 0.9994; all six substitutions are exact
  distance ties.
- DBpedia OpenAI's verified archive is 975,000 base and 5,000 query rather than
  a literal million, 1536D; evaluation uses the first 1,000 queries. Base/query
  maximum norm errors are `1.79e-7`/`1.19e-7`. Exact ground truth is recomputed
  because no trusted top-k file accompanies this prepared split.

## Environment

- AWS `54.67.30.55`, Ubuntu 24.04, Linux `7.0.0-1011-aws`.
- Neoverse-V2, one socket, 16 physical cores/threads, 64 KiB L1d and 2 MiB L2
  per core, 36 MiB shared L3; 66,190,868,480 bytes RAM, no swap.
- GCC 13.3.0, CMake 3.28.3, OpenBLAS pthread 0.3.26.
- Official Faiss base `6bbb068ad8dab612b84304400846da1b14539085` plus commits
  `3342ef6c`, `8fb93980`, `ecf61d04`, and `50ef9041`.
- Release C++20 flags: `-O3 -DNDEBUG -fPIC -march=armv8-a+sve -fopenmp`;
  SDOT translation unit additionally uses `-march=armv8.2-a+dotprod`.
- `OMP_NUM_THREADS=16`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`,
  `OPENBLAS_NUM_THREADS=1`. SQ8 and RaBitQ use the same SVE library and thread
  budget. Timings include query allocation, all transforms, quantization,
  traversal, and top-k; offline code/cache construction is reported separately.

## Independent affine comparator and PCA extension

The earlier affine implementation remains isolated on AWS branch
`qinco-independent` at commit `63900b9e`; none of its scorer code was copied
into the RaBitQ integration. Its prior same-machine report measured speedups
over SQ8 of 2.494x/2.442x on Cohere, 2.201x/2.274x on GIST,
2.607x/2.594x on DBpedia, and 1.710x/1.373x on SIFT (90%/95%). Those results
used an older Faiss base (`00928e3f`) and a different representation, so they
are a separate engineering comparator, not a paired claim against this build.

That branch also tested PCA plus int8-head/int4-tail separately. Cohere
768->384 with a 192D int8 head used 292 bytes/vector but regressed to 0.606x at
95%; DBpedia 1536->768 with a 384D int8 head used 580 bytes/vector, achieved
0.795x at 90%, and did not reach 95% through ef=512. The extension was rejected:
lower memory and reconstruction error did not satisfy matched-recall throughput.
The present handoff explicitly preserves RaBitQ document quantization, so that
affine/PCA codec was not merged into `main`.

## Reproduction and raw data

Build:

```sh
cmake -S /home/ubuntu/faiss_rabitq_pr1 \
  -B /home/ubuntu/faiss_rabitq_build_main \
  -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON \
  -DBUILD_TESTING=ON -DFAISS_OPT_LEVEL=sve \
  -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build /home/ubuntu/faiss_rabitq_build_main \
  --target faiss_sve swigfaiss_sve faiss_test -j16
```

Representative main run:

```sh
cd /home/ubuntu/faiss_rabitq_pr1
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp/faiss-rabitq-python \
python3 benchs/bench_hnsw_rabitq_vs_sq8.py \
  --dataset cohere-1m-768d-normalized \
  --base /home/ubuntu/data/cohere_medium_1m/train_f32_norm.npy \
  --query /home/ubuntu/data/cohere_medium_1m/test_f32_norm.npy \
  --graph /home/ubuntu/qinco_graphs/cohere_hnsw_M32_efc100.faissindex \
  --nb 1000000 --nq 1000 --bits 7 --threads 16 \
  --transform-threads 16 --block-size 64 \
  --ef 16 24 32 48 64 96 128 192 256 --floors .90 .95 \
  --output /home/ubuntu/rabitq_results/combined_cohere1m.jsonl
```

Raw JSONL beside this report contains metadata hashes, every ef candidate,
actual recall, every confirmation timing, memory, and batch=1 samples:
`combined_cohere1m.jsonl`, `combined_gist1m.jsonl`,
`combined_gist1m_high_ef.jsonl`, `combined_dbpedia_openai.jsonl`,
`combined_sift1m.jsonl`, and `gist1m_filters.jsonl`. Earlier PR-specific raw
files preserve construction, scorer, and rotation ablations.

No research-novelty claim is made. The implementation combines existing Faiss
RaBitQ/HNSW semantics with standard asymmetric integer scoring, native ARM dot
products, and query-level parallel scheduling; its main unresolved weaknesses
are additional cache memory, lack of a navigation-aware filter, and modest
full-dimensional gains on 128D SIFT.
