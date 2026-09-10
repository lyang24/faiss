# PR 2: expanded RaBitQ integer ADC on ARM

This checkpoint compares three scorers on one byte-identical RQ7 database and
one FP32-constructed HNSW topology. It is a Cohere 100K screening run, not the
final 1M confirmation.

## Environment

- AWS `c8g.4xlarge`: 16 physical Neoverse-V2 cores, 61 GiB RAM.
- Ubuntu 24.04, Linux `7.0.0-1011-aws`.
- GCC 13.3.0, CMake 3.28.3, OpenBLAS 0.3.26.
- Faiss base `6bbb068ad8dab612b84304400846da1b14539085`.
- `FAISS_OPT_LEVEL=sve`, release build, OpenMP enabled; runtime reports
  `OPTIMIZE ARM_SVE` and `native_dotprod=true`.
- Fixed thread budget: `OMP_NUM_THREADS=16`, `OMP_PROC_BIND=close`,
  `OMP_PLACES=cores`, `OPENBLAS_NUM_THREADS=1`.

The dedicated ARM translation unit is compiled with
`-march=armv8.2-a+dotprod`. Runtime dispatch requires Linux
`HWCAP_ASIMDDP`; ordinary NEON alone does not select it. Object disassembly
contains native instructions such as:

```text
8c:  4e839440  sdot v0.4s, v2.16b, v3.16b
98:  4e839441  sdot v1.4s, v2.16b, v3.16b
```

## Correctness

- The unchanged handoff `check_arm_reference.sh` passed its 18-file manifest,
  all 896 scalar-oracle comparisons, and rebuilt `integer_adc.so`,
  `threaded_transform.so`, and `filtered_adc.so` against this library.
- The production C++ test independently trains actual `IndexRaBitQ` codes and
  passed 896 exact float comparisons. It covers dimensions
  1, 7, 8, 15, 16, 17, 31, 32, 33, 127, 768, 769, 1024, and 4096; total bits
  2, 4, 7, and 8; zero residuals; tails; and arbitrary batch IDs `{3,0,2,1}`.
- All 14 Python `tests.test_hnsw_rabitq` lifecycle and integration tests pass.
- A focused ASan+UBSan build of the exact production ARM translation unit
  passed 112 single/batch-4 comparisons, including unaligned inputs, zero
  dimensions, every SIMD-tail boundary, 4096-chunk boundaries, and 100,000D.
- The accumulator is reduced to `int64_t` every 4096 dimensions. At most 1024
  signed-byte products reach any `int32` lane per chunk, so the lane bound is
  far below `INT32_MAX`.

## Cohere 100K screening result

Dataset: first 100,000 normalized 768D Cohere base vectors, 1,000 normalized
queries, exact L2 ground truth recomputed for the subset, RR1234, RQ7, M=32,
efConstruction=100, k=10. Every search includes rotation, query centering and
quantization, allocation, HNSW traversal, and top-k. Values below are medians
of three raw timings after one warmup.

| Recall floor | Scorer | efSearch | Actual recall@10 | Batch-1000 QPS | Seconds |
|---|---|---:|---:|---:|---:|
| >=90% | packed/staged | 32 | 0.9085 | 24,540.5 | 0.0407489 |
| >=90% | expanded float | 32 | 0.9101 | 16,943.1 | 0.0590212 |
| >=90% | expanded int8/SDOT | 32 | 0.9091 | 37,625.2 | 0.0265780 |
| >=95% | packed/staged | 64 | 0.9505 | 21,206.6 | 0.0471550 |
| >=95% | expanded float | 64 | 0.9517 | 11,472.5 | 0.0871650 |
| >=95% | expanded int8/SDOT | 64 | 0.9511 | 30,553.6 | 0.0327294 |

Paired speedups of integer ADC over the existing packed/staged scorer are
`1.533x` at >=90% and `1.441x` at >=95%. Against the same expanded layout with
a scalar floating dot, the speedups are `2.221x` and `2.663x`. The negative
expanded-float result is important: removing bit extraction without a suitable
native dot product regresses end-to-end throughput.

For batch=1, using the ef selected above, integer ADC reaches 9,004.8 QPS at
ef=32 versus 3,039.9 for packed (`2.962x`), and 6,361.2 at ef=64 versus
2,513.6 (`2.531x`). Each sample contains 200 complete single-query calls; the
one-query recall happens to be 1.0 and is not used to select ef.

## Memory and setup

The packed RQ7 codes occupy 69,200,000 bytes and the graph 27,220,616 bytes.
The expanded cache adds exactly 77,600,000 bytes (`d+8 = 776` bytes/vector),
for 174,020,616 logical serving bytes while the authoritative packed codes are
retained. Cache construction took 49.3 ms on first allocation and 29.8 ms when
reusing capacity. This one-time document conversion is not included in query
latency; query quantization is included.

## Reproduction

```sh
cd /home/ubuntu/faiss_rabitq_pr1
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp/faiss-rabitq-python \
python3 benchs/bench_hnsw_rabitq_integer_adc.py \
  --base /home/ubuntu/data/cohere_medium_1m/train_f32_norm.npy \
  --query /home/ubuntu/data/cohere_medium_1m/test_f32_norm.npy \
  --nb 100000 --nq 1000 --m 32 --efc 100 --bits 7 --threads 16 \
  --ef 16 24 32 48 64 96 128 --repeats 3 --batch1-iterations 200 \
  --rotation --output /home/ubuntu/rabitq_results/pr2_cohere100k_rq7.jsonl
```

Raw records, including every timing sample and all ef candidates, are in
`pr2_cohere100k_rq7.jsonl` beside this report. PR 3 will separately replace the
serial rotation control and the final study will rerun 1M paired confirmation.
