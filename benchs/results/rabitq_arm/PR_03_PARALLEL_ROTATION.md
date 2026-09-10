# PR 3: block-parallel existing query rotation

This PR changes only how the existing `IndexPreTransform` chain is scheduled.
It does not change RR1234, dimensions, transform order, RaBitQ training,
document codes, center, factors, or the downstream graph search.

## Interface and concurrency contract

`SearchParametersPreTransform` now accepts `transform_threads` and
`transform_block_size`. The default thread value is zero and preserves the
old serial call exactly. A positive explicit budget enables a per-transform
OpenMP loop over independent query blocks; every loop joins before the next
transform or sub-index search.

The implementation allocates one normal output matrix per chain element and
includes that allocation in end-to-end timing. It never mutates global OpenMP
or BLAS thread settings. The benchmark fixes BLAS to one thread to avoid nested
oversubscription. The number of active transform threads is capped by the
number of blocks, and batches no larger than one block stay on the serial path.
This avoids startup overhead for batch=1. Concurrent search calls share only
const transform state, and exceptions raised inside a worker are captured and
re-thrown after the OpenMP barrier.

The existing `index_params` pointer is still forwarded to the sub-index, so
`efSearch`, selectors, and other per-search controls keep their prior scope.
`range_search` and `search_and_reconstruct` use the same opt-in transform
scheduling; calls without `SearchParametersPreTransform` remain serial.

## Correctness

- Direct transform checks cover query counts 1, 7, 63, 64, 65, 127, 128, 129,
  and 1000, including every boundary around the 64-query reference block.
- Serial and parallel `IndexPreTransform` searches return identical IDs and
  close scores. An additional 7-query/block-3 case forces multiple workers for
  a small tail.
- A real `IndexPreTransform -> IndexHNSWRaBitQ` Python integration test uses
  FP32 graph construction, RQ7 expanded integer ADC, and the same query-count
  set; all IDs and scores match. The full Python file passes 15/15 tests.
- A filtered HNSW test verifies that `IDSelectorRange` and `efSearch` reach the
  sub-index through the new outer parameters.
- Invalid budgets throw, and an intentionally malformed transform proves that
  an exception inside an OpenMP worker reaches the caller rather than calling
  `std::terminate`.
- The full SVE library was rebuilt with ASan+UBSan, not just the test harness.
  An end-to-end RR1234/RQ7/FP32-HNSW program passed 15,840 serial-versus-parallel
  result comparisons across the query-count set plus a filtered search, with
  leak detection enabled.

## Cohere 100K scheduling sweep

The machine and build match the PR 2 report: 16-core Neoverse-V2, GCC 13.3,
OpenBLAS 0.3.26, SVE release build, OpenMP 16 threads, BLAS 1 thread. Dataset is
normalized Cohere 100K/768D with 1,000 queries, exact subset L2 ground truth,
RR1234, RQ7 integer ADC, FP32 HNSW M=32/efConstruction=100, and k=10. Each cell
is the median of three end-to-end samples after a warmup.

| Transform schedule | Best block | QPS ef=32 | recall@10 | QPS ef=64 | recall@10 |
|---|---:|---:|---:|---:|---:|
| Serial | - | 36,981.2 | 0.9100 | 30,281.1 | 0.9512 |
| 2 threads | 128 | 55,370.6 | 0.9100 | 41,800.6 | 0.9512 |
| 4 threads | 128 | 72,180.0 | 0.9100 | 50,940.8 | 0.9512 |
| 8 threads | 128 | 89,380.5 | 0.9100 | 59,004.7 | 0.9512 |
| 16 threads | 64 | 100,224.2 | 0.9100 | 65,253.3 | 0.9512 |

The selected 16-thread/block-64 schedule is `2.710x` faster than serial at the
>=90% point and `2.155x` at >=95%. Block 128 underuses the 16-core machine
because 1,000 queries produce only eight blocks; its ef=32 throughput falls to
89,241.6 QPS. Smaller blocks provide more tasks but pay more BLAS packing and
call overhead. The best block therefore depends on both batch and thread count,
not just transform arithmetic intensity.

At batch=1, the selected parameters automatically use the serial path: ef=32
measured 8,761.7 versus 8,716.9 QPS for the serial control, and ef=64 measured
6,228.3 versus 6,216.8. These differences are noise-sized; no small-batch gain
is claimed.

After freezing 16 threads/block 64, an isolated run repeated three samples for
serial-before, parallel, and serial-after. Pooling the six bracketing serial
samples gives 37,341 QPS at ef=32 versus 102,677 for parallel (`2.750x`), with
recall 0.9094 for both. At ef=64 the pooled serial control is 30,698 QPS versus
64,145 (`2.090x`), with recall 0.9514 for both. The separate raw confirmation
file preserves every sample. An earlier confirmation attempted during the
sanitizer compilation was rejected and is not included in repository results.

## Reproduction

```sh
cd /home/ubuntu/faiss_rabitq_pr1
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
OPENBLAS_NUM_THREADS=1 PYTHONPATH=/tmp/faiss-rabitq-python \
python3 benchs/bench_hnsw_rabitq_parallel_rotation.py \
  --base /home/ubuntu/data/cohere_medium_1m/train_f32_norm.npy \
  --query /home/ubuntu/data/cohere_medium_1m/test_f32_norm.npy \
  --nb 100000 --nq 1000 --m 32 --efc 100 --bits 7 \
  --search-threads 16 --transform-threads 2 4 8 16 \
  --block-size 16 32 64 128 --ef 32 64 --repeats 3 \
  --batch1-iterations 200 \
  --output /home/ubuntu/rabitq_results/pr3_cohere100k_rotation_sweep.jsonl
```

The complete 69-record sweep is stored in
`pr3_cohere100k_rotation_sweep.jsonl`; the isolated bracket is in
`pr3_cohere100k_rotation_confirm.jsonl`. The final combined 1M experiment must
reselect/freeze ef independently for each scorer and use bracketed confirmation;
this 100K scheduling screen does not substitute for that result.
