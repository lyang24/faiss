# HNSW RaBitQ: support FP32 graph construction

## Behavior

`IndexHNSWRaBitQ::add_with_fp32_graph` is an opt-in, batch-only construction
API. It encodes the supplied vectors with the existing trained `IndexRaBitQ`
storage, but uses a temporary `IndexFlatL2` distance computer while constructing
the HNSW topology. The FP32 copy is released when construction returns and is
not part of serving storage.

The index records this construction policy. Appending with ordinary `add()` is
rejected instead of silently mixing FP32 and RaBitQ construction distances.
`reset()` clears the policy and restores ordinary mutable behavior. Clone and
serialization preserve it; a separate `IHNg` tag leaves existing `IHNr` files
backward readable. Files written with the new opt-in mode require a reader that
knows `IHNg`.

This PR does not change RaBitQ training, codes, serving scores, staged search,
query transforms, or HNSW navigation.

## Correctness

On AWS Neoverse-V2 with the SVE build from Faiss
`6bbb068ad8dab612b84304400846da1b14539085`:

```sh
PYTHONPATH=/tmp/faiss-rabitq-python \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
python3 -m unittest tests.test_hnsw_rabitq -v
```

All 12 tests passed. The added tests cover:

- identical packed RaBitQ bytes between native and FP32 construction;
- exact topology equality with `IndexHNSWFlat` under deterministic build;
- levels, offsets, entry point and maximum level;
- clone, full serialization, storage-skipping serialization and ownership;
- append rejection, repeat-build rejection and reset/rebuild behavior.

## Graph-only ablation

Command:

```sh
PYTHONPATH=/tmp/faiss-rabitq-python \
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
OPENBLAS_NUM_THREADS=1 \
python3 benchs/bench_hnsw_rabitq_fp32_graph.py \
  --base /home/ubuntu/data/cohere_medium_1m/train_f32_norm.npy \
  --query /home/ubuntu/data/cohere_medium_1m/test_f32_norm.npy \
  --nb 100000 --nq 1000 --m 32 --efc 100 --bits 7 --threads 16 \
  --ef 16 24 32 48 64 96 128 192 256 --repeats 3 --rotation \
  --output /home/ubuntu/rabitq_results/pr1_cohere100k_rq7.jsonl
```

The experiment uses normalized Cohere vectors, RR1234, RQ7, k=10, the existing
staged serving scorer and independently swept `efSearch`. The document-code
SHA256 is identical for both construction policies.

| Recall floor | Native graph | FP32 graph | paired QPS gain |
|---|---:|---:|---:|
| 90% | ef=48, 0.9089, 22,210 QPS | ef=32, 0.9098, 24,758 QPS | 1.115x |
| 95% | ef=128, 0.9586, 16,269 QPS | ef=64, 0.9503, 21,065 QPS | 1.295x |

Native construction took 16.880 s and FP32 construction took 7.370 s for this
100K subset. Both serving indexes used 96,420,616 bytes for codes plus graph
arrays. FP32 construction temporarily used an additional 307,200,000-byte flat
vector copy. These are subset results, not the final 1M combined benchmark.
Raw rows are in `pr1_cohere100k_rq7.jsonl`.
