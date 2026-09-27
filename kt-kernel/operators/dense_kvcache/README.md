# GLM-4.5-Air dense CPU decode attention

This operator is the BF16/AVX512 adaptation of the Qwen3 dense paged-attention
kernel. It is intentionally narrow: decode only (`q_len == 1`) and the
GLM-4.5-Air head shape (`96` query heads, `8` KV heads, `head_dim == 128`,
GQA group size `12`). A full model has 46 layers; `layer_num` remains
configurable so deployments can offload only part of the model.

The caller supplies post-RoPE Q/K. GLM-4.5-Air applies RoPE to only part of each
head, but that does not change this kernel because RoPE is already complete at
the API boundary.

## Layout and precision

- Q/output: BF16 `[batch, 1, 96, 128]`
- new K/V: BF16 `[batch, q_len, 8, 128]`
- K pages: BF16 `[block_len, 128]`
- V pages: BF16 `[128, block_len]`
- scores, probabilities, partial outputs and LSE: FP32
- block table: int32 `[batch, block_table_stride]`

`block_len` must be a multiple of 16 because it is the reduction dimension of
the AVX512 BF16-by-FP32 PV GEMM. It is the CPU cache page size and is independent
from MiniSGL's GPU KV-cache `--page-size` unless an integration layer explicitly
shares the same page table.

On AVX512F CPUs, Q is converted once per decode call and QK/PV use
BF16-by-FP32 GEMM with FP32 accumulation. If AVX512-BF16 is available, QK uses
native BF16 dot products and skips that conversion; PV keeps FP32 probabilities
for accuracy.

The hot path uses 1/2/4/8-page tasks selected at runtime to keep roughly two
tasks per worker. Softmax is vectorized, probabilities reuse the score buffer,
and task-local blocks accumulate an unnormalized numerator and softmax mass so
normalization happens only once. The second reduction remains stable. Logical
pages may map to arbitrary physical pages. The cache object and its worker pool
must not be used by overlapping calls.

When the worker pool spans distinct NUMA nodes, the block GEMMs are split
across all subpools in proportion to their thread counts, with disjoint scratch
slots. Pools mapped to the same NUMA node retain the lower-overhead subpool-0
path. The small final reduction also stays in subpool 0. With
`--kt-cpuinfer 64 --kt-threadpool-count 2`, the usual `[0, 1]` mapping therefore
uses both 32-thread subpools for QK/PV instead of silently using only 32 threads.

## Build and test

The main CMake build auto-detects AVX512. On the target machine it can also be
requested explicitly:

```bash
cmake -S kt-kernel -B build/kt-kernel -DLLAMA_AVX512=ON
```

The standalone numerical test compiles the exact AVX512 path and compares
ragged paged attention against an FP64 reference:

```bash
bash kt-kernel/test/dense_kvcache/run.sh 4
bash kt-kernel/test/dense_kvcache/run.sh 64
# On a non-AVX512 development host, compile without executing:
bash kt-kernel/test/dense_kvcache/run.sh --build-only
# Compile the optional native BF16-dot path without executing it:
DENSE_TEST_ARCH=avx512bf16 bash kt-kernel/test/dense_kvcache/run.sh --build-only
```

For a numerical check on an AVX2-only development host, use
`DENSE_TEST_ARCH=avx2 bash kt-kernel/test/dense_kvcache/run.sh 4`. The target
deployment build and benchmark should still use the default AVX512 mode.
Use `--bench` for a repeatable single-layer batch-5/context-2048 microbenchmark:

```bash
bash kt-kernel/test/dense_kvcache/run.sh --bench 64
# Exercise two NUMA subpools (threads first, subpool count second):
bash kt-kernel/test/dense_kvcache/run.sh --bench 64 2
```

The Python extension exposes `kt_kernel_ext.dense_kvcache`. Pass CPU addresses
of contiguous BF16 tensors to `update_kvcache_bf16`, `attn`, or
`attn_with_kvcache`; block tables and sequence lengths are contiguous int32 CPU
tensors, and LSE is FP32.
