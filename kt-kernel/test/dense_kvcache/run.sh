#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
arch="${DENSE_TEST_ARCH:-avx512}"
if [[ "$arch" == "avx512" ]]; then
  arch_flags=(-mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx2 -mfma -mf16c)
elif [[ "$arch" == "avx512bf16" ]]; then
  arch_flags=(-mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512bf16 -mavx2 -mfma -mf16c)
elif [[ "$arch" == "avx2" ]]; then
  arch_flags=(-mavx2 -mfma -mf16c)
else
  echo "DENSE_TEST_ARCH must be avx512, avx512bf16 or avx2" >&2
  exit 2
fi

build_dir="build/dense_kvcache_$arch"
mkdir -p "$build_dir"

common=(
  -std=c++20 -O3 -pthread -D_GNU_SOURCE
  "${arch_flags[@]}"
  -Ithird_party -Ithird_party/llama.cpp -Ikt-kernel
)

gcc_common=(
  -std=gnu11 -O2 -D_GNU_SOURCE
  "${arch_flags[@]}"
  -Ithird_party -Ithird_party/llama.cpp -Ikt-kernel
)

for source in third_party/llama.cpp/ggml.c third_party/llama.cpp/ggml-quants.c; do
  gcc "${gcc_common[@]}" -c "$source" -o "$build_dir/$(basename "$source").o"
done

sources=(
  third_party/llamafile/iqk_mul_mat_amd_avx2.cpp
  third_party/llamafile/flags.cpp
  kt-kernel/cpu_backend/worker_pool.cpp
  kt-kernel/operators/dense_kvcache/dense_kvcache_attn.cpp
  kt-kernel/operators/dense_kvcache/dense_kvcache_utils.cpp
  kt-kernel/operators/dense_kvcache/dense_kvcache_read_write.cpp
  kt-kernel/operators/dense_kvcache/dense_kvcache_load_dump.cpp
  kt-kernel/test/dense_kvcache/gemm_driver.cpp
  kt-kernel/test/dense_kvcache/test_glm45_air.cpp
)

for source in "${sources[@]}"; do
  object="$build_dir/$(basename "$source").o"
  g++ "${common[@]}" -include cstring -c "$source" -o "$object"
done

g++ -pthread "$build_dir"/*.o -lnuma -lhwloc -o "$build_dir/test_glm45_air"
if [[ "${1:-}" != "--build-only" ]]; then
  if [[ $# == 0 ]]; then
    "$build_dir/test_glm45_air" 4
  else
    "$build_dir/test_glm45_air" "$@"
  fi
fi
