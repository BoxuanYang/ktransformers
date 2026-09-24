#!/usr/bin/env bash
# 在 Linux x86 上运行原生 CPU 回归；先执行 install.sh build，应用 GGML 补丁。
set -euo pipefail
cd "$(dirname "$0")/../.."
out=kt-kernel/build/test-llamafile-iq4
mkdir -p "$out"
flags=(-O2 -mavx2 -mfma -mf16c -ffunction-sections -fdata-sections)
gcc "${flags[@]}" -D_GNU_SOURCE -c third_party/llama.cpp/ggml.c -o "$out/ggml.o"
gcc "${flags[@]}" -c third_party/llama.cpp/ggml-quants.c -o "$out/quants.o"
g++ "${flags[@]}" -std=c++20 -I third_party -c third_party/llamafile/iqk_mul_mat_amd_avx2.cpp -o "$out/iqk.o"
g++ "${flags[@]}" -std=c++20 -I third_party -I third_party/llama.cpp -I kt-kernel \
    kt-kernel/test/test_llamafile_iq4_nl.cpp third_party/llamafile/flags.cpp \
    kt-kernel/cpu_backend/worker_pool.cpp kt-kernel/cpu_backend/shared_mem_buffer.cpp \
    "$out/ggml.o" "$out/quants.o" "$out/iqk.o" \
    -Wl,--gc-sections -pthread -lnuma -lhwloc -lm -o "$out/test"
"$out/test"
