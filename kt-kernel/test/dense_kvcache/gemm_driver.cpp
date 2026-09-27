#include <atomic>
#include <cstddef>
#include <cstdint>

#include "llama.cpp/ggml.h"

// The main CMake build applies the repository's MXFP4 compatibility patch to
// llama.cpp. Keep this standalone test buildable before that configure step.
#ifndef QK_MXFP4
#define QK_MXFP4 32
#define GGML_TYPE_MXFP4 GGML_TYPE_COUNT
struct block_mxfp4 {
  uint8_t e;
  uint8_t qs[QK_MXFP4 / 2];
};
inline void ggml_vec_dot_mxfp4_q8_0(int, float* result, size_t, const void*,
                                    size_t, const void*, size_t, int) {
  *result = 0.0f;
}
#endif

#define llamafile_sgemm dense_test_sgemm
#include "llamafile/tinyblas_cpu_sgemm.inc"
#undef llamafile_sgemm

extern "C" bool llamafile_sgemm(long m, long n, long k, const void* a, long lda,
                                 const void* b, long ldb, void* c, long ldc,
                                 int ith, int nth, int task, int a_type,
                                 int b_type, int c_type, int precision) {
  return dense_test_sgemm(m, n, k, a, lda, b, ldb, c, ldc, ith, nth, task,
                          a_type, b_type, c_type, precision);
}
