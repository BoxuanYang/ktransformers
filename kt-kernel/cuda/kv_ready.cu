#include <cuda/atomic>
#include <cuda_runtime.h>

namespace {
__global__ void publish_ready(int* flag) {
  cuda::atomic_ref<int, cuda::thread_scope_device> ready(*flag);
  ready.store(1, cuda::memory_order_release);
}
}  // namespace

cudaError_t publish_kv_ready(int* flag, cudaStream_t stream) {
  publish_ready<<<1, 1, 0, stream>>>(flag);
  return cudaGetLastError();
}
