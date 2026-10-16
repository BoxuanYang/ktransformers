/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-16 10:43:18
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-08-07 09:47:43
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_CPUINFER_H
#define CPUINFER_CPUINFER_H

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS)
#include "vendors/cuda.h"
#elif KTRANSFORMERS_USE_MUSA
#include "vendors/musa.h"
#elif KTRANSFORMERS_USE_ROCM
#define __HIP_PLATFORM_AMD__
#include "vendors/hip.h"
#elif KTRANSFORMERS_USE_MACA
#include "vendors/maca.h"
#elif KTRANSFORMERS_USE_ASCEND_NPU
#include "vendors/ascend_npu.h"
#endif

#include "./vendors/vendor.h"
#include "backend_io_manager.h"
#include "llama.cpp/ggml-impl.h"
#include "task_queue.h"
#include "worker_pool.h"

#if defined(KTRANSFORMERS_USE_CUDA)
#include <nvtx3/nvToolsExt.h>
#endif

class CPUInfer {
 public:
  struct KVScatterPlan {
    std::byte* kv_dst;
    const std::byte* src;
    const int32_t* indices;
    size_t count;
    size_t row_bytes;
    size_t capacity;
    size_t page_size;
    bool is_prefill;
  };

  struct SyncArgs {
    CPUInfer* cpuinfer;
    size_t allow_n_pending;
    std::vector<KVScatterPlan> kv_scatter_plans;
    BackendIOManager* io_manager;
    const std::byte* moe_src;
    std::byte* moe_dst;
    size_t moe_nbytes;
    bool delete_after_run;
  };

  CPUInfer(int thread_num) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(thread_num);
    task_queue_ = new TaskQueue();
    for (int i = 0; i < (1 << 16); ++i) {
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
    }
  }
  CPUInfer(int thread_num, int numa_id) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(thread_num, numa_id);
    task_queue_ = new TaskQueue();
    for (int i = 0; i < (1 << 16); ++i) {
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
    }
  }

  CPUInfer(WorkerPoolConfig config) {
    printf("CPUInfer[0x%lx]: Hello\n", (intptr_t)this);
    backend_ = new WorkerPool(config);
    task_queue_ = new TaskQueue();
    for (int i = 0; i < (1 << 16); ++i) {
      ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
    }
  }

  ~CPUInfer() {
    printf("CPUInfer[0x%lx]: Goodbye\n", (intptr_t)this);
    for (SyncArgs* args : captured_sync_args_) {
      delete args;
    }
    delete backend_;
    delete task_queue_;
  }

  CPUInfer(const CPUInfer&) = delete;
  CPUInfer& operator=(const CPUInfer&) = delete;
  CPUInfer(CPUInfer&&) = delete;
  CPUInfer& operator=(CPUInfer&&) = delete;

  template <typename Func, typename Obj, typename... Args>
  void enqueue(Func f, Obj* obj, Args... args) {
    task_queue_->enqueue([=]() { std::invoke(f, *obj, args...); });
  }

  void submit(std::pair<intptr_t, intptr_t> params) {
    void (*func)(void*) = (void (*)(void*))params.first;
    void* args = (void*)params.second;
    *((CPUInfer**)args) = this;
    func(args);
  }
#ifndef KTRANSFORMERS_CPU_ONLY
  void submit_with_cuda_stream(intptr_t user_cuda_stream, std::pair<intptr_t, intptr_t> params) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS) || \
    defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM) || defined(KTRANSFORMERS_USE_MACA) || \
    defined(KTRANSFORMERS_USE_ASCEND_NPU)
    void (*func)(void*) = (void (*)(void*))params.first;
    void* args = (void*)params.second;
    *((CPUInfer**)args) = this;
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)func, args);
#endif
  }
#endif

  static void sync_(void* sync_args) {
    SyncArgs* args = (SyncArgs*)sync_args;
    // The KV D2H is ordered before this already-required KT sync callback.
    // Scatter first so its CPU copy overlaps any remaining CPU MoE work, then
    // perform the original queue drain. Prefill and decode share this path, so
    // KV shadowing adds no CUDA host node in either phase.
    if (!args->kv_scatter_plans.empty()) {
#if defined(KTRANSFORMERS_USE_CUDA)
      nvtxRangePushA(args->kv_scatter_plans.front().is_prefill ? "KVShadow/Prefill_CPU_Scatter"
                                                               : "KVShadow/Decode_CPU_Scatter");
#endif
      for (const KVScatterPlan& plan : args->kv_scatter_plans) {
        const size_t src_stride = 2 * plan.row_bytes;
        for (size_t i = 0; i < plan.count; ++i) {
          const int64_t index = plan.indices[i];
          if (index < 0 || static_cast<size_t>(index) >= plan.capacity) {
            continue;
          }
          const std::byte* row = plan.src + i * src_stride;
          const size_t page = static_cast<size_t>(index) / plan.page_size;
          const size_t offset = static_cast<size_t>(index) % plan.page_size;
          const size_t page_stride = 2 * plan.page_size * plan.row_bytes;
          std::byte* page_base = plan.kv_dst + page * page_stride;
          std::memcpy(page_base + offset * plan.row_bytes, row, plan.row_bytes);
          std::memcpy(page_base + (plan.page_size + offset) * plan.row_bytes,
                      row + plan.row_bytes, plan.row_bytes);
        }
      }
#if defined(KTRANSFORMERS_USE_CUDA)
      nvtxRangePop();
#endif
    }
    args->cpuinfer->task_queue_->sync(args->allow_n_pending);
    // output_cpu becomes valid only after the CPU task queue has drained.  Put
    // the activation H2D on the shared priority executor here, inside KT's
    // existing sync host callback, so CUDA Graph replay does not need any
    // additional host nodes.
    if (args->io_manager != nullptr && args->moe_src != nullptr && args->moe_dst != nullptr &&
        args->moe_nbytes != 0) {
      const std::uint64_t task_id = args->io_manager->submit_moe_h2d(
          reinterpret_cast<std::uintptr_t>(args->moe_src), reinterpret_cast<std::uintptr_t>(args->moe_dst),
          args->moe_nbytes);
      args->io_manager->sync_moe_h2d(task_id);
    }
    if (args->delete_after_run) {
      delete args;
    }
  }

  void sync(size_t allow_n_pending = 0) {
    SyncArgs args{this, allow_n_pending, {}, nullptr, nullptr, nullptr, 0, false};
    sync_(&args);
  }
#ifndef KTRANSFORMERS_CPU_ONLY
  void sync_with_cuda_stream(intptr_t user_cuda_stream, size_t allow_n_pending = 0,
                             const std::vector<std::vector<uintptr_t>>& raw_kv_scatter_plans = {},
                             const std::vector<uintptr_t>& raw_moe_h2d_plan = {}) {
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS) || \
    defined(KTRANSFORMERS_USE_MUSA) || defined(KTRANSFORMERS_USE_ROCM) || defined(KTRANSFORMERS_USE_MACA) || \
    defined(KTRANSFORMERS_USE_ASCEND_NPU)
    std::vector<KVScatterPlan> kv_scatter_plans;
    kv_scatter_plans.reserve(raw_kv_scatter_plans.size());
    for (const auto& raw : raw_kv_scatter_plans) {
      if (raw.size() != 8 || !raw[0] || !raw[1] || !raw[2] || !raw[3] || !raw[4] || !raw[5] ||
          !raw[6]) {
        throw std::invalid_argument("KV scatter plan must contain 7 pointer/size values and a phase flag");
      }
      kv_scatter_plans.push_back(KVScatterPlan{
          reinterpret_cast<std::byte*>(raw[0]),
          reinterpret_cast<const std::byte*>(raw[1]),
          reinterpret_cast<const int32_t*>(raw[2]),
          static_cast<size_t>(raw[3]),
          static_cast<size_t>(raw[4]),
          static_cast<size_t>(raw[5]),
          static_cast<size_t>(raw[6]),
          raw[7] != 0,
      });
    }

    BackendIOManager* io_manager = nullptr;
    const std::byte* moe_src = nullptr;
    std::byte* moe_dst = nullptr;
    size_t moe_nbytes = 0;
    if (!raw_moe_h2d_plan.empty()) {
      if (raw_moe_h2d_plan.size() != 4 || !raw_moe_h2d_plan[0] || !raw_moe_h2d_plan[1] ||
          !raw_moe_h2d_plan[2] || !raw_moe_h2d_plan[3]) {
        throw std::invalid_argument("MoE H2D plan must contain manager, source, destination and byte count");
      }
      io_manager = reinterpret_cast<BackendIOManager*>(raw_moe_h2d_plan[0]);
      moe_src = reinterpret_cast<const std::byte*>(raw_moe_h2d_plan[1]);
      moe_dst = reinterpret_cast<std::byte*>(raw_moe_h2d_plan[2]);
      moe_nbytes = static_cast<size_t>(raw_moe_h2d_plan[3]);
    }

    auto* stream = reinterpret_cast<cudaStream_t>(user_cuda_stream);
    bool captured = false;
#if defined(KTRANSFORMERS_USE_CUDA) || defined(KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS)
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture_status) == cudaSuccess) {
      captured = capture_status != cudaStreamCaptureStatusNone;
    }
#endif
    SyncArgs* args = new SyncArgs{this, allow_n_pending, std::move(kv_scatter_plans), io_manager,
                                  moe_src, moe_dst, moe_nbytes, !captured};
    cudaError_t status = cudaLaunchHostFunc(stream, (cudaHostFn_t)&sync_, (void*)args);
    if (status != cudaSuccess) {
      delete args;
      throw std::runtime_error(cudaGetErrorString(status));
    }
    if (captured) {
      std::lock_guard<std::mutex> lock(captured_sync_args_mutex_);
      captured_sync_args_.push_back(args);
    }
#endif
  }
#endif
 public:
  WorkerPool* backend_;
  TaskQueue* task_queue_;

 private:
  std::mutex captured_sync_args_mutex_;
  std::vector<SyncArgs*> captured_sync_args_;
};

#endif
