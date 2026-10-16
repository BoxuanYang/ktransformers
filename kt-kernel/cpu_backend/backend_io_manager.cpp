#include "backend_io_manager.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

#if defined(KTRANSFORMERS_USE_CUDA)
cudaError_t publish_kv_ready(int* flag, cudaStream_t stream);
#endif

namespace {

#if defined(KTRANSFORMERS_USE_CUDA)
void check_cuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) {
    return;
  }
  std::ostringstream message;
  message << operation << " failed: " << cudaGetErrorString(status);
  throw std::runtime_error(message.str());
}
#endif

}  // namespace

BackendIOManager::BackendIOManager(int device_id) : device_id_(device_id) {
#if defined(KTRANSFORMERS_USE_CUDA)
  try {
    check_cuda(cudaSetDevice(device_id_), "cudaSetDevice");
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    check_cuda(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming), "cudaEventCreateWithFlags");
    worker_ = std::thread(&BackendIOManager::worker_main, this);
  } catch (...) {
    if (event_) {
      cudaEventDestroy(event_);
      event_ = nullptr;
    }
    if (stream_) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
    throw;
  }
#else
  throw std::runtime_error("BackendIOManager requires a CUDA build of kt-kernel");
#endif
}

BackendIOManager::~BackendIOManager() {
  try {
    close();
  } catch (...) {
    // Destructors must not throw. Explicit close() still reports failures.
  }
}

void BackendIOManager::validate_task(std::uintptr_t src, std::uintptr_t dst, std::size_t nbytes) const {
  if (!src || !dst || !nbytes) {
    throw std::invalid_argument("H2D task requires non-null pointers and a positive byte count");
  }
}

std::uint64_t BackendIOManager::submit_kv_fetch(std::uintptr_t src, std::uintptr_t dst, std::size_t nbytes,
                                                 std::uint64_t group_id, std::uint64_t completion_tag,
                                                 std::uintptr_t ready_flag) {
  validate_task(src, dst, nbytes);
  if (!group_id) {
    throw std::invalid_argument("KV fetch group_id must be non-zero");
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_ || stopping_) {
    throw std::runtime_error("BackendIOManager is closed");
  }
  if (!fatal_error_.empty()) {
    throw std::runtime_error(fatal_error_);
  }
  const std::uint64_t id = next_task_id_++;
  kv_queue_.push_back(Task{id, src, dst, nbytes, group_id, completion_tag, nullptr, ready_flag});
  work_cv_.notify_one();
  return id;
}

std::uint64_t BackendIOManager::submit_moe_h2d(std::uintptr_t src, std::uintptr_t dst, std::size_t nbytes) {
  validate_task(src, dst, nbytes);
  auto completion = std::make_shared<Completion>();
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_ || stopping_) {
    throw std::runtime_error("BackendIOManager is closed");
  }
  if (!fatal_error_.empty()) {
    throw std::runtime_error(fatal_error_);
  }
  const std::uint64_t id = next_task_id_++;
  moe_completions_.emplace(id, completion);
  moe_queue_.push_back(Task{id, src, dst, nbytes, 0, 0, std::move(completion)});
  work_cv_.notify_one();
  return id;
}

void BackendIOManager::sync_moe_h2d(std::uint64_t task_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = moe_completions_.find(task_id);
  if (it == moe_completions_.end()) {
    throw std::invalid_argument("Unknown MoE H2D task id");
  }
  std::shared_ptr<Completion> completion = it->second;
  completion->cv.wait(lock, [&] { return completion->done; });
  const std::string error = completion->error;
  moe_completions_.erase(task_id);
  if (!error.empty()) {
    throw std::runtime_error(error);
  }
}

void BackendIOManager::cancel_kv(std::uint64_t group_id) {
  if (!group_id) {
    return;
  }
  std::unique_lock<std::mutex> lock(mutex_);
  kv_queue_.erase(std::remove_if(kv_queue_.begin(), kv_queue_.end(),
                                 [group_id](const Task& task) { return task.group_id == group_id; }),
                  kv_queue_.end());
  active_cv_.notify_all();
  active_cv_.wait(lock, [&] { return active_kv_group_ != group_id; });
}

void BackendIOManager::sync_kv(std::uint64_t group_id) {
  std::unique_lock<std::mutex> lock(mutex_);
  active_cv_.wait(lock, [&] {
    return !fatal_error_.empty() ||
           (active_kv_group_ != group_id &&
            std::none_of(kv_queue_.begin(), kv_queue_.end(),
                         [group_id](const Task& t) { return t.group_id == group_id; }));
  });
  if (!fatal_error_.empty()) throw std::runtime_error(fatal_error_);
}

void BackendIOManager::pause_kv() {
  std::unique_lock<std::mutex> lock(mutex_);
  kv_paused_ = true;
  active_cv_.wait(lock, [&] { return active_kv_group_ == 0; });
}

void BackendIOManager::resume_kv() {
  std::lock_guard<std::mutex> lock(mutex_);
  kv_paused_ = false;
  work_cv_.notify_one();
}

std::vector<std::uint64_t> BackendIOManager::take_completed_kv() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!fatal_error_.empty()) {
    throw std::runtime_error(fatal_error_);
  }
  std::vector<std::uint64_t> result(completed_kv_.begin(), completed_kv_.end());
  completed_kv_.clear();
  return result;
}

void BackendIOManager::fail_task(const Task& task, const std::string& error) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (fatal_error_.empty()) {
    fatal_error_ = error;
  }
  if (task.completion) {
    task.completion->error = error;
    task.completion->done = true;
    task.completion->cv.notify_all();
  }
}

void BackendIOManager::worker_main() {
#if defined(KTRANSFORMERS_USE_CUDA)
  try {
    check_cuda(cudaSetDevice(device_id_), "cudaSetDevice(worker)");
    while (true) {
      Task task;
      bool is_kv = false;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        work_cv_.wait(lock, [&] {
          return stopping_ || !moe_queue_.empty() || (!kv_paused_ && !kv_queue_.empty());
        });
        if (stopping_ && moe_queue_.empty() && kv_queue_.empty()) {
          break;
        }
        // Re-arbitrate after every physical transfer. MoE is strict priority;
        // both queues remain FIFO internally.
        if (!moe_queue_.empty()) {
          task = std::move(moe_queue_.front());
          moe_queue_.pop_front();
        } else {
          task = std::move(kv_queue_.front());
          kv_queue_.pop_front();
          is_kv = true;
          active_kv_group_ = task.group_id;
        }
      }

      bool succeeded = true;
      try {
        check_cuda(cudaMemcpyAsync(reinterpret_cast<void*>(task.dst), reinterpret_cast<const void*>(task.src),
                                   task.nbytes, cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync(H2D)");
        // Same-stream publication follows the final DMA. The compute-side gate
        // acquires this flag once per layer; it never waits for a missing layer.
        if (task.ready_flag) {
          check_cuda(publish_kv_ready(reinterpret_cast<int*>(task.ready_flag), stream_),
                     "publish_kv_ready");
        }
        check_cuda(cudaEventRecord(event_, stream_), "cudaEventRecord");
        check_cuda(cudaEventSynchronize(event_), "cudaEventSynchronize");
      } catch (const std::exception& error) {
        succeeded = false;
        fail_task(task, error.what());
        // Do not publish a terminal layer flag after an earlier frame failed.
        std::lock_guard<std::mutex> lock(mutex_);
        kv_queue_.clear();
        moe_queue_.clear();
        stopping_ = true;
        for (auto& [_, completion] : moe_completions_) {
          completion->error = fatal_error_;
          completion->done = true;
          completion->cv.notify_all();
        }
        active_cv_.notify_all();
      }

      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (is_kv) {
          if (succeeded && task.completion_tag) {
            completed_kv_.push_back(task.completion_tag);
          }
          active_kv_group_ = 0;
          active_cv_.notify_all();
        } else if (task.completion) {
          task.completion->done = true;
          task.completion->cv.notify_all();
        }
      }
    }
  } catch (const std::exception& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (fatal_error_.empty()) {
      fatal_error_ = error.what();
    }
    for (auto& [_, completion] : moe_completions_) {
      completion->error = fatal_error_;
      completion->done = true;
      completion->cv.notify_all();
    }
    active_kv_group_ = 0;
    active_cv_.notify_all();
  }
#endif
}

void BackendIOManager::close() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
      return;
    }
    stopping_ = true;
    work_cv_.notify_all();
  }
  if (worker_.joinable()) {
    worker_.join();
  }
#if defined(KTRANSFORMERS_USE_CUDA)
  if (event_) {
    check_cuda(cudaEventDestroy(event_), "cudaEventDestroy");
    event_ = nullptr;
  }
  if (stream_) {
    check_cuda(cudaStreamDestroy(stream_), "cudaStreamDestroy");
    stream_ = nullptr;
  }
#endif
  std::lock_guard<std::mutex> lock(mutex_);
  closed_ = true;
  if (!fatal_error_.empty()) {
    throw std::runtime_error(fatal_error_);
  }
}
