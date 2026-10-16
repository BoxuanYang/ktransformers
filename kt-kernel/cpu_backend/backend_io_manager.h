#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(KTRANSFORMERS_USE_CUDA)
#include "vendors/cuda.h"
#endif

// A single-stream H2D executor shared by KV restoration and CPU-MoE output
// copies.  Only one physical transfer is submitted at a time.  After every
// transfer the worker re-arbitrates the two queues, always preferring MoE.
// This bounds a newly-arrived MoE task's queueing delay to the currently
// executing KV frame.
class BackendIOManager {
 public:
  explicit BackendIOManager(int device_id);
  ~BackendIOManager();

  BackendIOManager(const BackendIOManager&) = delete;
  BackendIOManager& operator=(const BackendIOManager&) = delete;
  BackendIOManager(BackendIOManager&&) = delete;
  BackendIOManager& operator=(BackendIOManager&&) = delete;

  // completion_tag is returned by take_completed_kv() after this frame has
  // completed.  Use zero for non-terminal frames.
  std::uint64_t submit_kv_fetch(std::uintptr_t src, std::uintptr_t dst, std::size_t nbytes,
                                std::uint64_t group_id, std::uint64_t completion_tag = 0,
                                std::uintptr_t ready_flag = 0);

  // Returns a task id consumed by sync_moe_h2d().
  std::uint64_t submit_moe_h2d(std::uintptr_t src, std::uintptr_t dst, std::size_t nbytes);
  void sync_moe_h2d(std::uint64_t task_id);

  // Remove queued KV work for one request generation and wait only for its
  // currently executing frame, if any.  The caller may then safely recycle
  // the reserved destination frames.
  void cancel_kv(std::uint64_t group_id);
  void sync_kv(std::uint64_t group_id);
  void pause_kv();
  void resume_kv();
  std::vector<std::uint64_t> take_completed_kv();
  void close();
  std::uintptr_t native_handle() { return reinterpret_cast<std::uintptr_t>(this); }

 private:
  struct Completion {
    bool done = false;
    std::string error;
    std::condition_variable cv;
  };

  struct Task {
    std::uint64_t id = 0;
    std::uintptr_t src = 0;
    std::uintptr_t dst = 0;
    std::size_t nbytes = 0;
    std::uint64_t group_id = 0;
    std::uint64_t completion_tag = 0;
    std::shared_ptr<Completion> completion;
    std::uintptr_t ready_flag = 0;
  };

  void worker_main();
  void validate_task(std::uintptr_t src, std::uintptr_t dst, std::size_t nbytes) const;
  void fail_task(const Task& task, const std::string& error);

  int device_id_;
  std::mutex mutex_;
  std::condition_variable work_cv_;
  std::condition_variable active_cv_;
  std::deque<Task> kv_queue_;
  std::deque<Task> moe_queue_;
  std::deque<std::uint64_t> completed_kv_;
  std::unordered_map<std::uint64_t, std::shared_ptr<Completion>> moe_completions_;
  std::uint64_t next_task_id_ = 1;
  std::uint64_t active_kv_group_ = 0;
  bool stopping_ = false;
  bool kv_paused_ = false;
  bool closed_ = false;
  std::string fatal_error_;
  std::thread worker_;

#if defined(KTRANSFORMERS_USE_CUDA)
  cudaStream_t stream_ = nullptr;
  cudaEvent_t event_ = nullptr;
#endif
};
