#include "dense_kvcache.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "ggml-impl.h"
#include "llamafile/sgemm.h"

namespace dense {

namespace {

using ThreadFn = std::function<void(int)>;
using TaskFn = std::function<void(int, int)>;

bool has_multiple_numa_nodes(const WorkerPool* backend) {
  const auto& nodes = backend->config.subpool_numa_map;
  return std::any_of(nodes.begin() + 1, nodes.end(),
                     [&](int node) { return node != nodes.front(); });
}

int attention_worker_threads(const WorkerPool* backend) {
  const auto& config = backend->config;
  if (config.subpool_count <= 0 ||
      config.subpool_numa_map.size() != size_t(config.subpool_count) ||
      config.subpool_thread_count.size() != size_t(config.subpool_count)) {
    throw std::invalid_argument("invalid worker pool configuration");
  }

  int total = 0;
  for (int count : config.subpool_thread_count) {
    if (count <= 0 || count > std::numeric_limits<int>::max() - total) {
      throw std::invalid_argument("invalid worker pool thread count");
    }
    total += count;
  }
  return has_multiple_numa_nodes(backend) ? total
                                          : config.subpool_thread_count.front();
}

// WorkerPool::do_work_stealing_job() only uses subpool 0. Dense attention owns
// one scratch slot per worker, so it can safely fan a job out to every subpool
// after assigning each one disjoint task and scratch ranges.
void run_all_subpools(WorkerPool* backend, int task_count,
                      const ThreadFn& init, const TaskFn& compute,
                      const ThreadFn& finalize) {
  if (task_count <= 0) return;
  const int pool_count = backend->config.subpool_count;
  if (pool_count == 1 || !has_multiple_numa_nodes(backend)) {
    backend->do_work_stealing_job(
        task_count, init,
        [&](int task_id) { compute(task_id, WorkerPool::thread_local_id); },
        finalize);
    return;
  }

  std::vector<int> thread_offsets(pool_count + 1, 0);
  for (int pool = 0; pool < pool_count; ++pool) {
    thread_offsets[pool + 1] =
        thread_offsets[pool] + backend->config.subpool_thread_count[pool];
  }
  std::vector<int> task_offsets(pool_count + 1, 0);
  for (int pool = 1; pool < pool_count; ++pool) {
    task_offsets[pool] = int(int64_t(task_count) * thread_offsets[pool] /
                             thread_offsets.back());
  }
  task_offsets.back() = task_count;

  backend->dispense_backend()->do_numa_job([&](int pool) {
    const int begin = task_offsets[pool];
    const int count = task_offsets[pool + 1] - begin;
    if (count == 0) return;
    const int thread_base = thread_offsets[pool];
    backend->get_subpool(pool)->do_work_stealing_job(
        count,
        init ? ThreadFn([&](int local_thread) {
          init(thread_base + local_thread);
        }) : ThreadFn{},
        [&](int local_task) {
          compute(begin + local_task,
                  thread_base + WorkerPool::thread_local_id);
        },
        finalize ? ThreadFn([&](int local_thread) {
          finalize(thread_base + local_thread);
        }) : ThreadFn{});
  });
}

#if defined(__AVX512F__) && defined(__AVX512DQ__)
inline __m512 fast_exp_f32(__m512 x) {
  x = _mm512_max_ps(x, _mm512_set1_ps(-80.0f));
  const __m512 r = _mm512_set1_ps(0x1.8p23f);
  const __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
  const __m512 n = _mm512_sub_ps(z, r);
  const __m512 b = _mm512_fnmadd_ps(
      n, _mm512_set1_ps(0x1.7f7d1cp-20f),
      _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
  const __m512 u = _mm512_mul_ps(b, b);
  const __m512 polynomial = _mm512_fmadd_ps(
      _mm512_fmadd_ps(
          _mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b,
                          _mm512_set1_ps(0x1.573e2ep-5f)),
          u,
          _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b,
                          _mm512_set1_ps(0x1.fffdb6p-2f))),
      u, _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), b,
                         _mm512_set1_ps(1.0f)));
  return _mm512_scalef_ps(polynomial, n);
}
#elif defined(__AVX2__) && defined(__FMA__)
inline __m256 fast_exp_f32(__m256 x) {
  x = _mm256_max_ps(x, _mm256_set1_ps(-80.0f));
  const __m256 r = _mm256_set1_ps(0x1.8p23f);
  const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
  const __m256 n = _mm256_sub_ps(z, r);
  const __m256 b = _mm256_fnmadd_ps(
      n, _mm256_set1_ps(0x1.7f7d1cp-20f),
      _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
  const __m256i exponent = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
  const __m256 power = _mm256_castsi256_ps(
      _mm256_add_epi32(exponent, _mm256_castps_si256(_mm256_set1_ps(1.0f))));
  const __m256 u = _mm256_mul_ps(b, b);
  const __m256 polynomial = _mm256_fmadd_ps(
      _mm256_fmadd_ps(
          _mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                          _mm256_set1_ps(0x1.573e2ep-5f)),
          u,
          _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                          _mm256_set1_ps(0x1.fffdb6p-2f))),
      u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
  return _mm256_fmadd_ps(polynomial, power, power);
}
#endif

inline float scale_softmax_f32(float* row, int valid_tokens, int padded_tokens,
                               float scale) {
  int i = 0;
  float max_score = -std::numeric_limits<float>::infinity();

#if defined(__AVX512F__) && defined(__AVX512DQ__)
  __m512 max_vector = _mm512_set1_ps(max_score);
  const __m512 scale_vector = _mm512_set1_ps(scale);
  for (; i + 16 <= valid_tokens; i += 16) {
    const __m512 value = _mm512_mul_ps(_mm512_loadu_ps(row + i), scale_vector);
    _mm512_storeu_ps(row + i, value);
    max_vector = _mm512_max_ps(max_vector, value);
  }
  max_score = _mm512_reduce_max_ps(max_vector);
#elif defined(__AVX2__) && defined(__FMA__)
  __m256 max_vector = _mm256_set1_ps(max_score);
  const __m256 scale_vector = _mm256_set1_ps(scale);
  for (; i + 8 <= valid_tokens; i += 8) {
    const __m256 value = _mm256_mul_ps(_mm256_loadu_ps(row + i), scale_vector);
    _mm256_storeu_ps(row + i, value);
    max_vector = _mm256_max_ps(max_vector, value);
  }
  alignas(32) float max_lanes[8];
  _mm256_store_ps(max_lanes, max_vector);
  for (float value : max_lanes) max_score = std::max(max_score, value);
#endif
  for (; i < valid_tokens; ++i) {
    row[i] *= scale;
    max_score = std::max(max_score, row[i]);
  }

  i = 0;
  float sum = 0.0f;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
  __m512 sum_vector = _mm512_setzero_ps();
  const __m512 max_vector_for_exp = _mm512_set1_ps(max_score);
  for (; i + 16 <= valid_tokens; i += 16) {
    const __m512 value = fast_exp_f32(
        _mm512_sub_ps(_mm512_loadu_ps(row + i), max_vector_for_exp));
    _mm512_storeu_ps(row + i, value);
    sum_vector = _mm512_add_ps(sum_vector, value);
  }
  sum = _mm512_reduce_add_ps(sum_vector);
#elif defined(__AVX2__) && defined(__FMA__)
  __m256 sum_vector = _mm256_setzero_ps();
  const __m256 max_vector_for_exp = _mm256_set1_ps(max_score);
  for (; i + 8 <= valid_tokens; i += 8) {
    const __m256 value = fast_exp_f32(
        _mm256_sub_ps(_mm256_loadu_ps(row + i), max_vector_for_exp));
    _mm256_storeu_ps(row + i, value);
    sum_vector = _mm256_add_ps(sum_vector, value);
  }
  alignas(32) float sum_lanes[8];
  _mm256_store_ps(sum_lanes, sum_vector);
  for (float value : sum_lanes) sum += value;
#endif
  for (; i < valid_tokens; ++i) {
    row[i] = std::exp(row[i] - max_score);
    sum += row[i];
  }

  ggml_vec_scale_f32(valid_tokens, row, 1.0f / sum);
  std::fill(row + valid_tokens, row + padded_tokens, 0.0f);
  return max_score + std::log(sum);
}

inline void combine_scaled_f32(int n, float* dst, const float* src,
                               float dst_scale, float src_scale) {
  int i = 0;
#if defined(__AVX512F__)
  const __m512 a = _mm512_set1_ps(dst_scale);
  const __m512 b = _mm512_set1_ps(src_scale);
  for (; i + 16 <= n; i += 16) {
    const __m512 x = _mm512_loadu_ps(dst + i);
    const __m512 y = _mm512_loadu_ps(src + i);
    _mm512_storeu_ps(dst + i, _mm512_fmadd_ps(y, b, _mm512_mul_ps(x, a)));
  }
#elif defined(__AVX2__)
  const __m256 a = _mm256_set1_ps(dst_scale);
  const __m256 b = _mm256_set1_ps(src_scale);
  for (; i + 8 <= n; i += 8) {
    const __m256 x = _mm256_loadu_ps(dst + i);
    const __m256 y = _mm256_loadu_ps(src + i);
    _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(y, b, _mm256_mul_ps(x, a)));
  }
#endif
  for (; i < n; ++i) dst[i] = dst[i] * dst_scale + src[i] * src_scale;
}

inline void merge_task_block(int head_count, int head_dim, float* current_output,
                             float* current_max, float* current_sum,
                             const float* block_output, const float* block_lse) {
  alignas(64) float current_scale[16];
  alignas(64) float block_scale[16];

#if defined(__AVX512F__) && defined(__AVX512DQ__)
  const __mmask16 mask = static_cast<__mmask16>((1u << head_count) - 1u);
  const __m512 old_max = _mm512_maskz_loadu_ps(mask, current_max);
  const __m512 next_lse = _mm512_maskz_loadu_ps(mask, block_lse);
  const __m512 merged_max = _mm512_max_ps(old_max, next_lse);
  const __m512 old_scale = fast_exp_f32(_mm512_sub_ps(old_max, merged_max));
  const __m512 next_scale = fast_exp_f32(_mm512_sub_ps(next_lse, merged_max));
  const __m512 old_sum = _mm512_maskz_loadu_ps(mask, current_sum);
  const __m512 merged_sum =
      _mm512_fmadd_ps(old_sum, old_scale, next_scale);
  _mm512_mask_storeu_ps(current_max, mask, merged_max);
  _mm512_mask_storeu_ps(current_sum, mask, merged_sum);
  _mm512_mask_store_ps(current_scale, mask, old_scale);
  _mm512_mask_store_ps(block_scale, mask, next_scale);
#else
  for (int i = 0; i < head_count; ++i) {
    const float merged_max = std::max(current_max[i], block_lse[i]);
    current_scale[i] = std::exp(current_max[i] - merged_max);
    block_scale[i] = std::exp(block_lse[i] - merged_max);
    current_sum[i] = current_sum[i] * current_scale[i] + block_scale[i];
    current_max[i] = merged_max;
  }
#endif

  for (int i = 0; i < head_count; ++i) {
    combine_scaled_f32(head_dim, current_output + i * head_dim,
                       block_output + i * head_dim,
                       current_scale[i], block_scale[i]);
  }
}

inline void multiply_add_f32(int n, float* dst, const float* src, float scale) {
  int i = 0;
#if defined(__AVX512F__)
  const __m512 weight = _mm512_set1_ps(scale);
  for (; i + 16 <= n; i += 16) {
    _mm512_storeu_ps(dst + i,
                     _mm512_fmadd_ps(_mm512_loadu_ps(src + i), weight,
                                     _mm512_loadu_ps(dst + i)));
  }
#elif defined(__AVX2__)
  const __m256 weight = _mm256_set1_ps(scale);
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(dst + i,
                     _mm256_fmadd_ps(_mm256_loadu_ps(src + i), weight,
                                     _mm256_loadu_ps(dst + i)));
  }
#endif
  for (; i < n; ++i) dst[i] += src[i] * scale;
}

inline void fp32_to_bf16_row(const float* input, ggml_bf16_t* output, int n) {
  int i = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
  const __m512i absolute_mask = _mm512_set1_epi32(0x7fffffff);
  const __m512i exponent_mask = _mm512_set1_epi32(0x7f800000);
  const __m512i sign_mask = _mm512_set1_epi32(0x80000000);
  const __m512i infinity = _mm512_set1_epi32(0x7f800000);
  const __m512i quiet_nan = _mm512_set1_epi32(64);
  const __m512i round_bias = _mm512_set1_epi32(0x7fff);
  const __m512i one = _mm512_set1_epi32(1);
  const __m512i zero = _mm512_setzero_si512();

  for (; i + 16 <= n; i += 16) {
    const __m512i bits = _mm512_castps_si512(_mm512_loadu_ps(input + i));
    const __m512i absolute = _mm512_and_si512(bits, absolute_mask);
    const __mmask16 nan_mask =
        _mm512_cmp_epu32_mask(absolute, infinity, _MM_CMPINT_GT);
    const __mmask16 subnormal_mask = _mm512_cmp_epi32_mask(
        _mm512_and_si512(bits, exponent_mask), zero, _MM_CMPINT_EQ);

    const __m512i odd = _mm512_and_si512(_mm512_srli_epi32(bits, 16), one);
    __m512i result = _mm512_srli_epi32(
        _mm512_add_epi32(bits, _mm512_add_epi32(round_bias, odd)), 16);
    const __m512i nan =
        _mm512_or_si512(_mm512_srli_epi32(bits, 16), quiet_nan);
    const __m512i signed_zero =
        _mm512_srli_epi32(_mm512_and_si512(bits, sign_mask), 16);
    result = _mm512_mask_mov_epi32(result, nan_mask, nan);
    result = _mm512_mask_mov_epi32(result, subnormal_mask, signed_zero);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i),
                        _mm512_cvtepi32_epi16(result));
  }
#endif
  if (i < n) ggml_fp32_to_bf16_row(input + i, output + i, n - i);
}

}  // namespace

void KVCache::set_parallel_reduce(bool enabled) {
  parallel_reduce_ = enabled;
}

// task 在 1/2/4/8 个连续逻辑 block 间自适应，兼顾长上下文开销和 64 线程并行度。
void KVCache::attention_kvhead_(const ggml_bf16_t* q_in, ggml_bf16_t* output,
                              float* attn_lse, int batch_size, WorkerPool* backend) {
  const int head_dim = config_.head_dim;
  const int block_len = config_.block_len;
  const int output_size = n_gqa_ * head_dim;
  std::fill(thread_local_failed_.begin(), thread_local_failed_.end(), 0);
  if (batch_size > 0) {
#if !defined(__AVX512BF16__)
    ggml_bf16_to_fp32_row(q_in, query_fp32_.data(),
                          size_t(batch_size) * config_.q_head_num * head_dim);
#endif
  }

  // 两个提交位置共用代码；锁的对象、粒度和提交时机保持不变。
  auto flush_thread_result = [&](int thread_id) {
    const auto [batch_id, head_id] = thread_cur_head_idx_[thread_id];
    if (batch_id == -1) return;
    auto& dst = output_fp32_[batch_id][head_id];
    auto& dst_lse = attn_lse_[batch_id][head_id];
    auto& src = thread_local_cur_output_fp32_[thread_id];
    auto& src_lse = thread_local_cur_attn_lse_[thread_id];
    std::lock_guard<std::mutex> lock(*mutex_[batch_id][head_id]);
    // LSE 可以合法地等于 0，使用独立有效标志判断首次提交。
    if (!output_valid_[batch_id][head_id]) {
      std::copy(src.begin(), src.end(), dst.begin());
      std::copy(src_lse.begin(), src_lse.end(), dst_lse.begin());
      output_valid_[batch_id][head_id] = 1;
    } else {
      for (int i = 0; i < n_gqa_; ++i) {
        const float hi = std::max(dst_lse[i], src_lse[i]);
        const float lo = std::min(dst_lse[i], src_lse[i]);
        const float merged_lse = hi + std::log(1.0 + std::exp(lo - hi));
        combine_scaled_f32(head_dim, dst.data() + i * head_dim,
                           src.data() + i * head_dim,
                           std::exp(dst_lse[i] - merged_lse),
                           std::exp(src_lse[i] - merged_lse));
        dst_lse[i] = merged_lse;
      }
    }
  };

  // 空 batch / 全空 KV 不提交任务，避免线程池处理 task_num=0。
  if (task_offsets_[batch_size] > 0) {
    run_all_subpools(
        backend,
        task_offsets_[batch_size],
        [&](int thread_id) {
          thread_cur_head_idx_[thread_id] = {-1, -1};
        },
        [&](int task_id, int thread_id) {
          // 重复前缀对应空序列，由 upper_bound 自动跳过。
          const int batch_id = int(std::upper_bound(task_offsets_.begin(),
              task_offsets_.begin() + batch_size + 1, task_id) - task_offsets_.begin()) - 1;
          const int chunks = (task_offsets_[batch_id + 1] - task_offsets_[batch_id]) / config_.kv_head_num;
          const int local_id = task_id - task_offsets_[batch_id];
          const int head_id = local_id / chunks;
          const int chunk_id = local_id % chunks;
          const int len = cache_seqlens_[batch_id];
          const int blocks = len / block_len + (len % block_len != 0);
          const int block_begin = chunk_id * blocks_per_task_;
          const int block_end = block_begin + std::min(blocks_per_task_, blocks - block_begin);
          const size_t query_offset =
              (size_t(batch_id) * config_.kv_head_num + head_id) * output_size;
#if defined(__AVX512BF16__)
          const void* query = q_in + query_offset;
#else
          const void* query =
              query_fp32_.data() + query_offset;
#endif
          for (int block_id = block_begin; block_id < block_end; ++block_id) {
            // 有效 block 数与 block 表行步长是两个不同的量。
            const int block_idx = block_table_[size_t(batch_id) * block_num_per_seq_ + block_id];
            if (block_idx < 0 || block_idx >= config_.max_block_num) {
              thread_local_failed_[thread_id] = 1;
              return;
            }
            const int valid_tokens = std::min(block_len, cache_seqlens_[batch_id] - block_id * block_len);
            auto& block_output = thread_local_output_fp32_[thread_id];
            auto& block_lse = thread_local_attn_lse_[thread_id];
            float* block_output_data = block_output.data();
            float* block_lse_data = block_lse.data();
            const ggml_bf16_t* k_block =
                k_cache_bf16_[layer_id_][head_id][block_idx].data();
            const ggml_bf16_t* v_block =
                v_cache_bf16_[layer_id_][head_id][block_idx].data();
            const bool ok = attn_with_kvcache_one_block_(
                head_dim, n_gqa_, query,
                block_len, valid_tokens,
                k_block, v_block,
                thread_local_attn_score_[thread_id].data(), block_output_data, block_lse_data);
            if (!ok) {
              // 不从工作线程抛异常；所有线程完成后，由调用线程报告错误。
              thread_local_failed_[thread_id] = 1;
              return;
            }

            const auto [cur_batch_id, cur_head_id] = thread_cur_head_idx_[thread_id];
            auto& cur_output = thread_local_cur_output_fp32_[thread_id];
            auto& cur_lse = thread_local_cur_attn_lse_[thread_id];
            auto& cur_sum = thread_local_cur_attn_sum_[thread_id];
            if (parallel_reduce_ && block_id != block_begin) {
              // 任务内累计未归一化 numerator/mass，只在任务结束时计算一次 LSE 和归一化。
              merge_task_block(n_gqa_, head_dim, cur_output.data(), cur_lse.data(),
                               cur_sum.data(), block_output.data(), block_lse.data());
            } else if (!parallel_reduce_ && batch_id == cur_batch_id && head_id == cur_head_id) {
              for (int i = 0; i < n_gqa_; ++i) {
                const float hi = std::max(cur_lse[i], block_lse[i]);
                const float lo = std::min(cur_lse[i], block_lse[i]);
                const float merged_lse = hi + std::log(1.0f + std::exp(lo - hi));
                combine_scaled_f32(head_dim, cur_output.data() + i * head_dim,
                                   block_output.data() + i * head_dim,
                                   std::exp(cur_lse[i] - merged_lse),
                                   std::exp(block_lse[i] - merged_lse));
                cur_lse[i] = merged_lse;
              }
            } else {
              if (!parallel_reduce_) flush_thread_result(thread_id);
              thread_cur_head_idx_[thread_id] = {batch_id, head_id};
              std::copy(block_output.begin(), block_output.end(), cur_output.begin());
              std::copy(block_lse.begin(), block_lse.end(), cur_lse.begin());
              if (parallel_reduce_) std::fill(cur_sum.begin(), cur_sum.end(), 1.0f);
            }
          }
          if (parallel_reduce_) {
            for (int i = 0; i < n_gqa_; ++i) {
              ggml_vec_scale_f32(head_dim, thread_local_cur_output_fp32_[thread_id].data() + i * head_dim,
                                 1.0f / thread_local_cur_attn_sum_[thread_id][i]);
              thread_local_cur_attn_lse_[thread_id][i] +=
                  std::log(thread_local_cur_attn_sum_[thread_id][i]);
            }
            // task 内先归并，只有完整成功的 task 才提交一份独占结果。
            std::copy_n(thread_local_cur_output_fp32_[thread_id].data(), output_size,
                        reduce_task_output_.data() + size_t(task_id) * output_size);
            std::copy_n(thread_local_cur_attn_lse_[thread_id].data(), n_gqa_,
                        reduce_task_lse_.data() + size_t(task_id) * kLseStride);
          }
        },
        [&](int thread_id) {
          if (!parallel_reduce_) flush_thread_result(thread_id);
        });
  }
  if (std::find(thread_local_failed_.begin(), thread_local_failed_.end(), 1) != thread_local_failed_.end()) {
    throw std::runtime_error("Dense attention failed: invalid physical block or unsupported AVX512 BF16/F32 GEMM");
  }

  // 上面的同步任务池调用已经等所有 block 完成；失败时不会读取未完成的块结果。
  if (parallel_reduce_ && task_offsets_[batch_size] > 0) {
    // This pass is small and cache-resident; a second NUMA-distributor round is
    // slower than reducing it in subpool 0.
    backend->do_work_stealing_job(
        batch_size * config_.q_head_num,
        [&](int task_id) { reduce_one_query_head_(task_id); });
  }

  // 空序列输出为 0、LSE 为 -inf。输出在此转为 BF16，LSE 始终为 FP32。
  for (int b = 0; b < batch_size; ++b) {
    for (int h = 0; h < config_.kv_head_num; ++h) {
      const size_t group = size_t(b) * config_.kv_head_num + h;
      fp32_to_bf16_row(output_fp32_[b][h].data(), output + group * output_size,
                       output_size);
      if (attn_lse) std::copy(attn_lse_[b][h].begin(), attn_lse_[b][h].end(), attn_lse + group * n_gqa_);
    }
  }
}

// 每个 reduce task 独占一个 query head 的输出，动态领取；不进行 QK/PV 计算。
void KVCache::reduce_one_query_head_(int task_id) {
  int batch_id = task_id / config_.q_head_num;
  int query_head = task_id % config_.q_head_num;
  int head_id = query_head / n_gqa_;
  int group_id = query_head % n_gqa_;
  int chunks = (task_offsets_[batch_id + 1] - task_offsets_[batch_id]) / config_.kv_head_num;
  if (chunks == 0) return;  // 初始化已将空序列输出设为 0、LSE 设为 -inf。
  int first = task_offsets_[batch_id] + head_id * chunks;
  int head_dim = config_.head_dim;
  int output_size = n_gqa_ * head_dim;
  float* dst = output_fp32_[batch_id][head_id].data() + group_id * head_dim;

  // 用各 task LSE 的稳定 softmax 作为 task 输出权重。只读暂存结果，不修改其他 task 的数据。
  float max_lse = reduce_task_lse_[size_t(first) * kLseStride + group_id];
  for (int b = 1; b < chunks; ++b) {
    float value = reduce_task_lse_[size_t(first + b) * kLseStride + group_id];
    if (value > max_lse) max_lse = value;
  }
  float sum = 0.0f;
  for (int b = 0; b < chunks; ++b) {
    float weight = std::exp(reduce_task_lse_[size_t(first + b) * kLseStride + group_id] - max_lse);
    const float* src = reduce_task_output_.data() + size_t(first + b) * output_size + group_id * head_dim;
    sum += weight;
    multiply_add_f32(head_dim, dst, src, weight);
  }
  ggml_vec_scale_f32(head_dim, dst, 1.0f / sum);
  attn_lse_[batch_id][head_id][group_id] = max_lse + std::log(sum);
}

// 在原有 batch 循环内构建任务前缀和，不分配临时任务对象。
void KVCache::attn_initialize_kvhead_(int batch_size, int layer_idx, const int* block_table,
                                    int block_table_stride, const int* cache_seqlens) {
  layer_id_ = layer_idx;
  block_table_ = block_table;
  block_num_per_seq_ = block_table_stride;
  task_offsets_[0] = 0;
  for (int b = 0; b < batch_size; ++b) {
    const int len = cache_seqlens[b];
    if (len < 0) throw std::invalid_argument("cache_seqlens must be nonnegative");
    const int blocks = len / config_.block_len + (len % config_.block_len != 0);
    if (blocks > block_table_stride) throw std::invalid_argument("block table is too short");
    const int chunks = blocks / blocks_per_task_ + (blocks % blocks_per_task_ != 0);
    const int64_t tasks = int64_t(task_offsets_[b]) + int64_t(chunks) * config_.kv_head_num;
    if (tasks > std::numeric_limits<int>::max()) throw std::overflow_error("too many attention tasks");
    task_offsets_[b + 1] = int(tasks);
    cache_seqlens_[b] = len;
    for (int h = 0; h < config_.kv_head_num; ++h) {
      output_valid_[b][h] = 0;
      std::fill(output_fp32_[b][h].begin(), output_fp32_[b][h].end(), 0.0f);
      std::fill(attn_lse_[b][h].begin(), attn_lse_[b][h].end(), -std::numeric_limits<float>::infinity());
    }
  }
  if (parallel_reduce_) {
    size_t output_count = size_t(task_offsets_[batch_size]) * n_gqa_ * config_.head_dim;
    size_t lse_count = size_t(task_offsets_[batch_size]) * kLseStride;
    if (reduce_task_output_.size() < output_count) reduce_task_output_.resize(output_count);
    if (reduce_task_lse_.size() < lse_count) reduce_task_lse_.resize(lse_count);
  }
}

// 公开签名保留兼容性；generate_token_idx 不参与计算，q_len 只允许为 1。
void KVCache::attn(const ggml_bf16_t* q_in, ggml_bf16_t* output, float* attn_lse,
                   int layer_idx, int generate_token_idx, int q_len, int batch_size,
                   int max_block_num, int* block_table, int* cache_seqlens, WorkerPool* backend) {
  (void)generate_token_idx;
  if (q_len != 1) throw std::invalid_argument("Dense attention supports decode q_len=1 only");
  if (!backend || batch_size < 0 ||
      batch_size > config_.max_batch_size || layer_idx < 0 || layer_idx >= config_.layer_num ||
      max_block_num < 0) throw std::invalid_argument("invalid dense attention dimensions or capacity");
  const int worker_threads = attention_worker_threads(backend);
  if (worker_threads > config_.max_thread_num) {
    throw std::invalid_argument("dense attention scratch capacity is smaller than the worker pool");
  }
  if (batch_size > 0 && (!q_in || !output || !cache_seqlens || !block_table)) {
    throw std::invalid_argument("null dense attention input/output");
  }
  const int target_tasks = 2 * worker_threads;
  for (int candidate = kMaxBlocksPerTask; candidate >= 1; candidate /= 2) {
    int64_t task_count = 0;
    for (int batch = 0; batch < batch_size; ++batch) {
      const int blocks = (cache_seqlens[batch] + config_.block_len - 1) / config_.block_len;
      task_count += int64_t(config_.kv_head_num) *
                    ((blocks + candidate - 1) / candidate);
    }
    if (task_count >= target_tasks || candidate == 1) {
      blocks_per_task_ = candidate;
      break;
    }
  }
  attn_initialize_kvhead_(batch_size, layer_idx, block_table, max_block_num, cache_seqlens);
  attention_kvhead_(q_in, output, attn_lse, batch_size, backend);
}

// cache_seqlens 为写入前长度，本函数每次增加 1；跨层调用需由调用方提供各层正确的长度。
void KVCache::attn_with_kvcache(
    const ggml_bf16_t* q_in, const ggml_bf16_t* k_in, const ggml_bf16_t* v_in,
    ggml_bf16_t* output, float* attn_lse, int layer_idx, int generate_token_idx,
    int q_len, int batch_size, int max_block_num, int* block_table,
    int* cache_seqlens, WorkerPool* backend) {
  if (q_len != 1) throw std::invalid_argument("Dense attention supports decode q_len=1 only");
  if (batch_size > 0 && (!q_in || !output)) throw std::invalid_argument("null query/output");
  update_kvcache_bf16(k_in, v_in, layer_idx, block_table, batch_size,
                      max_block_num, cache_seqlens, 1, backend);
  for (int b = 0; b < batch_size; ++b) ++cache_seqlens[b];
  attn(q_in, output, attn_lse, layer_idx, generate_token_idx, 1, batch_size,
       max_block_num, block_table, cache_seqlens, backend);
}

// Q [G,D], K [T,D], V [D,T] 均为 BF16，Q/K 已完成 Norm 和 RoPE。
// score/probability [G,T]，output [G,D] 为单块结果，不能与线程累计结果共用。
// 尾块只改变有效范围，两个 GEMM 仍使用完整 block_len 及原来的行步长。
bool KVCache::attn_with_kvcache_one_block_(
    int head_dim, int n_gqa, const void* q, int block_len, int valid_tokens,
    const ggml_bf16_t* k_cache, const ggml_bf16_t* v_cache, float* attn_score,
    float* output, float* lse) {
  if (!llamafile_sgemm(block_len, n_gqa, head_dim, k_cache, head_dim, q, head_dim,
                       attn_score, block_len, 0, 1, GGML_TASK_TYPE_COMPUTE,
                       GGML_TYPE_BF16,
#if defined(__AVX512BF16__)
                       GGML_TYPE_BF16,
#else
                       GGML_TYPE_F32,
#endif
                       GGML_TYPE_F32, GGML_PREC_DEFAULT)) return false;
  const float scale = 1.0f / std::sqrt(float(head_dim));
  for (int i = 0; i < n_gqa; ++i) {
    float* row = attn_score + i * block_len;
    lse[i] = scale_softmax_f32(row, valid_tokens, block_len, scale);
  }
  // Softmax 已原地覆盖 score，直接作为 PV 的 FP32 probability。
  return llamafile_sgemm(head_dim, n_gqa, block_len, v_cache, block_len,
                         attn_score, block_len, output, head_dim, 0, 1,
                         GGML_TASK_TYPE_COMPUTE, GGML_TYPE_BF16, GGML_TYPE_F32,
                         GGML_TYPE_F32, GGML_PREC_DEFAULT);
}

}  // namespace dense

