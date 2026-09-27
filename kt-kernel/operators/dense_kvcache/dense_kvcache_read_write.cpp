#include "dense_kvcache.h"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace dense {

/**
 * @brief 在同一 token-major BF16 缓冲区中导出历史 KV，并把紧随其后的新 KV 写回 cache。
 *
 * 函数按 (batch, logical block, KV head) 并行。对 [0, cache_seqlens) 范围从内部 BF16 cache 读取；
 * 对 [cache_seqlens, cache_seqlens + q_len) 范围则从同一缓冲区读取新 K/V 并写回物理 block。
 * 这个函数来自原 get_and_update_kvcache_bf16()，删除了 Q4_0/Q8_0 反量化和量化写回分支。
 *
 * @param k_in 输入/输出 BF16 Key 缓冲区，布局为 [batch_size, max_tokens, kv_head_num, head_dim]。
 * @param v_in 输入/输出 BF16 Value 缓冲区，布局与 k_in 相同。
 * @param layer_id 要读取和更新的模型层编号。
 * @param block_table 行优先 [batch_size, max_block_num] 逻辑到物理 block 映射。
 * @param batch_size 序列数量。
 * @param max_block_num 每个序列的 block 表宽度，同时决定缓冲区最大 token 容量。
 * @param cache_seqlens 每个序列在写入新 token 前的有效 token 数。
 * @param q_len 紧随已有 cache、需要写回的新 token 数；当前 Decode 调用约定为 1。
 * @param backend 用于并行处理 batch、block 和 KV head 的工作线程池。
 */
void KVCache::get_and_update_kvcache_bf16(ggml_bf16_t* k_in, ggml_bf16_t* v_in,
                                          int layer_id, int* block_table,
                                          int batch_size, int max_block_num,
                                          int* cache_seqlens, int q_len,
                                          WorkerPool* backend) {
  backend->do_work_stealing_job(
      config_.kv_head_num * max_block_num * batch_size, nullptr,
      [&](int task_id) {
        int batch_id = task_id / (config_.kv_head_num * max_block_num);
        int block_id = (task_id / config_.kv_head_num) % max_block_num;
        int head_id = task_id % config_.kv_head_num;
        int block_idx = block_table[size_t(batch_id) * max_block_num + block_id];
        int seq_len = cache_seqlens[batch_id];
        int block_l = block_id * config_.block_len;
        int block_r = block_id * config_.block_len + config_.block_len;
        ggml_bf16_t* k_block = k_cache_bf16_[layer_id][head_id][block_idx].data();
        ggml_bf16_t* v_block = v_cache_bf16_[layer_id][head_id][block_idx].data();
        if (block_l < seq_len) {
          for (int k = 0; k < config_.block_len; k++) {
            if (block_id * config_.block_len + k >= seq_len) break;
            for (int l = 0; l < config_.head_dim; l++) {
              k_in[size_t(batch_id) * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                      block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                      k * (config_.kv_head_num * config_.head_dim) + head_id * config_.head_dim + l] =
                  k_block[k * config_.head_dim + l];
              v_in[size_t(batch_id) * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                      block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                      k * (config_.kv_head_num * config_.head_dim) + head_id * config_.head_dim + l] =
                  v_block[l * config_.block_len + k];
            }
          }
        }
        if (block_r > seq_len && block_l < seq_len + q_len) {
          for (int k = 0; k < config_.block_len; k++) {
            if (block_id * config_.block_len + k >= seq_len + q_len ||
                block_id * config_.block_len + k < seq_len)
              continue;
            for (int l = 0; l < config_.head_dim; l++) {
              k_block[k * config_.head_dim + l] =
                  k_in[size_t(batch_id) * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                          block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                          k * (config_.kv_head_num * config_.head_dim) + head_id * config_.head_dim + l];
              v_block[l * config_.block_len + k] =
                  v_in[size_t(batch_id) * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                          block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                          k * (config_.kv_head_num * config_.head_dim) + head_id * config_.head_dim + l];
            }
          }
        }
      },
      nullptr);
}

/**
 * @brief 按逻辑顺序把一层 paged KV cache 导出为连续 BF16 K/V。
 *
 * 函数按 (batch, logical block, KV head) 并行，通过 block_table 读取物理 block，并将内部转置保存的
 * V 恢复为调用方使用的 token-major 布局。超出各序列 cache_seqlens 的位置不会写入。
 * 这个函数来自原 get_kvcache_bf16()，删除了 Q4_0/Q8_0 cache 的反量化分支。
 *
 * @param k_in 接收 Key 的 BF16 输出缓冲区，布局为 [batch_size, max_tokens, kv_head_num, head_dim]。
 * @param v_in 接收 Value 的 BF16 输出缓冲区，布局与 k_in 相同。
 * @param layer_id 要导出的模型层编号。
 * @param block_table 行优先 [batch_size, max_block_num] 逻辑到物理 block 映射。
 * @param batch_size 序列数量。
 * @param max_block_num 每个序列的 block 表宽度和输出最大 block 数。
 * @param cache_seqlens 每个序列的有效 token 数，用于裁剪尾块。
 * @param backend 用于并行读取 batch、block 和 KV head 的工作线程池。
 */
void KVCache::get_kvcache_bf16(ggml_bf16_t* k_in, ggml_bf16_t* v_in, int layer_id,
                               int* block_table, int batch_size, int max_block_num,
                               int* cache_seqlens, WorkerPool* backend) {
  backend->do_work_stealing_job(
      config_.kv_head_num * max_block_num * batch_size, nullptr,
      [&](int task_id) {
        int batch_id = task_id / (config_.kv_head_num * max_block_num);
        int block_id = (task_id / config_.kv_head_num) % max_block_num;
        int head_id = task_id % config_.kv_head_num;
        int block_idx = block_table[size_t(batch_id) * max_block_num + block_id];
        int seq_len = cache_seqlens[batch_id];
        int block_l = block_id * config_.block_len;
        int block_r = block_id * config_.block_len + config_.block_len;
        const ggml_bf16_t* k_block = k_cache_bf16_[layer_id][head_id][block_idx].data();
        const ggml_bf16_t* v_block = v_cache_bf16_[layer_id][head_id][block_idx].data();
        if (block_l < seq_len) {
          for (int k = 0; k < config_.block_len; k++) {
            if (block_id * config_.block_len + k >= seq_len) break;
            for (int l = 0; l < config_.head_dim; l++) {
              k_in[size_t(batch_id) * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                      block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                      k * (config_.kv_head_num * config_.head_dim) + head_id * config_.head_dim + l] =
                  k_block[k * config_.head_dim + l];
              v_in[size_t(batch_id) * (max_block_num * config_.block_len * config_.kv_head_num * config_.head_dim) +
                      block_id * (config_.block_len * config_.kv_head_num * config_.head_dim) +
                      k * (config_.kv_head_num * config_.head_dim) + head_id * config_.head_dim + l] =
                  v_block[l * config_.block_len + k];
            }
          }
        }
      },
      nullptr);
}

/**
 * @brief 把连续 BF16 K/V token 追加写入一层 paged KV cache。
 *
 * 函数按 (batch, KV head, Query token) 并行，根据 cache_seqlens 和 token 偏移计算目标逻辑 block、
 * 物理 block 及 block 内位置；K 直接写入 token-major 布局，V 转置写入 channel-major 布局。
 * 这个函数来自原 update_kvcache_bf16()，删除了 Q4_0/Q8_0 的量化及 Value 读改写分支。
 *
 * @param k_in 已完成 GLM-4.5-Air K Norm 和 RoPE 的新增 BF16 Key，布局为
 * [batch_size, q_len, kv_head_num, head_dim]。
 * @param v_in 新增 BF16 Value，布局与 k_in 相同。
 * @param layer_id 目标模型层编号。
 * @param block_table 行优先 [batch_size, max_block_num] 逻辑到物理 block 映射。
 * @param batch_size 序列数量。
 * @param max_block_num 每个序列的 block 表宽度。
 * @param cache_seqlens 每个序列写入前的有效 token 数。
 * @param q_len 每个序列要追加的 token 数；decode 为 1，历史 KV 导入可大于 1。
 * @param backend 用于并行写入 batch、KV head 和 token 的工作线程池。
 */
void KVCache::update_kvcache_bf16(const ggml_bf16_t* k_in, const ggml_bf16_t* v_in,
                                  int layer_id, int* block_table, int batch_size,
                                  int max_block_num, int* cache_seqlens, int q_len,
                                  WorkerPool* backend) {
  if (!backend || (backend->config.subpool_thread_count.empty() || backend->config.subpool_thread_count[0] <= 0 ||
      backend->config.subpool_thread_count[0] > config_.max_thread_num) || batch_size < 0 ||
      batch_size > config_.max_batch_size || layer_id < 0 || layer_id >= config_.layer_num ||
      max_block_num < 0 || q_len < 0) throw std::invalid_argument("invalid KV update dimensions or capacity");
  if (batch_size == 0 || q_len == 0) return;
  if (!k_in || !v_in || !block_table || !cache_seqlens) throw std::invalid_argument("null KV update input");
  if (int64_t(batch_size) * config_.kv_head_num * q_len > std::numeric_limits<int>::max()) {
    throw std::overflow_error("too many KV update tasks");
  }
  for (int b = 0; b < batch_size; ++b) {
    const int64_t end = int64_t(cache_seqlens[b]) + q_len;
    if (cache_seqlens[b] < 0 || end > std::numeric_limits<int>::max() ||
        end > int64_t(max_block_num) * config_.block_len) throw std::invalid_argument("KV update exceeds block table");
    for (int block = cache_seqlens[b] / config_.block_len; block <= (end - 1) / config_.block_len; ++block) {
      const int physical = block_table[size_t(b) * max_block_num + block];
      if (physical < 0 || physical >= config_.max_block_num) throw std::invalid_argument("invalid physical KV block");
    }
  }
  backend->do_work_stealing_job(
      batch_size * config_.kv_head_num * q_len, nullptr,
      [&](int task_id) {
        int batch_id = task_id / (config_.kv_head_num * q_len);
        int head_id = task_id / q_len % config_.kv_head_num;
        int seq_len = cache_seqlens[batch_id] + task_id % q_len;
        int q_offset = task_id % q_len;
        int block_id = seq_len / config_.block_len;
        int block_idx = block_table[size_t(batch_id) * max_block_num + block_id];
        int pos_in_block = seq_len % config_.block_len;
        ggml_bf16_t* k_block = k_cache_bf16_[layer_id][head_id][block_idx].data();
        ggml_bf16_t* v_block = v_cache_bf16_[layer_id][head_id][block_idx].data();
        const size_t input_offset =
            size_t(batch_id) * q_len * config_.kv_head_num * config_.head_dim +
            q_offset * config_.kv_head_num * config_.head_dim + head_id * config_.head_dim;
        std::memcpy(k_block + pos_in_block * config_.head_dim, k_in + input_offset,
                    size_t(config_.head_dim) * sizeof(ggml_bf16_t));
        for (int l = 0; l < config_.head_dim; l++) {
          v_block[l * config_.block_len + pos_in_block] =
              v_in[input_offset + l];
        }
      },
      nullptr);
}

}  // namespace dense

