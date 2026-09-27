#include "dense_kvcache.h"

#include <stdexcept>
#include <cstdio>

#include "ggml-impl.h"

namespace dense {

/**
 * @brief 将 KV cache 支持的 GGML 数据类型转换为可读字符串。
 *
 * 该辅助函数用于构造配置日志；未识别或尚未实现的枚举值返回 "UNDIFINED"。
 * 这个函数来自原 ggml_type_to_string()，删除了当前 Dense 版本未实现的量化类型分支。
 *
 * @param type 要转换的 GGML 数据类型枚举。
 * @return 与 type 对应的固定字符串。
 */
std::string ggml_type_to_string(ggml_type type) {
  switch (type) {
    case GGML_TYPE_BF16:
      return "GGML_TYPE_BF16";
  }
  return "UNDIFINED";
}

/**
 * @brief 保存 Dense KV cache 的静态配置。
 *
 * 保存并打印配置；KVCache 构造函数在分配前统一检查形状和容量。
 * 这个函数来自原 KVCacheConfig 构造函数，删除了 anchor、retrieval 和稀疏复用步长参数。
 *
 * @param layer_num 模型层数，也是 KV cache 的层维度。
 * @param kv_head_num 每层 Key/Value head 数。
 * @param q_head_num 每层 Query head 数。
 * @param head_dim 每个 Attention head 的特征维度。
 * @param block_len 每个物理 cache block 容纳的 token 数。
 * @param kv_type 内部 K/V 存储类型；当前 KVCache 实例只接受 BF16。
 * @param max_block_num 每层最多预分配的物理 block 数。
 * @param max_batch_size 最多预分配的 batch 数。
 * @param max_thread_num 最多预分配的工作线程及线程局部缓冲区数量。
 */
KVCacheConfig::KVCacheConfig(int layer_num, int kv_head_num, int q_head_num, int head_dim,
                             int block_len, ggml_type kv_type, int max_block_num,
                             int max_batch_size, int max_thread_num)
    : layer_num(layer_num),
      kv_head_num(kv_head_num),
      q_head_num(q_head_num),
      head_dim(head_dim),
      block_len(block_len),
      kv_type(kv_type),
      max_block_num(max_block_num),
      max_batch_size(max_batch_size),
      max_thread_num(max_thread_num) {
  printf("layer_num: %d, kv_head_num: %d, q_head_num: %d, head_dim: %d, "
         "block_len: %d, kv_type: %s, max_block_num: %d, max_batch_size: %d, "
         "max_thread_num: %d\n",
         layer_num, kv_head_num, q_head_num, head_dim, block_len,
         ggml_type_to_string(kv_type).c_str(), max_block_num, max_batch_size,
         max_thread_num);
}

/**
 * @brief 按配置创建 KV cache，并预分配 BF16 Attention 所需状态。
 *
 * 构造函数计算 GQA 分组大小，创建四级 BF16 K/V 容器，再调用 ThreadResize()、BatchResize() 和
 * BlockResize() 建立全部线程、batch 和 block 维度的缓冲区。
 * 这个函数来自原 KVCache 构造函数，删除了 sparse retrieval、anchor、importance 和量化存储初始化。
 *
 * @param config 模型形状和容量；在本构造函数中统一校验。
 */
KVCache::KVCache(KVCacheConfig config) {
  // 在创建缓存前检查，包括由可写配置字段传入的值；Release 构建不能依赖 assert。
  if (config.kv_type != GGML_TYPE_BF16 || config.head_dim != 128 || config.kv_head_num != 8 ||
      config.q_head_num != 96 || config.layer_num <= 0 ||
      config.block_len <= 0 || config.block_len % 16 != 0 || config.max_block_num <= 0 ||
      config.max_batch_size <= 0 || config.max_thread_num <= 0) {
    throw std::invalid_argument(
        "Dense cache requires BF16 GLM-4.5-Air heads (96 Q, 8 KV, D=128), "
        "positive capacities and block_len divisible by 16");
  }
  config_ = config;
  n_gqa_ = config_.q_head_num / config_.kv_head_num;
  k_cache_bf16_.resize(config_.layer_num);
  v_cache_bf16_.resize(config_.layer_num);
  past_block_num_.resize(config_.layer_num, 0);
  ThreadResize(config.max_thread_num);
  BatchResize(config.max_batch_size);
  BlockResize(config.max_block_num);
}

/**
 * @brief 并不真正调整线程数量，只调整每个工作线程独享的 Attention 临时缓冲区数量和大小。
 *
 * 每个线程获得 Attention score、FP32 输出、LSE 和在线 block 归并状态，
 * 避免并行 block 计算时发生临时内存竞争。
 * 这个函数来自原 ThreadResize()，仅删除了量化 Attention 使用的 Q8 输出缓冲区。
 *
 * @param thread_num 需要支持的最大并行工作线程数；应不小于 WorkerPool 的线程数。
 */
void KVCache::ThreadResize(int thread_num) {
  if (thread_num <= 0) throw std::invalid_argument("thread_num must be positive");
  config_.max_thread_num = thread_num;
  thread_local_attn_score_.resize(thread_num);
  thread_local_output_fp32_.resize(thread_num);
  thread_local_attn_lse_.resize(thread_num);
  thread_local_cur_output_fp32_.resize(thread_num);
  thread_local_cur_attn_lse_.resize(thread_num);
  thread_local_cur_attn_sum_.resize(thread_num);
  thread_local_failed_.resize(thread_num);
  thread_cur_head_idx_.resize(thread_num);
  for (int i = 0; i < thread_num; i++) {
    thread_local_attn_score_[i].resize(n_gqa_ * config_.block_len);
    thread_local_output_fp32_[i].resize(n_gqa_ * config_.head_dim);
    thread_local_attn_lse_[i].resize(n_gqa_);
    thread_local_cur_output_fp32_[i].resize(n_gqa_ * config_.head_dim);
    thread_local_cur_attn_lse_[i].resize(n_gqa_);
    thread_local_cur_attn_sum_[i].resize(n_gqa_);
  }
}

/**
 * @brief 调整所有按 batch 索引的 Query、输出和同步状态。
 *
 * 函数为每个 (batch, KV head) 创建 mutex、FP32 输出和 LSE 缓冲区，并分配每个 batch 的有效序列长度。
 * 这个函数来自原 BatchResize()，删除了检索表、相似度、稀疏度统计和量化 Query 缓冲区。
 *
 * @param batch_size 需要预分配的最大 batch 数；运行时 batch 不得超过该值。
 */
void KVCache::BatchResize(int batch_size) {
  if (batch_size <= 0) throw std::invalid_argument("batch_size must be positive");
  config_.max_batch_size = batch_size;
  task_offsets_.resize(size_t(batch_size) + 1);
  output_valid_.resize(batch_size);
  mutex_.resize(batch_size);
  output_fp32_.resize(batch_size);
  attn_lse_.resize(batch_size);
  cache_seqlens_.resize(batch_size);
  query_fp32_.resize(size_t(batch_size) * config_.q_head_num * config_.head_dim);
  for (int i = 0; i < batch_size; i++) {
    output_valid_[i].resize(config_.kv_head_num);
    mutex_[i].resize(config_.kv_head_num);
    output_fp32_[i].resize(config_.kv_head_num);
    attn_lse_[i].resize(config_.kv_head_num);
    for (int j = 0; j < config_.kv_head_num; j++) {
      if (!mutex_[i][j]) mutex_[i][j] = std::make_unique<std::mutex>();
      output_fp32_[i][j].resize(n_gqa_ * config_.head_dim);
      attn_lse_[i][j].resize(n_gqa_);
    }
  }
}

/**
 * @brief 调整全部按物理 block 索引的 BF16 KV 缓冲区。
 *
 * 函数分配每层、每 KV head 的 BF16 K/V block。K 使用 token-major 布局，V 使用适合 PV GEMM
 * 的 [head_dim, block_len] 转置布局。K block 保存调用方预先完成 RoPE 的 Key。
 * 这个函数来自原 BlockResize()，删除了稀疏检索表、anchor、importance 和 Q4_0/Q8_0 存储分支。
 *
 * @param max_block_num 每层要支持的最大物理 block 数，通常等于 config_.max_block_num。
 */
void KVCache::BlockResize(int max_block_num) {
  if (max_block_num <= 0) throw std::invalid_argument("block_num must be positive");
  config_.max_block_num = max_block_num;
  for (int layer_id = 0; layer_id < config_.layer_num; layer_id++) {
    k_cache_bf16_[layer_id].resize(config_.kv_head_num);
    v_cache_bf16_[layer_id].resize(config_.kv_head_num);
    for (int i = 0; i < config_.kv_head_num; i++) {
      k_cache_bf16_[layer_id][i].resize(max_block_num);
      v_cache_bf16_[layer_id][i].resize(max_block_num);
      for (int j = 0; j < max_block_num; j++) {
        k_cache_bf16_[layer_id][i][j].resize(config_.block_len * config_.head_dim);
        v_cache_bf16_[layer_id][i][j].resize(config_.block_len * config_.head_dim);
      }
    }
  }
}

/**
 * @brief 清零全部层、全部 KV head 中由 block 表引用的有效 K/V block。
 *
 * 函数按 (layer, batch, logical block, KV head) 并行，将 cache_seqlens 覆盖的 BF16 K/V block
 * 全部置零。
 * 这个函数来自原 clear_kvcache_all_layers()，删除了 Q4_0/Q8_0 cache 的清零分支。
 *
 * @param block_table 行优先 [batch_size, max_block_num] 逻辑到物理 block 映射。
 * @param cache_seqlens 每个 batch 的有效 token 数，用于确定应清零的 block 范围。
 * @param batch_size 序列数量。
 * @param max_block_num 每个序列参与遍历的 block 表宽度。
 * @param backend 用于并行清零层、block 和 KV head 的工作线程池。
 */
void KVCache::clear_kvcache_all_layers(int* block_table, int* cache_seqlens,
                                       int batch_size, int max_block_num,
                                       WorkerPool* backend) {
  backend->do_work_stealing_job(
      config_.layer_num * batch_size * max_block_num * config_.kv_head_num, nullptr,
      [&](int task_id) {
        int layer_id = task_id / (batch_size * max_block_num * config_.kv_head_num);
        int batch_id = (task_id / (max_block_num * config_.kv_head_num)) % batch_size;
        int block_id = task_id / config_.kv_head_num % max_block_num;
        int head_id = task_id % config_.kv_head_num;
        const int len = cache_seqlens[batch_id];
        const int blocks = len / config_.block_len + (len % config_.block_len != 0);
        if (block_id >= blocks) return;
        int block_idx = block_table[batch_id * max_block_num + block_id];
        std::fill(k_cache_bf16_[layer_id][head_id][block_idx].begin(),
                  k_cache_bf16_[layer_id][head_id][block_idx].end(), ggml_bf16_t{0});
        std::fill(v_cache_bf16_[layer_id][head_id][block_idx].begin(),
                  v_cache_bf16_[layer_id][head_id][block_idx].end(), ggml_bf16_t{0});
      },
      nullptr);
}

/**
 * @brief 将一个连续 FP32 向量原地乘以标量。
 *
 * 实现优先使用 Accelerate 或 GGML SIMD 向量指令，并用标量循环处理 SIMD 尾部；无加速后端时退化为
 * 普通循环。Attention 的跨 block LSE 归并使用该函数重缩放局部输出。
 * 这个函数来自原 ggml_vec_scale_f32()，实现未做算法修改，仅放入 dense namespace 以避免符号冲突。
 *
 * @param n 向量 y 中要缩放的元素数量。
 * @param y 输入/输出 FP32 向量。
 * @param v 乘到每个元素上的缩放系数。
 */
void ggml_vec_scale_f32(const int n, float* y, const float v) {
#if defined(GGML_USE_ACCELERATE)
  vDSP_vsmul(y, 1, &v, y, 1, n);
#elif defined(GGML_SIMD)
  const int np = (n & ~(GGML_F32_STEP - 1));

  GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

  GGML_F32_VEC ay[GGML_F32_ARR];

  for (int i = 0; i < np; i += GGML_F32_STEP) {
    for (int j = 0; j < GGML_F32_ARR; j++) {
      ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);
      ay[j] = GGML_F32_VEC_MUL(ay[j], vx);

      GGML_F32_VEC_STORE(y + i + j * GGML_F32_EPR, ay[j]);
    }
  }

  // leftovers
  for (int i = np; i < n; ++i) {
    y[i] *= v;
  }
#else
  // scalar
  for (int i = 0; i < n; ++i) {
    y[i] *= v;
  }
#endif
}

}  // namespace dense

