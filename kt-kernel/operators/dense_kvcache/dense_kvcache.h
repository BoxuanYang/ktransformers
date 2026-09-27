/**
 * @file dense_kvcache.h
 * @brief 原 KVCache 的 BF16 Dense GQA 删节版声明。
 *
 * 本文件保持 kvcache.h 的 paged-cache 设计、数据布局、运行时缓冲区和函数命名，只删除
 * anchor、importance、block retrieval、稀疏度统计以及 INT4/INT8 存储。为避免与原全局类重名，
 * 相同的 KVCacheConfig/KVCache 类放在 dense 命名空间中。
 */
#ifndef CPUINFER_OPERATOR_DENSE_KVCACHE_H
#define CPUINFER_OPERATOR_DENSE_KVCACHE_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "../../cpu_backend/worker_pool.h"
#include "llama.cpp/ggml.h"

namespace dense {

/**
 * @brief 将 KV cache 支持的 GGML 数据类型转换为可读字符串。
 *
 * 这个函数来自原 ggml_type_to_string()，删除了当前 Dense 版本未实现的量化类型分支。
 *
 * @param type 要转换的 GGML 数据类型枚举。
 * @return 与 type 对应的固定字符串；未实现的类型返回 "UNDIFINED"。
 */
std::string ggml_type_to_string(ggml_type type);

/**
 * @struct KVCacheConfig
 * @brief Dense KV cache 的配置结构体。
 *
 * 该结构保存模型形状、BF16 KV 类型以及 block、batch 和线程容量上限。
 * KVCache 接受 GLM-4.5-Air 的 Q=96、KV=8、D=128，block_len 为正的 16 倍数。
 * 这个结构来自原 KVCacheConfig，删除了 anchor、retrieval 和稀疏复用配置成员。
 */
struct KVCacheConfig {
  int layer_num;
  int kv_head_num;
  int q_head_num;
  int head_dim;
  int block_len; // 每个物理 block 容纳的 token 数
  ggml_type kv_type;
  int max_block_num; // 每层最多预分配的物理 block 数
  int max_batch_size; // 最多预分配的 batch 数
  int max_thread_num; // 最多预分配的工作线程数

  /**
   * @brief KVCacheConfig 的默认构造函数。
   *
   * 这个函数来自原 KVCacheConfig 默认构造函数，实现保持不变；成员由调用方随后填写。
   */
  KVCacheConfig() = default;

  /**
   * @brief 使用指定值构造 Dense KV cache 配置。
   *
   * 这个函数来自原 KVCacheConfig 带参构造函数，删除了 anchor、retrieval 和稀疏复用参数。
   *
   * @param layer_num 模型层数。
   * @param kv_head_num 每层 KV head 数。
   * @param q_head_num 每层 Query head 数。
   * @param head_dim 每个 Attention head 的特征维度。
   * @param block_len 每个物理 cache block 的 token 数。
   * @param kv_type KV cache 数据类型；当前实例只实现 BF16。
   * @param max_block_num 每层最多预分配的物理 block 数。
   * @param max_batch_size 最多预分配的 batch 数。
   * @param max_thread_num 最多预分配的工作线程数。
   */
  KVCacheConfig(int layer_num, int kv_head_num, int q_head_num, int head_dim, int block_len,
                ggml_type kv_type, int max_block_num, int max_batch_size, int max_thread_num);
};

/**
 * @class KVCache
 * @brief 管理 Dense GQA Attention 使用的 BF16 paged KV cache。
 *
 * 这个类来自原 KVCache，保留 paged-cache 布局和 block-parallel Attention，只删除稀疏及量化状态。
 */
class KVCache {
 public:
  /**
   * @brief 使用给定配置构造 KVCache 对象。
   *
   * 这个函数来自原 KVCache 构造函数，删除了 sparse retrieval、anchor、importance 和量化存储初始化。
   *
   * @param config 包含模型形状以及 block、batch、线程容量的配置对象。
   */
  KVCache(KVCacheConfig config);

  // 仅在同步 attn 调用之间切换归并模式。
  // task 大小在 1/2/4/8 个连续逻辑 block 间自适应；false 使用带锁归并作对照。
  void set_parallel_reduce(bool enabled);

  /**
   * @brief 调整线程工作区容量，不改变 WorkerPool 的线程数。
   *
   * 这个函数来自原 ThreadResize()，仅删除量化 Attention 使用的线程私有缓冲区。
   *
   * @param thread_num 调整后的最大线程数。
   */
  void ThreadResize(int thread_num);


  /**
   * @brief 调整 Cache 管理的 Batch Size。
   *
   * 这个函数来自原 BatchResize()，删除了 sparse retrieval 和量化 Query 的 batch 状态。
   *
   * @param batch_size 调整后的最大 Batch Size。
   */
  void BatchResize(int batch_size);


  /**
   * @brief 调整 Cache 管理的物理 block 数量。
   *
   * 这个函数来自原 BlockResize()，只保留 BF16 K/V block 的分配。
   *
   * @param block_num 调整后的最大物理 block 数。
   */
  void BlockResize(int block_num);

  /**
   * @brief 获取 Cache 的层数。
   *
   * 这个函数来自原 get_layer_num()，实现保持不变。
   *
   * @return Cache 配置的层数。
   */
  int get_layer_num() { return config_.layer_num; }


  /**
   * @brief 获取 Cache 的 KV head 数量。
   *
   * 这个函数来自原 get_kv_head_num()，实现保持不变。
   *
   * @return Cache 配置的 KV head 数量。
   */
  int get_kv_head_num() { return config_.kv_head_num; }


  /**
   * @brief 获取 Cache 的 Query head 数量。
   *
   * 这个函数来自原 get_q_head_num()，实现保持不变。
   *
   * @return Cache 配置的 Query head 数量。
   */
  int get_q_head_num() { return config_.q_head_num; }


  /**
   * @brief 获取 Cache 中每个 head 的维度。
   *
   * 这个函数来自原 get_head_dim()，实现保持不变。
   *
   * @return 每个 Attention head 的特征维度。
   */
  int get_head_dim() { return config_.head_dim; }


  /**
   * @brief 获取 Cache 中每个 block 的长度。
   *
   * 这个函数来自原 get_block_len()，实现保持不变。
   *
   * @return 每个物理 block 容纳的 token 数。
   */
  int get_block_len() { return config_.block_len; }


  /**
   * @brief 获取指定层已记录的 block 数量。
   *
   * 这个函数来自原 get_block_num()，实现保持不变。
   *
   * @param layer_id 要查询的模型层编号。
   * @return 指定层的 past_block_num_。
   */
  int get_block_num(int layer_id) { return past_block_num_[layer_id]; }
  
  
  /**
   * @brief 获取 Cache 的总 token 长度。
   *
   * 这个函数来自原 get_cache_total_len()，实现保持不变。
   *
   * @return Cache 当前记录的总 token 数。
   */
  int get_cache_total_len() { return cache_total_len_; }

  
  /**
   * @brief 获取 Cache 的总 block 数。
   *
   * 这个函数来自原 get_cache_total_block_num()，保留按 block_len 向上取整的公式。
   *
   * @return 覆盖 cache_total_len_ 所需的 block 数。
   */
  int get_cache_total_block_num() {
    return (cache_total_len_ + config_.block_len - 1) / config_.block_len;
  }
  /**
   * @brief 更新 Cache 的总 token 长度。
   *
   * 这个函数来自原 update_cache_total_len()，实现保持不变。
   *
   * @param cache_total_len Cache 的新总长度。
   */
  void update_cache_total_len(int cache_total_len) { cache_total_len_ = cache_total_len; }

  /**
   * @brief 使用已有 paged KV cache 计算 Decode GQA Attention。
   *
   * 这个函数来自原 attn()，删除了 retrieval_type 分派、Top-K 检索和稀疏参数，改为遍历全部有效 block。
   *
   * @param q_in 已完成 GLM-4.5-Air Q Norm 和 RoPE 的 BF16 Query，布局为
   * [batch_size, q_len, q_head_num, head_dim]。
   * @param output BF16 Attention 输出，布局与 q_in 相同。
   * @param attn_lse FP32 log-sum-exp 输出，布局为 [batch_size, q_len, q_head_num]。
   * @param layer_idx 当前模型层编号。
   * @param generate_token_idx 保留的原 Decode token 编号参数；Dense 路径不用于检索复用。
   * @param q_len Query token 数；必须为 1，其他值在入口报错。
   * @param batch_size 并发序列数量。
   * @param max_block_num block_table 每行的表项数。
   * @param block_table 行优先的逻辑 block 到物理 block 完整映射。
   * @param cache_seqlens 每个序列当前有效的 KV token 数。
   * @param backend 执行 block kernel 和结果归并的工作线程池。
   */
  void attn(const ggml_bf16_t* q_in, ggml_bf16_t* output, float* attn_lse, int layer_idx,
            int generate_token_idx, int q_len, int batch_size, int max_block_num,
            int* block_table, int* cache_seqlens, WorkerPool* backend);

  /**
   * @brief 从二进制文件恢复完整的 BF16 KV cache。
   *
   * 这个函数来自原 load_kvcache()，删除了 anchor、importance 和量化 KV 数据读取。
   *
   * @param tensor_file_path 要读取的二进制文件路径。
   * @param backend 保留的工作线程池参数；当前串行读取不使用该参数。
   */
  void load_kvcache(std::string tensor_file_path, WorkerPool* backend);
  /**
   * @brief 按逻辑 block 顺序写出完整的 BF16 KV cache。
   *
   * 这个函数来自原 dump_kvcache()，删除了 anchor、importance 和量化 KV 数据写出。
   *
   * @param block_table 一维逻辑 block 到物理 block 映射。
   * @param cache_total_len 要写出的有效 token 总数。
   * @param tensor_file_path 输出二进制文件路径。
   * @param backend 保留的工作线程池参数；当前串行写入不使用该参数。
   */
  void dump_kvcache(int* block_table, int cache_total_len, std::string tensor_file_path,
                    WorkerPool* backend);

  /**
   * @brief 导出历史 BF16 K/V，并把紧随其后的新 K/V 写回 cache。
   *
   * 这个函数来自原 get_and_update_kvcache_bf16()，删除了 Q4_0/Q8_0 读写分支。
   *
   * @param k_in 输入/输出 BF16 Key，布局为 [batch_size, max_tokens, kv_head_num, head_dim]。
   * @param v_in 输入/输出 BF16 Value，布局与 k_in 相同。
   * @param layer_id 当前模型层编号。
   * @param block_table 行优先的逻辑 block 到物理 block 映射。
   * @param batch_size 序列数量。
   * @param max_block_num block_table 每行的表项数。
   * @param cache_seqlens 每个序列写入前的有效 token 数。
   * @param q_len 要追加的新 token 数；当前 Decode 调用约定为 1。
   * @param backend 并行处理 batch、block 和 KV head 的工作线程池。
   */
  void get_and_update_kvcache_bf16(ggml_bf16_t* k_in, ggml_bf16_t* v_in, int layer_id,
                                   int* block_table, int batch_size, int max_block_num,
                                   int* cache_seqlens, int q_len, WorkerPool* backend);
  /**
   * @brief 按逻辑顺序导出一层 paged BF16 K/V。
   *
   * 这个函数来自原 get_kvcache_bf16()，删除了 Q4_0/Q8_0 反量化分支。
   *
   * @param k_in 接收连续 BF16 Key 的输出缓冲区。
   * @param v_in 接收连续 BF16 Value 的输出缓冲区。
   * @param layer_id 要导出的模型层编号。
   * @param block_table 行优先的逻辑 block 到物理 block 映射。
   * @param batch_size 序列数量。
   * @param max_block_num block_table 每行的表项数。
   * @param cache_seqlens 每个序列的有效 token 数。
   * @param backend 并行读取 batch、block 和 KV head 的工作线程池。
   */
  void get_kvcache_bf16(ggml_bf16_t* k_in, ggml_bf16_t* v_in, int layer_id,
                        int* block_table, int batch_size, int max_block_num,
                        int* cache_seqlens, WorkerPool* backend);
  /**
   * @brief 把连续 BF16 K/V token 追加写入一层 paged KV cache。
   *
   * 这个函数来自原 update_kvcache_bf16()，删除了 Q4_0/Q8_0 量化写入分支。
   *
   * @param k_in 已完成 GLM-4.5-Air K Norm 和 RoPE 的新增 BF16 Key，布局为
   * [batch_size, q_len, kv_head_num, head_dim]。
   * @param v_in 新增 BF16 Value，布局与 k_in 相同。
   * @param layer_id 目标模型层编号。
   * @param block_table 行优先的逻辑 block 到物理 block 映射。
   * @param batch_size 序列数量。
   * @param max_block_num block_table 每行的表项数。
   * @param cache_seqlens 每个序列写入前的有效 token 数。
   * @param q_len 要追加的 token 数；decode 为 1，历史 KV 批量导入可大于 1。
   * @param backend 并行写入 batch、KV head 和 token 的工作线程池。
   */
  void update_kvcache_bf16(const ggml_bf16_t* k_in, const ggml_bf16_t* v_in, int layer_id,
                           int* block_table, int batch_size, int max_block_num,
                           int* cache_seqlens, int q_len, WorkerPool* backend);

  /**
   * @brief 追加当前 Decode K/V，并基于更新后的完整 cache 计算 Attention。
   *
   * 这个函数来自原 attn_with_kvcache()，删除了 topk、local、init block 和稀疏检索调用。
   *
   * @param q_in 已完成 GLM-4.5-Air Q Norm 和 RoPE 的当前 token BF16 Query。
   * @param k_in 已完成 GLM-4.5-Air K Norm 和 RoPE 的当前 token BF16 Key。
   * @param v_in 当前 token 的 BF16 Value。
   * @param output BF16 Attention 输出。
   * @param attn_lse FP32 log-sum-exp 输出。
   * @param layer_idx 当前模型层编号。
   * @param generate_token_idx 保留的原 Decode token 编号参数。
   * @param q_len Query 长度；该 Decode 接口要求为 1。
   * @param batch_size 并发序列数量。
   * @param max_block_num block_table 每行的表项数。
   * @param block_table 行优先的逻辑 block 到物理 block 映射。
   * @param cache_seqlens 输入/输出有效 token 数。
   * @param backend 用于写入 cache 和计算 Attention 的工作线程池。
   */
  void attn_with_kvcache(const ggml_bf16_t* q_in, const ggml_bf16_t* k_in,
                         const ggml_bf16_t* v_in, ggml_bf16_t* output, float* attn_lse,
                         int layer_idx, int generate_token_idx, int q_len, int batch_size,
                         int max_block_num, int* block_table, int* cache_seqlens,
                         WorkerPool* backend);

  /**
   * @brief 清零全部层中由 block 表引用的有效 BF16 K/V block。
   *
   * 这个函数来自原 clear_kvcache_all_layers()，删除了 Q4_0/Q8_0 清零分支。
   *
   * @param block_table 行优先的逻辑 block 到物理 block 映射。
   * @param cache_seqlens 每个序列的有效 token 数。
   * @param batch_size 序列数量。
   * @param max_block_num block_table 每行的表项数。
   * @param backend 并行清零层、block 和 KV head 的工作线程池。
   */
  void clear_kvcache_all_layers(int* block_table, int* cache_seqlens, int batch_size,
                                int max_block_num, WorkerPool* backend);
 private:
  static constexpr int kMaxBlocksPerTask = 8;
  static constexpr int kLseStride = 16;
  int blocks_per_task_ = kMaxBlocksPerTask;
  bool parallel_reduce_ = true;
  // 每个 task 独占一个局部归并结果槽。容量只在需要增长时调整，后续调用复用。
  std::vector<float> reduce_task_output_;
  // 每 task 预留 16 个 float（一个缓存行），避免相邻 task 写 LSE 时发生伪共享。
  std::vector<float> reduce_task_lse_;
  void reduce_one_query_head_(int task_id);

  KVCacheConfig config_;
  int n_gqa_;
  int cache_total_len_ = 0;
  std::vector<uint64_t> past_block_num_;

  // 与原 BF16 分支完全相同的四级 paged-cache 结构和布局。

  // [layer_num][kv_head_num][physical_block][block_len * head_dim]
  // Key cache 只保存已经完成位置编码（post-RoPE）的 Key。
  std::vector<std::vector<std::vector<std::vector<ggml_bf16_t>>>> k_cache_bf16_;
  
  // [layer_num][kv_head_num][physical_block][head_dim * block_len]
  std::vector<std::vector<std::vector<std::vector<ggml_bf16_t>>>> v_cache_bf16_;

  int64_t layer_id_;
  const int* block_table_;
  // 每个 sequence 在完整 block_table 中分配的 block 槽位数，不表示当前已经使用的 block 数；
  // 对应原稀疏代码库中的 max_block_num_after_retrieval_。
  int block_num_per_seq_;

  // 保持原 block-parallel Attention 所需的 batch/head、锁和线程私有状态。
  std::vector<int> cache_seqlens_;
  // [batch + 1] 自适应分块任务的前缀和；不是 block_table 的行偏移或有效 block 数。
  std::vector<int> task_offsets_;
  // 每个 batch/head 是否已有全局结果；使用字节数组，避免 vector<bool> 的位共享。
  std::vector<std::vector<uint8_t>> output_valid_;
  std::vector<uint8_t> thread_local_failed_;

  // 这里的mutex，指的是为每一个batch的每一个kv head 分配一个 mutex 
  std::vector<std::vector<std::unique_ptr<std::mutex>>> mutex_;
  std::vector<std::vector<std::vector<float>>> output_fp32_;
  std::vector<std::vector<std::vector<float>>> attn_lse_;
  std::vector<float> query_fp32_;
  std::vector<std::pair<int, int>> thread_cur_head_idx_;
  std::vector<std::vector<float>> thread_local_attn_score_;
  std::vector<std::vector<float>> thread_local_output_fp32_;
  std::vector<std::vector<float>> thread_local_attn_lse_;
  std::vector<std::vector<float>> thread_local_cur_output_fp32_;
  std::vector<std::vector<float>> thread_local_cur_attn_lse_;
  std::vector<std::vector<float>> thread_local_cur_attn_sum_;

  // 重置累计状态，并按 blocks_per_task_ 构建每个 head 的任务前缀和。
  void attn_initialize_kvhead_(int batch_size, int layer_idx, const int* block_table,
                               int block_table_stride, const int* cache_seqlens);

  // 一个任务计算同一 (batch, KV head) 的最多八个连续逻辑 block，并在本地归并。
  void attention_kvhead_(const ggml_bf16_t* q_in, ggml_bf16_t* output,
                         float* attn_lse, int batch_size, WorkerPool* backend);

  // Q [G,D], K [T,D], V [D,T] 为 BF16；有效 token 为 [0, valid_tokens)。
  // score/probability [G,T]；output [G,D]、lse [G] 为 FP32 单块结果。
  // 返回 false 表示当前 GEMM 不支持该配置，由调用线程在任务完成后报告。
  bool attn_with_kvcache_one_block_(int head_dim, int n_gqa, const void* q,
                                    int block_len, int valid_tokens,
                                    const ggml_bf16_t* k_cache, const ggml_bf16_t* v_cache,
                                    float* attn_score, float* output, float* lse);
};

/**
 * @brief 将一个连续 FP32 向量原地乘以标量。
 *
 * 这个函数来自原 ggml_vec_scale_f32()，实现不变，仅放入 dense namespace 以避免符号冲突。
 *
 * @param n 向量 y 中要缩放的元素数量。
 * @param y 输入/输出 FP32 向量。
 * @param v 乘到每个元素上的缩放系数。
 */
void ggml_vec_scale_f32(const int n, float* y, const float v);

}  // namespace dense

#endif

