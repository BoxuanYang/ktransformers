#include "dense_kvcache.h"

#include <chrono>
#include <fstream>
#include <iostream>


/*
这个文件是KVCache的SSD存储与读取，与目前工作无关，无需关注
*/

namespace dense {

/**
 * @brief 从二进制文件恢复完整的 BF16 KV cache。
 *
 * 文件开头保存有效 token 总数，随后按层、KV head 和物理 block 顺序保存有效 BF16 K/V。函数根据
 * 有效 token 数恢复每层 past_block_num_；加载时必须使用与写出时一致的模型和 cache 配置。
 * 这个函数来自原 load_kvcache()，删除了 anchor、importance 和量化 KV 数据的读取。
 *
 * @param tensor_file_path 要读取的二进制 KV cache 文件路径。
 * @param backend 预留的工作线程池参数；与原实现相同，当前采用串行文件读取且不使用该参数。
 */
void KVCache::load_kvcache(std::string tensor_file_path, WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();
  std::ifstream ifs_tensor(tensor_file_path, std::ios::binary);
  if (!ifs_tensor) throw std::runtime_error("Failed to open tensor file");
  ifs_tensor.read(reinterpret_cast<char*>(&cache_total_len_), sizeof(cache_total_len_));
  int past_block_num = (cache_total_len_ + config_.block_len - 1) / config_.block_len;
  printf("cache_total_len: %d, past_block_num: %d\n", cache_total_len_, past_block_num);
  for (int i = 0; i < config_.layer_num; ++i) past_block_num_[i] = past_block_num;
  for (int i = 0; i < config_.layer_num; ++i) {
    for (int j = 0; j < config_.kv_head_num; ++j) {
      for (int k = 0; k < past_block_num_[i]; ++k) {
        ifs_tensor.read(reinterpret_cast<char*>(k_cache_bf16_[i][j][k].data()),
                        k_cache_bf16_[i][j][k].size() * sizeof(ggml_bf16_t));
        ifs_tensor.read(reinterpret_cast<char*>(v_cache_bf16_[i][j][k].data()),
                        v_cache_bf16_[i][j][k].size() * sizeof(ggml_bf16_t));
      }
    }
  }
  ifs_tensor.close();
  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  printf("time of load: %f s\n", diff.count());
}

/**
 * @brief 按逻辑 block 顺序把 BF16 KV cache 写入二进制文件。
 *
 * 函数先写有效 token 总数，再通过 block_table 将逻辑 block 映射为物理 block，按层、KV head 和
 * block 顺序写出 K/V。输出可由 load_kvcache() 在配置完全一致的 KVCache 实例中恢复。
 * 这个函数来自原 dump_kvcache()，删除了 anchor、importance 和量化 KV 数据的写出。
 *
 * @param block_table 一维逻辑到物理 block 映射；至少包含覆盖 cache_total_len 的 block 数。
 * @param cache_total_len 要持久化的有效 token 总数，用于计算应写出的 block 数和尾块范围。
 * @param tensor_file_path 输出二进制文件路径；文件存在时会被覆盖。
 * @param backend 预留的工作线程池参数；与原实现相同，当前采用串行文件写入且不使用该参数。
 */
void KVCache::dump_kvcache(int* block_table, int cache_total_len,
                           std::string tensor_file_path, WorkerPool* backend) {
  // Timer start
  auto start = std::chrono::high_resolution_clock::now();
  std::ofstream ofs(tensor_file_path, std::ios::binary);
  printf("dump_kvcache: %s\n", tensor_file_path.c_str());
  if (!ofs.is_open()) {
    std::cerr << "Cannot open file " << tensor_file_path << std::endl;
    return;
  }
  ofs.write(reinterpret_cast<const char*>(&cache_total_len), sizeof(cache_total_len));
  int past_block_num = (cache_total_len + config_.block_len - 1) / config_.block_len;
  printf("cache_total_len: %d, past_block_num: %d\n", cache_total_len, past_block_num);
  for (int i = 0; i < config_.layer_num; ++i) {
    for (int j = 0; j < config_.kv_head_num; ++j) {
      for (int k = 0; k < past_block_num; ++k) {
        int block_idx = block_table[k];
        ofs.write(reinterpret_cast<const char*>(k_cache_bf16_[i][j][block_idx].data()),
                  k_cache_bf16_[i][j][block_idx].size() * sizeof(ggml_bf16_t));
        ofs.write(reinterpret_cast<const char*>(v_cache_bf16_[i][j][block_idx].data()),
                  v_cache_bf16_[i][j][block_idx].size() * sizeof(ggml_bf16_t));
      }
    }
  }
  ofs.close();
  // Timer end
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  printf("time of dump: %f s\n", diff.count());
}

}  // namespace dense

