#include "operators/dense_kvcache/dense_kvcache.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "ggml-impl.h"

namespace {

constexpr int kQueryHeads = 96;
constexpr int kKvHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kGqa = kQueryHeads / kKvHeads;
constexpr int kBlockLen = 32;
constexpr int kBatch = 3;
constexpr int kTableStride = 10;
constexpr int kTokens = kBlockLen * kTableStride;

using BFloat16 = ggml_bf16_t;

BFloat16 bf16(float value) { return GGML_FP32_TO_BF16(value); }
float fp32(BFloat16 value) { return GGML_BF16_TO_FP32(value); }

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

size_t kv_offset(int batch, int token, int head, int dim) {
  return ((size_t(batch) * kTokens + token) * kKvHeads + head) * kHeadDim + dim;
}

size_t q_offset(int batch, int head, int dim) {
  return (size_t(batch) * kQueryHeads + head) * kHeadDim + dim;
}

void check_reference(const std::vector<BFloat16>& keys,
                     const std::vector<BFloat16>& values,
                     const std::vector<BFloat16>& queries,
                     const std::vector<BFloat16>& output,
                     const std::vector<float>& lse,
                     const std::vector<int>& lengths) {
  for (int batch = 0; batch < kBatch; ++batch) {
    for (int query_head = 0; query_head < kQueryHeads; ++query_head) {
      const int kv_head = query_head / kGqa;
      std::vector<double> scores(lengths[batch]);

      for (int token = 0; token < lengths[batch]; ++token) {
        double dot = 0.0;
        for (int dim = 0; dim < kHeadDim; ++dim) {
          dot += double(fp32(queries[q_offset(batch, query_head, dim)])) *
                 fp32(keys[kv_offset(batch, token, kv_head, dim)]);
        }
        scores[token] = dot / std::sqrt(double(kHeadDim));
      }

      const double max_score = *std::max_element(scores.begin(), scores.end());
      double sum = 0.0;
      for (double& score : scores) {
        score = std::exp(score - max_score);
        sum += score;
      }

      const size_t head_offset = size_t(batch) * kQueryHeads + query_head;
      require(std::abs(lse[head_offset] - (max_score + std::log(sum))) < 0.02,
              "LSE differs from the FP64 reference");

      for (int dim = 0; dim < kHeadDim; ++dim) {
        double expected = 0.0;
        for (int token = 0; token < lengths[batch]; ++token) {
          expected += scores[token] / sum * fp32(values[kv_offset(batch, token, kv_head, dim)]);
        }
        const float actual = fp32(output[q_offset(batch, query_head, dim)]);
        require(std::isfinite(actual) && std::abs(actual - expected) < 0.02,
                "output differs from the FP64 reference");
      }
    }
  }
}

WorkerPoolConfig worker_config(int thread_count, int pool_count) {
  require(pool_count > 0 && pool_count <= thread_count,
          "pool count must be between 1 and the thread count");
  WorkerPoolConfig config;
  config.subpool_count = pool_count;
  config.subpool_numa_map.resize(pool_count);
  config.subpool_thread_count.resize(pool_count);
  for (int pool = 0; pool < pool_count; ++pool) {
    config.subpool_numa_map[pool] = pool;
    config.subpool_thread_count[pool] =
        thread_count / pool_count + (pool < thread_count % pool_count);
  }
  return config;
}

void run(int thread_count, int pool_count) {
  const std::vector<int> lengths = {1, 129, 289};
  const int physical_blocks = 16;
  WorkerPool pool(worker_config(thread_count, pool_count));
  dense::KVCache cache(dense::KVCacheConfig(
      1, kKvHeads, kQueryHeads, kHeadDim, kBlockLen, GGML_TYPE_BF16,
      physical_blocks, kBatch, thread_count));

  std::vector<int> block_table(kBatch * kTableStride, -1);
  int next_block = 0;
  for (int batch = 0; batch < kBatch; ++batch) {
    const int block_count = (lengths[batch] + kBlockLen - 1) / kBlockLen;
    for (int block = 0; block < block_count; ++block) {
      block_table[batch * kTableStride + block] = next_block++;
    }
  }

  std::mt19937 random(45);
  std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
  std::vector<BFloat16> keys(size_t(kBatch) * kTokens * kKvHeads * kHeadDim);
  std::vector<BFloat16> values(keys.size());
  std::vector<BFloat16> queries(size_t(kBatch) * kQueryHeads * kHeadDim);
  std::vector<BFloat16> output(queries.size());
  std::vector<float> lse(size_t(kBatch) * kQueryHeads);
  for (BFloat16& value : keys) value = bf16(distribution(random));
  for (BFloat16& value : values) value = bf16(distribution(random));
  for (BFloat16& value : queries) value = bf16(distribution(random));

  for (int batch = 0; batch < kBatch; ++batch) {
    int empty_length = 0;
    cache.update_kvcache_bf16(
        keys.data() + kv_offset(batch, 0, 0, 0),
        values.data() + kv_offset(batch, 0, 0, 0), 0,
        block_table.data() + batch * kTableStride, 1, kTableStride,
        &empty_length, lengths[batch], &pool);
  }

  std::vector<int> attention_lengths = lengths;
  cache.attn(queries.data(), output.data(), lse.data(), 0, 0, 1, kBatch,
             kTableStride, block_table.data(), attention_lengths.data(), &pool);
  check_reference(keys, values, queries, output, lse, lengths);

  // Exercise stable softmax and cross-task reduction with very large score gaps.
  std::fill(queries.begin(), queries.end(), bf16(0.0f));
  for (int batch = 0; batch < kBatch; ++batch) {
    for (int head = 0; head < kQueryHeads; ++head) {
      queries[q_offset(batch, head, 0)] = bf16(1.0f);
    }
    for (int token = 0; token < lengths[batch]; ++token) {
      const float score = (token / kBlockLen) % 2 == 0 ? -200.0f : 200.0f;
      for (int head = 0; head < kKvHeads; ++head) {
        for (int dim = 0; dim < kHeadDim; ++dim) {
          keys[kv_offset(batch, token, head, dim)] =
              bf16(dim == 0 ? score * std::sqrt(float(kHeadDim)) : 0.0f);
          values[kv_offset(batch, token, head, dim)] =
              bf16(float((token + dim) % 7) / 7.0f);
        }
      }
    }
    int empty_length = 0;
    cache.update_kvcache_bf16(
        keys.data() + kv_offset(batch, 0, 0, 0),
        values.data() + kv_offset(batch, 0, 0, 0), 0,
        block_table.data() + batch * kTableStride, 1, kTableStride,
        &empty_length, lengths[batch], &pool);
  }
  attention_lengths = lengths;
  cache.attn(queries.data(), output.data(), lse.data(), 0, 0, 1, kBatch,
             kTableStride, block_table.data(), attention_lengths.data(), &pool);
  check_reference(keys, values, queries, output, lse, lengths);
  std::cout << "PASS: GLM-4.5-Air BF16 GQA=12, ragged paged KV, threads="
            << thread_count << " pools=" << pool_count << '\n';
}

void benchmark(int thread_count, int pool_count) {
  constexpr int batch_size = 5;
  constexpr int sequence_length = 2048;
  constexpr int block_len = 128;
  constexpr int blocks_per_sequence = sequence_length / block_len;
  constexpr int physical_blocks = batch_size * blocks_per_sequence;

  WorkerPool pool(worker_config(thread_count, pool_count));
  dense::KVCache cache(dense::KVCacheConfig(
      1, kKvHeads, kQueryHeads, kHeadDim, block_len, GGML_TYPE_BF16,
      physical_blocks, batch_size, thread_count));

  std::vector<int> block_table(physical_blocks);
  for (int block = 0; block < physical_blocks; ++block) block_table[block] = block;
  std::vector<int> lengths(batch_size, sequence_length);
  std::vector<BFloat16> keys(size_t(batch_size) * sequence_length * kKvHeads * kHeadDim);
  std::vector<BFloat16> values(keys.size());
  std::vector<BFloat16> queries(size_t(batch_size) * kQueryHeads * kHeadDim);
  std::vector<BFloat16> output(queries.size());
  std::vector<float> lse(size_t(batch_size) * kQueryHeads);

  std::mt19937 random(45);
  std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
  for (BFloat16& value : keys) value = bf16(distribution(random));
  for (BFloat16& value : values) value = bf16(distribution(random));
  for (BFloat16& value : queries) value = bf16(distribution(random));

  const size_t sequence_elements = size_t(sequence_length) * kKvHeads * kHeadDim;
  for (int batch = 0; batch < batch_size; ++batch) {
    int empty_length = 0;
    cache.update_kvcache_bf16(
        keys.data() + size_t(batch) * sequence_elements,
        values.data() + size_t(batch) * sequence_elements, 0,
        block_table.data() + batch * blocks_per_sequence, 1, blocks_per_sequence,
        &empty_length, sequence_length, &pool);
  }

  auto attention = [&] {
    cache.attn(queries.data(), output.data(), lse.data(), 0, 0, 1, batch_size,
               blocks_per_sequence, block_table.data(), lengths.data(), &pool);
  };
  for (int iteration = 0; iteration < 8; ++iteration) attention();

  std::vector<double> samples;
  for (int iteration = 0; iteration < 31; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    attention();
    samples.push_back(std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - start)
                          .count());
  }
  std::sort(samples.begin(), samples.end());
  std::cout << "BENCH: batch=" << batch_size << " context=" << sequence_length
            << " threads=" << thread_count << " pools=" << pool_count
            << " median_us=" << samples[15]
            << " p95_us=" << samples[29] << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const bool run_benchmark = argc >= 2 && std::string(argv[1]) == "--bench";
    const int threads = run_benchmark ? (argc >= 3 ? std::stoi(argv[2]) : 4)
                                      : (argc == 2 ? std::stoi(argv[1]) : 4);
    const int pools = run_benchmark ? (argc >= 4 ? std::stoi(argv[3]) : 1)
                                    : (argc >= 3 ? std::stoi(argv[2]) : 1);
    require(threads > 0, "thread count must be positive");
    if (run_benchmark) benchmark(threads, pools);
    else run(threads, pools);
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
