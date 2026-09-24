// 独立 CPU 数值回归：直接编译实际 Llamafile 内核，不需要 GPU 或模型权重。
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "llamafile/tinyblas_cpu_sgemm.inc"
#include "operators/llamafile/moe.hpp"

int test_matmul() {
  std::mt19937 rng(42);
  std::normal_distribution<float> dist;
  int failures = 0;
  for (ggml_type type : {GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K, GGML_TYPE_Q8_0}) {
    const auto traits = ggml_internal_get_type_traits(type);
    const auto input_traits = ggml_internal_get_type_traits(traits.vec_dot_type);
    for (int width : {704, 1408, 1536, 4096}) {
      if (width % ggml_blck_size(type) != 0) continue;
      for (int tokens : {1, 13}) {
        const int rows = 32, blocks = width / ggml_blck_size(type);
        const int lda = blocks + 1, ldb = blocks + 2, ldc = rows + 3;
        const size_t a_stride = lda * ggml_type_size(type);
        const size_t b_stride = ldb * ggml_type_size(traits.vec_dot_type);
        std::vector<uint8_t> a(rows * a_stride), b(tokens * b_stride);
        std::vector<float> dense_a(rows * width), dense_b(tokens * width), source(width);
        std::vector<float> output(tokens * ldc, std::numeric_limits<float>::quiet_NaN());
        for (int row = 0; row < rows; ++row) {
          for (float& v : source) v = dist(rng);
          traits.from_float_reference(source.data(), a.data() + row * a_stride, width);
          traits.to_float(a.data() + row * a_stride, dense_a.data() + row * width, width);
        }
        for (int token = 0; token < tokens; ++token) {
          for (float& v : source) v = dist(rng);
          input_traits.from_float(source.data(), b.data() + token * b_stride, width);
          if (traits.vec_dot_type == GGML_TYPE_Q8_K) {
            dequantize_row_q8_K(reinterpret_cast<const block_q8_K*>(b.data() + token * b_stride),
                                dense_b.data() + token * width, width);
          } else {
            input_traits.to_float(b.data() + token * b_stride, dense_b.data() + token * width, width);
          }
        }
        bool supported = true;
        // 检查多线程任务划分和非连续行距，输出填 NaN 可捕获漏算。
        for (int thread = 0; thread < 3; ++thread) {
          supported &= llamafile_sgemm(rows, tokens, blocks, a.data(), lda, b.data(), ldb,
                                      output.data(), ldc, thread, 3, GGML_TASK_TYPE_COMPUTE,
                                      type, traits.vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        }
        double error = 0, norm = 0;
        bool padding_ok = true;
        for (int token = 0; token < tokens; ++token) {
          for (int row = 0; row < rows; ++row) {
            double expected = 0;
            for (int col = 0; col < width; ++col) {
              expected += double(dense_a[row * width + col]) * dense_b[token * width + col];
            }
            error += std::pow(output[token * ldc + row] - expected, 2);
            norm += expected * expected;
          }
          for (int row = rows; row < ldc; ++row) padding_ok &= std::isnan(output[token * ldc + row]);
        }
        const double relative_error = std::sqrt(error / norm);
        bool pass = supported && padding_ok && relative_error < 2e-5;
        std::printf("%s K=%d tokens=%d supported=%d relative_error=%.3g %s\n",
                    ggml_type_name(type), width, tokens, supported, relative_error, pass ? "PASS" : "FAIL");
        failures += !pass;
      }
    }
  }
  return failures;
}

int test_moe(ggml_type gate_type, ggml_type down_type, int width) {
  constexpr int hidden = 512, experts = 4, topk = 2, max_tokens = 13;
  std::mt19937 rng(123);
  std::normal_distribution<float> dist(0, 0.05f);
  // 使用同一份量化权重计算参考值，避免把权重量化误差误判为内核错误。
  auto quantize = [&](ggml_type type) {
    std::vector<float> values(experts * hidden * width);
    for (float& value : values) value = dist(rng);
    std::vector<uint8_t> packed(values.size() * ggml_type_size(type) / ggml_blck_size(type));
    ggml_internal_get_type_traits(type).from_float_reference(values.data(), packed.data(), values.size());
    return packed;
  };
  auto gate = quantize(gate_type), up = quantize(gate_type), down = quantize(down_type);
  std::vector<ggml_bf16_t> input(max_tokens * hidden), output(input.size());
  for (auto& value : input) value = ggml_fp32_to_bf16(20 * dist(rng));
  std::vector<int64_t> ids(max_tokens * topk);
  std::vector<float> weights(ids.size());
  for (int t = 0; t < max_tokens; ++t) {
    ids[t * topk] = t % experts;
    ids[t * topk + 1] = (t + 1) % experts;
    weights[t * topk] = 0.3f;
    weights[t * topk + 1] = 0.7f;
  }
  auto gt = ggml_internal_get_type_traits(gate_type);
  auto dt = ggml_internal_get_type_traits(down_type);
  const size_t gate_row = hidden * ggml_type_size(gate_type) / ggml_blck_size(gate_type);
  const size_t down_row = width * ggml_type_size(down_type) / ggml_blck_size(down_type);
  std::vector<uint8_t> qinput(hidden * ggml_type_size(gt.vec_dot_type) / ggml_blck_size(gt.vec_dot_type));
  std::vector<uint8_t> qmid(width * ggml_type_size(dt.vec_dot_type) / ggml_blck_size(dt.vec_dot_type));
  std::vector<float> x(hidden), mid(width), expected(input.size(), 0);
  for (int t = 0; t < max_tokens; ++t) {
    for (int h = 0; h < hidden; ++h) x[h] = ggml_bf16_to_fp32(input[t * hidden + h]);
    from_float(x.data(), qinput.data(), hidden, gt.vec_dot_type);
    for (int k = 0; k < topk; ++k) {
      int e = ids[t * topk + k];
      for (int row = 0; row < width; ++row) {
        float g, u;
        gt.vec_dot(hidden, &g, 0, gate.data() + (e * width + row) * gate_row, 0, qinput.data(), 0, 1);
        gt.vec_dot(hidden, &u, 0, up.data() + (e * width + row) * gate_row, 0, qinput.data(), 0, 1);
        mid[row] = g / (1 + std::exp(-g)) * u;
      }
      from_float(mid.data(), qmid.data(), width, dt.vec_dot_type);
      for (int h = 0; h < hidden; ++h) {
        float value;
        dt.vec_dot(width, &value, 0, down.data() + (e * hidden + h) * down_row, 0, qmid.data(), 0, 1);
        expected[t * hidden + h] += weights[t * topk + k] * value;
      }
    }
  }
  int failures = 0;
  for (int pools : {1, 2}) {
    WorkerPool pool(WorkerPoolConfig{pools, std::vector<int>(pools, 0), std::vector<int>(pools, 2)});
    GeneralMOEConfig config(experts, topk, hidden, width);
    config.pool = &pool;
    config.gate_proj = gate.data();
    config.up_proj = up.data();
    config.down_proj = down.data();
    config.gate_type = config.up_type = gate_type;
    config.down_type = down_type;
    config.hidden_type = GGML_TYPE_BF16;
    config.m_block = 32;
    config.group_min_len = 2;
    config.group_max_len = max_tokens;
    TP_MOE<LLAMA_MOE_TP> moe(config);
    moe.load_weights();
    // 交替执行 decode 和 prefill，检查共享缓冲区复用及双线程池切分。
    for (int tokens : {1, max_tokens, 1}) {
      moe.forward(tokens, topk, ids.data(), weights.data(), input.data(), output.data());
      double error = 0, norm = 0;
      for (int i = 0; i < tokens * hidden; ++i) {
        error += std::pow(ggml_bf16_to_fp32(output[i]) - expected[i], 2);
        norm += double(expected[i]) * expected[i];
      }
      double relative_error = std::sqrt(error / norm);
      bool pass = relative_error < 0.005;
      std::printf("MoE %s/%s intermediate=%d pools=%d tokens=%d relative_error=%.3g %s\n",
                  ggml_type_name(gate_type), ggml_type_name(down_type), width, pools, tokens,
                  relative_error, pass ? "PASS" : "FAIL");
      failures += !pass;
    }
  }
  return failures;
}

int main() {
  ggml_context* ctx = ggml_init({0, nullptr, true});
  ggml_free(ctx);
  int failures = test_matmul();
  failures += test_moe(GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, 1408);
  failures += test_moe(GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS, 1536);
  failures += test_moe(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 1408);
  failures += test_moe(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 1536);
  return failures != 0;
}
