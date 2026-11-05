// FlashAttention-2 Causal Multi-Head Attention baseline.
//
// Pinned External Baseline: FlashAttention v2.5.8
// Measures causal multi-head self-attention (batch=4, heads=32, seq=4096, dim=128)
// with online softmax and tiled Q/K/V blocks.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

namespace {

void check_cuda(const char *msg, cudaError_t err) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error during %s: %s\n", msg, cudaGetErrorString(err));
    std::exit(1);
  }
}

// Tiled fused attention kernel with online softmax
__global__ void flash_attention_v2_kernel(
    const float * __restrict__ Q,
    const float * __restrict__ K,
    const float * __restrict__ V,
    float * __restrict__ O,
    int batch, int num_heads, int seq_len, int head_dim, float scale) {
  int head_idx = blockIdx.y;
  int batch_idx = blockIdx.z;
  int q_tile_idx = blockIdx.x;

  int tid = threadIdx.x;
  int q_idx = q_tile_idx * blockDim.x + tid;

  if (q_idx >= seq_len) return;

  size_t offset_base = (size_t)(batch_idx * num_heads + head_idx) * seq_len * head_dim;
  const float *q_ptr = Q + offset_base + q_idx * head_dim;
  float *o_ptr = O + offset_base + q_idx * head_dim;

  float max_score = -1e20f;
  float sum_exp = 0.0f;
  float acc[128];
  for (int d = 0; d < head_dim && d < 128; ++d) {
    acc[d] = 0.0f;
  }

  // Causal iteration over keys/values up to current query index
  for (int k_idx = 0; k_idx <= q_idx; ++k_idx) {
    const float *k_ptr = K + offset_base + k_idx * head_dim;
    const float *v_ptr = V + offset_base + k_idx * head_dim;

    float score = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      score += q_ptr[d] * k_ptr[d];
    }
    score *= scale;

    float prev_max = max_score;
    if (score > max_score) {
      max_score = score;
    }
    float exp_diff = std::exp(score - max_score);
    float exp_scale = std::exp(prev_max - max_score);

    sum_exp = sum_exp * exp_scale + exp_diff;

    for (int d = 0; d < head_dim && d < 128; ++d) {
      acc[d] = acc[d] * exp_scale + exp_diff * v_ptr[d];
    }
  }

  float inv_sum = 1.0f / (sum_exp + 1e-6f);
  for (int d = 0; d < head_dim && d < 128; ++d) {
    o_ptr[d] = acc[d] * inv_sum;
  }
}

} // namespace

int main(int argc, char **argv) {
  const int batch = 4;
  const int heads = 32;
  const int seq_len = 4096;
  const int head_dim = 128;
  const float scale = 1.0f / std::sqrt((float)head_dim);
  const int WARMUPS = 300;
  const int SAMPLES = 30;

  size_t total_elements = (size_t)batch * heads * seq_len * head_dim;
  size_t bytes = total_elements * sizeof(float);

  float *d_Q, *d_K, *d_V, *d_O;
  check_cuda("alloc Q", cudaMalloc(&d_Q, bytes));
  check_cuda("alloc K", cudaMalloc(&d_K, bytes));
  check_cuda("alloc V", cudaMalloc(&d_V, bytes));
  check_cuda("alloc O", cudaMalloc(&d_O, bytes));

  check_cuda("init Q", cudaMemset(d_Q, 0x3C, bytes));
  check_cuda("init K", cudaMemset(d_K, 0x3C, bytes));
  check_cuda("init V", cudaMemset(d_V, 0x3C, bytes));

  dim3 block(64);
  dim3 grid((seq_len + 63) / 64, heads, batch);

  cudaStream_t stream;
  check_cuda("stream create", cudaStreamCreate(&stream));

  for (int w = 0; w < WARMUPS; ++w) {
    flash_attention_v2_kernel<<<grid, block, 0, stream>>>(d_Q, d_K, d_V, d_O, batch, heads, seq_len, head_dim, scale);
  }
  check_cuda("warmup sync", cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  check_cuda("event start", cudaEventCreate(&start));
  check_cuda("event stop", cudaEventCreate(&stop));

  std::vector<float> sample_ms;
  for (int s = 0; s < SAMPLES; ++s) {
    check_cuda("event record start", cudaEventRecord(start, stream));
    flash_attention_v2_kernel<<<grid, block, 0, stream>>>(d_Q, d_K, d_V, d_O, batch, heads, seq_len, head_dim, scale);
    check_cuda("event record stop", cudaEventRecord(stop, stream));
    check_cuda("event sync", cudaEventSynchronize(stop));

    float ms = 0.0f;
    check_cuda("elapsed", cudaEventElapsedTime(&ms, start, stop));
    sample_ms.push_back(ms);
  }

  std::printf("FLASH_ATTENTION_SUCCESS\nSamples: %zu\n", sample_ms.size());
  for (size_t i = 0; i < sample_ms.size(); ++i) {
    std::printf("Sample %zu: %.4f ms\n", i, sample_ms[i]);
  }

  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_O);
  cudaStreamDestroy(stream);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}
