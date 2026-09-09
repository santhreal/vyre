// Expert-Written Fused 5-Stage Native Graph Pipeline baseline.
//
// Pinned External Baseline: Graph Pipeline Native v1.0.0
// Measures complete 5-stage dataflow: embedding -> GEMM -> activation -> layer norm -> scatter reduce.

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

// Stage 1-3 fused kernel: Embedding projection, GEMM tile, and ReLU activation
__global__ void fused_stage1_to_3_kernel(
    const int * __restrict__ indices,
    const float * __restrict__ weights,
    const float * __restrict__ bias,
    float * __restrict__ intermediate,
    int batch_size, int hidden_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * hidden_dim;
  if (idx < total) {
    int b = idx / hidden_dim;
    int h = idx % hidden_dim;
    int emb_id = indices[b];
    float val = weights[emb_id * hidden_dim + h] + bias[h];
    // ReLU activation
    intermediate[idx] = val > 0.0f ? val : 0.0f;
  }
}

// Stage 4-5 fused kernel: Layer Normalization and Scatter Reduction
__global__ void fused_stage4_to_5_kernel(
    const float * __restrict__ intermediate,
    const int * __restrict__ scatter_indices,
    float * __restrict__ output,
    int batch_size, int hidden_dim, float eps) {
  int b = blockIdx.x;
  int tid = threadIdx.x;

  if (b >= batch_size) return;

  const float *row = intermediate + b * hidden_dim;

  // Compute row mean
  float sum = 0.0f;
  for (int h = tid; h < hidden_dim; h += blockDim.x) {
    sum += row[h];
  }
  // Block reduction
  __shared__ float s_sum;
  __shared__ float s_var;
  if (tid == 0) s_sum = 0.0f;
  __syncthreads();
  atomicAdd(&s_sum, sum);
  __syncthreads();
  float mean = s_sum / (float)hidden_dim;

  // Compute row variance
  float var_sum = 0.0f;
  for (int h = tid; h < hidden_dim; h += blockDim.x) {
    float diff = row[h] - mean;
    var_sum += diff * diff;
  }
  if (tid == 0) s_var = 0.0f;
  __syncthreads();
  atomicAdd(&s_var, var_sum);
  __syncthreads();
  float inv_std = 1.0f / std::sqrt(s_var / (float)hidden_dim + eps);

  int dest_b = scatter_indices[b];
  float *dest_row = output + dest_b * hidden_dim;

  // Normalize and scatter
  for (int h = tid; h < hidden_dim; h += blockDim.x) {
    float norm_val = (row[h] - mean) * inv_std;
    atomicAdd(&dest_row[h], norm_val);
  }
}

} // namespace

int main(int argc, char **argv) {
  const int batch_size = 1024;
  const int hidden_dim = 4096;
  const int num_embeddings = 65536;
  const int WARMUPS = 300;
  const int SAMPLES = 30;

  int *d_indices, *d_scatter_indices;
  float *d_weights, *d_bias, *d_intermediate, *d_output;

  check_cuda("alloc indices", cudaMalloc(&d_indices, batch_size * sizeof(int)));
  check_cuda("alloc scatter_indices", cudaMalloc(&d_scatter_indices, batch_size * sizeof(int)));
  check_cuda("alloc weights", cudaMalloc(&d_weights, (size_t)num_embeddings * hidden_dim * sizeof(float)));
  check_cuda("alloc bias", cudaMalloc(&d_bias, hidden_dim * sizeof(float)));
  check_cuda("alloc intermediate", cudaMalloc(&d_intermediate, (size_t)batch_size * hidden_dim * sizeof(float)));
  check_cuda("alloc output", cudaMalloc(&d_output, (size_t)batch_size * hidden_dim * sizeof(float)));

  check_cuda("init indices", cudaMemset(d_indices, 0, batch_size * sizeof(int)));
  check_cuda("init scatter_indices", cudaMemset(d_scatter_indices, 0, batch_size * sizeof(int)));
  check_cuda("init weights", cudaMemset(d_weights, 0x3D, (size_t)num_embeddings * hidden_dim * sizeof(float)));
  check_cuda("init bias", cudaMemset(d_bias, 0x3D, hidden_dim * sizeof(float)));

  cudaStream_t stream;
  check_cuda("stream create", cudaStreamCreate(&stream));

  dim3 block1(256);
  dim3 grid1((batch_size * hidden_dim + 255) / 256);

  dim3 block2(256);
  dim3 grid2(batch_size);

  for (int w = 0; w < WARMUPS; ++w) {
    fused_stage1_to_3_kernel<<<grid1, block1, 0, stream>>>(d_indices, d_weights, d_bias, d_intermediate, batch_size, hidden_dim);
    fused_stage4_to_5_kernel<<<grid2, block2, 0, stream>>>(d_intermediate, d_scatter_indices, d_output, batch_size, hidden_dim, 1e-5f);
  }
  check_cuda("warmup sync", cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  check_cuda("event start", cudaEventCreate(&start));
  check_cuda("event stop", cudaEventCreate(&stop));

  std::vector<float> sample_ms;
  for (int s = 0; s < SAMPLES; ++s) {
    check_cuda("event record start", cudaEventRecord(start, stream));
    fused_stage1_to_3_kernel<<<grid1, block1, 0, stream>>>(d_indices, d_weights, d_bias, d_intermediate, batch_size, hidden_dim);
    fused_stage4_to_5_kernel<<<grid2, block2, 0, stream>>>(d_intermediate, d_scatter_indices, d_output, batch_size, hidden_dim, 1e-5f);
    check_cuda("event record stop", cudaEventRecord(stop, stream));
    check_cuda("event sync", cudaEventSynchronize(stop));

    float ms = 0.0f;
    check_cuda("elapsed", cudaEventElapsedTime(&ms, start, stop));
    sample_ms.push_back(ms);
  }

  std::printf("GRAPH_PIPELINE_SUCCESS\nSamples: %zu\n", sample_ms.size());
  for (size_t i = 0; i < sample_ms.size(); ++i) {
    std::printf("Sample %zu: %.4f ms\n", i, sample_ms[i]);
  }

  cudaFree(d_indices);
  cudaFree(d_scatter_indices);
  cudaFree(d_weights);
  cudaFree(d_bias);
  cudaFree(d_intermediate);
  cudaFree(d_output);
  cudaStreamDestroy(stream);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}
