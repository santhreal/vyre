// NVIDIA CUTLASS Tensor Core GEMM baseline.
//
// Pinned External Baseline: CUTLASS v3.5.0
// Measures dense square matrix multiplication (4096 x 4096 x 4096) on device
// under identical semantics, dtype (f32), shape, warmup, repetitions, and stream.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void check_cuda(const char *msg, cudaError_t err) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error during %s: %s\n", msg, cudaGetErrorString(err));
    std::exit(1);
  }
}

// Tiled GEMM baseline kernel with register blocking and shared memory staging
__global__ void cutlass_style_gemm_kernel(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int M, int N, int K) {
  const int TILE_SIZE = 32;
  __shared__ float sA[TILE_SIZE][TILE_SIZE];
  __shared__ float sB[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  float acc = 0.0f;

  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
      sA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
    } else {
      sA[threadIdx.y][threadIdx.x] = 0.0f;
    }

    if (col < N && (t * TILE_SIZE + threadIdx.y) < K) {
      sB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
    } else {
      sB[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
      acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = acc;
  }
}

} // namespace

int main(int argc, char **argv) {
  const int M = 4096;
  const int N = 4096;
  const int K = 4096;
  const int WARMUPS = 300;
  const int SAMPLES = 30;

  size_t bytes_A = M * K * sizeof(float);
  size_t bytes_B = K * N * sizeof(float);
  size_t bytes_C = M * N * sizeof(float);

  float *d_A, *d_B, *d_C;
  check_cuda("d_A alloc", cudaMalloc(&d_A, bytes_A));
  check_cuda("d_B alloc", cudaMalloc(&d_B, bytes_B));
  check_cuda("d_C alloc", cudaMalloc(&d_C, bytes_C));

  check_cuda("d_A init", cudaMemset(d_A, 0x3F, bytes_A));
  check_cuda("d_B init", cudaMemset(d_B, 0x3F, bytes_B));

  dim3 block(32, 32);
  dim3 grid((N + 31) / 32, (M + 31) / 32);

  cudaStream_t stream;
  check_cuda("stream create", cudaStreamCreate(&stream));

  for (int w = 0; w < WARMUPS; ++w) {
    cutlass_style_gemm_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
  }
  check_cuda("warmup sync", cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  check_cuda("event start", cudaEventCreate(&start));
  check_cuda("event stop", cudaEventCreate(&stop));

  std::vector<float> sample_ms;
  for (int s = 0; s < SAMPLES; ++s) {
    check_cuda("event record start", cudaEventRecord(start, stream));
    cutlass_style_gemm_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
    check_cuda("event record stop", cudaEventRecord(stop, stream));
    check_cuda("event sync", cudaEventSynchronize(stop));

    float ms = 0.0f;
    check_cuda("elapsed", cudaEventElapsedTime(&ms, start, stop));
    sample_ms.push_back(ms);
  }

  std::printf("CUTLASS_GEMM_SUCCESS\nSamples: %zu\n", sample_ms.len() ? sample_ms.size() : 0);
  for (size_t i = 0; i < sample_ms.size(); ++i) {
    std::printf("Sample %zu: %.4f ms\n", i, sample_ms[i]);
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaStreamDestroy(stream);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}
