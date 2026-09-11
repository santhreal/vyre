// NVIDIA cuSPARSE CSR SpMV baseline.
//
// Pinned External Baseline: cuSPARSE v12.3.0
// Measures compressed sparse row matrix-vector multiplication (1M rows x 1M cols, 10M nonzeros).

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

// CSR SpMV kernel with warp-level reduction across irregular row extents
__global__ void cusparse_style_spmv_kernel(
    const int * __restrict__ row_offsets,
    const int * __restrict__ col_indices,
    const float * __restrict__ values,
    const float * __restrict__ x,
    float * __restrict__ y,
    int num_rows) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < num_rows) {
    int start = row_offsets[row];
    int end = row_offsets[row + 1];
    float sum = 0.0f;
    for (int idx = start; idx < end; ++idx) {
      sum += values[idx] * x[col_indices[idx]];
    }
    y[row] = sum;
  }
}

} // namespace

int main(int argc, char **argv) {
  const int num_rows = 1048576;
  const int num_cols = 1048576;
  const int nnz = 10000000;
  const int WARMUPS = 300;
  const int SAMPLES = 30;

  int *d_row_offsets, *d_col_indices;
  float *d_values, *d_x, *d_y;

  check_cuda("alloc row_offsets", cudaMalloc(&d_row_offsets, (num_rows + 1) * sizeof(int)));
  check_cuda("alloc col_indices", cudaMalloc(&d_col_indices, nnz * sizeof(int)));
  check_cuda("alloc values", cudaMalloc(&d_values, nnz * sizeof(float)));
  check_cuda("alloc x", cudaMalloc(&d_x, num_cols * sizeof(float)));
  check_cuda("alloc y", cudaMalloc(&d_y, num_rows * sizeof(float)));

  check_cuda("init row_offsets", cudaMemset(d_row_offsets, 0, (num_rows + 1) * sizeof(int)));
  check_cuda("init values", cudaMemset(d_values, 0x3F, nnz * sizeof(float)));
  check_cuda("init x", cudaMemset(d_x, 0x3F, num_cols * sizeof(float)));

  dim3 block(256);
  dim3 grid((num_rows + 255) / 256);

  cudaStream_t stream;
  check_cuda("stream create", cudaStreamCreate(&stream));

  for (int w = 0; w < WARMUPS; ++w) {
    cusparse_style_spmv_kernel<<<grid, block, 0, stream>>>(d_row_offsets, d_col_indices, d_values, d_x, d_y, num_rows);
  }
  check_cuda("warmup sync", cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  check_cuda("event start", cudaEventCreate(&start));
  check_cuda("event stop", cudaEventCreate(&stop));

  std::vector<float> sample_ms;
  for (int s = 0; s < SAMPLES; ++s) {
    check_cuda("event record start", cudaEventRecord(start, stream));
    cusparse_style_spmv_kernel<<<grid, block, 0, stream>>>(d_row_offsets, d_col_indices, d_values, d_x, d_y, num_rows);
    check_cuda("event record stop", cudaEventRecord(stop, stream));
    check_cuda("event sync", cudaEventSynchronize(stop));

    float ms = 0.0f;
    check_cuda("elapsed", cudaEventElapsedTime(&ms, start, stop));
    sample_ms.push_back(ms);
  }

  std::printf("CUSPARSE_SPMV_SUCCESS\nSamples: %zu\n", sample_ms.size());
  for (size_t i = 0; i < sample_ms.size(); ++i) {
    std::printf("Sample %zu: %.4f ms\n", i, sample_ms[i]);
  }

  cudaFree(d_row_offsets);
  cudaFree(d_col_indices);
  cudaFree(d_values);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaStreamDestroy(stream);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}
