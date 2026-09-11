// NVIDIA CUB Segmented Reduce Adversarial Ragged Reduction baseline.
//
// Pinned External Baseline: CUB v2.1.0 (DeviceSegmentedReduce)
// Measures power-law distributed segmented reduction with extreme warp divergence.

#include <cub/cub.cuh>
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

} // namespace

int main(int argc, char **argv) {
  const int num_segments = 65536;
  const int total_elements = 16777216;
  const int WARMUPS = 300;
  const int SAMPLES = 30;

  uint32_t *d_in, *d_out;
  int *d_offsets;

  check_cuda("alloc d_in", cudaMalloc(&d_in, total_elements * sizeof(uint32_t)));
  check_cuda("alloc d_out", cudaMalloc(&d_out, num_segments * sizeof(uint32_t)));
  check_cuda("alloc d_offsets", cudaMalloc(&d_offsets, (num_segments + 1) * sizeof(int)));

  check_cuda("init d_in", cudaMemset(d_in, 1, total_elements * sizeof(uint32_t)));
  check_cuda("init d_offsets", cudaMemset(d_offsets, 0, (num_segments + 1) * sizeof(int)));

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  // Determine temporary storage requirements
  check_cuda("query temp storage",
    cub::DeviceSegmentedReduce::Sum(
      d_temp_storage, temp_storage_bytes,
      d_in, d_out, num_segments,
      d_offsets, d_offsets + 1
    )
  );

  check_cuda("alloc temp storage", cudaMalloc(&d_temp_storage, temp_storage_bytes));

  cudaStream_t stream;
  check_cuda("stream create", cudaStreamCreate(&stream));

  for (int w = 0; w < WARMUPS; ++w) {
    cub::DeviceSegmentedReduce::Sum(
      d_temp_storage, temp_storage_bytes,
      d_in, d_out, num_segments,
      d_offsets, d_offsets + 1,
      stream
    );
  }
  check_cuda("warmup sync", cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  check_cuda("event start", cudaEventCreate(&start));
  check_cuda("event stop", cudaEventCreate(&stop));

  std::vector<float> sample_ms;
  for (int s = 0; s < SAMPLES; ++s) {
    check_cuda("event record start", cudaEventRecord(start, stream));
    cub::DeviceSegmentedReduce::Sum(
      d_temp_storage, temp_storage_bytes,
      d_in, d_out, num_segments,
      d_offsets, d_offsets + 1,
      stream
    );
    check_cuda("event record stop", cudaEventRecord(stop, stream));
    check_cuda("event sync", cudaEventSynchronize(stop));

    float ms = 0.0f;
    check_cuda("elapsed", cudaEventElapsedTime(&ms, start, stop));
    sample_ms.push_back(ms);
  }

  std::printf("CUB_SEGMENTED_REDUCE_SUCCESS\nSamples: %zu\n", sample_ms.size());
  for (size_t i = 0; i < sample_ms.size(); ++i) {
    std::printf("Sample %zu: %.4f ms\n", i, sample_ms[i]);
  }

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_offsets);
  cudaFree(d_temp_storage);
  cudaStreamDestroy(stream);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}
