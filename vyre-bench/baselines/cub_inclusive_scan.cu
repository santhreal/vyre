// CUB device-wide inclusive scan, measured the way the vyre case measures its
// own scan: the same element count, the same warmup and sample counts, device
// time from CUDA events, and a checksum so the two are compared for agreement
// before they are compared for speed.
//
// This is compiled by `vyre-bench` at measurement time rather than by a build
// script. CUB is header-only but needs nvcc, and a build script that needs nvcc
// makes every CPU-only build of this workspace need a CUDA toolkit for a
// benchmark it will never run.
//
// Temporary storage is allocated once, outside the timed region, because
// `DeviceScan` sizes it from a query call and charging that allocation to CUB
// would measure the allocator rather than the scan.

#include <cub/cub.cuh>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void fail(const char *what, cudaError_t status) {
  std::fprintf(stderr, "Fix: %s failed: %s\n", what, cudaGetErrorString(status));
  std::exit(1);
}

void check(const char *what, cudaError_t status) {
  if (status != cudaSuccess) {
    fail(what, status);
  }
}

// The same deterministic input the Rust side builds, so the checksums are
// comparable. A random fill would make the two agree only by luck.
std::uint32_t element(std::uint32_t index) { return (index % 7u) + 1u; }

} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::fprintf(stderr, "Fix: usage: %s <elements> <warmups> <samples>\n", argv[0]);
    return 1;
  }
  const std::size_t elements = std::strtoull(argv[1], nullptr, 10);
  const std::size_t warmups = std::strtoull(argv[2], nullptr, 10);
  const std::size_t samples = std::strtoull(argv[3], nullptr, 10);
  if (elements == 0 || samples == 0) {
    std::fprintf(stderr, "Fix: elements and samples must both be positive\n");
    return 1;
  }

  std::vector<std::uint32_t> host(elements);
  for (std::size_t index = 0; index < elements; ++index) {
    host[index] = element(static_cast<std::uint32_t>(index));
  }

  std::uint32_t *input = nullptr;
  std::uint32_t *output = nullptr;
  check("cudaMalloc(input)", cudaMalloc(&input, elements * sizeof(std::uint32_t)));
  check("cudaMalloc(output)", cudaMalloc(&output, elements * sizeof(std::uint32_t)));
  check("cudaMemcpy(input)",
        cudaMemcpy(input, host.data(), elements * sizeof(std::uint32_t),
                   cudaMemcpyHostToDevice));

  void *scratch = nullptr;
  std::size_t scratch_bytes = 0;
  check("DeviceScan::InclusiveSum(size query)",
        cub::DeviceScan::InclusiveSum(scratch, scratch_bytes, input, output,
                                      static_cast<int>(elements)));
  check("cudaMalloc(scratch)", cudaMalloc(&scratch, scratch_bytes));

  cudaEvent_t start;
  cudaEvent_t stop;
  check("cudaEventCreate(start)", cudaEventCreate(&start));
  check("cudaEventCreate(stop)", cudaEventCreate(&stop));

  for (std::size_t warmup = 0; warmup < warmups; ++warmup) {
    check("DeviceScan::InclusiveSum(warmup)",
          cub::DeviceScan::InclusiveSum(scratch, scratch_bytes, input, output,
                                        static_cast<int>(elements)));
  }
  check("cudaDeviceSynchronize(warmup)", cudaDeviceSynchronize());

  std::vector<double> measured(samples);
  for (std::size_t sample = 0; sample < samples; ++sample) {
    check("cudaEventRecord(start)", cudaEventRecord(start));
    check("DeviceScan::InclusiveSum(sample)",
          cub::DeviceScan::InclusiveSum(scratch, scratch_bytes, input, output,
                                        static_cast<int>(elements)));
    check("cudaEventRecord(stop)", cudaEventRecord(stop));
    check("cudaEventSynchronize", cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    check("cudaEventElapsedTime", cudaEventElapsedTime(&milliseconds, start, stop));
    measured[sample] = static_cast<double>(milliseconds);
  }

  std::vector<std::uint32_t> result(elements);
  check("cudaMemcpy(output)",
        cudaMemcpy(result.data(), output, elements * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost));

  // Wrapping u32 sum of the whole scanned buffer. The Rust side computes the
  // same value over its own device result, so a scan that is fast and wrong
  // fails the comparison instead of winning it.
  std::uint32_t checksum = 0;
  for (std::size_t index = 0; index < elements; ++index) {
    checksum += result[index];
  }

  int device = 0;
  check("cudaGetDevice", cudaGetDevice(&device));
  cudaDeviceProp properties{};
  check("cudaGetDeviceProperties", cudaGetDeviceProperties(&properties, device));

  std::printf("{\"elements\":%zu,\"checksum\":%u,\"device\":\"%s\",", elements,
              checksum, properties.name);
  std::printf("\"compute_capability\":\"%d.%d\",\"cub_version\":%d,", properties.major,
              properties.minor, CUB_VERSION);
  std::printf("\"samples_ms\":[");
  for (std::size_t sample = 0; sample < samples; ++sample) {
    std::printf("%s%.9f", sample == 0 ? "" : ",", measured[sample]);
  }
  std::printf("]}\n");

  check("cudaFree(scratch)", cudaFree(scratch));
  check("cudaFree(output)", cudaFree(output));
  check("cudaFree(input)", cudaFree(input));
  check("cudaEventDestroy(start)", cudaEventDestroy(start));
  check("cudaEventDestroy(stop)", cudaEventDestroy(stop));
  return 0;
}
