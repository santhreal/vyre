# CUDA release path proof

This artifact backs `cuda-first-path`.

Evidence sources:

Required generated evidence:

- `release/evidence/backends/backend-matrix.json`
- `release/evidence/benchmarks/cuda-release-suite.json`
- `release/evidence/benchmarks/cuda-ptx-patterns.json`
- `release/evidence/benchmarks/bench-release-axes.json`

Release contract:

- CUDA must dispatch, acquire successfully, and have higher release precedence than WGPU.
- `nvidia-smi -L` must report a GPU, and the GPU probe must record NVIDIA driver and CUDA runtime versions.
- Every CUDA benchmark artifact must include benchmark-local `environment.gpu_devices`, `environment.nvidia_driver_version`, and `environment.nvidia_cuda_version` provenance captured from `nvidia-smi`, so benchmark numbers remain tied to a concrete GPU model and driver/runtime pair.
- CUDA evidence must include feature/runtime markers covering PTX lowering, PTX source caching before module load, vector load/store fusion, fused vector-load gap scheduling, compute load-use gap scheduling, resident dispatch, resident input/output buffers with sparse readback batching, CUDA graphs, module cache, PTX target probing, and paired megakernel speculation.
- The PTX source cache marker must expose `CudaPtxSourceCacheSnapshot` counters for entries, hits, and misses through the public CUDA backend surface, and must prove a bounded `PTX_SOURCE_CACHE_SOFT_CAP` with eviction; release evidence must prove repeated dispatch can be measured for source-cache reuse before module-cache lookup without an unbounded source-text cache.
- CUDA PTX pattern evidence must include positive predication, vector load/store, cp.async, ldmatrix-capable target, tensor-core, scheduling, predicated-store, emitted cp.async, emitted MMA, and emitted-byte metrics.
- `cuda-ptx-patterns.json` must prove `ptx_corpus_kernels >= 8` and `ptx_branch_labels == 0` so the release proof covers the full fast-path corpus and predication avoids branch/reconvergence labels.
- `cuda-ptx-patterns.json` must prove vector load/store fusion in emitted PTX: positive p50 `ptx_vectorized_loads_emitted` and `ptx_vectorized_stores_emitted`, plus zero p50 `ptx_vector_kernel_scalar_loads`, `ptx_vector_kernel_scalar_stores`, and `ptx_vector_kernel_scalar_index_adds`.
- WGPU fallback evidence must include persistent dispatch, megakernel dispatch, sparse readback rings, async dispatch prefetch, dispatch scratch reuse, paired megakernel speculation, disk cache, and no-hidden-CPU-fallback contracts.
- The release gate requires the named CUDA and WGPU marker ids, not only a minimum marker count.
- CUDA and WGPU suite artifacts must prove zero benchmark failures, selected-backend parity, at least 30 `wall_ns` and `baseline_wall_ns` samples per case, positive `p50`/`p95`/`p99` latency percentiles for metrics used as speed proof, and family/case provenance in `artifact_statuses`.
- `backend-matrix.json` must prove `preferred_backend_gpu_only = true` and `preferred_backend_id` is `cuda` or `wgpu`; `cpu-ref`/`reference` may exist only as explicit conformance oracles, never as implicit runtime fallback.
- Production backend source must not contain hidden CPU fallback language.
