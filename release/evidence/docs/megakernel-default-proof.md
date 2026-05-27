# Megakernel default proof

This artifact backs `megakernel-default`.

Evidence sources:

Required generated evidence:

- `release/evidence/backends/backend-matrix.json`
- `release/evidence/benchmarks/release-workload-matrix.json`
- `release/evidence/benchmarks/workload-10-megakernel-queued-batches.json`
- `release/evidence/benchmarks/megakernel-condition-cuda.json`
- `release/evidence/benchmarks/megakernel-latency-cuda.json`
- `release/evidence/benchmarks/megakernel-condition-100x-proof.json`

Release contract:

- Megakernel condition evaluation must be represented in the release workload matrix, including the active contract case id `release.megakernel_queue.1m`.
- `release-workload-matrix.json` must record a non-empty `dispatch_policy` for every required workload; workload `megakernel-queued-batches` must use `megakernel`, and every non-megakernel required workload must carry a concrete architectural or measured justification.
- CUDA must be the release path for megakernel benchmark proof.
- The release matrix must include a reproducible `cargo_full` CUDA command for the megakernel workload and must bind it to CUDA as the release backend.
- `backend-matrix.json` must prove `megakernel-paired-speculation` in both CUDA and WGPU feature markers, with no missing implementation tokens or unresolved scaffold markers.
- `megakernel-condition-cuda.json` must report `selected_backend = cuda`, zero failed cases, at least 30 `wall_ns` samples, positive `p50`/`p95`/`p99` `wall_ns` latency percentiles, and positive p50 `megakernel_condition_slots`, `megakernel_condition_fired`, and `megakernel_condition_slots_per_sec_x1000` metrics.
- `megakernel-latency-cuda.json` must report `selected_backend = cuda`, zero failed cases, at least 30 `wall_ns` samples, positive `p50`/`p95`/`p99` `wall_ns` latency percentiles, and positive p50 `megakernel_slots`, `megakernel_dispatch_latency_ns`, `megakernel_slots_per_sec_x1000`, `megakernel_roundtrip_buffers`, `megakernel_speculation_samples`, `megakernel_speculation_adopted`, `megakernel_speculation_rejected`, `megakernel_speculation_side_compile_cost_ns`, and `megakernel_speculation_autotune_records` metrics.
