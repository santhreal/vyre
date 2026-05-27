# Benchmark documentation proof

This artifact backs benchmark-related release requirements.

Evidence sources:

Required generated evidence:

- `release/evidence/benchmarks/release-workload-matrix.json`
- `release/evidence/benchmarks/compiler-grade-thesis-workloads.json`
- `release/evidence/benchmarks/bench-release-axes.json`
- `release/evidence/benchmarks/cuda-release-suite.json`
- `release/evidence/benchmarks/wgpu-fallback-suite.json`
- `release/evidence/optimization/pass-family-benchmark-manifest.json`

Release contract:

- Benchmark commands must use `cargo_full`.
- CUDA is the primary release benchmark backend.
- WGPU evidence is fallback evidence, not the main release path.
- Every claimed speedup must include sample count, positive `p50`/`p95`/`p99` latency percentiles, correctness oracle, and explicit baseline contract.
- Every workload benchmark artifact must carry source provenance through `git.commit`, an explicit source fingerprint, or source artifact provenance; host CPU model provenance; workload/dataset/corpus/input fingerprint provenance per case; cache hit-rate field; cold compile or cold wall timing; host-to-device bytes; device-to-host bytes; kernel launch count; and the optimization or pipeline stages applied.
- `release-workload-matrix.json` must prove at least 12 release workload families, unique `release_plan_workload` numbers, one unique workload benchmark artifact per family, matched release cases, canonical `BENCH_TARGETS.toml` target ids, named CPU-SOTA baseline provenance, fair CPU-SOTA baseline crate provenance bound to CUDA, and a reproducible `cargo_full` CUDA benchmark command naming each evidence artifact. The current matrix carries 13 required workload rows.
- `compiler-grade-thesis-workloads.json` must prove compound release axes for parsing throughput, dataflow convergence, graph traversal, pattern matching, fixpoint witness propagation, persistent megakernel batches, and optimizer saturation. Each axis must resolve to release-suite macro workloads that require GPU execution, carry CUDA-bound CPU-SOTA baseline contracts, and operate at at least 1 MiB input scale.
- `pass-family-benchmark-manifest.json` must map every required optimization family to at least one CUDA benchmark artifact, and every manifest case must prove optimized `wall_ns.p50` beats `baseline_wall_ns.p50` through `min_wall_speedup_x1000 > 1000`.
- `bench-release-axes.json` must include real correctness ULP evidence; `ulp_drift_max=0` without a source correctness metric is not acceptable.
- `cuda-release-suite.json` and `wgpu-fallback-suite.json` must cover at least 12 workload families and include per-artifact semantic statuses with source family id, requested case id, source fingerprint, host CPU model, selected backend, case count, failed count, backend mismatch count, `wall_ns` sample count, `baseline_wall_ns` sample count, positive `p50`/`p95`/`p99` latency percentiles for speed-proof metrics, CPU-SOTA 100x contract counts where required, and zero blockers.
