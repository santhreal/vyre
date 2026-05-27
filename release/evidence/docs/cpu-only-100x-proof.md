# CPU-only 100x proof contract

This artifact backs `cpu-only-100x-proof`.

Evidence sources:

Required generated evidence:

- `release/evidence/benchmarks/release-workload-matrix.json`
- `release/evidence/benchmarks/cpu-only-100x-proof.json`
- `release/evidence/benchmarks/megakernel-condition-100x-proof.json`
- `release/evidence/benchmarks/workload-01-condition-eval.json`
- `release/evidence/benchmarks/workload-02-string-bitmap-scatter.json`
- `release/evidence/benchmarks/workload-03-offset-count-aggregation.json`
- `release/evidence/benchmarks/workload-05-entropy-window.json`
- `release/evidence/benchmarks/workload-06-quantified-condition-loops.json`
- `release/evidence/benchmarks/workload-07-alias-reaching-def.json`
- `release/evidence/benchmarks/workload-08-ifds-witness.json`
- `release/evidence/benchmarks/workload-09-c-ast-traversal.json`
- `release/evidence/benchmarks/workload-10-megakernel-queued-batches.json`
- `release/evidence/benchmarks/workload-11-egraph-saturation.json`
- `release/evidence/benchmarks/workload-12-sparse-output-compaction.json`
- CUDA release-suite reports for the current `release.*` workload case ids.

Release contract:

- The current required 100x case ids are `release.condition_eval.1m`, `release.string_bitmap_scatter.1m`, `release.offset_count_aggregation.1m`, `release.entropy_window.1m`, `release.quantified_condition_loops.1m`, `release.alias_reaching_def.1m`, `release.ifds_witness.1m`, `release.c_ast_traversal.1m`, `release.megakernel_queue.1m`, `release.egraph_saturation.1m`, and `sparse.compaction.count.1m`.
- Each required 100x case must declare a `CpuSota` baseline with `min_speedup_x >= 100.0`.
- The aggregate 100x proof artifact must include correctness oracle evidence and CUDA benchmark samples for every required 100x contract case.
- The aggregate 100x proof artifact must preserve source identity and CUDA environment provenance from its source workload reports, including `source_fingerprint`, `git`, GPU model, NVIDIA driver, CUDA runtime, and host CPU model where source reports provide it.
- The aggregate proof must list those required cases in `required_cpu_sota_100x_cases` and report an empty `missing_required_cpu_sota_100x_cases` array.
- Benchmark cases must report at least `30` wall-clock samples and positive `p50`/`p95`/`p99` `wall_ns` latency percentiles.
- CPU-SOTA 100x suite entries must also report at least `30` `baseline_wall_ns` samples, positive `p50`/`p95`/`p99` `baseline_wall_ns` latency percentiles, and passing 100x contract status for every required 100x case in the CUDA suite `artifact_statuses`.
- CPU-SOTA baseline metadata must include a concrete baseline crate and CUDA backend binding.
- The release workload matrix must expose a reproducible `cargo_full` CUDA benchmark command.
- The release workload matrix must name all required CPU-SOTA 100x families, must carry named CPU-SOTA baseline provenance for every required workload family, and must list active contract case ids for every required 100x case; a 100x aggregate artifact without those matrix contracts is not release evidence.
