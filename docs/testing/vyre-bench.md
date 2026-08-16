# Testing `vyre-bench`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-bench
```

Own reproducible workload benchmarks against the best available native baseline for each class, not against vyre's own unfused output.

The crate lives at `vyre-bench`. The `benchmarks` owner maintains its
`tooling` testing contract.

## Commands

```console
./cargo_full test -p vyre-bench
```

```console
./cargo_full test -p vyre-bench --all-features
```

```console
./cargo_full run --bin xtask -- release-benchmarks --backend cuda
```

## Feature sets

- Default feature members: `cli`
- Available manifest features: `cli`, `default`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bench` | `release` | `vyre-bench/benches/release.rs` | None | `./cargo_full test -p vyre-bench --bench release` |
| `bin` | `vyre-bench` | `vyre-bench/src/main.rs` | None | `./cargo_full test -p vyre-bench --bin vyre-bench` |
| `bin` | `vyre-bench` | `vyre-bench/src/main.rs` | `cli` | `./cargo_full test -p vyre-bench --bin vyre-bench` |
| `lib` | `vyre_bench` | `vyre-bench/src/lib.rs` | None | `./cargo_full test -p vyre-bench` |
| `test` | `ann_research_planners` | `vyre-bench/tests/ann_research_planners.rs` | None | `./cargo_full test -p vyre-bench --test ann_research_planners` |
| `test` | `baseline_determinism` | `vyre-bench/tests/baseline_determinism.rs` | None | `./cargo_full test -p vyre-bench --test baseline_determinism` |
| `test` | `benchmark_methodology_contracts` | `vyre-bench/tests/benchmark_methodology_contracts.rs` | None | `./cargo_full test -p vyre-bench --test benchmark_methodology_contracts` |
| `test` | `benchmark_target_contracts` | `vyre-bench/tests/benchmark_target_contracts.rs` | None | `./cargo_full test -p vyre-bench --test benchmark_target_contracts` |
| `test` | `cache_hit_rate_reporting` | `vyre-bench/tests/cache_hit_rate_reporting.rs` | None | `./cargo_full test -p vyre-bench --test cache_hit_rate_reporting` |
| `test` | `case_declaration_contracts` | `vyre-bench/tests/case_declaration_contracts.rs` | None | `./cargo_full test -p vyre-bench --test case_declaration_contracts` |
| `test` | `cli_audit_surface` | `vyre-bench/tests/cli_audit_surface.rs` | None | `./cargo_full test -p vyre-bench --test cli_audit_surface` |
| `test` | `clone_family_contracts` | `vyre-bench/tests/clone_family_contracts.rs` | None | `./cargo_full test -p vyre-bench --test clone_family_contracts` |
| `test` | `corpus_paging_planner` | `vyre-bench/tests/corpus_paging_planner.rs` | None | `./cargo_full test -p vyre-bench --test corpus_paging_planner` |
| `test` | `coverage_sanitizer_matrix` | `vyre-bench/tests/coverage_sanitizer_matrix.rs` | None | `./cargo_full test -p vyre-bench --test coverage_sanitizer_matrix` |
| `test` | `cross_backend_counter_schema` | `vyre-bench/tests/cross_backend_counter_schema.rs` | None | `./cargo_full test -p vyre-bench --test cross_backend_counter_schema` |
| `test` | `cross_backend_matrix` | `vyre-bench/tests/cross_backend_matrix.rs` | None | `./cargo_full test -p vyre-bench --test cross_backend_matrix` |
| `test` | `cross_emitter_property` | `vyre-bench/tests/cross_emitter_property.rs` | None | `./cargo_full test -p vyre-bench --test cross_emitter_property` |
| `test` | `cuda_event_timing` | `vyre-bench/tests/cuda_event_timing.rs` | None | `./cargo_full test -p vyre-bench --test cuda_event_timing` |
| `test` | `dataset_lineage_catalog` | `vyre-bench/tests/dataset_lineage_catalog.rs` | None | `./cargo_full test -p vyre-bench --test dataset_lineage_catalog` |
| `test` | `determinism_gate` | `vyre-bench/tests/determinism_gate.rs` | None | `./cargo_full test -p vyre-bench --test determinism_gate` |
| `test` | `dfa_full_coverage` | `vyre-bench/tests/dfa_full_coverage.rs` | None | `./cargo_full test -p vyre-bench --test dfa_full_coverage` |
| `test` | `feature_cfg_contract` | `vyre-bench/tests/feature_cfg_contract.rs` | None | `./cargo_full test -p vyre-bench --test feature_cfg_contract` |
| `test` | `finite_queue_artifact` | `vyre-bench/tests/finite_queue_artifact.rs` | None | `./cargo_full test -p vyre-bench --test finite_queue_artifact` |
| `test` | `metrics_exposition_contracts` | `vyre-bench/tests/metrics_exposition_contracts.rs` | None | `./cargo_full test -p vyre-bench --test metrics_exposition_contracts` |
| `test` | `min_samples_gate` | `vyre-bench/tests/min_samples_gate.rs` | None | `./cargo_full test -p vyre-bench --test min_samples_gate` |
| `test` | `nvme_gpu_ingest_telemetry` | `vyre-bench/tests/nvme_gpu_ingest_telemetry.rs` | None | `./cargo_full test -p vyre-bench --test nvme_gpu_ingest_telemetry` |
| `test` | `parser_structural_index_prepass` | `vyre-bench/tests/parser_structural_index_prepass.rs` | None | `./cargo_full test -p vyre-bench --test parser_structural_index_prepass` |
| `test` | `perf_analyses_snapshot` | `vyre-bench/tests/perf_analyses_snapshot.rs` | None | `./cargo_full test -p vyre-bench --test perf_analyses_snapshot` |
| `test` | `performance_contract_baseline_truth` | `vyre-bench/tests/performance_contract_baseline_truth.rs` | None | `./cargo_full test -p vyre-bench --test performance_contract_baseline_truth` |
| `test` | `prototype_kernel_comparator` | `vyre-bench/tests/prototype_kernel_comparator.rs` | None | `./cargo_full test -p vyre-bench --test prototype_kernel_comparator` |
| `test` | `regex_cpu_gpu_partition_registry` | `vyre-bench/tests/regex_cpu_gpu_partition_registry.rs` | None | `./cargo_full test -p vyre-bench --test regex_cpu_gpu_partition_registry` |
| `test` | `regex_engine_comparator_registry` | `vyre-bench/tests/regex_engine_comparator_registry.rs` | None | `./cargo_full test -p vyre-bench --test regex_engine_comparator_registry` |
| `test` | `regex_external_accelerator_routes` | `vyre-bench/tests/regex_external_accelerator_routes.rs` | None | `./cargo_full test -p vyre-bench --test regex_external_accelerator_routes` |
| `test` | `registry_closure` | `vyre-bench/tests/registry_closure.rs` | None | `./cargo_full test -p vyre-bench --test registry_closure` |
| `test` | `relation_engine_comparators` | `vyre-bench/tests/relation_engine_comparators.rs` | None | `./cargo_full test -p vyre-bench --test relation_engine_comparators` |
| `test` | `release_bench_release_macro` | `vyre-bench/tests/release_bench_release_macro.rs` | None | `./cargo_full test -p vyre-bench --test release_bench_release_macro` |
| `test` | `release_macro_cuda_live` | `vyre-bench/tests/release_macro_cuda_live.rs` | None | `./cargo_full test -p vyre-bench --test release_macro_cuda_live` |
| `test` | `release_matrix_contracts` | `vyre-bench/tests/release_matrix_contracts.rs` | None | `./cargo_full test -p vyre-bench --test release_matrix_contracts` |
| `test` | `reproducibility_capsules` | `vyre-bench/tests/reproducibility_capsules.rs` | None | `./cargo_full test -p vyre-bench --test reproducibility_capsules` |
| `test` | `result_schema` | `vyre-bench/tests/result_schema.rs` | None | `./cargo_full test -p vyre-bench --test result_schema` |
| `test` | `roofline_counter_evidence` | `vyre-bench/tests/roofline_counter_evidence.rs` | None | `./cargo_full test -p vyre-bench --test roofline_counter_evidence` |
| `test` | `scan_counter_evidence_registry` | `vyre-bench/tests/scan_counter_evidence_registry.rs` | None | `./cargo_full test -p vyre-bench --test scan_counter_evidence_registry` |
| `test` | `section_189_hardware_regression_evidence_and_pmu_policy` | `vyre-bench/tests/section_189_hardware_regression_evidence_and_pmu_policy.rs` | None | `./cargo_full test -p vyre-bench --test section_189_hardware_regression_evidence_and_pmu_policy` |
| `test` | `snapshot_persistence` | `vyre-bench/tests/snapshot_persistence.rs` | None | `./cargo_full test -p vyre-bench --test snapshot_persistence` |
| `test` | `source_fingerprint_operator_files` | `vyre-bench/tests/source_fingerprint_operator_files.rs` | None | `./cargo_full test -p vyre-bench --test source_fingerprint_operator_files` |
| `test` | `statistical_regression_gates` | `vyre-bench/tests/statistical_regression_gates.rs` | None | `./cargo_full test -p vyre-bench --test statistical_regression_gates` |
| `test` | `suite_completeness` | `vyre-bench/tests/suite_completeness.rs` | None | `./cargo_full test -p vyre-bench --test suite_completeness` |
| `test` | `sweep_suite_matrix` | `vyre-bench/tests/sweep_suite_matrix.rs` | None | `./cargo_full test -p vyre-bench --test sweep_suite_matrix` |
| `test` | `tail_latency_monotonicity` | `vyre-bench/tests/tail_latency_monotonicity.rs` | None | `./cargo_full test -p vyre-bench --test tail_latency_monotonicity` |
| `test` | `thermal_normalization` | `vyre-bench/tests/thermal_normalization.rs` | None | `./cargo_full test -p vyre-bench --test thermal_normalization` |
| `test` | `thesis_workload_contracts` | `vyre-bench/tests/thesis_workload_contracts.rs` | None | `./cargo_full test -p vyre-bench --test thesis_workload_contracts` |
| `test` | `throughput_consistency` | `vyre-bench/tests/throughput_consistency.rs` | None | `./cargo_full test -p vyre-bench --test throughput_consistency` |

## Test classes

- Command and policy behavior
- Evidence schema and regeneration contracts
- Failure diagnostics and repository boundaries

## Hardware requirements

Benchmark unit tests are host-capable. Release measurements require the backend and device named by the benchmark command; probe failure invalidates the run.

## Evidence outputs

- `release/evidence/benchmarks/`
- Raw per-sample benchmark records
- Generated suite summaries and source-tree fingerprints

## Skips and failures

Ignored physical benchmarks are absent from the default test command. A release benchmark command must execute its requested device and preserve raw samples; it cannot report a synthetic or skipped result.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
