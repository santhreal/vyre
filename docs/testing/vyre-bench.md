# Testing `vyre-bench`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench
```

Own reproducible workload benchmarks, comparisons, budgets, and raw benchmark evidence.

The crate lives at `vyre-bench`. The `benchmarks` owner maintains its
`tooling` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench
```

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --all-features
```

```console
CARGO_BUILD_JOBS=1 ./cargo_full run --bin xtask -- release-benchmarks --backend cuda
```

## Feature sets

- Default feature members: `cli`
- Available manifest features: `cli`, `default`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bench` | `release` | `vyre-bench/benches/release.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --bench release` |
| `bin` | `vyre-bench` | `vyre-bench/src/main.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --bin vyre-bench` |
| `bin` | `vyre-bench` | `vyre-bench/src/main.rs` | `cli` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --bin vyre-bench` |
| `lib` | `vyre_bench` | `vyre-bench/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench` |
| `test` | `ann_research_planners` | `vyre-bench/tests/ann_research_planners.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test ann_research_planners` |
| `test` | `baseline_determinism` | `vyre-bench/tests/baseline_determinism.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test baseline_determinism` |
| `test` | `benchmark_methodology_contracts` | `vyre-bench/tests/benchmark_methodology_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test benchmark_methodology_contracts` |
| `test` | `benchmark_target_contracts` | `vyre-bench/tests/benchmark_target_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test benchmark_target_contracts` |
| `test` | `corpus_paging_planner` | `vyre-bench/tests/corpus_paging_planner.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test corpus_paging_planner` |
| `test` | `coverage_sanitizer_matrix` | `vyre-bench/tests/coverage_sanitizer_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test coverage_sanitizer_matrix` |
| `test` | `cross_backend_counter_schema` | `vyre-bench/tests/cross_backend_counter_schema.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test cross_backend_counter_schema` |
| `test` | `cross_emitter_property` | `vyre-bench/tests/cross_emitter_property.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test cross_emitter_property` |
| `test` | `dataset_lineage_catalog` | `vyre-bench/tests/dataset_lineage_catalog.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test dataset_lineage_catalog` |
| `test` | `dfa_full_coverage` | `vyre-bench/tests/dfa_full_coverage.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test dfa_full_coverage` |
| `test` | `feature_cfg_contract` | `vyre-bench/tests/feature_cfg_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test feature_cfg_contract` |
| `test` | `finite_queue_artifact` | `vyre-bench/tests/finite_queue_artifact.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test finite_queue_artifact` |
| `test` | `full_pipeline_snapshot` | `vyre-bench/tests/full_pipeline_snapshot.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test full_pipeline_snapshot` |
| `test` | `g10_cross_backend` | `vyre-bench/tests/g10_cross_backend.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test g10_cross_backend` |
| `test` | `g12_cli` | `vyre-bench/tests/g12_cli.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test g12_cli` |
| `test` | `g1_cuda_events` | `vyre-bench/tests/g1_cuda_events.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test g1_cuda_events` |
| `test` | `g2_tail_latency` | `vyre-bench/tests/g2_tail_latency.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test g2_tail_latency` |
| `test` | `g3_determinism` | `vyre-bench/tests/g3_determinism.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test g3_determinism` |
| `test` | `g5_cache_hit` | `vyre-bench/tests/g5_cache_hit.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test g5_cache_hit` |
| `test` | `g6_snapshot` | `vyre-bench/tests/g6_snapshot.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test g6_snapshot` |
| `test` | `g7_thermal` | `vyre-bench/tests/g7_thermal.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test g7_thermal` |
| `test` | `g9_sweep` | `vyre-bench/tests/g9_sweep.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test g9_sweep` |
| `test` | `metrics_exposition_contracts` | `vyre-bench/tests/metrics_exposition_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test metrics_exposition_contracts` |
| `test` | `min_samples_gate` | `vyre-bench/tests/min_samples_gate.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test min_samples_gate` |
| `test` | `nvme_gpu_ingest_telemetry` | `vyre-bench/tests/nvme_gpu_ingest_telemetry.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test nvme_gpu_ingest_telemetry` |
| `test` | `paged_corpus_multi_gpu_benchmark` | `vyre-bench/tests/paged_corpus_multi_gpu_benchmark.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test paged_corpus_multi_gpu_benchmark` |
| `test` | `parser_structural_index_prepass` | `vyre-bench/tests/parser_structural_index_prepass.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test parser_structural_index_prepass` |
| `test` | `perf_analyses_snapshot` | `vyre-bench/tests/perf_analyses_snapshot.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test perf_analyses_snapshot` |
| `test` | `prototype_kernel_comparator` | `vyre-bench/tests/prototype_kernel_comparator.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test prototype_kernel_comparator` |
| `test` | `regex_cpu_gpu_partition_registry` | `vyre-bench/tests/regex_cpu_gpu_partition_registry.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test regex_cpu_gpu_partition_registry` |
| `test` | `regex_engine_comparator_registry` | `vyre-bench/tests/regex_engine_comparator_registry.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test regex_engine_comparator_registry` |
| `test` | `regex_external_accelerator_routes` | `vyre-bench/tests/regex_external_accelerator_routes.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test regex_external_accelerator_routes` |
| `test` | `relation_engine_comparators` | `vyre-bench/tests/relation_engine_comparators.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test relation_engine_comparators` |
| `test` | `release_bench_release_macro` | `vyre-bench/tests/release_bench_release_macro.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test release_bench_release_macro` |
| `test` | `release_macro_cuda_live` | `vyre-bench/tests/release_macro_cuda_live.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test release_macro_cuda_live` |
| `test` | `release_matrix_contracts` | `vyre-bench/tests/release_matrix_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test release_matrix_contracts` |
| `test` | `reproducibility_capsules` | `vyre-bench/tests/reproducibility_capsules.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test reproducibility_capsules` |
| `test` | `result_schema` | `vyre-bench/tests/result_schema.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test result_schema` |
| `test` | `roofline_counter_evidence` | `vyre-bench/tests/roofline_counter_evidence.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test roofline_counter_evidence` |
| `test` | `scan_counter_evidence_registry` | `vyre-bench/tests/scan_counter_evidence_registry.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test scan_counter_evidence_registry` |
| `test` | `scan_roofline_bandwidth_cuda` | `vyre-bench/tests/scan_roofline_bandwidth_cuda.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test scan_roofline_bandwidth_cuda` |
| `test` | `scan_roofline_model_cuda` | `vyre-bench/tests/scan_roofline_model_cuda.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test scan_roofline_model_cuda` |
| `test` | `scan_roofline_operating_point_cuda` | `vyre-bench/tests/scan_roofline_operating_point_cuda.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test scan_roofline_operating_point_cuda` |
| `test` | `source_fingerprint_operator_files` | `vyre-bench/tests/source_fingerprint_operator_files.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test source_fingerprint_operator_files` |
| `test` | `statistical_regression_gates` | `vyre-bench/tests/statistical_regression_gates.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test statistical_regression_gates` |
| `test` | `suite_completeness` | `vyre-bench/tests/suite_completeness.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test suite_completeness` |
| `test` | `thesis_workload_contracts` | `vyre-bench/tests/thesis_workload_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test thesis_workload_contracts` |
| `test` | `throughput_consistency` | `vyre-bench/tests/throughput_consistency.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-bench --test throughput_consistency` |

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
