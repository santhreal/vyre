//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/ann_research_planners.rs`.
#[path = "ann_research_planners.rs"]
pub mod ann_research_planners;

/// Integration tests from `tests/baseline_determinism.rs`.
#[cfg(feature = "device-tests")]
#[allow(clippy::field_reassign_with_default)]
#[path = "baseline_determinism.rs"]
pub mod baseline_determinism;

/// Integration tests from `tests/benchmark_methodology_contracts.rs`.
#[path = "benchmark_methodology_contracts.rs"]
pub mod benchmark_methodology_contracts;

/// Integration tests from `tests/benchmark_target_contracts.rs`.
#[path = "benchmark_target_contracts.rs"]
pub mod benchmark_target_contracts;

/// Integration tests from `tests/cache_hit_rate_reporting.rs`.
#[cfg(feature = "device-tests")]
#[allow(clippy::field_reassign_with_default, missing_docs)]
#[path = "cache_hit_rate_reporting.rs"]
pub mod cache_hit_rate_reporting;

/// Integration tests from `tests/case_declaration_contracts.rs`.
#[path = "case_declaration_contracts.rs"]
pub mod case_declaration_contracts;

/// Integration tests from `tests/cli_audit_surface.rs`.
#[path = "cli_audit_surface.rs"]
pub mod cli_audit_surface;

/// Integration tests from `tests/clone_family_contracts.rs`.
#[path = "clone_family_contracts.rs"]
pub mod clone_family_contracts;

/// Integration tests from `tests/corpus_paging_planner.rs`.
#[path = "corpus_paging_planner.rs"]
pub mod corpus_paging_planner;

/// Integration tests from `tests/coverage_sanitizer_matrix.rs`.
#[path = "coverage_sanitizer_matrix.rs"]
pub mod coverage_sanitizer_matrix;

/// Integration tests from `tests/cross_backend_counter_schema.rs`.
#[path = "cross_backend_counter_schema.rs"]
pub mod cross_backend_counter_schema;

/// Integration tests from `tests/cross_backend_matrix.rs`.
#[cfg(feature = "device-tests")]
#[allow(clippy::field_reassign_with_default)]
#[path = "cross_backend_matrix.rs"]
pub mod cross_backend_matrix;

/// Integration tests from `tests/cross_emitter_property.rs`.
#[path = "cross_emitter_property.rs"]
pub mod cross_emitter_property;

/// Integration tests from `tests/cuda_event_timing.rs`.
#[cfg(feature = "device-tests")]
#[allow(clippy::field_reassign_with_default, missing_docs)]
#[path = "cuda_event_timing.rs"]
pub mod cuda_event_timing;

/// Integration tests from `tests/dataset_lineage_catalog.rs`.
#[path = "dataset_lineage_catalog.rs"]
pub mod dataset_lineage_catalog;

/// Integration tests from `tests/device_profile_admissibility.rs`.
#[cfg(feature = "device-tests")]
#[path = "device_profile_admissibility.rs"]
pub mod device_profile_admissibility;

/// Integration tests from `tests/dfa_full_coverage.rs`.
#[cfg(feature = "device-tests")]
#[allow(clippy::field_reassign_with_default)]
#[path = "dfa_full_coverage.rs"]
pub mod dfa_full_coverage;


/// Integration tests from `tests/evidence_receipt_contracts.rs`.
#[path = "evidence_receipt_contracts.rs"]
pub mod evidence_receipt_contracts;
/// Integration tests from `tests/feature_cfg_contract.rs`.
#[path = "feature_cfg_contract.rs"]
pub mod feature_cfg_contract;

/// Integration tests from `tests/finite_queue_artifact.rs`.
#[cfg(not(target_os = "macos"))]
#[path = "finite_queue_artifact.rs"]
pub mod finite_queue_artifact;

/// Integration tests from `tests/metrics_exposition_contracts.rs`.
#[path = "metrics_exposition_contracts.rs"]
pub mod metrics_exposition_contracts;

/// Integration tests from `tests/nvme_gpu_ingest_telemetry.rs`.
#[cfg(target_os = "linux")]
#[path = "nvme_gpu_ingest_telemetry.rs"]
pub mod nvme_gpu_ingest_telemetry;

/// Integration tests from `tests/parser_structural_index_prepass.rs`.
#[path = "parser_structural_index_prepass.rs"]
pub mod parser_structural_index_prepass;

/// Integration tests from `tests/perf_analyses_snapshot.rs`.
#[path = "perf_analyses_snapshot.rs"]
pub mod perf_analyses_snapshot;

/// Integration tests from `tests/performance_contract_baseline_truth.rs`.
#[path = "performance_contract_baseline_truth.rs"]
pub mod performance_contract_baseline_truth;

/// Integration tests from `tests/prototype_kernel_comparator.rs`.
#[path = "prototype_kernel_comparator.rs"]
pub mod prototype_kernel_comparator;

/// Integration tests from `tests/regex_cpu_gpu_partition_registry.rs`.
#[path = "regex_cpu_gpu_partition_registry.rs"]
pub mod regex_cpu_gpu_partition_registry;

/// Integration tests from `tests/regex_engine_comparator_registry.rs`.
#[path = "regex_engine_comparator_registry.rs"]
pub mod regex_engine_comparator_registry;

/// Integration tests from `tests/regex_external_accelerator_routes.rs`.
#[path = "regex_external_accelerator_routes.rs"]
pub mod regex_external_accelerator_routes;

/// Integration tests from `tests/registry_closure.rs`.
#[path = "registry_closure.rs"]
pub mod registry_closure;

/// Integration tests from `tests/relation_engine_comparators.rs`.
#[path = "relation_engine_comparators.rs"]
pub mod relation_engine_comparators;

/// Integration tests from `tests/release_bench_release_macro.rs`.
#[path = "release_bench_release_macro.rs"]
pub mod release_bench_release_macro;

/// Integration tests from `tests/release_macro_cuda_live.rs`.
#[cfg(all(not(target_os = "macos"), feature = "device-tests"))]
#[path = "release_macro_cuda_live.rs"]
pub mod release_macro_cuda_live;

/// Integration tests from `tests/release_matrix_contracts/mod.rs`.
#[path = "release_matrix_contracts/mod.rs"]
pub mod release_matrix_contracts;

/// Integration tests from `tests/release_producer_cuda_contracts.rs`.
#[path = "release_producer_cuda_contracts.rs"]
pub mod release_producer_cuda_contracts;

/// Integration tests from `tests/reproducibility_capsules.rs`.
#[path = "reproducibility_capsules.rs"]
pub mod reproducibility_capsules;

/// Integration tests from `tests/result_schema.rs`.
#[cfg(feature = "device-tests")]
#[allow(clippy::field_reassign_with_default)]
#[path = "result_schema.rs"]
pub mod result_schema;

/// Integration tests from `tests/roofline_counter_evidence.rs`.
#[path = "roofline_counter_evidence.rs"]
pub mod roofline_counter_evidence;

/// Integration tests from `tests/scan_counter_evidence_registry.rs`.
#[path = "scan_counter_evidence_registry.rs"]
pub mod scan_counter_evidence_registry;

/// Integration tests from `tests/section_189_hardware_regression_evidence_and_pmu_policy.rs`.
#[path = "section_189_hardware_regression_evidence_and_pmu_policy.rs"]
pub mod section_189_hardware_regression_evidence_and_pmu_policy;

/// Integration tests from `tests/snapshot_persistence.rs`.
#[cfg(feature = "device-tests")]
#[allow(clippy::field_reassign_with_default)]
#[path = "snapshot_persistence.rs"]
pub mod snapshot_persistence;

/// Integration tests from `tests/source_fingerprint_operator_files.rs`.
#[path = "source_fingerprint_operator_files.rs"]
pub mod source_fingerprint_operator_files;

/// Integration tests from `tests/statistical_regression_gates.rs`.
#[path = "statistical_regression_gates.rs"]
pub mod statistical_regression_gates;

/// Integration tests from `tests/suite_completeness.rs`.
#[path = "suite_completeness.rs"]
pub mod suite_completeness;

/// Integration tests from `tests/sweep_suite_matrix.rs`.
#[cfg(feature = "device-tests")]
#[allow(clippy::field_reassign_with_default)]
#[path = "sweep_suite_matrix.rs"]
pub mod sweep_suite_matrix;

/// Integration tests from `tests/tail_latency_monotonicity.rs`.
#[cfg(feature = "device-tests")]
#[allow(clippy::field_reassign_with_default, missing_docs)]
#[path = "tail_latency_monotonicity.rs"]
pub mod tail_latency_monotonicity;

/// Integration tests from `tests/thermal_normalization.rs`.
#[cfg(feature = "device-tests")]
#[allow(clippy::field_reassign_with_default)]
#[path = "thermal_normalization.rs"]
pub mod thermal_normalization;

/// Integration tests from `tests/thesis_workload_contracts.rs`.
#[path = "thesis_workload_contracts.rs"]
pub mod thesis_workload_contracts;

/// Integration tests from `tests/throughput_consistency.rs`.
#[cfg(feature = "device-tests")]
#[allow(clippy::field_reassign_with_default)]
#[path = "throughput_consistency.rs"]
pub mod throughput_consistency;
