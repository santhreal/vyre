//! Semantic checks for benchmark release evidence.
//!
//! `data` owns the tables the checks are written against and the enums they
//! report, so changing what release evidence must contain is a change to that
//! file alone. Every other module here decides one question about a report, and
//! its doc comment states which: whether the source fingerprint is precise and
//! current, whether a suite status row is proven by the artifact it names,
//! whether the two backend suites measured the same work, whether a release
//! axis survives recomputation from the artifacts cited for it.
//!
//! Callers see one module either way. The names below are the whole surface the
//! release gate and the benchmark subcommands import, so a check that grows a
//! second caller is re-exported here rather than reached through its file.

mod artifact_reader;
mod backend_identity;
mod backend_suite_artifact_status;
mod backend_suite_inventory;
mod backend_suite_parity;
mod case_summary;
mod cpu_sota_100x;
mod data;
mod evidence_blockers;
mod fused_execution_dag;
mod hygiene_surface;
mod json_reader;
mod optimization_analysis;
mod release_axes_artifact_metrics;
mod release_axes_cpu_sota;
mod release_axes_cuda;
mod release_axes_scalars;
mod schema_digest_chain;
mod source_artifact;
mod source_artifact_integrity;
mod source_artifact_provenance;
mod source_fingerprint;
mod suite_reader;
mod telemetry_labels;

pub(crate) use data::*;

pub(crate) use backend_identity::{
    backend_consistency_issues, backend_suite_backend_issue, contract_backend_issues,
    expected_backend_for_suite_evidence,
};
pub(crate) use backend_suite_artifact_status::backend_suite_artifact_status_issues;
pub(crate) use backend_suite_inventory::{
    backend_suite_inventory_issues, backend_suite_matrix_coverage_issues,
    describe_backend_suite_inventory_issue, describe_backend_suite_matrix_coverage_issue,
};
pub(crate) use backend_suite_parity::backend_suite_parity_issues;
pub(crate) use case_summary::{
    benchmark_case_failure_reason, benchmark_failed_case_summaries,
    benchmark_report_summary_case_evidence_mismatch,
    benchmark_report_summary_matches_case_evidence,
};
pub(crate) use cpu_sota_100x::{
    benchmark_case_claims_contract_win, benchmark_case_has_cpu_sota_contract,
    benchmark_case_proves_cpu_sota_100x, cpu_sota_100x_case_counts,
    inspect_cpu_sota_100x_case_count_consistency,
};
pub(crate) use evidence_blockers::benchmark_evidence_blocker_issues;
pub(crate) use fused_execution_dag::benchmark_fused_execution_dag_issues;
pub(crate) use hygiene_surface::inspect_hygiene_release_surface_coverage;
pub(crate) use json_reader::{
    duplicate_nonblank_object_array_field_values, duplicate_nonblank_string_array_values,
    metrics_has_any, metrics_has_positive_any, metrics_has_zero_any,
};
pub(crate) use optimization_analysis::{
    benchmark_before_after_semantic_win, inspect_optimization_analysis_fixture,
};
pub(crate) use release_axes_cpu_sota::cpu_sota_100x_source_artifact_issues;
pub(crate) use release_axes_cuda::cuda_release_axes_source_artifact_issues;
pub(crate) use schema_digest_chain::{
    benchmark_schema_digest_chain_issues, benchmark_schema_digest_chain_value,
};
pub(crate) use source_artifact::{
    benchmark_duplicate_source_artifact_paths, benchmark_report_has_source_provenance,
    benchmark_source_artifact_count, benchmark_source_artifact_entry_count,
    benchmark_source_artifact_path_issue, benchmark_source_artifact_paths,
    benchmark_suite_artifact_path_issue,
};
pub(crate) use source_artifact_integrity::inspect_source_artifact_case_integrity;
pub(crate) use source_fingerprint::{
    current_freshness_fingerprint_for_report, report_freshness_fingerprint,
    source_fingerprint_freshness_issues, source_fingerprint_issues,
};
pub(crate) use suite_reader::report_status_for_path;
pub(crate) use telemetry_labels::{
    cuda_forbidden_telemetry_issues, cuda_telemetry_label_issues, launch_plan_label_issues,
};
