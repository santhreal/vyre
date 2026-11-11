//! Whether one suite status row is proven by the artifact it names.
//!
//! A status row is a summary of an artifact, written at suite generation time,
//! and every field in it is re-derived from the artifact here: the provenance
//! fingerprints, the selected backend, the case and failure counts, the metric
//! minima, the host and device attribution, the CPU-SOTA counts, and the
//! requested case that must appear exactly once. A field the artifact can prove
//! and the row omits is a gap, not a default, so it is reported as missing.
//!
//! The findings are returned in the order the fields are checked, because the
//! callers print them in that order and the tests pin it.

use serde_json::Value;

use super::artifact_reader::{
    artifact_case_failed_count, artifact_case_passed_count, artifact_environment_first_gpu_str,
    artifact_environment_first_gpu_u64, artifact_environment_host_cpu_model,
    artifact_environment_str, artifact_min_metric_percentile, artifact_min_metric_samples,
    artifact_nonmatching_case_backend_count,
};
use super::cpu_sota_100x::cpu_sota_100x_case_counts;
use super::data::BackendSuiteArtifactStatusIssue;
use super::json_reader::non_empty_str;

pub(crate) fn backend_suite_artifact_status_issues(
    status: &Value,
    artifact_report: &Value,
) -> Vec<BackendSuiteArtifactStatusIssue> {
    let path = status
        .get("path")
        .and_then(Value::as_str)
        .unwrap_or("<unknown>")
        .to_string();
    let mut issues = Vec::new();

    let status_source = status.get("source_fingerprint").and_then(non_empty_str);
    let artifact_source = artifact_report
        .get("source_fingerprint")
        .and_then(non_empty_str);
    match (status_source, artifact_source) {
        (None, Some(_)) => issues.push(BackendSuiteArtifactStatusIssue::MissingField {
            path: path.clone(),
            field: "source_fingerprint",
        }),
        (Some(status_source), Some(artifact_source)) if status_source != artifact_source => {
            issues.push(BackendSuiteArtifactStatusIssue::SourceFingerprintMismatch {
                path: path.clone(),
                status_source_fingerprint: status_source.to_string(),
                artifact_source_fingerprint: artifact_source.to_string(),
            });
        }
        _ => {}
    }

    let status_source_tree = status
        .get("source_tree_fingerprint")
        .and_then(non_empty_str);
    let artifact_source_tree = artifact_report
        .get("source_tree_fingerprint")
        .and_then(non_empty_str);
    match (status_source_tree, artifact_source_tree) {
        (None, Some(_)) => issues.push(BackendSuiteArtifactStatusIssue::MissingField {
            path: path.clone(),
            field: "source_tree_fingerprint",
        }),
        (Some(status_source_tree), Some(artifact_source_tree))
            if status_source_tree != artifact_source_tree =>
        {
            issues.push(
                BackendSuiteArtifactStatusIssue::SourceTreeFingerprintMismatch {
                    path: path.clone(),
                    status_source_tree_fingerprint: status_source_tree.to_string(),
                    artifact_source_tree_fingerprint: artifact_source_tree.to_string(),
                },
            );
        }
        _ => {}
    }

    let status_backend = status.get("selected_backend").and_then(non_empty_str);
    let artifact_backend = artifact_report
        .get("selected_backend")
        .and_then(non_empty_str);
    match (status_backend, artifact_backend) {
        (None, Some(_)) => issues.push(BackendSuiteArtifactStatusIssue::MissingField {
            path: path.clone(),
            field: "selected_backend",
        }),
        (Some(status_backend), Some(artifact_backend)) if status_backend != artifact_backend => {
            issues.push(BackendSuiteArtifactStatusIssue::SelectedBackendMismatch {
                path: path.clone(),
                status_selected_backend: status_backend.to_string(),
                artifact_selected_backend: artifact_backend.to_string(),
            });
        }
        _ => {}
    }

    let status_case_count = status.get("case_count").and_then(Value::as_u64);
    let artifact_case_count = artifact_report
        .get("cases")
        .and_then(Value::as_array)
        .map(|cases| cases.len() as u64);
    let artifact_summary_total_cases = artifact_report
        .get("summary")
        .and_then(|summary| summary.get("total_cases"))
        .and_then(Value::as_u64);
    if artifact_case_count.is_some() && artifact_summary_total_cases.is_none() {
        issues.push(BackendSuiteArtifactStatusIssue::MissingField {
            path: path.clone(),
            field: "summary.total_cases",
        });
    }
    if let (Some(summary_total_cases), Some(case_count)) =
        (artifact_summary_total_cases, artifact_case_count)
    {
        if summary_total_cases != case_count {
            issues.push(BackendSuiteArtifactStatusIssue::NumericFieldMismatch {
                path: path.clone(),
                field: "summary.total_cases",
                status_value: summary_total_cases,
                artifact_value: case_count,
            });
        }
    }
    match (status_case_count, artifact_case_count) {
        (None, Some(_)) => issues.push(BackendSuiteArtifactStatusIssue::MissingField {
            path: path.clone(),
            field: "case_count",
        }),
        (Some(status_case_count), Some(artifact_case_count))
            if status_case_count != artifact_case_count =>
        {
            issues.push(BackendSuiteArtifactStatusIssue::CaseCountMismatch {
                path: path.clone(),
                status_case_count,
                artifact_case_count,
            });
        }
        _ => {}
    }

    let status_nonmatching_backend_count = status
        .get("nonmatching_case_backend_count")
        .and_then(Value::as_u64);
    let artifact_nonmatching_backend_count =
        artifact_nonmatching_case_backend_count(status, artifact_report);
    match (
        status_nonmatching_backend_count,
        artifact_nonmatching_backend_count,
    ) {
        (None, Some(_)) => issues.push(BackendSuiteArtifactStatusIssue::MissingField {
            path: path.clone(),
            field: "nonmatching_case_backend_count",
        }),
        (Some(status_value), Some(artifact_value)) if status_value != artifact_value => {
            issues.push(BackendSuiteArtifactStatusIssue::NumericFieldMismatch {
                path: path.clone(),
                field: "nonmatching_case_backend_count",
                status_value,
                artifact_value,
            });
        }
        _ => {}
    }

    let status_failed_count = status.get("failed_count").and_then(Value::as_u64);
    let artifact_summary_passed_count = artifact_report
        .get("summary")
        .and_then(|summary| summary.get("passed"))
        .and_then(Value::as_u64);
    let artifact_case_passed_count = artifact_case_passed_count(artifact_report);
    if artifact_case_passed_count.is_some() && artifact_summary_passed_count.is_none() {
        issues.push(BackendSuiteArtifactStatusIssue::MissingField {
            path: path.clone(),
            field: "summary.passed",
        });
    }
    if let (Some(summary_passed_count), Some(case_passed_count)) =
        (artifact_summary_passed_count, artifact_case_passed_count)
    {
        if summary_passed_count != case_passed_count {
            issues.push(BackendSuiteArtifactStatusIssue::NumericFieldMismatch {
                path: path.clone(),
                field: "summary.passed",
                status_value: summary_passed_count,
                artifact_value: case_passed_count,
            });
        }
    }
    let artifact_summary_failed_count = artifact_report
        .get("summary")
        .and_then(|summary| summary.get("failed"))
        .and_then(Value::as_u64);
    let artifact_case_failed_count = artifact_case_failed_count(artifact_report);
    if artifact_case_failed_count.is_some() && artifact_summary_failed_count.is_none() {
        issues.push(BackendSuiteArtifactStatusIssue::MissingField {
            path: path.clone(),
            field: "summary.failed",
        });
    }
    if let (Some(summary_failed_count), Some(case_failed_count)) =
        (artifact_summary_failed_count, artifact_case_failed_count)
    {
        if summary_failed_count != case_failed_count {
            issues.push(BackendSuiteArtifactStatusIssue::NumericFieldMismatch {
                path: path.clone(),
                field: "summary.failed",
                status_value: summary_failed_count,
                artifact_value: case_failed_count,
            });
        }
    }
    let artifact_failed_count = artifact_case_failed_count.or(artifact_summary_failed_count);
    match (status_failed_count, artifact_failed_count) {
        (None, Some(_)) => issues.push(BackendSuiteArtifactStatusIssue::MissingField {
            path: path.clone(),
            field: "failed_count",
        }),
        (Some(status_failed_count), Some(artifact_failed_count))
            if status_failed_count != artifact_failed_count =>
        {
            issues.push(BackendSuiteArtifactStatusIssue::FailedCountMismatch {
                path: path.clone(),
                status_failed_count,
                artifact_failed_count,
            });
        }
        _ => {}
    }

    for (field, artifact_value) in backend_suite_numeric_artifact_fields(artifact_report) {
        match status.get(field).and_then(Value::as_u64) {
            None => issues.push(BackendSuiteArtifactStatusIssue::MissingField {
                path: path.clone(),
                field,
            }),
            Some(status_value) if status_value != artifact_value => {
                issues.push(BackendSuiteArtifactStatusIssue::NumericFieldMismatch {
                    path: path.clone(),
                    field,
                    status_value,
                    artifact_value,
                });
            }
            _ => {}
        }
    }
    for (field, artifact_value) in backend_suite_string_artifact_fields(artifact_report) {
        match (status.get(field).and_then(non_empty_str), artifact_value) {
            (_, None) => issues.push(BackendSuiteArtifactStatusIssue::MissingField {
                path: path.clone(),
                field,
            }),
            (None, Some(_)) => issues.push(BackendSuiteArtifactStatusIssue::MissingField {
                path: path.clone(),
                field,
            }),
            (Some(status_value), Some(artifact_value)) if status_value != artifact_value => {
                issues.push(BackendSuiteArtifactStatusIssue::StringFieldMismatch {
                    path: path.clone(),
                    field,
                    status_value: status_value.to_string(),
                    artifact_value,
                });
            }
            _ => {}
        }
    }

    let (artifact_contract_cases, artifact_passing_cases) =
        cpu_sota_100x_case_counts(artifact_report);
    match status
        .get("cpu_sota_100x_contract_cases")
        .and_then(Value::as_u64)
    {
        None if artifact_contract_cases > 0 => {
            issues.push(BackendSuiteArtifactStatusIssue::MissingField {
                path: path.clone(),
                field: "cpu_sota_100x_contract_cases",
            });
        }
        Some(status_contract_cases) if status_contract_cases != artifact_contract_cases => {
            issues.push(
                BackendSuiteArtifactStatusIssue::CpuSota100xContractCaseCountMismatch {
                    path: path.clone(),
                    status_contract_cases,
                    artifact_contract_cases,
                },
            );
        }
        _ => {}
    }
    match status
        .get("cpu_sota_100x_passing_cases")
        .and_then(Value::as_u64)
    {
        None if artifact_passing_cases > 0 => {
            issues.push(BackendSuiteArtifactStatusIssue::MissingField {
                path: path.clone(),
                field: "cpu_sota_100x_passing_cases",
            });
        }
        Some(status_passing_cases) if status_passing_cases != artifact_passing_cases => {
            issues.push(
                BackendSuiteArtifactStatusIssue::CpuSota100xPassingCaseCountMismatch {
                    path: path.clone(),
                    status_passing_cases,
                    artifact_passing_cases,
                },
            );
        }
        _ => {}
    }

    if let Some(requested_case_id) = status.get("requested_case_id").and_then(non_empty_str) {
        let requested_case_count =
            artifact_report
                .get("cases")
                .and_then(Value::as_array)
                .map(|cases| {
                    cases
                        .iter()
                        .filter(|case| {
                            case.get("id").and_then(Value::as_str) == Some(requested_case_id)
                        })
                        .count()
                });
        match requested_case_count {
            Some(0) => issues.push(BackendSuiteArtifactStatusIssue::MissingRequestedCase {
                path: path.clone(),
                requested_case_id: requested_case_id.to_string(),
            }),
            Some(count) if count > 1 => {
                issues.push(BackendSuiteArtifactStatusIssue::DuplicateRequestedCase {
                    path,
                    requested_case_id: requested_case_id.to_string(),
                    count,
                });
            }
            _ => {}
        }
    }

    issues
}

fn backend_suite_numeric_artifact_fields(artifact_report: &Value) -> Vec<(&'static str, u64)> {
    let fields = [
        (
            "min_wall_samples",
            artifact_min_metric_samples(artifact_report, "wall_ns"),
        ),
        (
            "min_baseline_wall_samples",
            artifact_min_metric_samples(artifact_report, "baseline_wall_ns"),
        ),
        (
            "min_wall_p50",
            artifact_min_metric_percentile(artifact_report, "wall_ns", "p50"),
        ),
        (
            "min_wall_p95",
            artifact_min_metric_percentile(artifact_report, "wall_ns", "p95"),
        ),
        (
            "min_wall_p99",
            artifact_min_metric_percentile(artifact_report, "wall_ns", "p99"),
        ),
        (
            "min_baseline_wall_p50",
            artifact_min_metric_percentile(artifact_report, "baseline_wall_ns", "p50"),
        ),
        (
            "min_baseline_wall_p95",
            artifact_min_metric_percentile(artifact_report, "baseline_wall_ns", "p95"),
        ),
        (
            "min_baseline_wall_p99",
            artifact_min_metric_percentile(artifact_report, "baseline_wall_ns", "p99"),
        ),
        (
            "min_kernel_launches",
            artifact_min_metric_percentile(artifact_report, "kernel_launches", "p50"),
        ),
        (
            "min_cuda_ptx_source_cache_entries",
            artifact_min_metric_percentile(artifact_report, "cuda_ptx_source_cache_entries", "p50"),
        ),
        (
            "min_cuda_ptx_source_cache_hits",
            artifact_min_metric_percentile(artifact_report, "cuda_ptx_source_cache_hits", "p50"),
        ),
        (
            "min_cuda_ptx_source_cache_misses",
            artifact_min_metric_percentile(artifact_report, "cuda_ptx_source_cache_misses", "p50"),
        ),
        (
            "gpu_memory_total_mib",
            artifact_environment_first_gpu_u64(artifact_report, "memory_total_mib"),
        ),
        (
            "gpu_compute_capability_major",
            artifact_environment_first_gpu_u64(artifact_report, "compute_capability_major"),
        ),
        (
            "gpu_compute_capability_minor",
            artifact_environment_first_gpu_u64(artifact_report, "compute_capability_minor"),
        ),
    ];
    fields
        .into_iter()
        .filter_map(|(field, value)| value.map(|value| (field, value)))
        .collect()
}

fn backend_suite_string_artifact_fields(
    artifact_report: &Value,
) -> Vec<(&'static str, Option<String>)> {
    let fields = [
        (
            "host_cpu_model",
            artifact_environment_host_cpu_model(artifact_report),
        ),
        (
            "gpu_model",
            artifact_environment_first_gpu_str(artifact_report, "name"),
        ),
        (
            "nvidia_driver_version",
            artifact_environment_str(artifact_report, "nvidia_driver_version"),
        ),
        (
            "nvidia_cuda_version",
            artifact_environment_str(artifact_report, "nvidia_cuda_version"),
        ),
    ];
    fields
        .into_iter()
        .filter(|(_, value)| value.is_some())
        .map(|(field, value)| (field, value.flatten()))
        .collect()
}

#[cfg(test)]
mod tests;
