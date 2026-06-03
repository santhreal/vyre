use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use serde_json::{Map, Value};

static CURRENT_SOURCE_FINGERPRINTS: OnceLock<Mutex<BTreeMap<PathBuf, String>>> = OnceLock::new();
static CURRENT_SOURCE_TREE_FINGERPRINTS: OnceLock<Mutex<BTreeMap<PathBuf, String>>> =
    OnceLock::new();

pub(crate) fn benchmark_case_failure_reason(case: &Value) -> Option<String> {
    let status = case.get("status").and_then(Value::as_str);
    let contract_failed = case
        .get("performance")
        .and_then(|performance| performance.get("contract_passed"))
        .and_then(Value::as_bool)
        == Some(false);
    let invalid_reason = case
        .get("correctness")
        .and_then(|correctness| correctness.get("Invalid"))
        .and_then(|invalid| invalid.get("reason"))
        .and_then(Value::as_str)
        .filter(|reason| !reason.is_empty())
        .map(str::to_string);
    let violation_reason = case
        .get("performance")
        .and_then(|performance| performance.get("violations"))
        .and_then(Value::as_array)
        .map(|violations| {
            violations
                .iter()
                .filter_map(Value::as_str)
                .filter(|violation| !violation.is_empty())
                .collect::<Vec<_>>()
        })
        .and_then(|violations| (!violations.is_empty()).then(|| violations.join("; ")));
    invalid_reason
        .or(violation_reason)
        .or_else(|| match status {
            Some("pass") => None,
            Some(status) if !status.is_empty() => Some(format!("status `{status}`")),
            _ => Some("missing pass status".to_string()),
        })
        .or_else(|| contract_failed.then(|| "performance contract failed".to_string()))
}

pub(crate) fn benchmark_case_passes_summary_evidence(case: &Value) -> bool {
    case.get("status").and_then(Value::as_str) == Some("pass")
        && benchmark_case_failure_reason(case).is_none()
}

pub(crate) fn benchmark_report_summary_matches_case_evidence(report: &Value) -> bool {
    benchmark_report_summary_case_evidence_mismatch(report).is_none()
}

pub(crate) fn benchmark_report_has_source_provenance(report: &Value) -> bool {
    [
        "source_fingerprint",
        "source_revision",
        "source_artifact_fingerprint",
        "commit_fingerprint",
    ]
    .iter()
    .any(|field| report.get(*field).and_then(non_empty_str).is_some())
        || report
            .get("source_artifacts")
            .and_then(Value::as_array)
            .is_some_and(|items| items.iter().any(|item| non_empty_str(item).is_some()))
        || report
            .get("git")
            .is_some_and(|git| git.get("commit").and_then(non_empty_str).is_some())
}

pub(crate) fn benchmark_source_artifact_count(report: &Value) -> usize {
    report
        .get("source_artifacts")
        .and_then(Value::as_array)
        .map_or(0, |items| {
            items
                .iter()
                .filter_map(non_empty_str)
                .collect::<BTreeSet<_>>()
                .len()
        })
}

pub(crate) fn benchmark_report_summary_case_evidence_mismatch(report: &Value) -> Option<String> {
    let Some(cases) = report.get("cases").and_then(Value::as_array) else {
        return Some("missing cases array".to_string());
    };
    let Some(summary) = report.get("summary") else {
        return Some("missing summary".to_string());
    };
    let passed = cases
        .iter()
        .filter(|case| benchmark_case_passes_summary_evidence(case))
        .count() as u64;
    let failed = cases.len() as u64 - passed;
    let summary_total_cases = summary.get("total_cases").and_then(Value::as_u64);
    let summary_passed = summary.get("passed").and_then(Value::as_u64);
    let summary_failed = summary.get("failed").and_then(Value::as_u64);
    if summary_total_cases == Some(cases.len() as u64)
        && summary_passed == Some(passed)
        && summary_failed == Some(failed)
    {
        return None;
    }
    Some(format!(
        "summary total/pass/fail ({summary_total_cases:?}/{summary_passed:?}/{summary_failed:?}) contradicts case evidence ({}/{passed}/{failed})",
        cases.len()
    ))
}

pub(crate) fn benchmark_failed_case_summaries(report: &Value) -> Vec<String> {
    report
        .get("cases")
        .and_then(Value::as_array)
        .map(|cases| {
            cases
                .iter()
                .filter_map(|case| {
                    let id = case
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("<unknown>");
                    benchmark_case_failure_reason(case).map(|reason| format!("`{id}`: {reason}"))
                })
                .collect()
        })
        .unwrap_or_default()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum LaunchPlanLabelIssue {
    MissingSingle,
    SingleHasMulti,
    MissingMulti { launch_count: f64 },
    MultiHasSingle { launch_count: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum BackendConsistencyIssue {
    MissingCaseBackend {
        case_id: String,
        expected_backend: String,
    },
    CaseBackendMismatch {
        case_id: String,
        expected_backend: String,
        actual_backend: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ContractBackendIssue {
    MissingBaselines { case_id: String, backend_id: String },
    NoApplicableBaseline { case_id: String, backend_id: String },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CudaTelemetryLabelIssue {
    MissingLabel {
        case_id: String,
        label: &'static str,
    },
    LabelWithoutCounters {
        case_id: String,
        label: &'static str,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SourceFingerprintIssue {
    DirtyUnknownState {
        source_fingerprint: String,
    },
    DirtyMissingWorktree {
        source_fingerprint: String,
    },
    DirtyUnknownWorktree {
        source_fingerprint: String,
    },
    DirtyInvalidWorktree {
        source_fingerprint: String,
        worktree: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SourceFingerprintFreshnessIssue {
    Mismatch {
        source_fingerprint: String,
        current_source_fingerprint: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BackendSuiteParityIssue {
    MissingCudaPair {
        family_id: String,
        requested_case_id: String,
    },
    MissingWgpuPair {
        family_id: String,
        requested_case_id: String,
    },
    CountMismatch {
        cuda_count: usize,
        wgpu_count: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BackendSuiteInventoryIssue {
    CountMismatch {
        artifact_count: usize,
        status_count: usize,
    },
    MissingStatus {
        path: String,
    },
    MissingArtifact {
        path: String,
    },
    DuplicateArtifact {
        path: String,
    },
    DuplicateStatus {
        path: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BackendSuiteBackendIssue {
    Missing {
        expected_backend: String,
    },
    Mismatch {
        expected_backend: String,
        actual_backend: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BackendSuiteArtifactStatusIssue {
    MissingField {
        path: String,
        field: &'static str,
    },
    SourceFingerprintMismatch {
        path: String,
        status_source_fingerprint: String,
        artifact_source_fingerprint: String,
    },
    SourceTreeFingerprintMismatch {
        path: String,
        status_source_tree_fingerprint: String,
        artifact_source_tree_fingerprint: String,
    },
    SelectedBackendMismatch {
        path: String,
        status_selected_backend: String,
        artifact_selected_backend: String,
    },
    CaseCountMismatch {
        path: String,
        status_case_count: u64,
        artifact_case_count: u64,
    },
    FailedCountMismatch {
        path: String,
        status_failed_count: u64,
        artifact_failed_count: u64,
    },
    NumericFieldMismatch {
        path: String,
        field: &'static str,
        status_value: u64,
        artifact_value: u64,
    },
    StringFieldMismatch {
        path: String,
        field: &'static str,
        status_value: String,
        artifact_value: String,
    },
    CpuSota100xContractCaseCountMismatch {
        path: String,
        status_contract_cases: u64,
        artifact_contract_cases: u64,
    },
    CpuSota100xPassingCaseCountMismatch {
        path: String,
        status_passing_cases: u64,
        artifact_passing_cases: u64,
    },
    MissingRequestedCase {
        path: String,
        requested_case_id: String,
    },
}

pub(crate) fn source_fingerprint_issues(source_fingerprint: &str) -> Vec<SourceFingerprintIssue> {
    let Some(rest) = source_fingerprint.strip_prefix("git:") else {
        return Vec::new();
    };
    let mut issues = Vec::new();
    if rest.contains(":dirty=unknown") {
        issues.push(SourceFingerprintIssue::DirtyUnknownState {
            source_fingerprint: source_fingerprint.to_string(),
        });
    }
    let Some(dirty_offset) = rest.find(":dirty=true") else {
        return issues;
    };
    let after_dirty = &rest[dirty_offset + ":dirty=true".len()..];
    let Some(worktree) = after_dirty.strip_prefix(":worktree=") else {
        issues.push(SourceFingerprintIssue::DirtyMissingWorktree {
            source_fingerprint: source_fingerprint.to_string(),
        });
        return issues;
    };
    if worktree == "unknown" {
        issues.push(SourceFingerprintIssue::DirtyUnknownWorktree {
            source_fingerprint: source_fingerprint.to_string(),
        });
    } else if !is_blake3_hex_digest(worktree) {
        issues.push(SourceFingerprintIssue::DirtyInvalidWorktree {
            source_fingerprint: source_fingerprint.to_string(),
            worktree: worktree.to_string(),
        });
    }
    issues
}

pub(crate) fn source_fingerprint_freshness_issues(
    source_fingerprint: &str,
    current_source_fingerprint: &str,
) -> Vec<SourceFingerprintFreshnessIssue> {
    if source_fingerprint == current_source_fingerprint {
        Vec::new()
    } else {
        vec![SourceFingerprintFreshnessIssue::Mismatch {
            source_fingerprint: source_fingerprint.to_string(),
            current_source_fingerprint: current_source_fingerprint.to_string(),
        }]
    }
}

pub(crate) fn report_freshness_fingerprint(report: &Value) -> Option<(&'static str, &str)> {
    for field in ["source_tree_fingerprint", "source_fingerprint"] {
        if let Some(value) = report
            .get(field)
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
        {
            return Some((field, value));
        }
    }
    None
}

pub(crate) fn current_freshness_fingerprint_for_report(
    path: &Path,
    report: &Value,
) -> Option<String> {
    if report
        .get("source_tree_fingerprint")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
    {
        return Some(current_source_tree_fingerprint_for_evidence_path(path)?);
    }
    current_source_fingerprint_for_evidence_path(path)
}

pub(crate) fn current_source_tree_fingerprint_for_evidence_path(path: &Path) -> Option<String> {
    let workspace_root = workspace_root_for_evidence_path(path)?;
    Some(current_source_tree_fingerprint_at(&workspace_root))
}

pub(crate) fn current_source_fingerprint_for_evidence_path(path: &Path) -> Option<String> {
    let workspace_root = workspace_root_for_evidence_path(path)?;
    Some(current_source_fingerprint_at(&workspace_root))
}

fn current_source_fingerprint_at(workspace_root: &Path) -> String {
    let key = workspace_root
        .canonicalize()
        .unwrap_or_else(|_| workspace_root.to_path_buf());
    let cache = CURRENT_SOURCE_FINGERPRINTS.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Ok(cache) = cache.lock() {
        if let Some(fingerprint) = cache.get(&key) {
            return fingerprint.clone();
        }
    }

    let git = vyre_bench::probes::capture_git_info_at(workspace_root);
    let fingerprint = vyre_bench::probes::source_fingerprint(&git);
    if let Ok(mut cache) = cache.lock() {
        cache.insert(key, fingerprint.clone());
    }
    fingerprint
}

fn current_source_tree_fingerprint_at(workspace_root: &Path) -> String {
    let key = workspace_root
        .canonicalize()
        .unwrap_or_else(|_| workspace_root.to_path_buf());
    let cache = CURRENT_SOURCE_TREE_FINGERPRINTS.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Ok(cache) = cache.lock() {
        if let Some(fingerprint) = cache.get(&key) {
            return fingerprint.clone();
        }
    }

    let fingerprint = vyre_bench::probes::source_tree_fingerprint_at(workspace_root);
    if let Ok(mut cache) = cache.lock() {
        cache.insert(key, fingerprint.clone());
    }
    fingerprint
}

fn workspace_root_for_evidence_path(path: &Path) -> Option<PathBuf> {
    let mut cursor = if path.is_dir() { path } else { path.parent()? };
    loop {
        if cursor.join("Cargo.toml").is_file() && cursor.join("release").is_dir() {
            return Some(cursor.to_path_buf());
        }
        cursor = cursor.parent()?;
    }
}

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
        match status.get(field).and_then(non_empty_str) {
            None => issues.push(BackendSuiteArtifactStatusIssue::MissingField {
                path: path.clone(),
                field,
            }),
            Some(status_value) if status_value != artifact_value => {
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
        artifact_cpu_sota_100x_contract_counts(artifact_report);
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
        let contains_requested_case = artifact_report
            .get("cases")
            .and_then(Value::as_array)
            .is_some_and(|cases| {
                cases
                    .iter()
                    .any(|case| case.get("id").and_then(Value::as_str) == Some(requested_case_id))
            });
        if !contains_requested_case {
            issues.push(BackendSuiteArtifactStatusIssue::MissingRequestedCase {
                path,
                requested_case_id: requested_case_id.to_string(),
            });
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

fn backend_suite_string_artifact_fields(artifact_report: &Value) -> Vec<(&'static str, String)> {
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
        .filter_map(|(field, value)| value.map(|value| (field, value)))
        .collect()
}

fn artifact_environment<'a>(artifact_report: &'a Value) -> Option<&'a Value> {
    artifact_report.get("environment")
}

fn artifact_environment_str(artifact_report: &Value, field: &str) -> Option<String> {
    artifact_environment(artifact_report)?
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn artifact_environment_host_cpu_model(artifact_report: &Value) -> Option<String> {
    let environment = artifact_environment(artifact_report)?;
    environment
        .get("host_cpu_model")
        .or_else(|| environment.get("cpu_model"))
        .or_else(|| environment.get("host_cpu"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn artifact_environment_first_gpu<'a>(artifact_report: &'a Value) -> Option<&'a Value> {
    artifact_environment(artifact_report)?
        .get("gpu_devices")
        .and_then(Value::as_array)
        .and_then(|devices| devices.first())
}

fn artifact_environment_first_gpu_str(artifact_report: &Value, field: &str) -> Option<String> {
    artifact_environment_first_gpu(artifact_report)?
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn artifact_environment_first_gpu_u64(artifact_report: &Value, field: &str) -> Option<u64> {
    artifact_environment_first_gpu(artifact_report)?
        .get(field)
        .and_then(Value::as_u64)
}

fn artifact_nonmatching_case_backend_count(status: &Value, artifact_report: &Value) -> Option<u64> {
    let expected_backend = status
        .get("selected_backend")
        .and_then(non_empty_str)
        .or_else(|| {
            artifact_report
                .get("selected_backend")
                .and_then(non_empty_str)
        })?;
    let cases = artifact_report.get("cases").and_then(Value::as_array)?;
    Some(
        cases
            .iter()
            .filter(|case| case.get("backend_id").and_then(Value::as_str) != Some(expected_backend))
            .count() as u64,
    )
}

fn artifact_case_failed_count(artifact_report: &Value) -> Option<u64> {
    let cases = artifact_report.get("cases").and_then(Value::as_array)?;
    Some(
        cases
            .iter()
            .filter(|case| !benchmark_case_passes_summary_evidence(case))
            .count() as u64,
    )
}

fn artifact_case_passed_count(artifact_report: &Value) -> Option<u64> {
    let cases = artifact_report.get("cases").and_then(Value::as_array)?;
    Some(
        cases
            .iter()
            .filter(|case| benchmark_case_passes_summary_evidence(case))
            .count() as u64,
    )
}

fn artifact_min_metric_samples(artifact_report: &Value, metric_name: &str) -> Option<u64> {
    let cases = artifact_report.get("cases").and_then(Value::as_array)?;
    if cases.is_empty() {
        return None;
    }
    let mut seen_metric = false;
    let min = cases
        .iter()
        .map(|case| {
            let metric = case
                .get("metrics")
                .and_then(|metrics| metrics.get(metric_name));
            if metric.is_some() {
                seen_metric = true;
            }
            metric
                .and_then(|metric| metric.get("samples"))
                .and_then(Value::as_u64)
                .unwrap_or(0)
        })
        .min()
        .unwrap_or(0);
    seen_metric.then_some(min)
}

fn artifact_min_metric_percentile(
    artifact_report: &Value,
    metric_name: &str,
    percentile: &str,
) -> Option<u64> {
    let cases = artifact_report.get("cases").and_then(Value::as_array)?;
    if cases.is_empty() {
        return None;
    }
    let mut seen_metric = false;
    let min = cases
        .iter()
        .map(|case| {
            let metric = case
                .get("metrics")
                .and_then(|metrics| metrics.get(metric_name));
            if metric.is_some() {
                seen_metric = true;
            }
            metric
                .and_then(|metric| metric.get(percentile))
                .and_then(nonnegative_json_number_as_u64)
                .unwrap_or(0)
        })
        .min()
        .unwrap_or(0);
    seen_metric.then_some(min)
}

fn nonnegative_json_number_as_u64(value: &Value) -> Option<u64> {
    value.as_u64().or_else(|| {
        value
            .as_f64()
            .filter(|value| *value >= 0.0)
            .map(|value| value as u64)
    })
}

fn artifact_cpu_sota_100x_contract_counts(artifact_report: &Value) -> (u64, u64) {
    let report_backend = artifact_report
        .get("selected_backend")
        .and_then(Value::as_str);
    let Some(cases) = artifact_report.get("cases").and_then(Value::as_array) else {
        return (0, 0);
    };
    cases
        .iter()
        .fold((0, 0), |(contract_count, passing_count), case| {
            let case_backend = case
                .get("backend_id")
                .and_then(Value::as_str)
                .or(report_backend);
            if !case_has_cpu_sota_contract(case, case_backend, 100.0) {
                return (contract_count, passing_count);
            }
            let contract_passed = case
                .get("performance")
                .and_then(|performance| performance.get("contract_passed"))
                .and_then(Value::as_bool)
                == Some(true);
            let speedup_passed = case
                .get("performance")
                .and_then(|performance| performance.get("speedup_x"))
                .and_then(Value::as_f64)
                .is_some_and(|speedup| speedup >= 100.0);
            (
                contract_count + 1,
                passing_count
                    + u64::from(
                        benchmark_case_passes_summary_evidence(case)
                            && contract_passed
                            && speedup_passed,
                    ),
            )
        })
}

fn case_has_cpu_sota_contract(
    case: &Value,
    backend_id: Option<&str>,
    required_speedup: f64,
) -> bool {
    case.get("contract")
        .and_then(|contract| contract.get("baselines"))
        .and_then(Value::as_array)
        .is_some_and(|baselines| {
            baselines.iter().any(|baseline| {
                baseline.get("class").and_then(Value::as_str) == Some("CpuSota")
                    && baseline
                        .get("min_speedup_x")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0)
                        >= required_speedup
                    && baseline_applies_to_backend(baseline, backend_id)
            })
        })
}

pub(crate) fn baseline_applies_to_backend(baseline: &Value, backend_id: Option<&str>) -> bool {
    let Some(backend_ids) = baseline.get("backend_ids").and_then(Value::as_array) else {
        return true;
    };
    if backend_ids.is_empty() {
        return true;
    }
    let Some(backend_id) = backend_id else {
        return false;
    };
    backend_ids
        .iter()
        .any(|candidate| candidate.as_str() == Some(backend_id))
}

pub(crate) fn backend_suite_inventory_issues(suite: &Value) -> Vec<BackendSuiteInventoryIssue> {
    let artifact_count = suite_array_len(suite, "artifacts");
    let status_count = suite_array_len(suite, "artifact_statuses");
    let artifact_counts = suite_artifact_path_counts(suite);
    let status_counts = suite_status_path_counts(suite);
    let artifact_paths = artifact_counts.keys().cloned().collect::<BTreeSet<_>>();
    let status_paths = status_counts.keys().cloned().collect::<BTreeSet<_>>();
    let mut issues = Vec::new();

    if artifact_count != status_count {
        issues.push(BackendSuiteInventoryIssue::CountMismatch {
            artifact_count,
            status_count,
        });
    }
    for (path, count) in artifact_counts {
        if count > 1 {
            issues.push(BackendSuiteInventoryIssue::DuplicateArtifact { path });
        }
    }
    for (path, count) in status_counts {
        if count > 1 {
            issues.push(BackendSuiteInventoryIssue::DuplicateStatus { path });
        }
    }
    for path in artifact_paths.difference(&status_paths) {
        issues.push(BackendSuiteInventoryIssue::MissingStatus { path: path.clone() });
    }
    for path in status_paths.difference(&artifact_paths) {
        issues.push(BackendSuiteInventoryIssue::MissingArtifact { path: path.clone() });
    }
    issues
}

pub(crate) fn backend_suite_parity_issues(
    cuda_suite: &Value,
    wgpu_suite: &Value,
) -> Vec<BackendSuiteParityIssue> {
    let cuda_count = suite_artifact_status_count(cuda_suite);
    let wgpu_count = suite_artifact_status_count(wgpu_suite);
    let cuda_pairs = suite_family_case_pairs(cuda_suite);
    let wgpu_pairs = suite_family_case_pairs(wgpu_suite);
    let mut issues = Vec::new();
    if cuda_count != wgpu_count || cuda_pairs.len() != wgpu_pairs.len() {
        issues.push(BackendSuiteParityIssue::CountMismatch {
            cuda_count,
            wgpu_count,
        });
    }
    for (family_id, requested_case_id) in cuda_pairs.difference(&wgpu_pairs) {
        issues.push(BackendSuiteParityIssue::MissingWgpuPair {
            family_id: family_id.clone(),
            requested_case_id: requested_case_id.clone(),
        });
    }
    for (family_id, requested_case_id) in wgpu_pairs.difference(&cuda_pairs) {
        issues.push(BackendSuiteParityIssue::MissingCudaPair {
            family_id: family_id.clone(),
            requested_case_id: requested_case_id.clone(),
        });
    }
    issues
}

pub(crate) fn expected_backend_for_suite_evidence(evidence: &str) -> Option<&'static str> {
    if evidence.ends_with("cuda-release-suite.json") {
        Some("cuda")
    } else if evidence.ends_with("wgpu-fallback-suite.json") {
        Some("wgpu")
    } else {
        None
    }
}

pub(crate) fn backend_suite_backend_issue(
    suite: &Value,
    expected_backend: &str,
) -> Option<BackendSuiteBackendIssue> {
    match suite.get("backend").and_then(non_empty_str) {
        None => Some(BackendSuiteBackendIssue::Missing {
            expected_backend: expected_backend.to_string(),
        }),
        Some(actual_backend) if actual_backend != expected_backend => {
            Some(BackendSuiteBackendIssue::Mismatch {
                expected_backend: expected_backend.to_string(),
                actual_backend: actual_backend.to_string(),
            })
        }
        Some(_) => None,
    }
}

pub(crate) fn backend_consistency_issues(report: &Value) -> Vec<BackendConsistencyIssue> {
    let Some(expected_backend) = report
        .get("selected_backend")
        .and_then(Value::as_str)
        .filter(|backend| !backend.trim().is_empty())
    else {
        return Vec::new();
    };
    let Some(cases) = report.get("cases").and_then(Value::as_array) else {
        return Vec::new();
    };

    cases
        .iter()
        .filter_map(|case| {
            let case_id = case_id(case);
            match case
                .get("backend_id")
                .and_then(Value::as_str)
                .filter(|backend| !backend.trim().is_empty())
            {
                Some(actual_backend) if actual_backend == expected_backend => None,
                Some(actual_backend) => Some(BackendConsistencyIssue::CaseBackendMismatch {
                    case_id,
                    expected_backend: expected_backend.to_string(),
                    actual_backend: actual_backend.to_string(),
                }),
                None => Some(BackendConsistencyIssue::MissingCaseBackend {
                    case_id,
                    expected_backend: expected_backend.to_string(),
                }),
            }
        })
        .collect()
}

pub(crate) fn contract_backend_issues(report: &Value) -> Vec<ContractBackendIssue> {
    let report_backend = report.get("selected_backend").and_then(non_empty_str);
    let Some(cases) = report.get("cases").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut issues = Vec::new();
    for case in cases {
        let case_id = case_id(case);
        let Some(backend_id) = case
            .get("backend_id")
            .and_then(non_empty_str)
            .or(report_backend)
        else {
            continue;
        };
        let Some(contract) = case.get("contract").filter(|contract| !contract.is_null()) else {
            continue;
        };
        let Some(baselines) = contract.get("baselines").and_then(Value::as_array) else {
            issues.push(ContractBackendIssue::MissingBaselines {
                case_id,
                backend_id: backend_id.to_string(),
            });
            continue;
        };
        if baselines.is_empty() {
            issues.push(ContractBackendIssue::MissingBaselines {
                case_id,
                backend_id: backend_id.to_string(),
            });
            continue;
        }
        let applies = baselines
            .iter()
            .any(|baseline| baseline_applies_to_backend(baseline, Some(backend_id)));
        if !applies {
            issues.push(ContractBackendIssue::NoApplicableBaseline {
                case_id,
                backend_id: backend_id.to_string(),
            });
        }
    }
    issues
}

pub(crate) fn cuda_telemetry_label_issues(report: &Value) -> Vec<CudaTelemetryLabelIssue> {
    if report.get("selected_backend").and_then(Value::as_str) != Some("cuda") {
        return Vec::new();
    }
    let Some(cases) = report.get("cases").and_then(Value::as_array) else {
        return Vec::new();
    };

    const CHECKS: &[(&str, &[&str])] = &[
        (
            "cuda-ptx-source-cache",
            &[
                "cuda_ptx_source_cache_entries",
                "cuda_ptx_source_cache_hits",
                "cuda_ptx_source_cache_misses",
            ],
        ),
        ("cuda-graph-replay", &["cuda_graph_launches"]),
        (
            "cuda-graph-materialized-output-cache",
            &["cuda_graph_materialized_cache_hits"],
        ),
        (
            "cuda-transfer-operation-telemetry",
            &[
                "cuda_host_upload_operations",
                "cuda_device_readback_operations",
            ],
        ),
    ];

    cases
        .iter()
        .flat_map(|case| {
            let metrics = case.get("metrics").and_then(Value::as_object);
            let case_id = case_id(case);
            CHECKS.iter().filter_map(move |(label, counters)| {
                let counters_active =
                    metric_value_any(metrics, counters).is_some_and(|value| value > 0.0);
                let label_present = optimization_passes_contain(case, label);
                match (counters_active, label_present) {
                    (true, false) => Some(CudaTelemetryLabelIssue::MissingLabel {
                        case_id: case_id.clone(),
                        label,
                    }),
                    (false, true) => Some(CudaTelemetryLabelIssue::LabelWithoutCounters {
                        case_id: case_id.clone(),
                        label,
                    }),
                    _ => None,
                }
            })
        })
        .collect()
}

pub(crate) fn launch_plan_label_issues(
    case: &Value,
    metrics: Option<&Map<String, Value>>,
) -> Vec<LaunchPlanLabelIssue> {
    let Some(launch_count) =
        metric_value_any(metrics, &["kernel_launches", "launch_count", "launches"])
    else {
        return Vec::new();
    };
    let has_single = optimization_passes_contain(case, "single-dispatch-launch-plan");
    let has_multi = optimization_passes_contain(case, "multi-dispatch-launch-plan");
    let mut issues = Vec::new();
    if launch_count == 1.0 {
        if !has_single {
            issues.push(LaunchPlanLabelIssue::MissingSingle);
        }
        if has_multi {
            issues.push(LaunchPlanLabelIssue::SingleHasMulti);
        }
    } else if launch_count > 1.0 {
        if !has_multi {
            issues.push(LaunchPlanLabelIssue::MissingMulti { launch_count });
        }
        if has_single {
            issues.push(LaunchPlanLabelIssue::MultiHasSingle { launch_count });
        }
    }
    issues
}

fn metric_value_any(metrics: Option<&Map<String, Value>>, fields: &[&str]) -> Option<f64> {
    let metrics = metrics?;
    fields
        .iter()
        .filter_map(|field| metrics.get(*field))
        .find_map(metric_value)
}

fn metric_value(metric: &Value) -> Option<f64> {
    metric
        .get("p50")
        .and_then(Value::as_f64)
        .or_else(|| {
            metric
                .get("p50")
                .and_then(Value::as_u64)
                .map(|value| value as f64)
        })
        .or_else(|| metric.as_f64())
        .or_else(|| metric.as_u64().map(|value| value as f64))
}

fn optimization_passes_contain(case: &Value, expected: &str) -> bool {
    ["optimization_passes_applied", "optimization_passes"]
        .iter()
        .any(|field| {
            case.get(*field)
                .and_then(Value::as_array)
                .is_some_and(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .any(|item| item == expected)
                })
        })
}

fn case_id(case: &Value) -> String {
    case.get("id")
        .and_then(Value::as_str)
        .unwrap_or("<unknown>")
        .to_string()
}

fn suite_family_case_pairs(suite: &Value) -> BTreeSet<(String, String)> {
    suite
        .get("artifact_statuses")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|status| {
            let family_id = status
                .get("family_id")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())?;
            let requested_case_id = status
                .get("requested_case_id")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())?;
            Some((family_id.to_string(), requested_case_id.to_string()))
        })
        .collect()
}

fn suite_artifact_status_count(suite: &Value) -> usize {
    suite_array_len(suite, "artifact_statuses")
}

fn suite_array_len(suite: &Value, field: &str) -> usize {
    suite
        .get(field)
        .and_then(Value::as_array)
        .map_or(0, Vec::len)
}

fn suite_artifact_path_counts(suite: &Value) -> BTreeMap<String, usize> {
    suite
        .get("artifacts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(non_empty_str)
        .fold(BTreeMap::new(), |mut counts, path| {
            *counts.entry(path.to_string()).or_default() += 1;
            counts
        })
}

fn suite_status_path_counts(suite: &Value) -> BTreeMap<String, usize> {
    suite
        .get("artifact_statuses")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|status| status.get("path").and_then(non_empty_str))
        .fold(BTreeMap::new(), |mut counts, path| {
            *counts.entry(path.to_string()).or_default() += 1;
            counts
        })
}

fn non_empty_str(value: &Value) -> Option<&str> {
    value.as_str().filter(|value| !value.trim().is_empty())
}

fn is_blake3_hex_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_case_summary_rejects_pass_status_with_invalid_correctness() {
        let case = serde_json::json!({
            "id": "release.condition_eval.1m",
            "status": "pass",
            "correctness": {
                "Invalid": {
                    "reason": "dispatch output mismatch at row 17"
                }
            },
            "performance": {"contract_passed": true}
        });

        assert_eq!(
            benchmark_case_failure_reason(&case),
            Some("dispatch output mismatch at row 17".to_string()),
            "Fix: explicit invalid correctness evidence must not be hidden by a contradictory pass status."
        );
    }

    #[test]
    fn failed_case_summary_rejects_pass_status_with_performance_violations() {
        let case = serde_json::json!({
            "id": "release.condition_eval.1m",
            "status": "pass",
            "correctness": {"Valid": {}},
            "performance": {
                "contract_passed": true,
                "violations": [
                    "speedup below CUDA release floor",
                    "p95 latency regression"
                ]
            }
        });

        assert_eq!(
            benchmark_case_failure_reason(&case),
            Some("speedup below CUDA release floor; p95 latency regression".to_string()),
            "Fix: performance violation evidence must stay visible even when status is pass."
        );
    }

    #[test]
    fn failed_case_summary_reports_contract_failed_pass_as_contract_failure() {
        let case = serde_json::json!({
            "id": "release.condition_eval.1m",
            "status": "pass",
            "correctness": {"Valid": {}},
            "performance": {"contract_passed": false}
        });

        assert_eq!(
            benchmark_case_failure_reason(&case),
            Some("performance contract failed".to_string()),
            "Fix: contradictory pass status must not hide contract_passed=false evidence."
        );
    }

    #[test]
    fn failed_case_summary_rejects_missing_pass_status() {
        let case = serde_json::json!({
            "id": "release.condition_eval.1m",
            "correctness": {"Valid": {}},
            "performance": {"contract_passed": true}
        });

        assert_eq!(
            benchmark_case_failure_reason(&case),
            Some("missing pass status".to_string()),
            "Fix: benchmark evidence must require an explicit pass status before a case can prove release performance."
        );
    }

    #[test]
    fn benchmark_report_summary_mismatch_reports_total_pass_fail_drift() {
        let report = serde_json::json!({
            "summary": {"total_cases": 2, "passed": 0, "failed": 0},
            "cases": [
                {
                    "id": "release.condition_eval.1m",
                    "status": "pass",
                    "correctness": {"Valid": {}},
                    "performance": {"contract_passed": true}
                }
            ]
        });

        assert_eq!(
            benchmark_report_summary_case_evidence_mismatch(&report),
            Some(
                "summary total/pass/fail (Some(2)/Some(0)/Some(0)) contradicts case evidence (1/1/0)"
                    .to_string()
            ),
            "Fix: benchmark summary validation must expose stale total_cases and passed counts, not only summary.failed."
        );
        assert!(
            !benchmark_report_summary_matches_case_evidence(&report),
            "Fix: stale benchmark summaries must not be accepted by boolean reuse/gate predicates."
        );
    }

    #[test]
    fn launch_plan_issues_reject_single_label_for_multi_launch_count() {
        let case = serde_json::json!({
            "optimization_passes_applied": ["single-dispatch-launch-plan"],
            "metrics": {
                "kernel_launches": {"p50": 4, "samples": 30}
            }
        });
        let issues =
            launch_plan_label_issues(&case, case.get("metrics").and_then(Value::as_object));

        assert_eq!(
            issues,
            vec![
                LaunchPlanLabelIssue::MissingMulti { launch_count: 4.0 },
                LaunchPlanLabelIssue::MultiHasSingle { launch_count: 4.0 },
            ],
            "Fix: multi-launch evidence must require the multi label and reject the single label."
        );
    }

    #[test]
    fn launch_plan_issues_accept_matching_single_and_multi_counts() {
        for case in [
            serde_json::json!({
                "optimization_passes_applied": ["single-dispatch-launch-plan"],
                "metrics": {"kernel_launches": {"p50": 1, "samples": 30}}
            }),
            serde_json::json!({
                "optimization_passes_applied": ["multi-dispatch-launch-plan"],
                "metrics": {"launch_count": 4}
            }),
        ] {
            let issues =
                launch_plan_label_issues(&case, case.get("metrics").and_then(Value::as_object));
            assert!(
                issues.is_empty(),
                "Fix: matching launch-plan label/count evidence should pass: {issues:?}"
            );
        }
    }

    #[test]
    fn source_fingerprint_rejects_dirty_without_worktree_digest() {
        assert_eq!(
            source_fingerprint_issues("git:abc123:dirty=true"),
            vec![SourceFingerprintIssue::DirtyMissingWorktree {
                source_fingerprint: "git:abc123:dirty=true".to_string(),
            }],
            "Fix: dirty benchmark evidence must not collapse distinct dirty worktrees."
        );
    }

    #[test]
    fn source_fingerprint_rejects_unknown_dirty_state_and_digest() {
        assert_eq!(
            source_fingerprint_issues("git:abc123:dirty=unknown"),
            vec![SourceFingerprintIssue::DirtyUnknownState {
                source_fingerprint: "git:abc123:dirty=unknown".to_string(),
            }],
            "Fix: release evidence must fail closed when git status provenance is unavailable."
        );
        assert_eq!(
            source_fingerprint_issues("git:abc123:dirty=true:worktree=unknown"),
            vec![SourceFingerprintIssue::DirtyUnknownWorktree {
                source_fingerprint: "git:abc123:dirty=true:worktree=unknown".to_string(),
            }],
            "Fix: dirty source fingerprints must carry the actual worktree digest."
        );
    }

    #[test]
    fn source_fingerprint_accepts_clean_and_precise_dirty_git_fingerprints() {
        assert!(source_fingerprint_issues("git:abc123:dirty=false").is_empty());
        assert!(
            source_fingerprint_issues(&format!(
                "git:abc123:dirty=true:worktree={}",
                "a".repeat(64)
            ))
            .is_empty(),
            "Fix: precise dirty source fingerprints must remain valid release evidence."
        );
    }

    #[test]
    fn source_fingerprint_freshness_rejects_non_current_evidence() {
        assert_eq!(
            source_fingerprint_freshness_issues(
                "git:old:dirty=false",
                "git:new:dirty=true:worktree=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            ),
            vec![SourceFingerprintFreshnessIssue::Mismatch {
                source_fingerprint: "git:old:dirty=false".to_string(),
                current_source_fingerprint:
                    "git:new:dirty=true:worktree=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                        .to_string(),
            }],
            "Fix: release evidence must be regenerated after source changes, not carried forward by matching old artifact metadata."
        );
    }

    #[test]
    fn source_fingerprint_freshness_accepts_current_evidence() {
        let fingerprint =
            "git:abc:dirty=true:worktree=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

        assert!(
            source_fingerprint_freshness_issues(fingerprint, fingerprint).is_empty(),
            "Fix: current source evidence should not be rejected by the freshness gate."
        );
    }

    #[test]
    fn current_source_fingerprint_resolves_from_release_evidence_path() {
        let dir = tempfile::TempDir::new()
            .expect("Fix: create temp workspace for evidence source fingerprint test.");
        std::fs::write(dir.path().join("Cargo.toml"), "[workspace]\n")
            .expect("Fix: write temp workspace manifest.");
        std::fs::create_dir_all(dir.path().join("release/evidence/benchmarks"))
            .expect("Fix: create temp release evidence directory.");
        let evidence = dir.path().join("release/evidence/benchmarks/workload.json");

        let fingerprint = current_source_fingerprint_for_evidence_path(&evidence)
            .expect("Fix: resolve workspace source fingerprint from nested release evidence path.");

        assert!(
            fingerprint.starts_with("crate:"),
            "Fix: non-git test workspaces should still produce deterministic crate source provenance, got {fingerprint}."
        );
    }

    #[test]
    fn report_freshness_fingerprint_prefers_source_tree_scope() {
        let report = serde_json::json!({
            "source_fingerprint": "git:abc:dirty=false",
            "source_tree_fingerprint": "source-tree-v1:def",
        });

        assert_eq!(
            report_freshness_fingerprint(&report),
            Some(("source_tree_fingerprint", "source-tree-v1:def")),
            "Fix: current-source gates must prefer evidence-stable source tree provenance over commit-shaped legacy provenance."
        );
    }

    #[test]
    fn source_fingerprint_rejects_malformed_dirty_worktree_digest() {
        assert_eq!(
            source_fingerprint_issues("git:abc123:dirty=true:worktree=not-a-digest"),
            vec![SourceFingerprintIssue::DirtyInvalidWorktree {
                source_fingerprint: "git:abc123:dirty=true:worktree=not-a-digest".to_string(),
                worktree: "not-a-digest".to_string(),
            }],
            "Fix: dirty source fingerprints must carry a stable 64-hex digest."
        );
    }

    #[test]
    fn benchmark_source_provenance_rejects_blank_source_artifacts() {
        let report = serde_json::json!({
            "source_artifacts": ["", "   ", null]
        });

        assert!(
            !benchmark_report_has_source_provenance(&report),
            "Fix: blank or non-string source_artifacts entries must not satisfy benchmark source provenance."
        );
    }

    #[test]
    fn benchmark_source_provenance_accepts_valid_artifact_or_commit() {
        assert!(
            benchmark_report_has_source_provenance(&serde_json::json!({
                "source_artifacts": ["release/evidence/benchmarks/cuda.json"]
            })),
            "Fix: a non-empty source artifact path must satisfy benchmark source provenance."
        );
        assert!(
            benchmark_report_has_source_provenance(&serde_json::json!({
                "git": {"commit": "abcdef"}
            })),
            "Fix: git commit provenance must satisfy benchmark source provenance."
        );
    }

    #[test]
    fn benchmark_source_artifact_count_ignores_blank_entries() {
        let report = serde_json::json!({
            "source_artifacts": [
                "",
                null,
                "release/evidence/benchmarks/cuda-a.json",
                "   ",
                "release/evidence/benchmarks/cuda-a.json",
                "release/evidence/benchmarks/cuda-b.json"
            ]
        });

        assert_eq!(
            benchmark_source_artifact_count(&report),
            2,
            "Fix: source_artifact counts must count only unique usable non-empty string entries."
        );
    }

    #[test]
    fn backend_consistency_rejects_case_backend_drift() {
        let report = serde_json::json!({
            "selected_backend": "cuda",
            "cases": [
                {"id": "same", "backend_id": "cuda"},
                {"id": "fallback", "backend_id": "wgpu"},
                {"id": "missing"}
            ]
        });

        assert_eq!(
            backend_consistency_issues(&report),
            vec![
                BackendConsistencyIssue::CaseBackendMismatch {
                    case_id: "fallback".to_string(),
                    expected_backend: "cuda".to_string(),
                    actual_backend: "wgpu".to_string(),
                },
                BackendConsistencyIssue::MissingCaseBackend {
                    case_id: "missing".to_string(),
                    expected_backend: "cuda".to_string(),
                },
            ],
            "Fix: report-level backend selection must be proven by every benchmark case."
        );
    }

    #[test]
    fn backend_consistency_allows_non_benchmark_manifest_without_selected_backend() {
        let manifest = serde_json::json!({
            "cases": [
                {"id": "manifest-row"}
            ]
        });

        assert!(
            backend_consistency_issues(&manifest).is_empty(),
            "Fix: backend consistency applies to benchmark reports that declare selected_backend."
        );
    }

    #[test]
    fn contract_backend_issues_reject_cuda_only_contract_on_wgpu_case() {
        let report = serde_json::json!({
            "selected_backend": "wgpu",
            "cases": [
                {
                    "id": "release.condition_eval.1m",
                    "backend_id": "wgpu",
                    "contract": {
                        "primitive": "condition eval",
                        "baselines": [
                            {"backend_ids": ["cuda"], "min_speedup_x": 100.0}
                        ]
                    }
                }
            ]
        });

        assert_eq!(
            contract_backend_issues(&report),
            vec![ContractBackendIssue::NoApplicableBaseline {
                case_id: "release.condition_eval.1m".to_string(),
                backend_id: "wgpu".to_string(),
            }],
            "Fix: WGPU benchmark evidence must not pass a CUDA-only performance contract by omission."
        );
    }

    #[test]
    fn contract_backend_issues_accept_backend_agnostic_contract() {
        let report = serde_json::json!({
            "selected_backend": "wgpu",
            "cases": [
                {
                    "id": "release.condition_eval.1m",
                    "backend_id": "wgpu",
                    "contract": {
                        "primitive": "condition eval",
                        "baselines": [
                            {"backend_ids": [], "min_speedup_x": 2.0}
                        ]
                    }
                }
            ]
        });

        assert!(
            contract_backend_issues(&report).is_empty(),
            "Fix: backend-agnostic contracts must remain valid for fallback backends."
        );
    }

    #[test]
    fn contract_backend_issues_reject_empty_baseline_list() {
        let report = serde_json::json!({
            "selected_backend": "cuda",
            "cases": [
                {
                    "id": "release.condition_eval.1m",
                    "backend_id": "cuda",
                    "contract": {
                        "primitive": "condition eval",
                        "baselines": []
                    }
                }
            ]
        });

        assert_eq!(
            contract_backend_issues(&report),
            vec![ContractBackendIssue::MissingBaselines {
                case_id: "release.condition_eval.1m".to_string(),
                backend_id: "cuda".to_string(),
            }],
            "Fix: a performance contract with no baselines must not prove release performance."
        );
    }

    #[test]
    fn cuda_telemetry_labels_track_active_counters() {
        let report = serde_json::json!({
            "selected_backend": "cuda",
            "cases": [
                {
                    "id": "active-unlabeled",
                    "metrics": {"cuda_ptx_source_cache_misses": {"p50": 1}},
                    "optimization_passes_applied": ["cuda-explicit-backend-selection"]
                },
                {
                    "id": "inactive-labeled",
                    "metrics": {
                        "cuda_ptx_source_cache_entries": {"p50": 0},
                        "cuda_ptx_source_cache_hits": {"p50": 0},
                        "cuda_ptx_source_cache_misses": {"p50": 0}
                    },
                    "optimization_passes_applied": ["cuda-ptx-source-cache"]
                },
                {
                    "id": "active-labeled",
                    "metrics": {"cuda_ptx_source_cache_hits": {"p50": 2}},
                    "optimization_passes_applied": ["cuda-ptx-source-cache"]
                },
                {
                    "id": "graph-unlabeled",
                    "metrics": {"cuda_graph_launches": {"p50": 3}},
                    "optimization_passes_applied": ["cuda-explicit-backend-selection"]
                },
                {
                    "id": "transfer-false-label",
                    "metrics": {
                        "cuda_host_upload_operations": {"p50": 0},
                        "cuda_device_readback_operations": {"p50": 0}
                    },
                    "optimization_passes_applied": ["cuda-transfer-operation-telemetry"]
                }
            ]
        });

        assert_eq!(
            cuda_telemetry_label_issues(&report),
            vec![
                CudaTelemetryLabelIssue::MissingLabel {
                    case_id: "active-unlabeled".to_string(),
                    label: "cuda-ptx-source-cache",
                },
                CudaTelemetryLabelIssue::LabelWithoutCounters {
                    case_id: "inactive-labeled".to_string(),
                    label: "cuda-ptx-source-cache",
                },
                CudaTelemetryLabelIssue::MissingLabel {
                    case_id: "graph-unlabeled".to_string(),
                    label: "cuda-graph-replay",
                },
                CudaTelemetryLabelIssue::LabelWithoutCounters {
                    case_id: "transfer-false-label".to_string(),
                    label: "cuda-transfer-operation-telemetry",
                },
            ],
            "Fix: CUDA release telemetry labels must match measured backend counters."
        );
    }

    #[test]
    fn backend_suite_parity_rejects_missing_family_case_pairs() {
        let cuda = serde_json::json!({
            "artifact_statuses": [
                {"family_id": "condition-eval", "requested_case_id": "release.condition_eval.1m"},
                {"family_id": "entropy-window", "requested_case_id": "release.entropy_window.1m"}
            ]
        });
        let wgpu = serde_json::json!({
            "artifact_statuses": [
                {"family_id": "condition-eval", "requested_case_id": "release.condition_eval.1m"},
                {"family_id": "ifds-witness", "requested_case_id": "release.ifds_witness.1m"}
            ]
        });

        assert_eq!(
            backend_suite_parity_issues(&cuda, &wgpu),
            vec![
                BackendSuiteParityIssue::MissingWgpuPair {
                    family_id: "entropy-window".to_string(),
                    requested_case_id: "release.entropy_window.1m".to_string(),
                },
                BackendSuiteParityIssue::MissingCudaPair {
                    family_id: "ifds-witness".to_string(),
                    requested_case_id: "release.ifds_witness.1m".to_string(),
                },
            ],
            "Fix: CUDA and WGPU release suites must cover the same family/case contract."
        );
    }

    #[test]
    fn backend_suite_backend_identity_comes_from_release_suite_name() {
        assert_eq!(
            expected_backend_for_suite_evidence(
                "release/evidence/benchmarks/cuda-release-suite.json"
            ),
            Some("cuda"),
            "Fix: CUDA suite filenames define the required backend identity."
        );
        assert_eq!(
            expected_backend_for_suite_evidence(
                "release/evidence/benchmarks/wgpu-fallback-suite.json"
            ),
            Some("wgpu"),
            "Fix: WGPU fallback suite filenames define the required backend identity."
        );
    }

    #[test]
    fn backend_suite_backend_identity_rejects_missing_or_mismatched_field() {
        assert_eq!(
            backend_suite_backend_issue(&serde_json::json!({}), "cuda"),
            Some(BackendSuiteBackendIssue::Missing {
                expected_backend: "cuda".to_string(),
            }),
            "Fix: suite backend identity must be explicit, not inferred from artifact rows."
        );
        assert_eq!(
            backend_suite_backend_issue(&serde_json::json!({"backend": "wgpu"}), "cuda"),
            Some(BackendSuiteBackendIssue::Mismatch {
                expected_backend: "cuda".to_string(),
                actual_backend: "wgpu".to_string(),
            }),
            "Fix: a CUDA release suite must not self-report a WGPU backend."
        );
        assert_eq!(
            backend_suite_backend_issue(&serde_json::json!({"backend": "cuda"}), "cuda"),
            None,
            "Fix: matching suite backend identity should pass."
        );
    }

    #[test]
    fn backend_suite_inventory_rejects_missing_cross_entries() {
        let suite = serde_json::json!({
            "artifacts": [
                "release/evidence/benchmarks/cuda/condition.json",
                "release/evidence/benchmarks/cuda/entropy.json"
            ],
            "artifact_statuses": [
                {"path": "release/evidence/benchmarks/cuda/condition.json"},
                {"path": "release/evidence/benchmarks/cuda/ifds.json"}
            ]
        });

        assert_eq!(
            backend_suite_inventory_issues(&suite),
            vec![
                BackendSuiteInventoryIssue::MissingStatus {
                    path: "release/evidence/benchmarks/cuda/entropy.json".to_string(),
                },
                BackendSuiteInventoryIssue::MissingArtifact {
                    path: "release/evidence/benchmarks/cuda/ifds.json".to_string(),
                },
            ],
            "Fix: suite artifacts and artifact_statuses must describe the same file set."
        );
    }

    #[test]
    fn backend_suite_inventory_rejects_duplicate_paths_and_count_drift() {
        let suite = serde_json::json!({
            "artifacts": [
                "release/evidence/benchmarks/cuda/condition.json",
                "release/evidence/benchmarks/cuda/condition.json"
            ],
            "artifact_statuses": [
                {"path": "release/evidence/benchmarks/cuda/condition.json"}
            ]
        });

        assert_eq!(
            backend_suite_inventory_issues(&suite),
            vec![
                BackendSuiteInventoryIssue::CountMismatch {
                    artifact_count: 2,
                    status_count: 1,
                },
                BackendSuiteInventoryIssue::DuplicateArtifact {
                    path: "release/evidence/benchmarks/cuda/condition.json".to_string(),
                },
            ],
            "Fix: duplicate suite inventory entries must not prove artifact coverage."
        );
    }

    #[test]
    fn backend_suite_artifact_status_rejects_stale_artifact_metadata() {
        let status = serde_json::json!({
            "path": "release/evidence/benchmarks/workload-01-condition-eval.json",
            "source_fingerprint": "git:old:dirty=false",
            "source_tree_fingerprint": "source-tree-v1:old",
            "selected_backend": "cuda",
            "case_count": 2,
            "failed_count": 0,
            "nonmatching_case_backend_count": 0,
            "requested_case_id": "release.condition_eval.1m"
        });
        let artifact = serde_json::json!({
            "source_fingerprint": "git:new:dirty=false",
            "source_tree_fingerprint": "source-tree-v1:new",
            "selected_backend": "wgpu",
            "summary": {"total_cases": 1, "passed": 0, "failed": 1},
            "cases": [
                {"id": "release.other.1m"}
            ]
        });

        assert_eq!(
            backend_suite_artifact_status_issues(&status, &artifact),
            vec![
                BackendSuiteArtifactStatusIssue::SourceFingerprintMismatch {
                    path: "release/evidence/benchmarks/workload-01-condition-eval.json".to_string(),
                    status_source_fingerprint: "git:old:dirty=false".to_string(),
                    artifact_source_fingerprint: "git:new:dirty=false".to_string(),
                },
                BackendSuiteArtifactStatusIssue::SourceTreeFingerprintMismatch {
                    path: "release/evidence/benchmarks/workload-01-condition-eval.json".to_string(),
                    status_source_tree_fingerprint: "source-tree-v1:old".to_string(),
                    artifact_source_tree_fingerprint: "source-tree-v1:new".to_string(),
                },
                BackendSuiteArtifactStatusIssue::SelectedBackendMismatch {
                    path: "release/evidence/benchmarks/workload-01-condition-eval.json".to_string(),
                    status_selected_backend: "cuda".to_string(),
                    artifact_selected_backend: "wgpu".to_string(),
                },
                BackendSuiteArtifactStatusIssue::CaseCountMismatch {
                    path: "release/evidence/benchmarks/workload-01-condition-eval.json".to_string(),
                    status_case_count: 2,
                    artifact_case_count: 1,
                },
                BackendSuiteArtifactStatusIssue::NumericFieldMismatch {
                    path: "release/evidence/benchmarks/workload-01-condition-eval.json".to_string(),
                    field: "nonmatching_case_backend_count",
                    status_value: 0,
                    artifact_value: 1,
                },
                BackendSuiteArtifactStatusIssue::FailedCountMismatch {
                    path: "release/evidence/benchmarks/workload-01-condition-eval.json".to_string(),
                    status_failed_count: 0,
                    artifact_failed_count: 1,
                },
                BackendSuiteArtifactStatusIssue::MissingRequestedCase {
                    path: "release/evidence/benchmarks/workload-01-condition-eval.json".to_string(),
                    requested_case_id: "release.condition_eval.1m".to_string(),
                },
            ],
            "Fix: backend suite status rows must be proven against the listed artifact JSON."
        );
    }

    #[test]
    fn backend_suite_artifact_status_accepts_matching_metadata() {
        let status = serde_json::json!({
            "path": "release/evidence/benchmarks/workload-01-condition-eval.json",
            "source_fingerprint": "git:abc:dirty=false",
            "source_tree_fingerprint": "source-tree-v1:abc",
            "selected_backend": "cuda",
            "case_count": 1,
            "failed_count": 0,
            "nonmatching_case_backend_count": 0,
            "requested_case_id": "release.condition_eval.1m"
        });
        let artifact = serde_json::json!({
            "source_fingerprint": "git:abc:dirty=false",
            "source_tree_fingerprint": "source-tree-v1:abc",
            "selected_backend": "cuda",
            "summary": {"total_cases": 1, "passed": 1, "failed": 0},
            "cases": [
                {"id": "release.condition_eval.1m", "backend_id": "cuda", "status": "pass"}
            ]
        });

        assert!(
            backend_suite_artifact_status_issues(&status, &artifact).is_empty(),
            "Fix: matching suite status and artifact JSON should pass."
        );
    }

    #[test]
    fn backend_suite_artifact_status_accepts_float_metric_percentiles() {
        let status = serde_json::json!({
            "path": "release/evidence/benchmarks/workload-01-condition-eval.json",
            "selected_backend": "cuda",
            "case_count": 1,
            "failed_count": 0,
            "nonmatching_case_backend_count": 0,
            "requested_case_id": "release.condition_eval.1m",
            "min_wall_samples": 30,
            "min_wall_p50": 12,
            "min_wall_p95": 20,
            "min_wall_p99": 30
        });
        let artifact = serde_json::json!({
            "selected_backend": "cuda",
            "summary": {"total_cases": 1, "passed": 1, "failed": 0},
            "cases": [
                {
                    "id": "release.condition_eval.1m",
                    "backend_id": "cuda",
                    "status": "pass",
                    "metrics": {
                        "wall_ns": {"samples": 30, "p50": 12.75, "p95": 20.25, "p99": 30.875}
                    }
                }
            ]
        });

        assert!(
            backend_suite_artifact_status_issues(&status, &artifact).is_empty(),
            "Fix: backend suite status verification must parse benchmark float percentiles the same way suite generation does."
        );
    }

    #[test]
    fn backend_suite_artifact_status_rejects_backend_mismatch_counter_drift() {
        let status = serde_json::json!({
            "path": "release/evidence/benchmarks/workload-01-condition-eval.json",
            "selected_backend": "cuda",
            "case_count": 1,
            "failed_count": 0,
            "nonmatching_case_backend_count": 0,
            "requested_case_id": "release.condition_eval.1m"
        });
        let artifact = serde_json::json!({
            "selected_backend": "cuda",
            "summary": {"total_cases": 1, "passed": 1, "failed": 0},
            "cases": [
                {"id": "release.condition_eval.1m", "backend_id": "wgpu", "status": "pass"}
            ]
        });

        assert_eq!(
            backend_suite_artifact_status_issues(&status, &artifact),
            vec![BackendSuiteArtifactStatusIssue::NumericFieldMismatch {
                path: "release/evidence/benchmarks/workload-01-condition-eval.json".to_string(),
                field: "nonmatching_case_backend_count",
                status_value: 0,
                artifact_value: 1,
            }],
            "Fix: backend suite status rows must not hide case-level backend drift."
        );
    }

    #[test]
    fn backend_suite_artifact_status_rejects_summary_failed_count_hidden_by_pass_status() {
        let status = serde_json::json!({
            "path": "release/evidence/benchmarks/workload-01-condition-eval.json",
            "selected_backend": "cuda",
            "case_count": 1,
            "failed_count": 0,
            "nonmatching_case_backend_count": 0,
            "requested_case_id": "release.condition_eval.1m"
        });
        let artifact = serde_json::json!({
            "selected_backend": "cuda",
            "summary": {"total_cases": 1, "passed": 1, "failed": 0},
            "cases": [
                {
                    "id": "release.condition_eval.1m",
                    "backend_id": "cuda",
                    "status": "pass",
                    "correctness": {
                        "Invalid": {
                            "reason": "CUDA/WGPU output mismatch at row 17"
                        }
                    }
                }
            ]
        });

        assert_eq!(
            backend_suite_artifact_status_issues(&status, &artifact),
            vec![
                BackendSuiteArtifactStatusIssue::NumericFieldMismatch {
                    path: "release/evidence/benchmarks/workload-01-condition-eval.json"
                        .to_string(),
                    field: "summary.passed",
                    status_value: 1,
                    artifact_value: 0,
                },
                BackendSuiteArtifactStatusIssue::NumericFieldMismatch {
                    path: "release/evidence/benchmarks/workload-01-condition-eval.json"
                        .to_string(),
                    field: "summary.failed",
                    status_value: 0,
                    artifact_value: 1,
                },
                BackendSuiteArtifactStatusIssue::FailedCountMismatch {
                    path: "release/evidence/benchmarks/workload-01-condition-eval.json"
                        .to_string(),
                    status_failed_count: 0,
                    artifact_failed_count: 1,
                },
            ],
            "Fix: suite status must not trust summary.failed when case evidence exposes a contradictory benchmark failure."
        );
    }

    #[test]
    fn backend_suite_artifact_status_rejects_stale_artifact_summary_passed_count() {
        let status = serde_json::json!({
            "path": "release/evidence/benchmarks/workload-01-condition-eval.json",
            "selected_backend": "cuda",
            "case_count": 1,
            "failed_count": 0,
            "nonmatching_case_backend_count": 0,
            "requested_case_id": "release.condition_eval.1m"
        });
        let artifact = serde_json::json!({
            "selected_backend": "cuda",
            "summary": {"total_cases": 1, "passed": 1, "failed": 0},
            "cases": [
                {
                    "id": "release.condition_eval.1m",
                    "backend_id": "cuda",
                    "status": "pass",
                    "performance": {"contract_passed": false}
                }
            ]
        });

        assert_eq!(
            backend_suite_artifact_status_issues(&status, &artifact),
            vec![
                BackendSuiteArtifactStatusIssue::NumericFieldMismatch {
                    path: "release/evidence/benchmarks/workload-01-condition-eval.json"
                        .to_string(),
                    field: "summary.passed",
                    status_value: 1,
                    artifact_value: 0,
                },
                BackendSuiteArtifactStatusIssue::NumericFieldMismatch {
                    path: "release/evidence/benchmarks/workload-01-condition-eval.json"
                        .to_string(),
                    field: "summary.failed",
                    status_value: 0,
                    artifact_value: 1,
                },
                BackendSuiteArtifactStatusIssue::FailedCountMismatch {
                    path: "release/evidence/benchmarks/workload-01-condition-eval.json"
                        .to_string(),
                    status_failed_count: 0,
                    artifact_failed_count: 1,
                },
            ],
            "Fix: suite artifact validation must reject stale summary.passed counts derived without case contract evidence."
        );
    }

    #[test]
    fn backend_suite_artifact_status_rejects_stale_artifact_summary_total_cases() {
        let status = serde_json::json!({
            "path": "release/evidence/benchmarks/workload-01-condition-eval.json",
            "selected_backend": "cuda",
            "case_count": 1,
            "failed_count": 0,
            "nonmatching_case_backend_count": 0,
            "requested_case_id": "release.condition_eval.1m"
        });
        let artifact = serde_json::json!({
            "selected_backend": "cuda",
            "summary": {"total_cases": 2, "passed": 1, "failed": 0},
            "cases": [
                {
                    "id": "release.condition_eval.1m",
                    "backend_id": "cuda",
                    "status": "pass"
                }
            ]
        });

        assert_eq!(
            backend_suite_artifact_status_issues(&status, &artifact),
            vec![BackendSuiteArtifactStatusIssue::NumericFieldMismatch {
                path: "release/evidence/benchmarks/workload-01-condition-eval.json".to_string(),
                field: "summary.total_cases",
                status_value: 2,
                artifact_value: 1,
            }],
            "Fix: suite artifact validation must reject summary.total_cases drift even when status case_count matches the cases array."
        );
    }

    #[test]
    fn backend_suite_artifact_status_rejects_missing_artifact_summary_passed_count() {
        let status = serde_json::json!({
            "path": "release/evidence/benchmarks/workload-01-condition-eval.json",
            "selected_backend": "cuda",
            "case_count": 1,
            "failed_count": 0,
            "nonmatching_case_backend_count": 0,
            "requested_case_id": "release.condition_eval.1m"
        });
        let artifact = serde_json::json!({
            "selected_backend": "cuda",
            "summary": {"total_cases": 1, "failed": 0},
            "cases": [
                {
                    "id": "release.condition_eval.1m",
                    "backend_id": "cuda",
                    "status": "pass"
                }
            ]
        });

        assert_eq!(
            backend_suite_artifact_status_issues(&status, &artifact),
            vec![BackendSuiteArtifactStatusIssue::MissingField {
                path: "release/evidence/benchmarks/workload-01-condition-eval.json".to_string(),
                field: "summary.passed",
            }],
            "Fix: suite artifact validation must require the full summary total/pass/fail triplet, not accept partial failed-only summaries."
        );
    }

    #[test]
    fn backend_suite_artifact_status_rejects_unproven_case_pass_status() {
        let status = serde_json::json!({
            "path": "release/evidence/benchmarks/workload-01-condition-eval.json",
            "selected_backend": "cuda",
            "case_count": 1,
            "failed_count": 0,
            "nonmatching_case_backend_count": 0,
            "requested_case_id": "release.condition_eval.1m"
        });
        let artifact = serde_json::json!({
            "selected_backend": "cuda",
            "summary": {"total_cases": 1, "passed": 1, "failed": 0},
            "cases": [
                {
                    "id": "release.condition_eval.1m",
                    "backend_id": "cuda"
                }
            ]
        });

        assert_eq!(
            backend_suite_artifact_status_issues(&status, &artifact),
            vec![
                BackendSuiteArtifactStatusIssue::NumericFieldMismatch {
                    path: "release/evidence/benchmarks/workload-01-condition-eval.json"
                        .to_string(),
                    field: "summary.passed",
                    status_value: 1,
                    artifact_value: 0,
                },
                BackendSuiteArtifactStatusIssue::NumericFieldMismatch {
                    path: "release/evidence/benchmarks/workload-01-condition-eval.json"
                        .to_string(),
                    field: "summary.failed",
                    status_value: 0,
                    artifact_value: 1,
                },
                BackendSuiteArtifactStatusIssue::FailedCountMismatch {
                    path: "release/evidence/benchmarks/workload-01-condition-eval.json"
                        .to_string(),
                    status_failed_count: 0,
                    artifact_failed_count: 1,
                },
            ],
            "Fix: suite artifact validation must count only explicitly passing cases as release evidence."
        );
    }

    #[test]
    fn backend_suite_artifact_status_rejects_omitted_artifact_backed_fields() {
        let status = serde_json::json!({
            "path": "release/evidence/benchmarks/workload-01-condition-eval.json",
            "requested_case_id": "release.condition_eval.1m"
        });
        let artifact = serde_json::json!({
            "source_fingerprint": "git:abc:dirty=false",
            "source_tree_fingerprint": "source-tree-v1:abc",
            "selected_backend": "cuda",
            "summary": {"total_cases": 1, "passed": 1, "failed": 0},
            "environment": {
                "cpu_model": "AMD Ryzen 9 9950X 16-Core Processor",
                "gpu_devices": [
                    {
                        "name": "NVIDIA GeForce RTX 5090",
                        "memory_total_mib": 32607,
                        "compute_capability_major": 12,
                        "compute_capability_minor": 0
                    }
                ],
                "nvidia_driver_version": "570.211.01",
                "nvidia_cuda_version": "12.8"
            },
            "cases": [
                {
                    "id": "release.condition_eval.1m",
                    "backend_id": "cuda",
                    "status": "pass",
                    "metrics": {
                        "wall_ns": {"samples": 30, "p50": 10, "p95": 11, "p99": 12},
                        "baseline_wall_ns": {"samples": 30, "p50": 1000, "p95": 1001, "p99": 1002},
                        "kernel_launches": {"samples": 30, "p50": 1}
                    },
                    "contract": {
                        "baselines": [
                            {
                                "class": "CpuSota",
                                "backend_ids": ["cuda"],
                                "min_speedup_x": 100.0
                            }
                        ]
                    },
                    "performance": {"contract_passed": true, "speedup_x": 120.0}
                }
            ]
        });

        let missing_fields = backend_suite_artifact_status_issues(&status, &artifact)
            .into_iter()
            .filter_map(|issue| match issue {
                BackendSuiteArtifactStatusIssue::MissingField { field, .. } => Some(field),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(
            missing_fields,
            vec![
                "source_fingerprint",
                "source_tree_fingerprint",
                "selected_backend",
                "case_count",
                "nonmatching_case_backend_count",
                "failed_count",
                "min_wall_samples",
                "min_baseline_wall_samples",
                "min_wall_p50",
                "min_wall_p95",
                "min_wall_p99",
                "min_baseline_wall_p50",
                "min_baseline_wall_p95",
                "min_baseline_wall_p99",
                "min_kernel_launches",
                "gpu_memory_total_mib",
                "gpu_compute_capability_major",
                "gpu_compute_capability_minor",
                "host_cpu_model",
                "gpu_model",
                "nvidia_driver_version",
                "nvidia_cuda_version",
                "cpu_sota_100x_contract_cases",
                "cpu_sota_100x_passing_cases",
            ],
            "Fix: backend suite status rows must not omit artifact-backed proof fields."
        );
    }

    #[test]
    fn backend_suite_artifact_status_rejects_inflated_metric_minima() {
        let status = serde_json::json!({
            "path": "release/evidence/benchmarks/wgpu-workload-01-condition-eval.json",
            "selected_backend": "wgpu",
            "case_count": 1,
            "failed_count": 0,
            "nonmatching_case_backend_count": 0,
            "requested_case_id": "release.condition_eval.1m",
            "min_wall_samples": 35,
            "min_wall_p50": 100,
            "min_kernel_launches": 1
        });
        let artifact = serde_json::json!({
            "selected_backend": "wgpu",
            "summary": {"total_cases": 1, "passed": 1, "failed": 0},
            "cases": [
                {
                    "id": "release.condition_eval.1m",
                    "backend_id": "wgpu",
                    "status": "pass",
                    "metrics": {
                        "wall_ns": {"samples": 20, "p50": 150},
                        "kernel_launches": {"p50": 0}
                    }
                }
            ]
        });

        assert_eq!(
            backend_suite_artifact_status_issues(&status, &artifact),
            vec![
                BackendSuiteArtifactStatusIssue::NumericFieldMismatch {
                    path: "release/evidence/benchmarks/wgpu-workload-01-condition-eval.json"
                        .to_string(),
                    field: "min_wall_samples",
                    status_value: 35,
                    artifact_value: 20,
                },
                BackendSuiteArtifactStatusIssue::NumericFieldMismatch {
                    path: "release/evidence/benchmarks/wgpu-workload-01-condition-eval.json"
                        .to_string(),
                    field: "min_wall_p50",
                    status_value: 100,
                    artifact_value: 150,
                },
                BackendSuiteArtifactStatusIssue::MissingField {
                    path: "release/evidence/benchmarks/wgpu-workload-01-condition-eval.json"
                        .to_string(),
                    field: "min_wall_p95",
                },
                BackendSuiteArtifactStatusIssue::MissingField {
                    path: "release/evidence/benchmarks/wgpu-workload-01-condition-eval.json"
                        .to_string(),
                    field: "min_wall_p99",
                },
                BackendSuiteArtifactStatusIssue::NumericFieldMismatch {
                    path: "release/evidence/benchmarks/wgpu-workload-01-condition-eval.json"
                        .to_string(),
                    field: "min_kernel_launches",
                    status_value: 1,
                    artifact_value: 0,
                },
            ],
            "Fix: backend suite status metric minima must be recomputed from the artifact JSON, not trusted as independent proof."
        );
    }

    #[test]
    fn backend_suite_artifact_status_rejects_provenance_drift() {
        let status = serde_json::json!({
            "path": "release/evidence/benchmarks/workload-01-condition-eval.json",
            "selected_backend": "cuda",
            "case_count": 1,
            "failed_count": 0,
            "nonmatching_case_backend_count": 0,
            "requested_case_id": "release.condition_eval.1m",
            "host_cpu_model": "different CPU",
            "gpu_model": "different GPU",
            "gpu_memory_total_mib": 1,
            "gpu_compute_capability_major": 7,
            "gpu_compute_capability_minor": 5,
            "nvidia_driver_version": "000.000",
            "nvidia_cuda_version": "0.0"
        });
        let artifact = serde_json::json!({
            "selected_backend": "cuda",
            "summary": {"total_cases": 1, "passed": 1, "failed": 0},
            "environment": {
                "cpu_model": "AMD Ryzen 9 9950X 16-Core Processor",
                "gpu_devices": [
                    {
                        "name": "NVIDIA GeForce RTX 5090",
                        "memory_total_mib": 32607,
                        "compute_capability_major": 12,
                        "compute_capability_minor": 0
                    }
                ],
                "nvidia_driver_version": "570.211.01",
                "nvidia_cuda_version": "12.8"
            },
            "cases": [
                {"id": "release.condition_eval.1m", "backend_id": "cuda", "status": "pass"}
            ]
        });

        assert_eq!(
            backend_suite_artifact_status_issues(&status, &artifact),
            vec![
                BackendSuiteArtifactStatusIssue::NumericFieldMismatch {
                    path: "release/evidence/benchmarks/workload-01-condition-eval.json".to_string(),
                    field: "gpu_memory_total_mib",
                    status_value: 1,
                    artifact_value: 32607,
                },
                BackendSuiteArtifactStatusIssue::NumericFieldMismatch {
                    path: "release/evidence/benchmarks/workload-01-condition-eval.json".to_string(),
                    field: "gpu_compute_capability_major",
                    status_value: 7,
                    artifact_value: 12,
                },
                BackendSuiteArtifactStatusIssue::NumericFieldMismatch {
                    path: "release/evidence/benchmarks/workload-01-condition-eval.json".to_string(),
                    field: "gpu_compute_capability_minor",
                    status_value: 5,
                    artifact_value: 0,
                },
                BackendSuiteArtifactStatusIssue::StringFieldMismatch {
                    path: "release/evidence/benchmarks/workload-01-condition-eval.json".to_string(),
                    field: "host_cpu_model",
                    status_value: "different CPU".to_string(),
                    artifact_value: "AMD Ryzen 9 9950X 16-Core Processor".to_string(),
                },
                BackendSuiteArtifactStatusIssue::StringFieldMismatch {
                    path: "release/evidence/benchmarks/workload-01-condition-eval.json".to_string(),
                    field: "gpu_model",
                    status_value: "different GPU".to_string(),
                    artifact_value: "NVIDIA GeForce RTX 5090".to_string(),
                },
                BackendSuiteArtifactStatusIssue::StringFieldMismatch {
                    path: "release/evidence/benchmarks/workload-01-condition-eval.json".to_string(),
                    field: "nvidia_driver_version",
                    status_value: "000.000".to_string(),
                    artifact_value: "570.211.01".to_string(),
                },
                BackendSuiteArtifactStatusIssue::StringFieldMismatch {
                    path: "release/evidence/benchmarks/workload-01-condition-eval.json".to_string(),
                    field: "nvidia_cuda_version",
                    status_value: "0.0".to_string(),
                    artifact_value: "12.8".to_string(),
                },
            ],
            "Fix: backend suite status provenance must be proven by the artifact environment."
        );
    }

    #[test]
    fn backend_suite_artifact_status_rejects_unproven_contract_counts() {
        let status = serde_json::json!({
            "path": "release/evidence/benchmarks/wgpu-workload-06-quantified-condition-loops.json",
            "selected_backend": "wgpu",
            "case_count": 1,
            "failed_count": 0,
            "nonmatching_case_backend_count": 0,
            "requested_case_id": "release.quantified_condition_loops.1m",
            "cpu_sota_100x_contract_cases": 1,
            "cpu_sota_100x_passing_cases": 1
        });
        let artifact = serde_json::json!({
            "selected_backend": "wgpu",
            "summary": {"total_cases": 1, "passed": 1, "failed": 0},
            "cases": [
                {
                    "id": "release.quantified_condition_loops.1m",
                    "backend_id": "wgpu",
                    "status": "pass",
                    "contract": null,
                    "performance": {"contract_passed": true, "speedup_x": 1000.0}
                }
            ]
        });

        assert_eq!(
            backend_suite_artifact_status_issues(&status, &artifact),
            vec![
                BackendSuiteArtifactStatusIssue::CpuSota100xContractCaseCountMismatch {
                    path: "release/evidence/benchmarks/wgpu-workload-06-quantified-condition-loops.json".to_string(),
                    status_contract_cases: 1,
                    artifact_contract_cases: 0,
                },
                BackendSuiteArtifactStatusIssue::CpuSota100xPassingCaseCountMismatch {
                    path: "release/evidence/benchmarks/wgpu-workload-06-quantified-condition-loops.json".to_string(),
                    status_passing_cases: 1,
                    artifact_passing_cases: 0,
                },
            ],
            "Fix: backend suite status must not claim CPU-SOTA 100x contract proof absent from the artifact JSON."
        );
    }

    #[test]
    fn backend_suite_artifact_status_rejects_wrong_backend_contract_counts() {
        let status = serde_json::json!({
            "path": "release/evidence/benchmarks/wgpu-workload-01-condition-eval.json",
            "selected_backend": "wgpu",
            "case_count": 1,
            "failed_count": 0,
            "nonmatching_case_backend_count": 0,
            "requested_case_id": "release.condition_eval.1m",
            "cpu_sota_100x_contract_cases": 1,
            "cpu_sota_100x_passing_cases": 1
        });
        let artifact = serde_json::json!({
            "selected_backend": "wgpu",
            "summary": {"total_cases": 1, "passed": 1, "failed": 0},
            "cases": [
                {
                    "id": "release.condition_eval.1m",
                    "backend_id": "wgpu",
                    "status": "pass",
                    "contract": {
                        "primitive": "condition eval",
                        "baselines": [
                            {
                                "class": "CpuSota",
                                "backend_ids": ["cuda"],
                                "min_speedup_x": 100.0
                            }
                        ]
                    },
                    "performance": {"contract_passed": true, "speedup_x": 120.0}
                }
            ]
        });

        assert_eq!(
            backend_suite_artifact_status_issues(&status, &artifact),
            vec![
                BackendSuiteArtifactStatusIssue::CpuSota100xContractCaseCountMismatch {
                    path: "release/evidence/benchmarks/wgpu-workload-01-condition-eval.json"
                        .to_string(),
                    status_contract_cases: 1,
                    artifact_contract_cases: 0,
                },
                BackendSuiteArtifactStatusIssue::CpuSota100xPassingCaseCountMismatch {
                    path: "release/evidence/benchmarks/wgpu-workload-01-condition-eval.json"
                        .to_string(),
                    status_passing_cases: 1,
                    artifact_passing_cases: 0,
                },
            ],
            "Fix: WGPU suite status must not count a CUDA-only CpuSota baseline as WGPU proof."
        );
    }

    #[test]
    fn backend_suite_artifact_status_rejects_unproven_cpu_sota_pass_status() {
        let status = serde_json::json!({
            "path": "release/evidence/benchmarks/wgpu-workload-01-condition-eval.json",
            "selected_backend": "wgpu",
            "case_count": 1,
            "failed_count": 1,
            "nonmatching_case_backend_count": 0,
            "requested_case_id": "release.condition_eval.1m",
            "cpu_sota_100x_contract_cases": 1,
            "cpu_sota_100x_passing_cases": 1
        });
        let artifact = serde_json::json!({
            "selected_backend": "wgpu",
            "summary": {"total_cases": 1, "passed": 0, "failed": 1},
            "cases": [
                {
                    "id": "release.condition_eval.1m",
                    "backend_id": "wgpu",
                    "contract": {
                        "primitive": "condition eval",
                        "baselines": [
                            {
                                "class": "CpuSota",
                                "backend_ids": ["wgpu"],
                                "min_speedup_x": 100.0
                            }
                        ]
                    },
                    "performance": {"contract_passed": true, "speedup_x": 120.0}
                }
            ]
        });

        assert_eq!(
            backend_suite_artifact_status_issues(&status, &artifact),
            vec![BackendSuiteArtifactStatusIssue::CpuSota100xPassingCaseCountMismatch {
                path: "release/evidence/benchmarks/wgpu-workload-01-condition-eval.json"
                    .to_string(),
                status_passing_cases: 1,
                artifact_passing_cases: 0,
            }],
            "Fix: CPU-SOTA suite status must not count contract_passed speedup evidence without an explicit passing case status."
        );
    }

    #[test]
    fn backend_suite_artifact_status_accepts_proven_contract_counts() {
        let status = serde_json::json!({
            "path": "release/evidence/benchmarks/wgpu-workload-01-condition-eval.json",
            "selected_backend": "wgpu",
            "case_count": 1,
            "failed_count": 0,
            "nonmatching_case_backend_count": 0,
            "requested_case_id": "release.condition_eval.1m",
            "cpu_sota_100x_contract_cases": 1,
            "cpu_sota_100x_passing_cases": 1
        });
        let artifact = serde_json::json!({
            "selected_backend": "wgpu",
            "summary": {"total_cases": 1, "passed": 1, "failed": 0},
            "cases": [
                {
                    "id": "release.condition_eval.1m",
                    "backend_id": "wgpu",
                    "status": "pass",
                    "contract": {
                        "primitive": "condition eval",
                        "baselines": [
                            {
                                "class": "CpuSota",
                                "backend_ids": ["cuda", "wgpu"],
                                "min_speedup_x": 100.0
                            }
                        ]
                    },
                    "performance": {"contract_passed": true, "speedup_x": 120.0}
                }
            ]
        });

        assert!(
            backend_suite_artifact_status_issues(&status, &artifact).is_empty(),
            "Fix: suite status rows with contract counters should pass only when artifact cases prove the same counters."
        );
    }

    #[test]
    fn backend_suite_parity_rejects_count_drift_even_with_duplicate_metadata() {
        let cuda = serde_json::json!({
            "artifact_statuses": [
                {"family_id": "condition-eval", "requested_case_id": "release.condition_eval.1m"}
            ]
        });
        let wgpu = serde_json::json!({
            "artifact_statuses": [
                {"family_id": "condition-eval", "requested_case_id": "release.condition_eval.1m"},
                {"family_id": "condition-eval", "requested_case_id": "release.condition_eval.1m"}
            ]
        });

        assert_eq!(
            backend_suite_parity_issues(&cuda, &wgpu),
            vec![BackendSuiteParityIssue::CountMismatch {
                cuda_count: 1,
                wgpu_count: 2,
            }],
            "Fix: duplicate suite metadata should not silently prove artifact-count parity."
        );
    }
}
