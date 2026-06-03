use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use serde_json::{Map, Value};

static CURRENT_SOURCE_FINGERPRINTS: OnceLock<Mutex<BTreeMap<PathBuf, String>>> = OnceLock::new();

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
pub(crate) enum BackendSuiteArtifactStatusIssue {
    SourceFingerprintMismatch {
        path: String,
        status_source_fingerprint: String,
        artifact_source_fingerprint: String,
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
    if let (Some(status_source), Some(artifact_source)) = (status_source, artifact_source) {
        if status_source != artifact_source {
            issues.push(BackendSuiteArtifactStatusIssue::SourceFingerprintMismatch {
                path: path.clone(),
                status_source_fingerprint: status_source.to_string(),
                artifact_source_fingerprint: artifact_source.to_string(),
            });
        }
    }

    let status_backend = status.get("selected_backend").and_then(non_empty_str);
    let artifact_backend = artifact_report
        .get("selected_backend")
        .and_then(non_empty_str);
    if let (Some(status_backend), Some(artifact_backend)) = (status_backend, artifact_backend) {
        if status_backend != artifact_backend {
            issues.push(BackendSuiteArtifactStatusIssue::SelectedBackendMismatch {
                path: path.clone(),
                status_selected_backend: status_backend.to_string(),
                artifact_selected_backend: artifact_backend.to_string(),
            });
        }
    }

    let status_case_count = status.get("case_count").and_then(Value::as_u64);
    let artifact_case_count = artifact_report
        .get("cases")
        .and_then(Value::as_array)
        .map(|cases| cases.len() as u64);
    if let (Some(status_case_count), Some(artifact_case_count)) =
        (status_case_count, artifact_case_count)
    {
        if status_case_count != artifact_case_count {
            issues.push(BackendSuiteArtifactStatusIssue::CaseCountMismatch {
                path: path.clone(),
                status_case_count,
                artifact_case_count,
            });
        }
    }

    let status_failed_count = status.get("failed_count").and_then(Value::as_u64);
    let artifact_failed_count = artifact_report
        .get("summary")
        .and_then(|summary| summary.get("failed"))
        .and_then(Value::as_u64);
    if let (Some(status_failed_count), Some(artifact_failed_count)) =
        (status_failed_count, artifact_failed_count)
    {
        if status_failed_count != artifact_failed_count {
            issues.push(BackendSuiteArtifactStatusIssue::FailedCountMismatch {
                path: path.clone(),
                status_failed_count,
                artifact_failed_count,
            });
        }
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
        let applies = baselines.iter().any(|baseline| {
            baseline
                .get("backend_ids")
                .and_then(Value::as_array)
                .is_none_or(|backend_ids| {
                    backend_ids.is_empty()
                        || backend_ids
                            .iter()
                            .any(|candidate| candidate.as_str() == Some(backend_id))
                })
        });
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
            "selected_backend": "cuda",
            "case_count": 2,
            "failed_count": 0,
            "requested_case_id": "release.condition_eval.1m"
        });
        let artifact = serde_json::json!({
            "source_fingerprint": "git:new:dirty=false",
            "selected_backend": "wgpu",
            "summary": {"failed": 1},
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
            "selected_backend": "cuda",
            "case_count": 1,
            "failed_count": 0,
            "requested_case_id": "release.condition_eval.1m"
        });
        let artifact = serde_json::json!({
            "source_fingerprint": "git:abc:dirty=false",
            "selected_backend": "cuda",
            "summary": {"failed": 0},
            "cases": [
                {"id": "release.condition_eval.1m"}
            ]
        });

        assert!(
            backend_suite_artifact_status_issues(&status, &artifact).is_empty(),
            "Fix: matching suite status and artifact JSON should pass."
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
