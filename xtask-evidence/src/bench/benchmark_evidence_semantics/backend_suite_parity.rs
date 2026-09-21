//! Whether the CUDA and WGPU suites measured the same work.
//!
//! Two backends only compare if they ran the same family and case pairs the same
//! number of times, from the same source tree, with the same declared blockers
//! and the same CPU-SOTA counts. A pair present on one side only, a status field
//! that drifts, or an artifact path shared between the two suites means the
//! comparison is between different runs.

use serde_json::Value;

use super::backend_identity::backend_suite_backend_issue;
use super::data::BackendSuiteParityIssue;
use super::json_reader::non_empty_str;
use super::suite_reader::{
    suite_all_artifact_paths, suite_artifact_status_count, suite_family_case_pair_counts,
    suite_family_case_pairs, suite_status_blockers, suite_statuses_by_family_case_pair,
};

pub(crate) fn backend_suite_parity_issues(
    cuda_suite: &Value,
    wgpu_suite: &Value,
) -> Vec<BackendSuiteParityIssue> {
    let cuda_count = suite_artifact_status_count(cuda_suite);
    let wgpu_count = suite_artifact_status_count(wgpu_suite);
    let cuda_pairs = suite_family_case_pairs(cuda_suite);
    let wgpu_pairs = suite_family_case_pairs(wgpu_suite);
    let mut issues = Vec::new();
    if let Some(issue) = backend_suite_backend_issue(cuda_suite, "cuda") {
        issues.push(BackendSuiteParityIssue::CudaBackendIdentity { issue });
    }
    if let Some(issue) = backend_suite_backend_issue(wgpu_suite, "wgpu") {
        issues.push(BackendSuiteParityIssue::WgpuBackendIdentity { issue });
    }
    if cuda_count != wgpu_count || cuda_pairs.len() != wgpu_pairs.len() {
        issues.push(BackendSuiteParityIssue::CountMismatch {
            cuda_count,
            wgpu_count,
        });
    }
    for ((family_id, requested_case_id), count) in suite_family_case_pair_counts(cuda_suite) {
        if count > 1 {
            issues.push(BackendSuiteParityIssue::DuplicateCudaPair {
                family_id,
                requested_case_id,
                count,
            });
        }
    }
    for ((family_id, requested_case_id), count) in suite_family_case_pair_counts(wgpu_suite) {
        if count > 1 {
            issues.push(BackendSuiteParityIssue::DuplicateWgpuPair {
                family_id,
                requested_case_id,
                count,
            });
        }
    }
    let cuda_paths = suite_all_artifact_paths(cuda_suite);
    let wgpu_paths = suite_all_artifact_paths(wgpu_suite);
    for path in cuda_paths.intersection(&wgpu_paths) {
        issues.push(BackendSuiteParityIssue::SharedArtifactPath { path: path.clone() });
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
    let cuda_statuses = suite_statuses_by_family_case_pair(cuda_suite);
    let wgpu_statuses = suite_statuses_by_family_case_pair(wgpu_suite);
    for pair in cuda_pairs.intersection(&wgpu_pairs) {
        let Some(cuda_status) = cuda_statuses.get(pair) else {
            continue;
        };
        let Some(wgpu_status) = wgpu_statuses.get(pair) else {
            continue;
        };
        for field in [
            "case_count",
            "failed_count",
            "nonmatching_case_backend_count",
            "cpu_sota_100x_contract_cases",
            "cpu_sota_100x_passing_cases",
        ] {
            let cuda_value = cuda_status.get(field).and_then(Value::as_u64);
            let wgpu_value = wgpu_status.get(field).and_then(Value::as_u64);
            if cuda_value != wgpu_value {
                issues.push(BackendSuiteParityIssue::StatusFieldMismatch {
                    family_id: pair.0.clone(),
                    requested_case_id: pair.1.clone(),
                    field,
                    cuda_value,
                    wgpu_value,
                });
            }
        }
        let field = "source_tree_fingerprint";
        let cuda_value = cuda_status
            .get(field)
            .and_then(non_empty_str)
            .map(str::to_string);
        let wgpu_value = wgpu_status
            .get(field)
            .and_then(non_empty_str)
            .map(str::to_string);
        if cuda_value != wgpu_value {
            issues.push(BackendSuiteParityIssue::StatusStringFieldMismatch {
                family_id: pair.0.clone(),
                requested_case_id: pair.1.clone(),
                field,
                cuda_value,
                wgpu_value,
            });
        }
        let cuda_blockers = suite_status_blockers(cuda_status);
        let wgpu_blockers = suite_status_blockers(wgpu_status);
        if cuda_blockers != wgpu_blockers {
            issues.push(BackendSuiteParityIssue::StatusBlockersMismatch {
                family_id: pair.0.clone(),
                requested_case_id: pair.1.clone(),
                cuda_blockers,
                wgpu_blockers,
            });
        }
    }
    issues
}

#[cfg(test)]
mod tests {
    use super::super::data::BackendSuiteBackendIssue;
    use super::*;

    #[test]
    fn backend_suite_parity_rejects_missing_family_case_pairs() {
        let cuda = serde_json::json!({
            "backend": "cuda",
            "artifact_statuses": [
                {"family_id": "condition-eval", "requested_case_id": "release.condition_eval.1m"},
                {"family_id": "entropy-window", "requested_case_id": "release.entropy_window.1m"}
            ]
        });
        let wgpu = serde_json::json!({
            "backend": "wgpu",
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
    fn backend_suite_parity_rejects_status_field_drift_for_matching_pairs() {
        let cuda = serde_json::json!({
            "backend": "cuda",
            "artifact_statuses": [
                {
                    "family_id": "condition-eval",
                    "requested_case_id": "release.condition_eval.1m",
                    "case_count": 1,
                    "failed_count": 0,
                    "nonmatching_case_backend_count": 0,
                    "source_fingerprint": "git:cuda-source:dirty=false",
                    "source_tree_fingerprint": "source-tree-v1:shared"
                }
            ]
        });
        let wgpu = serde_json::json!({
            "backend": "wgpu",
            "artifact_statuses": [
                {
                    "family_id": "condition-eval",
                    "requested_case_id": "release.condition_eval.1m",
                    "case_count": 0,
                    "failed_count": 1,
                    "nonmatching_case_backend_count": 0,
                    "source_fingerprint": "git:wgpu-source:dirty=false",
                    "source_tree_fingerprint": "source-tree-v1:shared"
                }
            ]
        });

        assert_eq!(
            backend_suite_parity_issues(&cuda, &wgpu),
            vec![
                BackendSuiteParityIssue::StatusFieldMismatch {
                    family_id: "condition-eval".to_string(),
                    requested_case_id: "release.condition_eval.1m".to_string(),
                    field: "case_count",
                    cuda_value: Some(1),
                    wgpu_value: Some(0),
                },
                BackendSuiteParityIssue::StatusFieldMismatch {
                    family_id: "condition-eval".to_string(),
                    requested_case_id: "release.condition_eval.1m".to_string(),
                    field: "failed_count",
                    cuda_value: Some(0),
                    wgpu_value: Some(1),
                }
            ],
            "Fix: WGPU parity must compare proof strength for matching suite rows while tolerating evidence-only commit fingerprint drift."
        );
    }

    #[test]
    fn backend_suite_parity_rejects_source_tree_drift_not_evidence_commit_drift() {
        let cuda = serde_json::json!({
            "backend": "cuda",
            "artifact_statuses": [
                {
                    "family_id": "condition-eval",
                    "requested_case_id": "release.condition_eval.1m",
                    "source_fingerprint": "git:cuda-evidence-commit:dirty=false",
                    "source_tree_fingerprint": "source-tree-v1:cuda"
                }
            ]
        });
        let wgpu = serde_json::json!({
            "backend": "wgpu",
            "artifact_statuses": [
                {
                    "family_id": "condition-eval",
                    "requested_case_id": "release.condition_eval.1m",
                    "source_fingerprint": "git:wgpu-evidence-commit:dirty=false",
                    "source_tree_fingerprint": "source-tree-v1:wgpu"
                }
            ]
        });

        assert_eq!(
            backend_suite_parity_issues(&cuda, &wgpu),
            vec![
                BackendSuiteParityIssue::StatusStringFieldMismatch {
                    family_id: "condition-eval".to_string(),
                    requested_case_id: "release.condition_eval.1m".to_string(),
                    field: "source_tree_fingerprint",
                    cuda_value: Some("source-tree-v1:cuda".to_string()),
                    wgpu_value: Some("source-tree-v1:wgpu".to_string()),
                },
            ],
            "Fix: WGPU parity must reject source tree drift without treating benchmark evidence commits as backend source drift."
        );
    }

    #[test]
    fn backend_suite_parity_rejects_status_blocker_drift_for_matching_pairs() {
        let cuda = serde_json::json!({
            "backend": "cuda",
            "artifact_statuses": [
                {
                    "family_id": "condition-eval",
                    "requested_case_id": "release.condition_eval.1m",
                    "blockers": []
                }
            ]
        });
        let wgpu = serde_json::json!({
            "backend": "wgpu",
            "artifact_statuses": [
                {
                    "family_id": "condition-eval",
                    "requested_case_id": "release.condition_eval.1m",
                    "blockers": ["case `release.condition_eval.1m` failed: WGPU output drift"]
                }
            ]
        });

        assert_eq!(
            backend_suite_parity_issues(&cuda, &wgpu),
            vec![BackendSuiteParityIssue::StatusBlockersMismatch {
                family_id: "condition-eval".to_string(),
                requested_case_id: "release.condition_eval.1m".to_string(),
                cuda_blockers: Some(Vec::new()),
                wgpu_blockers: Some(vec![
                    "case `release.condition_eval.1m` failed: WGPU output drift".to_string()
                ]),
            }],
            "Fix: WGPU parity must reject matching suite rows with different blocker state."
        );
    }

    #[test]
    fn backend_suite_parity_rejects_cpu_sota_count_drift() {
        let cuda = serde_json::json!({
            "backend": "cuda",
            "artifact_statuses": [
                {
                    "family_id": "condition-eval",
                    "requested_case_id": "release.condition_eval.1m",
                    "cpu_sota_100x_contract_cases": 1,
                    "cpu_sota_100x_passing_cases": 1
                }
            ]
        });
        let wgpu = serde_json::json!({
            "backend": "wgpu",
            "artifact_statuses": [
                {
                    "family_id": "condition-eval",
                    "requested_case_id": "release.condition_eval.1m",
                    "cpu_sota_100x_contract_cases": 0,
                    "cpu_sota_100x_passing_cases": 0
                }
            ]
        });

        assert_eq!(
            backend_suite_parity_issues(&cuda, &wgpu),
            vec![
                BackendSuiteParityIssue::StatusFieldMismatch {
                    family_id: "condition-eval".to_string(),
                    requested_case_id: "release.condition_eval.1m".to_string(),
                    field: "cpu_sota_100x_contract_cases",
                    cuda_value: Some(1),
                    wgpu_value: Some(0),
                },
                BackendSuiteParityIssue::StatusFieldMismatch {
                    family_id: "condition-eval".to_string(),
                    requested_case_id: "release.condition_eval.1m".to_string(),
                    field: "cpu_sota_100x_passing_cases",
                    cuda_value: Some(1),
                    wgpu_value: Some(0),
                },
            ],
            "Fix: WGPU/CUDA parity must compare CPU-SOTA proof strength for matching suite rows."
        );
    }

    #[test]
    fn backend_suite_parity_rejects_mislabeled_suite_backends() {
        let cuda = serde_json::json!({
            "backend": "wgpu",
            "artifact_statuses": [
                {"family_id": "condition-eval", "requested_case_id": "release.condition_eval.1m"}
            ]
        });
        let wgpu = serde_json::json!({
            "artifact_statuses": [
                {"family_id": "condition-eval", "requested_case_id": "release.condition_eval.1m"}
            ]
        });

        assert_eq!(
            backend_suite_parity_issues(&cuda, &wgpu),
            vec![
                BackendSuiteParityIssue::CudaBackendIdentity {
                    issue: BackendSuiteBackendIssue::Mismatch {
                        expected_backend: "cuda".to_string(),
                        actual_backend: "wgpu".to_string(),
                    },
                },
                BackendSuiteParityIssue::WgpuBackendIdentity {
                    issue: BackendSuiteBackendIssue::Missing {
                        expected_backend: "wgpu".to_string(),
                    },
                },
            ],
            "Fix: WGPU/CUDA parity must reject mislabeled peer suite identities, not only row-level family/case coverage."
        );
    }

    #[test]
    fn backend_suite_parity_rejects_duplicate_family_case_pairs_with_equal_counts() {
        let cuda = serde_json::json!({
            "backend": "cuda",
            "artifact_statuses": [
                {
                    "path": "release/evidence/benchmarks/cuda-condition-a.json",
                    "family_id": "condition-eval",
                    "requested_case_id": "release.condition_eval.1m"
                },
                {
                    "path": "release/evidence/benchmarks/cuda-condition-b.json",
                    "family_id": "condition-eval",
                    "requested_case_id": "release.condition_eval.1m"
                }
            ]
        });
        let wgpu = serde_json::json!({
            "backend": "wgpu",
            "artifact_statuses": [
                {
                    "path": "release/evidence/benchmarks/wgpu-condition-a.json",
                    "family_id": "condition-eval",
                    "requested_case_id": "release.condition_eval.1m"
                },
                {
                    "path": "release/evidence/benchmarks/wgpu-condition-b.json",
                    "family_id": "condition-eval",
                    "requested_case_id": "release.condition_eval.1m"
                }
            ]
        });

        assert_eq!(
            backend_suite_parity_issues(&cuda, &wgpu),
            vec![
                BackendSuiteParityIssue::DuplicateCudaPair {
                    family_id: "condition-eval".to_string(),
                    requested_case_id: "release.condition_eval.1m".to_string(),
                    count: 2,
                },
                BackendSuiteParityIssue::DuplicateWgpuPair {
                    family_id: "condition-eval".to_string(),
                    requested_case_id: "release.condition_eval.1m".to_string(),
                    count: 2,
                },
            ],
            "Fix: WGPU parity must reject duplicate family/case rows even when CUDA and WGPU counts match."
        );
    }

    #[test]
    fn backend_suite_parity_rejects_count_drift_even_with_duplicate_metadata() {
        let cuda = serde_json::json!({
            "backend": "cuda",
            "artifact_statuses": [
                {"family_id": "condition-eval", "requested_case_id": "release.condition_eval.1m"}
            ]
        });
        let wgpu = serde_json::json!({
            "backend": "wgpu",
            "artifact_statuses": [
                {"family_id": "condition-eval", "requested_case_id": "release.condition_eval.1m"},
                {"family_id": "condition-eval", "requested_case_id": "release.condition_eval.1m"}
            ]
        });

        assert_eq!(
            backend_suite_parity_issues(&cuda, &wgpu),
            vec![
                BackendSuiteParityIssue::CountMismatch {
                    cuda_count: 1,
                    wgpu_count: 2,
                },
                BackendSuiteParityIssue::DuplicateWgpuPair {
                    family_id: "condition-eval".to_string(),
                    requested_case_id: "release.condition_eval.1m".to_string(),
                    count: 2,
                },
            ],
            "Fix: duplicate suite metadata should not silently prove artifact-count parity."
        );
    }

    #[test]
    fn backend_suite_parity_rejects_shared_artifact_paths() {
        let cuda = serde_json::json!({
            "backend": "cuda",
            "artifacts": ["release/evidence/benchmarks/workload-01-condition-eval.json"],
            "artifact_statuses": [
                {
                    "path": "release/evidence/benchmarks/workload-01-condition-eval.json",
                    "family_id": "condition-eval",
                    "requested_case_id": "release.condition_eval.1m"
                }
            ]
        });
        let wgpu = serde_json::json!({
            "backend": "wgpu",
            "artifacts": ["release/evidence/benchmarks/workload-01-condition-eval.json"],
            "artifact_statuses": [
                {
                    "path": "release/evidence/benchmarks/workload-01-condition-eval.json",
                    "family_id": "condition-eval",
                    "requested_case_id": "release.condition_eval.1m"
                }
            ]
        });

        assert_eq!(
            backend_suite_parity_issues(&cuda, &wgpu),
            vec![BackendSuiteParityIssue::SharedArtifactPath {
                path: "release/evidence/benchmarks/workload-01-condition-eval.json".to_string(),
            }],
            "Fix: WGPU fallback evidence must not reuse or overwrite CUDA release benchmark artifacts."
        );
    }
}
