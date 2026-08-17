//! Which backend a report or suite belongs to, and whether every case and
//! contract in it agrees.
//!
//! A suite's backend comes from its evidence name, not from a field it may
//! mislabel, and the artifact's own `selected_backend` is then checked against
//! it. Inside the artifact, each case's `backend_id` must match that backend and
//! each case's performance contract must have a baseline that applies to it,
//! since a contract written for another backend proves nothing here.

use std::collections::BTreeMap;

use serde_json::Value;

use super::cpu_sota_100x::baseline_applies_to_backend;
use super::data::{BackendConsistencyIssue, BackendSuiteBackendIssue, ContractBackendIssue};
use super::json_reader::{case_id, non_empty_str};

pub(crate) fn expected_backend_for_suite_evidence(evidence: &str) -> Option<&'static str> {
    if evidence == "cuda-release-suite.json" || evidence.ends_with("/cuda-release-suite.json") {
        Some("cuda")
    } else if evidence == "wgpu-fallback-suite.json" || evidence.ends_with("/wgpu-fallback-suite.json") {
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

    let mut issues = Vec::new();
    let mut case_id_counts = BTreeMap::new();
    for (case_index, case) in cases.iter().enumerate() {
        let case_id = case.get("id").and_then(non_empty_str).map(str::to_string);
        if case_id.is_none() {
            issues.push(BackendConsistencyIssue::MissingCaseId { case_index });
        }
        if let Some(case_id) = &case_id {
            *case_id_counts.entry(case_id.clone()).or_insert(0) += 1;
        }
        let case_id = case_id.unwrap_or_else(|| "<unknown>".to_string());
        match case
            .get("backend_id")
            .and_then(Value::as_str)
            .filter(|backend| !backend.trim().is_empty())
        {
            Some(actual_backend) if actual_backend == expected_backend => {}
            Some(actual_backend) => issues.push(BackendConsistencyIssue::CaseBackendMismatch {
                case_id,
                expected_backend: expected_backend.to_string(),
                actual_backend: actual_backend.to_string(),
            }),
            None => issues.push(BackendConsistencyIssue::MissingCaseBackend {
                case_id,
                expected_backend: expected_backend.to_string(),
            }),
        }
    }
    for (case_id, count) in case_id_counts {
        if count > 1 {
            issues.push(BackendConsistencyIssue::DuplicateCaseId { case_id, count });
        }
    }
    issues
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn backend_consistency_rejects_blank_case_identity() {
        let report = serde_json::json!({
            "selected_backend": "cuda",
            "cases": [
                {"id": "   ", "backend_id": "cuda"},
                {"backend_id": "cuda"}
            ]
        });

        assert_eq!(
            backend_consistency_issues(&report),
            vec![
                BackendConsistencyIssue::MissingCaseId { case_index: 0 },
                BackendConsistencyIssue::MissingCaseId { case_index: 1 },
            ],
            "Fix: backend consistency must require nonblank case ids before benchmark rows can prove release backend identity."
        );
    }

    #[test]
    fn backend_consistency_rejects_duplicate_case_identity() {
        let report = serde_json::json!({
            "selected_backend": "cuda",
            "cases": [
                {"id": "release.condition_eval.1m", "backend_id": "cuda"},
                {"id": "release.condition_eval.1m", "backend_id": "cuda"},
                {"id": "release.entropy_window.1m", "backend_id": "cuda"}
            ]
        });

        assert_eq!(
            backend_consistency_issues(&report),
            vec![BackendConsistencyIssue::DuplicateCaseId {
                case_id: "release.condition_eval.1m".to_string(),
                count: 2,
            }],
            "Fix: duplicate benchmark case ids must not prove distinct release cases."
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
    fn expected_backend_for_suite_evidence_rejects_non_boundary_matches() {
        assert_eq!(expected_backend_for_suite_evidence("cuda-release-suite.json"), Some("cuda"));
        assert_eq!(expected_backend_for_suite_evidence("evidence/benchmarks/cuda-release-suite.json"), Some("cuda"));
        assert_eq!(expected_backend_for_suite_evidence("mock-cuda-release-suite.json"), None);
        assert_eq!(expected_backend_for_suite_evidence("wgpu-fallback-suite.json"), Some("wgpu"));
        assert_eq!(expected_backend_for_suite_evidence("evidence/benchmarks/wgpu-fallback-suite.json"), Some("wgpu"));
        assert_eq!(expected_backend_for_suite_evidence("not-wgpu-fallback-suite.json"), None);
    }
}
