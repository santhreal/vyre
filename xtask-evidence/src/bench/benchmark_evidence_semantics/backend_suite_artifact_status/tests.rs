//! Status rows read against the artifacts that must prove them.

use super::*;
use crate::report_fixture::{cpu_sota_baseline, cpu_sota_baseline_for};

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
fn backend_suite_artifact_status_rejects_duplicate_requested_case_rows() {
    let status = serde_json::json!({
        "path": "release/evidence/benchmarks/workload-01-condition-eval.json",
        "requested_case_id": "release.condition_eval.1m",
        "case_count": 3,
        "failed_count": 0
    });
    let artifact = serde_json::json!({
        "summary": {"total_cases": 3, "passed": 3, "failed": 0},
        "cases": [
            {"id": "release.condition_eval.1m", "backend_id": "cuda", "status": "pass"},
            {"id": "release.condition_eval.1m", "backend_id": "cuda", "status": "pass"},
            {"id": "release.other.1m", "backend_id": "cuda", "status": "pass"}
        ]
    });

    assert_eq!(
        backend_suite_artifact_status_issues(&status, &artifact),
        vec![BackendSuiteArtifactStatusIssue::DuplicateRequestedCase {
            path: "release/evidence/benchmarks/workload-01-condition-eval.json".to_string(),
            requested_case_id: "release.condition_eval.1m".to_string(),
            count: 2,
        }],
        "Fix: suite status requested_case_id must identify exactly one benchmark row inside the artifact."
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
                "contract": cpu_sota_baseline(&["cuda"], 100.0),
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
fn backend_suite_artifact_status_rejects_blank_environment_provenance() {
    let status = serde_json::json!({
        "path": "release/evidence/benchmarks/workload-01-condition-eval.json",
        "selected_backend": "cuda",
        "case_count": 1,
        "failed_count": 0,
        "nonmatching_case_backend_count": 0,
        "requested_case_id": "release.condition_eval.1m",
        "host_cpu_model": "AMD Ryzen 9 9950X 16-Core Processor",
        "gpu_model": "NVIDIA GeForce RTX 5090",
        "gpu_memory_total_mib": 32607,
        "gpu_compute_capability_major": 12,
        "gpu_compute_capability_minor": 0,
        "nvidia_driver_version": "570.211.01",
        "nvidia_cuda_version": "12.8"
    });
    let artifact = serde_json::json!({
        "selected_backend": "cuda",
        "summary": {"total_cases": 1, "passed": 1, "failed": 0},
        "environment": {
            "cpu_model": "   ",
            "gpu_devices": [
                {
                    "name": "\t",
                    "memory_total_mib": 32607,
                    "compute_capability_major": 12,
                    "compute_capability_minor": 0
                }
            ],
            "nvidia_driver_version": " ",
            "nvidia_cuda_version": "\n"
        },
        "cases": [
            {"id": "release.condition_eval.1m", "backend_id": "cuda", "status": "pass"}
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
            "host_cpu_model",
            "gpu_model",
            "nvidia_driver_version",
            "nvidia_cuda_version"
        ],
        "Fix: whitespace-only benchmark artifact environment provenance must be treated as missing evidence."
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
                "contract": cpu_sota_baseline_for("condition eval", &["cuda"], 100.0),
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
                "contract": cpu_sota_baseline_for("condition eval", &["wgpu"], 100.0),
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
        "min_wall_samples": 30,
        "min_wall_p50": 10,
        "min_wall_p95": 11,
        "min_wall_p99": 12,
        "min_baseline_wall_samples": 30,
        "min_baseline_wall_p50": 1200,
        "min_baseline_wall_p95": 1201,
        "min_baseline_wall_p99": 1202,
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
                "contract": cpu_sota_baseline_for("condition eval", &["cuda", "wgpu"], 100.0),
                "metrics": {
                    "wall_ns": {"samples": 30, "p50": 10, "p95": 11, "p99": 12},
                    "baseline_wall_ns": {"samples": 30, "p50": 1200, "p95": 1201, "p99": 1202}
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
