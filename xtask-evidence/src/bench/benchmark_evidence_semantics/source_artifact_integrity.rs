//! Whether the cases inside one cited source artifact hold together.
//!
//! An aggregate that cites an artifact inherits everything wrong inside it, so
//! the artifact is opened and its cases are checked before any count derived
//! from it is believed: the summary against the cases, each case's backend
//! against the artifact's selected backend, each optimization label against the
//! counters that prove it, and the forbidden borrowed-resident escape hatch
//! against the native dispatch the aggregate claims.

use serde_json::Value;

use super::backend_identity::{backend_consistency_issues, contract_backend_issues};
use super::case_summary::benchmark_report_summary_case_evidence_mismatch;
use super::data::{
    BackendConsistencyIssue, ContractBackendIssue, CudaForbiddenTelemetryIssue,
    CudaTelemetryLabelIssue,
};
use super::telemetry_labels::{cuda_forbidden_telemetry_issues, cuda_telemetry_label_issues};

pub(crate) fn inspect_source_artifact_case_integrity(
    artifact: &str,
    report: &Value,
    native_dispatch_context: &str,
    issues: &mut Vec<String>,
) {
    for issue in backend_consistency_issues(report) {
        match issue {
            BackendConsistencyIssue::MissingCaseId { case_index } => issues.push(format!(
                "source_artifact `{artifact}` case index {case_index} must include a nonblank id"
            )),
            BackendConsistencyIssue::DuplicateCaseId { case_id, count } => issues.push(format!(
                "source_artifact `{artifact}` has {count} cases with id `{case_id}`"
            )),
            BackendConsistencyIssue::MissingCaseBackend {
                case_id,
                expected_backend,
            } => issues.push(format!(
                "source_artifact `{artifact}` case `{case_id}` must include backend_id `{expected_backend}` matching selected_backend"
            )),
            BackendConsistencyIssue::CaseBackendMismatch {
                case_id,
                expected_backend,
                actual_backend,
            } => issues.push(format!(
                "source_artifact `{artifact}` case `{case_id}` backend_id `{actual_backend}` does not match selected_backend `{expected_backend}`"
            )),
        }
    }
    for issue in contract_backend_issues(report) {
        match issue {
            ContractBackendIssue::MissingBaselines {
                case_id,
                backend_id,
            } => issues.push(format!(
                "source_artifact `{artifact}` case `{case_id}` backend `{backend_id}` has a performance contract with no baselines"
            )),
            ContractBackendIssue::NoApplicableBaseline {
                case_id,
                backend_id,
            } => issues.push(format!(
                "source_artifact `{artifact}` case `{case_id}` backend `{backend_id}` has no applicable performance contract baseline"
            )),
        }
    }
    for issue in cuda_forbidden_telemetry_issues(report) {
        match issue {
            CudaForbiddenTelemetryIssue::ResidentBorrowedEscapeHatch {
                case_id,
                observed_p50,
            } => issues.push(format!(
                "source_artifact `{artifact}` case `{case_id}` has cuda_resident_borrowed_fallback_dispatches p50={observed_p50}; {native_dispatch_context} must use native resident dispatch"
            )),
        }
    }
    for issue in cuda_telemetry_label_issues(report) {
        match issue {
            CudaTelemetryLabelIssue::MissingLabel { case_id, label } => issues.push(format!(
                "source_artifact `{artifact}` case `{case_id}` has positive CUDA telemetry counters but is missing `{label}`"
            )),
            CudaTelemetryLabelIssue::LabelWithoutCounters { case_id, label } => issues.push(format!(
                "source_artifact `{artifact}` case `{case_id}` lists `{label}` but all matching CUDA telemetry counters are zero or missing"
            )),
        }
    }
    if report
        .get("cases")
        .and_then(Value::as_array)
        .is_none_or(Vec::is_empty)
    {
        issues.push(format!(
            "source_artifact `{artifact}` has no benchmark cases"
        ));
    }
    if let Some(mismatch) = benchmark_report_summary_case_evidence_mismatch(report) {
        issues.push(format!(
            "source_artifact `{artifact}` summary does not match case evidence: {mismatch}"
        ));
    }
    if report
        .get("summary")
        .and_then(|summary| summary.get("failed"))
        .and_then(Value::as_u64)
        != Some(0)
    {
        issues.push(format!(
            "source_artifact `{artifact}` summary.failed must be 0"
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::super::release_axes_cuda::cuda_release_axes_source_artifact_issues;
    use crate::report_fixture::EvidenceWorkspace;

    #[test]
    fn cuda_release_axes_reject_case_backend_drift_inside_cuda_source_artifact() {
        let workspace = EvidenceWorkspace::new();
        let artifact = workspace.write_cuda_release_artifact(
            "workload-backend-drift.json",
            serde_json::json!([
                {
                    "id": "release.backend-drift",
                    "backend_id": "wgpu",
                    "status": "pass"
                }
            ]),
        );
        let axes = serde_json::json!({
            "source_artifacts": [artifact]
        });
        let cuda_suite = EvidenceWorkspace::cuda_release_suite(&[&artifact]);

        let issues = cuda_release_axes_source_artifact_issues(workspace.path(), &axes, &cuda_suite);

        assert!(
            issues.iter().any(|issue| issue.contains(
                "source_artifact `release/evidence/benchmarks/workload-backend-drift.json` case `release.backend-drift` backend_id `wgpu` does not match selected_backend `cuda`"
            )),
            "Fix: release-axis CUDA source artifact validation must reject case-level backend drift, not only artifact-level selected_backend; issues={issues:?}"
        );
    }

    #[test]
    fn cuda_release_axes_reject_borrowed_resident_fallback_source_artifact() {
        let workspace = EvidenceWorkspace::new();
        let artifact = workspace.write_cuda_release_artifact(
            "workload-borrowed-resident.json",
            serde_json::json!([
                {
                    "id": "release.borrowed-resident",
                    "backend_id": "cuda",
                    "status": "pass",
                    "optimization_passes_applied": ["cuda-resident-borrowed-escape-hatch"],
                    "metrics": {
                        "wall_ns": {"p50": 17_000},
                        "cold_compile_ns": {"p50": 2_000_000},
                        "wall_gb_s_x1000": {"p50": 4_000},
                        "memory_total_mib": {"p50": 24_576},
                        "cuda_resident_borrowed_fallback_dispatches": {"p50": 2.0}
                    }
                }
            ]),
        );
        let axes = serde_json::json!({
            "warm_us_per_file": 17.0,
            "cold_pipeline_build_ms": 2.0,
            "gbs_scan_throughput": 4.0,
            "ulp_drift_max": 0,
            "max_vram_mib": 24_576,
            "source_artifacts": [artifact]
        });
        let cuda_suite = EvidenceWorkspace::cuda_release_suite(&[&artifact]);

        let issues = cuda_release_axes_source_artifact_issues(workspace.path(), &axes, &cuda_suite);

        assert!(
            issues.iter().any(|issue| issue.contains(
                "source_artifact `release/evidence/benchmarks/workload-borrowed-resident.json` case `release.borrowed-resident` has cuda_resident_borrowed_fallback_dispatches p50=2"
            ) && issue.contains("canonical CUDA release axes must use native resident dispatch")),
            "Fix: canonical CUDA release axes must reject source artifacts measured through the borrowed resident fallback escape hatch; issues={issues:?}"
        );
    }
}
