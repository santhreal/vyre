//! The CUDA release axes read against the artifacts and the suite that prove
//! them.
//!
//! This is the entry point `bench-release` and the CUDA-first gate call: it walks
//! every cited artifact for a usable path, precise provenance, current
//! freshness, intact cases and the axis metrics, recomputes the scalars, and
//! cross-checks the CUDA suite's own inventory. One traversal, so an axis is
//! never accepted on the strength of artifacts a later check would reject.

use std::path::Path;

use serde_json::Value;

use super::backend_identity::backend_suite_backend_issue;
use super::backend_suite_inventory::{
    backend_suite_inventory_issues, describe_backend_suite_inventory_issue,
};
use super::data::BackendSuiteBackendIssue;
use super::json_reader::artifact_string_set;
use super::release_axes_artifact_metrics::inspect_release_axis_source_artifact_metrics;
use super::release_axes_scalars::inspect_release_axes_scalar_values;
use super::source_artifact::read_cited_source_artifact;
use super::source_artifact_integrity::inspect_source_artifact_case_integrity;
use super::source_artifact_provenance::inspect_release_axis_source_artifact_provenance;

pub(crate) fn cuda_release_axes_source_artifact_issues(
    workspace_root: &Path,
    axes: &Value,
    cuda_suite: &Value,
) -> Vec<String> {
    let mut issues = Vec::new();
    if let Some(issue) = backend_suite_backend_issue(cuda_suite, "cuda") {
        match issue {
            BackendSuiteBackendIssue::Missing { expected_backend } => issues.push(format!(
                "cuda-release-suite is missing backend identity `{expected_backend}`"
            )),
            BackendSuiteBackendIssue::Mismatch {
                expected_backend,
                actual_backend,
            } => issues.push(format!(
                "cuda-release-suite backend `{actual_backend}` does not match required `{expected_backend}`"
            )),
        }
    }
    let source_artifacts = artifact_string_set(
        axes,
        "source_artifacts",
        "source_artifacts",
        "source_artifacts array is missing",
        &mut issues,
    );
    if source_artifacts.len() < 12 {
        issues.push(format!(
            "source_artifacts has {} CUDA workload artifact(s), needs at least 12",
            source_artifacts.len()
        ));
    }

    let suite_artifacts = artifact_string_set(
        cuda_suite,
        "artifacts",
        "cuda-release-suite artifacts",
        "cuda-release-suite artifacts array is missing",
        &mut issues,
    );
    if suite_artifacts.is_empty() {
        issues.push("cuda-release-suite artifacts are empty or missing".to_string());
    }
    for issue in backend_suite_inventory_issues(cuda_suite) {
        issues.push(format!(
            "cuda-release-suite {}",
            describe_backend_suite_inventory_issue(&issue)
        ));
    }
    for artifact in source_artifacts.difference(&suite_artifacts) {
        issues.push(format!(
            "source_artifact `{artifact}` is not listed in cuda-release-suite artifacts"
        ));
    }
    for artifact in suite_artifacts.difference(&source_artifacts) {
        issues.push(format!(
            "cuda-release-suite artifact `{artifact}` is absent from bench-release-axes source_artifacts"
        ));
    }

    let mut source_reports = Vec::new();
    for artifact in source_artifacts {
        let Some((artifact_path, report)) =
            read_cited_source_artifact(workspace_root, &artifact, &mut issues)
        else {
            continue;
        };
        inspect_release_axis_source_artifact_provenance(
            &artifact,
            &artifact_path,
            &report,
            &mut issues,
        );
        if report.get("selected_backend").and_then(Value::as_str) != Some("cuda") {
            issues.push(format!(
                "source_artifact `{artifact}` selected_backend must be cuda"
            ));
        }
        inspect_source_artifact_case_integrity(
            &artifact,
            &report,
            "canonical CUDA release axes",
            &mut issues,
        );
        inspect_release_axis_source_artifact_metrics(&artifact, &report, &mut issues);
        source_reports.push(report);
    }
    inspect_release_axes_scalar_values(axes, &source_reports, &mut issues);
    issues
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report_fixture::EvidenceWorkspace;

    #[test]
    fn cuda_release_axes_reject_suite_status_inventory_drift() {
        let workspace = EvidenceWorkspace::new();
        let artifacts = workspace.cuda_release_axis_artifacts("release.inventory-drift", 0);
        let mut status_artifacts = artifacts.clone();
        status_artifacts[11] = "release/evidence/benchmarks/wgpu-workload-12.json".to_string();
        let artifact_statuses = status_artifacts
            .iter()
            .enumerate()
            .map(|(index, artifact)| {
                serde_json::json!({
                    "path": artifact,
                    "family_id": format!("family-{index:02}"),
                    "requested_case_id": format!("release.inventory-drift.{index}")
                })
            })
            .collect::<Vec<_>>();
        let axes = serde_json::json!({
            "warm_us_per_file": 17.0,
            "cold_pipeline_build_ms": 2.0,
            "gbs_scan_throughput": 4.0,
            "ulp_drift_max": 0,
            "max_vram_mib": 24576,
            "source_artifacts": artifacts.clone()
        });
        let cuda_suite = serde_json::json!({
            "backend": "cuda",
            "artifacts": artifacts,
            "artifact_statuses": artifact_statuses
        });

        let issues = cuda_release_axes_source_artifact_issues(workspace.path(), &axes, &cuda_suite);

        assert!(
            issues.iter().any(|issue| issue.contains(
                "cuda-release-suite lists artifact `release/evidence/benchmarks/workload-12.json` without matching artifact_statuses entry"
            )),
            "Fix: release axes must reject CUDA suite artifacts that lack matching status rows; issues={issues:?}"
        );
        assert!(
            issues.iter().any(|issue| issue.contains(
                "cuda-release-suite has artifact_statuses path `release/evidence/benchmarks/wgpu-workload-12.json` absent from artifacts"
            )),
            "Fix: release axes must reject stale or cross-backend suite status rows before bench-release consumes clean axes; issues={issues:?}"
        );
    }
}
