//! The CPU-SOTA 100x aggregate proof read against the artifacts it cites.
//!
//! The aggregate claims a source identity of its own, so each cited artifact is
//! checked against the aggregate's fingerprints as well as against the current
//! tree, and its cases are checked for backend drift, wrong-backend contracts
//! and borrowed CUDA dispatch. Source identity is decided on the source tree
//! fingerprint rather than the evidence commit, so an artifact measured on the
//! same source from a different commit is not rejected for that alone.

use std::path::Path;

use serde_json::Value;

use super::json_reader::non_empty_str;
use super::source_artifact::{benchmark_source_artifact_paths, read_cited_source_artifact};
use super::source_artifact_integrity::inspect_source_artifact_case_integrity;
use super::source_artifact_provenance::{
    describe_source_artifact_fingerprint_issues, describe_source_artifact_freshness_mismatch,
};
use super::source_fingerprint::{
    current_freshness_fingerprint_for_report, report_freshness_fingerprint,
};

pub(crate) fn cpu_sota_100x_source_artifact_issues(
    workspace_root: &Path,
    proof: &Value,
) -> Vec<String> {
    let mut issues = Vec::new();
    let aggregate_source_tree_fingerprint =
        proof.get("source_tree_fingerprint").and_then(non_empty_str);
    for artifact in benchmark_source_artifact_paths(proof) {
        let Some((artifact_path, report)) =
            read_cited_source_artifact(workspace_root, &artifact, &mut issues)
        else {
            continue;
        };
        if report.get("selected_backend").and_then(Value::as_str) != Some("cuda") {
            issues.push(format!(
                "source_artifact `{artifact}` was not produced for cuda"
            ));
        }
        inspect_source_artifact_case_integrity(
            &artifact,
            &report,
            "CPU-SOTA aggregate proof",
            &mut issues,
        );
        let report_source_fingerprint = report.get("source_fingerprint").and_then(non_empty_str);
        if let Some(fingerprint) = report_source_fingerprint {
            describe_source_artifact_fingerprint_issues(&artifact, fingerprint, &mut issues);
        } else {
            issues.push(format!(
                "source_artifact `{artifact}` has no source_fingerprint"
            ));
        }
        let report_source_tree_fingerprint = report
            .get("source_tree_fingerprint")
            .and_then(non_empty_str);
        match (
            aggregate_source_tree_fingerprint,
            report_source_tree_fingerprint,
        ) {
            (_, None) => issues.push(format!(
                "source_artifact `{artifact}` has no source_tree_fingerprint"
            )),
            (Some(aggregate), Some(fingerprint)) if fingerprint != aggregate => {
                issues.push(format!(
                    "source_artifact `{artifact}` source_tree_fingerprint `{fingerprint}` does not match aggregate source tree `{aggregate}`"
                ));
            }
            _ => {}
        }
        if let (Some((field, source_fingerprint)), Some(current_source_fingerprint)) = (
            report_freshness_fingerprint(&report),
            current_freshness_fingerprint_for_report(&artifact_path, &report),
        ) {
            describe_source_artifact_freshness_mismatch(
                &artifact,
                field,
                source_fingerprint,
                &current_source_fingerprint,
                &mut issues,
            );
        }
    }
    issues
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report_fixture::EvidenceWorkspace;

    #[test]
    fn cpu_sota_100x_source_artifacts_reject_weak_and_stale_provenance() {
        let workspace = EvidenceWorkspace::new();
        let weak_artifact = workspace.write_report(
            "cuda-weak-source.json",
            &serde_json::json!({
                "selected_backend": "cuda",
                "source_fingerprint": "git:abc123:dirty=true",
                "source_tree_fingerprint": workspace.source_tree_fingerprint(),
                "summary": {"total_cases": 0, "passed": 0, "failed": 0},
                "cases": []
            }),
        );
        let stale_artifact = workspace.write_report(
            "cuda-stale-source-tree.json",
            &serde_json::json!({
                "selected_backend": "cuda",
                "source_fingerprint": workspace.source_fingerprint(),
                "source_tree_fingerprint": "source-tree-v1:stale",
                "summary": {"total_cases": 0, "passed": 0, "failed": 0},
                "cases": []
            }),
        );
        let proof = serde_json::json!({
            "source_fingerprint": workspace.source_fingerprint(),
            "source_tree_fingerprint": workspace.source_tree_fingerprint(),
            "source_artifacts": [weak_artifact, stale_artifact]
        });

        let issues = cpu_sota_100x_source_artifact_issues(workspace.path(), &proof);

        assert!(
            issues.iter().any(|issue| issue.contains(
                "source_artifact `release/evidence/benchmarks/cuda-weak-source.json` source_fingerprint `git:abc123:dirty=true` is dirty but has no worktree digest"
            )),
            "Fix: CPU-SOTA aggregate source artifacts must reject weak dirty source_fingerprint provenance; issues={issues:?}"
        );
        assert!(
            !issues.iter().any(|issue| issue.contains(
                "source_artifact `release/evidence/benchmarks/cuda-weak-source.json` source_fingerprint `git:abc123:dirty=true` does not match aggregate source"
            )),
            "Fix: CPU-SOTA aggregate source artifacts must rely on source_tree_fingerprint for source identity instead of raw evidence commit equality; issues={issues:?}"
        );
        assert!(
            issues.iter().any(|issue| issue.contains(
                "source_artifact `release/evidence/benchmarks/cuda-stale-source-tree.json` source_tree_fingerprint `source-tree-v1:stale` does not match aggregate source tree"
            )),
            "Fix: CPU-SOTA aggregate source artifacts must match the aggregate source tree fingerprint; issues={issues:?}"
        );
        assert!(
            issues.iter().any(|issue| issue.contains(
                "source_artifact `release/evidence/benchmarks/cuda-stale-source-tree.json` source_tree_fingerprint `source-tree-v1:stale` does not match current workspace source"
            )),
            "Fix: CPU-SOTA aggregate source artifacts must be fresh against the current workspace; issues={issues:?}"
        );
    }

    #[test]
    fn cpu_sota_100x_source_artifacts_reject_backend_drift_and_borrowed_cuda_telemetry() {
        let workspace = EvidenceWorkspace::new();
        let artifact = workspace.write_cuda_release_artifact(
            "cuda-cpu-sota-drift.json",
            serde_json::json!([
                {
                    "id": "release.cpu-sota-drift",
                    "backend_id": "wgpu",
                    "status": "pass",
                    "optimization_passes_applied": ["cuda-resident-borrowed-escape-hatch"],
                    "metrics": {
                        "cuda_resident_borrowed_fallback_dispatches": {"p50": 3.0}
                    }
                }
            ]),
        );
        let proof = serde_json::json!({
            "source_fingerprint": workspace.source_fingerprint(),
            "source_tree_fingerprint": workspace.source_tree_fingerprint(),
            "source_artifacts": [artifact]
        });

        let issues = cpu_sota_100x_source_artifact_issues(workspace.path(), &proof);

        assert!(
            issues.iter().any(|issue| issue.contains(
                "source_artifact `release/evidence/benchmarks/cuda-cpu-sota-drift.json` case `release.cpu-sota-drift` backend_id `wgpu` does not match selected_backend `cuda`"
            )),
            "Fix: CPU-SOTA aggregate source artifacts must reject case-level backend drift before proof counts can imply CUDA coverage; issues={issues:?}"
        );
        assert!(
            issues.iter().any(|issue| issue.contains(
                "source_artifact `release/evidence/benchmarks/cuda-cpu-sota-drift.json` case `release.cpu-sota-drift` has cuda_resident_borrowed_fallback_dispatches p50=3"
            ) && issue.contains("CPU-SOTA aggregate proof must use native resident dispatch")),
            "Fix: CPU-SOTA aggregate source artifacts must reject borrowed resident CUDA dispatch evidence; issues={issues:?}"
        );
    }

    #[test]
    fn cpu_sota_100x_source_artifacts_reject_wrong_backend_contracts() {
        let workspace = EvidenceWorkspace::new();
        let artifact = workspace.write_cuda_release_artifact(
            "cuda-wrong-contract.json",
            serde_json::json!([
                {
                    "id": "release.cpu-sota-wrong-contract",
                    "backend_id": "cuda",
                    "status": "pass",
                    "contract": {
                        "baselines": [
                            {
                                "class": "CpuSota",
                                "backend_ids": ["wgpu"],
                                "min_speedup_x": 100.0
                            }
                        ]
                    },
                    "metrics": {
                        "wall_ns": {"p50": 10},
                        "baseline_wall_ns": {"p50": 2000}
                    },
                    "performance": {"contract_passed": true, "speedup_x": 200.0}
                }
            ]),
        );
        let proof = serde_json::json!({
            "source_fingerprint": workspace.source_fingerprint(),
            "source_tree_fingerprint": workspace.source_tree_fingerprint(),
            "source_artifacts": [artifact]
        });

        let issues = cpu_sota_100x_source_artifact_issues(workspace.path(), &proof);

        assert!(
            issues.iter().any(|issue| issue.contains(
                "source_artifact `release/evidence/benchmarks/cuda-wrong-contract.json` case `release.cpu-sota-wrong-contract` backend `cuda` has no applicable performance contract baseline"
            )),
            "Fix: CPU-SOTA aggregate source artifacts must reject CUDA cases whose performance contract only applies to WGPU; issues={issues:?}"
        );
    }
}
