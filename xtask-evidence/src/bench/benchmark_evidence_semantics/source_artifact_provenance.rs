//! How one cited artifact's source provenance findings are worded, and whether a
//! release-axis artifact names its source precisely and currently.
//!
//! Every aggregate that cites artifacts reports the same four imprecisions in a
//! `source_fingerprint` and the same freshness mismatch, in the same sentences,
//! so the wording lives here once and each walk decides only when to ask.
//!
//! The release-axis walk asks per artifact rather than once for the aggregate,
//! because an aggregate regenerated today can still cite an artifact measured on
//! a tree that no longer exists.

use std::path::Path;

use serde_json::Value;

use super::data::{SourceFingerprintFreshnessIssue, SourceFingerprintIssue};
use super::json_reader::non_empty_str;
use super::source_fingerprint::{
    current_freshness_fingerprint_for_report, report_freshness_fingerprint,
    source_fingerprint_freshness_issues, source_fingerprint_issues,
};

/// Name every imprecision in one cited artifact's `source_fingerprint`.
///
/// A fingerprint that cannot distinguish two dirty worktrees does not identify
/// the source the artifact was measured on, so each imprecision is reported
/// against the artifact that carries it.
pub(crate) fn describe_source_artifact_fingerprint_issues(
    artifact: &str,
    source_fingerprint: &str,
    issues: &mut Vec<String>,
) {
    for issue in source_fingerprint_issues(source_fingerprint) {
        match issue {
            SourceFingerprintIssue::DirtyUnknownState { source_fingerprint } => {
                issues.push(format!(
                    "source_artifact `{artifact}` source_fingerprint `{source_fingerprint}` has unknown dirty state"
                ));
            }
            SourceFingerprintIssue::DirtyMissingWorktree { source_fingerprint } => {
                issues.push(format!(
                    "source_artifact `{artifact}` source_fingerprint `{source_fingerprint}` is dirty but has no worktree digest"
                ));
            }
            SourceFingerprintIssue::DirtyUnknownWorktree { source_fingerprint } => {
                issues.push(format!(
                    "source_artifact `{artifact}` source_fingerprint `{source_fingerprint}` is dirty but has unknown worktree digest"
                ));
            }
            SourceFingerprintIssue::DirtyInvalidWorktree {
                source_fingerprint,
                worktree,
            } => {
                issues.push(format!(
                    "source_artifact `{artifact}` source_fingerprint `{source_fingerprint}` has invalid worktree digest `{worktree}`"
                ));
            }
        }
    }
}

/// Name a cited artifact's source drift against the tree the checker is on.
pub(crate) fn describe_source_artifact_freshness_mismatch(
    artifact: &str,
    field: &str,
    source_fingerprint: &str,
    current_source_fingerprint: &str,
    issues: &mut Vec<String>,
) {
    for issue in source_fingerprint_freshness_issues(source_fingerprint, current_source_fingerprint)
    {
        match issue {
            SourceFingerprintFreshnessIssue::Mismatch {
                source_fingerprint,
                current_source_fingerprint,
            } => issues.push(format!(
                "source_artifact `{artifact}` {field} `{source_fingerprint}` does not match current workspace source `{current_source_fingerprint}`"
            )),
        }
    }
}

pub(crate) fn inspect_release_axis_source_artifact_provenance(
    artifact: &str,
    artifact_path: &Path,
    report: &Value,
    issues: &mut Vec<String>,
) {
    let source_fingerprint = report.get("source_fingerprint").and_then(non_empty_str);
    let source_tree_fingerprint = report
        .get("source_tree_fingerprint")
        .and_then(non_empty_str);
    let Some(source_fingerprint) = source_fingerprint else {
        issues.push(format!(
            "source_artifact `{artifact}` has no source_fingerprint"
        ));
        return inspect_release_axis_source_artifact_freshness(
            artifact,
            artifact_path,
            report,
            issues,
        );
    };
    describe_source_artifact_fingerprint_issues(artifact, source_fingerprint, issues);
    if source_tree_fingerprint.is_none() {
        issues.push(format!(
            "source_artifact `{artifact}` has no source_tree_fingerprint"
        ));
    }
    inspect_release_axis_source_artifact_freshness(artifact, artifact_path, report, issues);
}

fn inspect_release_axis_source_artifact_freshness(
    artifact: &str,
    artifact_path: &Path,
    report: &Value,
    issues: &mut Vec<String>,
) {
    let Some((field, source_fingerprint)) = report_freshness_fingerprint(report) else {
        return;
    };
    let Some(current_source_fingerprint) =
        current_freshness_fingerprint_for_report(artifact_path, report)
    else {
        issues.push(format!(
            "source_artifact `{artifact}` current workspace source fingerprint could not be resolved"
        ));
        return;
    };
    describe_source_artifact_freshness_mismatch(
        artifact,
        field,
        source_fingerprint,
        &current_source_fingerprint,
        issues,
    );
}

#[cfg(test)]
mod tests {
    use super::super::release_axes_cuda::cuda_release_axes_source_artifact_issues;
    use crate::report_fixture::EvidenceWorkspace;

    #[test]
    fn cuda_release_axes_reject_stale_and_weak_source_artifact_provenance() {
        let workspace = EvidenceWorkspace::new();
        let stale_artifact = workspace.write_report(
            "workload-stale.json",
            &serde_json::json!({
                "selected_backend": "cuda",
                "source_tree_fingerprint": "source-tree-v1:stale",
                "summary": {"total_cases": 1, "passed": 1, "failed": 0},
                "cases": [{"id": "release.stale", "status": "pass"}]
            }),
        );
        let weak_artifact = workspace.write_report(
            "workload-weak.json",
            &serde_json::json!({
                "selected_backend": "cuda",
                "source_fingerprint": "git:abc123:dirty=true",
                "summary": {"total_cases": 1, "passed": 1, "failed": 0},
                "cases": [{"id": "release.weak", "status": "pass"}]
            }),
        );
        let axes = serde_json::json!({
            "source_artifacts": [stale_artifact, weak_artifact]
        });
        let cuda_suite = serde_json::json!({
            "artifacts": [stale_artifact, weak_artifact]
        });

        let issues = cuda_release_axes_source_artifact_issues(workspace.path(), &axes, &cuda_suite);

        assert!(
            issues.iter().any(|issue| issue.contains(
                "source_artifact `release/evidence/benchmarks/workload-stale.json` source_tree_fingerprint `source-tree-v1:stale` does not match current workspace source"
            )),
            "Fix: release-axis source artifacts must be fresh against the current workspace; issues={issues:?}"
        );
        assert!(
            issues.iter().any(|issue| issue.contains(
                "source_artifact `release/evidence/benchmarks/workload-stale.json` has no source_fingerprint"
            )),
            "Fix: release-axis source artifacts must preserve explicit source_fingerprint provenance; issues={issues:?}"
        );
        assert!(
            issues.iter().any(|issue| issue.contains(
                "source_artifact `release/evidence/benchmarks/workload-weak.json` source_fingerprint `git:abc123:dirty=true` is dirty but has no worktree digest"
            )),
            "Fix: release-axis source artifacts must reject legacy dirty source fingerprints; issues={issues:?}"
        );
        assert!(
            issues.iter().any(|issue| issue.contains(
                "source_artifact `release/evidence/benchmarks/workload-weak.json` has no source_tree_fingerprint"
            )),
            "Fix: release-axis source artifacts must preserve source_tree_fingerprint provenance; issues={issues:?}"
        );
    }
}
