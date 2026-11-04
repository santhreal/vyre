//! Whether a backend suite's own inventory is internally consistent, and whether
//! it covers the workload matrix.
//!
//! A suite declares its artifacts, its status rows, its family coverage and its
//! family count, and each of those is checkable against the others: an artifact
//! with no status row, a path or family listed twice, a declared count that does
//! not match the rows. Coverage is the same question asked outward, against the
//! workload families the matrix requires.

use std::collections::BTreeSet;

use serde_json::Value;

use super::data::{BackendSuiteInventoryIssue, BackendSuiteMatrixCoverageIssue};
use super::json_reader::non_empty_str;
use super::suite_reader::{suite_array_len, suite_artifact_path_counts, suite_status_counts};

pub(crate) fn backend_suite_inventory_issues(suite: &Value) -> Vec<BackendSuiteInventoryIssue> {
    let artifact_count = suite_array_len(suite, "artifacts");
    let status_count = suite_array_len(suite, "artifact_statuses");
    let artifact_counts = suite_artifact_path_counts(suite);
    let status_counts = suite_status_counts(suite, "path");
    let status_family_counts = suite_status_counts(suite, "family_id");
    let artifact_paths = artifact_counts.keys().cloned().collect::<BTreeSet<_>>();
    let status_paths = status_counts.keys().cloned().collect::<BTreeSet<_>>();
    let mut issues = Vec::new();

    if artifact_count != status_count {
        issues.push(BackendSuiteInventoryIssue::CountMismatch {
            artifact_count,
            status_count,
        });
    }
    if let Some(family_count) = suite.get("family_count").and_then(Value::as_u64) {
        if family_count as usize != artifact_count {
            issues.push(
                BackendSuiteInventoryIssue::DeclaredFamilyArtifactCountMismatch {
                    family_count,
                    artifact_count,
                },
            );
        }
        if family_count as usize != status_family_counts.len() {
            issues.push(
                BackendSuiteInventoryIssue::DeclaredFamilyStatusCountMismatch {
                    family_count,
                    status_family_count: status_family_counts.len(),
                },
            );
        }
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
    for (family_id, count) in status_family_counts {
        if count > 1 {
            issues.push(BackendSuiteInventoryIssue::DuplicateFamily { family_id, count });
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

pub(crate) fn backend_suite_matrix_coverage_issues(
    matrix: &Value,
    suite: &Value,
) -> Vec<BackendSuiteMatrixCoverageIssue> {
    let matrix_family_ids = matrix
        .get("families")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|family| family.get("id").and_then(non_empty_str))
        .map(str::to_string)
        .collect::<BTreeSet<_>>();
    let suite_family_ids = suite_status_counts(suite, "family_id")
        .into_keys()
        .collect::<BTreeSet<_>>();
    let mut issues = Vec::new();

    if matrix_family_ids.len() != suite_family_ids.len() {
        issues.push(BackendSuiteMatrixCoverageIssue::FamilyCountMismatch {
            matrix_family_count: matrix_family_ids.len(),
            suite_family_count: suite_family_ids.len(),
        });
    }
    for family_id in matrix_family_ids.difference(&suite_family_ids) {
        issues.push(BackendSuiteMatrixCoverageIssue::MissingMatrixFamily {
            family_id: family_id.clone(),
        });
    }
    for family_id in suite_family_ids.difference(&matrix_family_ids) {
        issues.push(BackendSuiteMatrixCoverageIssue::ExtraSuiteFamily {
            family_id: family_id.clone(),
        });
    }
    issues
}

pub(crate) fn describe_backend_suite_matrix_coverage_issue(
    issue: &BackendSuiteMatrixCoverageIssue,
) -> String {
    match issue {
        BackendSuiteMatrixCoverageIssue::FamilyCountMismatch {
            matrix_family_count,
            suite_family_count,
        } => format!(
            "covers {suite_family_count} workload family/families, but release-workload-matrix lists {matrix_family_count}"
        ),
        BackendSuiteMatrixCoverageIssue::MissingMatrixFamily { family_id } => {
            format!("is missing release-workload-matrix family `{family_id}`")
        }
        BackendSuiteMatrixCoverageIssue::ExtraSuiteFamily { family_id } => {
            format!("contains family `{family_id}` absent from release-workload-matrix")
        }
    }
}

pub(crate) fn describe_backend_suite_inventory_issue(issue: &BackendSuiteInventoryIssue) -> String {
    match issue {
        BackendSuiteInventoryIssue::CountMismatch {
            artifact_count,
            status_count,
        } => {
            format!("inventory count mismatch: artifacts={artifact_count}, artifact_statuses={status_count}")
        }
        BackendSuiteInventoryIssue::DeclaredFamilyArtifactCountMismatch {
            family_count,
            artifact_count,
        } => {
            format!("family_count={family_count}, but artifacts has {artifact_count} row(s)")
        }
        BackendSuiteInventoryIssue::DeclaredFamilyStatusCountMismatch {
            family_count,
            status_family_count,
        } => {
            format!(
                "family_count={family_count}, but artifact_statuses has {status_family_count} unique family_id row(s)"
            )
        }
        BackendSuiteInventoryIssue::MissingStatus { path } => {
            format!("lists artifact `{path}` without matching artifact_statuses entry")
        }
        BackendSuiteInventoryIssue::MissingArtifact { path } => {
            format!("has artifact_statuses path `{path}` absent from artifacts")
        }
        BackendSuiteInventoryIssue::DuplicateArtifact { path } => {
            format!("has duplicate artifact `{path}`")
        }
        BackendSuiteInventoryIssue::DuplicateStatus { path } => {
            format!("has duplicate artifact_statuses path `{path}`")
        }
        BackendSuiteInventoryIssue::DuplicateFamily { family_id, count } => {
            format!("has {count} artifact_statuses rows for family `{family_id}`")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn backend_suite_inventory_rejects_duplicate_family_coverage() {
        let suite = serde_json::json!({
            "artifacts": [
                "release/evidence/benchmarks/cuda/condition-fast.json",
                "release/evidence/benchmarks/cuda/condition-slow.json"
            ],
            "artifact_statuses": [
                {
                    "path": "release/evidence/benchmarks/cuda/condition-fast.json",
                    "family_id": "condition-eval",
                    "requested_case_id": "release.condition_eval.1m"
                },
                {
                    "path": "release/evidence/benchmarks/cuda/condition-slow.json",
                    "family_id": "condition-eval",
                    "requested_case_id": "release.condition_eval.10m"
                }
            ]
        });

        assert_eq!(
            backend_suite_inventory_issues(&suite),
            vec![BackendSuiteInventoryIssue::DuplicateFamily {
                family_id: "condition-eval".to_string(),
                count: 2,
            }],
            "Fix: backend suite family_count must represent unique workload families, not repeated family rows."
        );
    }

    #[test]
    fn backend_suite_inventory_rejects_declared_family_count_drift() {
        let artifacts = (0..12)
            .map(|index| format!("release/evidence/benchmarks/cuda/workload-{index}.json"))
            .collect::<Vec<_>>();
        let artifact_statuses = artifacts
            .iter()
            .enumerate()
            .map(|(index, path)| {
                serde_json::json!({
                    "path": path,
                    "family_id": format!("workload-{index}"),
                    "requested_case_id": format!("release.workload_{index}.1m")
                })
            })
            .collect::<Vec<_>>();
        let suite = serde_json::json!({
            "family_count": 13,
            "artifacts": artifacts,
            "artifact_statuses": artifact_statuses
        });

        assert_eq!(
            backend_suite_inventory_issues(&suite),
            vec![
                BackendSuiteInventoryIssue::DeclaredFamilyArtifactCountMismatch {
                    family_count: 13,
                    artifact_count: 12,
                },
                BackendSuiteInventoryIssue::DeclaredFamilyStatusCountMismatch {
                    family_count: 13,
                    status_family_count: 12,
                },
            ],
            "Fix: backend suite family_count must be derived from suite rows, not trusted as a stale release total."
        );
    }

    #[test]
    fn backend_suite_matrix_coverage_rejects_missing_optional_workloads() {
        let matrix = serde_json::json!({
            "families": [
                {"id": "condition-eval"},
                {"id": "compound-fused-filter"},
                {"id": "adaptive-routing"}
            ]
        });
        let suite = serde_json::json!({
            "artifact_statuses": [
                {"family_id": "condition-eval"}
            ]
        });

        assert_eq!(
            backend_suite_matrix_coverage_issues(&matrix, &suite),
            vec![
                BackendSuiteMatrixCoverageIssue::FamilyCountMismatch {
                    matrix_family_count: 3,
                    suite_family_count: 1,
                },
                BackendSuiteMatrixCoverageIssue::MissingMatrixFamily {
                    family_id: "adaptive-routing".to_string(),
                },
                BackendSuiteMatrixCoverageIssue::MissingMatrixFamily {
                    family_id: "compound-fused-filter".to_string(),
                },
            ],
            "Fix: backend suites must cover every release workload matrix family, including optional acceleration workloads."
        );
    }

    #[test]
    fn backend_suite_matrix_coverage_rejects_extra_suite_family() {
        let matrix = serde_json::json!({
            "families": [
                {"id": "condition-eval"}
            ]
        });
        let suite = serde_json::json!({
            "artifact_statuses": [
                {"family_id": "condition-eval"},
                {"family_id": "stale-family"}
            ]
        });

        assert_eq!(
            backend_suite_matrix_coverage_issues(&matrix, &suite),
            vec![
                BackendSuiteMatrixCoverageIssue::FamilyCountMismatch {
                    matrix_family_count: 1,
                    suite_family_count: 2,
                },
                BackendSuiteMatrixCoverageIssue::ExtraSuiteFamily {
                    family_id: "stale-family".to_string(),
                },
            ],
            "Fix: backend suites must not carry stale family rows outside the release workload matrix."
        );
    }
}
