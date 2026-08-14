//! Which source artifacts a report names, and whether each named path is one a
//! checker may open.
//!
//! An aggregate report proves itself by pointing at the per-workload artifacts
//! it was derived from, so the list is only as good as the paths in it: a blank
//! entry, a repeat, an absolute path, a parent traversal, a path outside
//! `release/`, or a symlink out of the workspace each make the citation
//! unusable. Resolution is refused rather than followed.

use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};

use serde_json::Value;

use super::data::{BenchmarkArtifactPathIssue, MAX_BENCHMARK_EVIDENCE_SEMANTIC_TEXT_BYTES};
use super::json_reader::non_empty_str;

pub(crate) fn benchmark_report_has_source_provenance(report: &Value) -> bool {
    report
        .get("source_fingerprint")
        .and_then(non_empty_str)
        .is_some()
}

pub(crate) fn benchmark_source_artifact_count(report: &Value) -> usize {
    benchmark_source_artifact_paths(report).len()
}

pub(crate) fn benchmark_source_artifact_entry_count(report: &Value) -> usize {
    report
        .get("source_artifacts")
        .and_then(Value::as_array)
        .map_or(0, |items| items.iter().filter_map(non_empty_str).count())
}

pub(crate) fn benchmark_source_artifact_paths(report: &Value) -> BTreeSet<String> {
    report
        .get("source_artifacts")
        .and_then(Value::as_array)
        .map_or_else(BTreeSet::new, |items| {
            items
                .iter()
                .filter_map(non_empty_str)
                .map(str::to_string)
                .collect::<BTreeSet<_>>()
        })
}

pub(crate) fn benchmark_duplicate_source_artifact_paths(report: &Value) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    report
        .get("source_artifacts")
        .and_then(Value::as_array)
        .map_or_else(BTreeSet::new, |items| {
            items
                .iter()
                .filter_map(non_empty_str)
                .filter_map(|artifact| {
                    if seen.insert(artifact.to_string()) {
                        None
                    } else {
                        Some(artifact.to_string())
                    }
                })
                .collect::<BTreeSet<_>>()
        })
}

pub(crate) fn benchmark_source_artifact_path_issue(
    workspace_root: &Path,
    artifact: &str,
) -> Option<BenchmarkArtifactPathIssue> {
    benchmark_release_artifact_path_issue(workspace_root, artifact)
}

pub(crate) fn benchmark_suite_artifact_path_issue(
    workspace_root: &Path,
    artifact: &str,
) -> Option<BenchmarkArtifactPathIssue> {
    benchmark_release_artifact_path_issue(workspace_root, artifact)
}

fn benchmark_release_artifact_path_issue(
    workspace_root: &Path,
    artifact: &str,
) -> Option<BenchmarkArtifactPathIssue> {
    let candidate = PathBuf::from(artifact);
    if candidate.is_absolute() {
        return Some(BenchmarkArtifactPathIssue::AbsolutePath);
    }
    if !artifact.starts_with("release/") {
        return Some(BenchmarkArtifactPathIssue::NonReleasePath);
    }
    if candidate
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Some(BenchmarkArtifactPathIssue::ParentTraversal);
    }
    let artifact_path = workspace_root.join(&candidate);
    if !artifact_path.is_file() {
        return Some(BenchmarkArtifactPathIssue::Missing { artifact_path });
    }
    let Ok(canonical_root) = workspace_root.canonicalize() else {
        return Some(BenchmarkArtifactPathIssue::Missing { artifact_path });
    };
    let Ok(canonical_artifact) = artifact_path.canonicalize() else {
        return Some(BenchmarkArtifactPathIssue::Missing { artifact_path });
    };
    if !canonical_artifact.starts_with(&canonical_root) {
        return Some(BenchmarkArtifactPathIssue::OutsideWorkspace {
            artifact_path: canonical_artifact,
            workspace_root: canonical_root,
        });
    }
    None
}

fn resolve_benchmark_artifact_path(workspace_root: &Path, artifact: &str) -> PathBuf {
    let candidate = PathBuf::from(artifact);
    if candidate.is_absolute() {
        candidate
    } else {
        workspace_root.join(candidate)
    }
}

/// Open one cited source artifact, or name why the citation is unusable.
///
/// A path the checker may not follow, a file it cannot read, and text that is
/// not JSON are one finding to a caller: this artifact proves nothing, so skip
/// it. Every aggregate that cites artifacts starts its walk here, which is what
/// keeps the three sentences it can answer with identical between them.
pub(crate) fn read_cited_source_artifact(
    workspace_root: &Path,
    artifact: &str,
    issues: &mut Vec<String>,
) -> Option<(PathBuf, Value)> {
    if let Some(issue) = benchmark_source_artifact_path_issue(workspace_root, artifact) {
        issues.push(issue.describe("source_artifact", artifact));
        return None;
    }
    let artifact_path = resolve_benchmark_artifact_path(workspace_root, artifact);
    let text = match xtask::output_arg::read_text_bounded(
        &artifact_path,
        MAX_BENCHMARK_EVIDENCE_SEMANTIC_TEXT_BYTES,
        "evidence",
    ) {
        Ok(text) => text,
        Err(error) => {
            issues.push(format!(
                "source_artifact `{artifact}` is unreadable: {error}"
            ));
            return None;
        }
    };
    match serde_json::from_str::<Value>(&text) {
        Ok(report) => Some((artifact_path, report)),
        Err(error) => {
            issues.push(format!(
                "source_artifact `{artifact}` is invalid JSON: {error}"
            ));
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_source_provenance_rejects_artifact_paths_without_source_fingerprint() {
        let report = serde_json::json!({
            "source_artifacts": ["release/evidence/benchmarks/cuda.json"]
        });

        assert!(
            !benchmark_report_has_source_provenance(&report),
            "Fix: source_artifact paths identify evidence inputs; they must not satisfy benchmark source provenance without source_fingerprint."
        );
    }

    #[test]
    fn benchmark_source_provenance_rejects_git_commit_without_source_fingerprint() {
        assert!(
            !benchmark_report_has_source_provenance(&serde_json::json!({
                "git": {"commit": "abcdef"}
            })),
            "Fix: git.commit metadata is not a freshness-checked source_fingerprint and must not satisfy benchmark source provenance."
        );
    }

    #[test]
    fn benchmark_source_provenance_accepts_explicit_source_fingerprint() {
        assert!(
            benchmark_report_has_source_provenance(&serde_json::json!({
                "source_fingerprint": "git:0123456789abcdef0123456789abcdef01234567:dirty=false",
                "source_artifacts": ["release/evidence/benchmarks/cuda.json"],
                "git": {"commit": "abcdef"}
            })),
            "Fix: explicit source_fingerprint must satisfy benchmark source provenance."
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
        assert_eq!(
            benchmark_source_artifact_paths(&report),
            BTreeSet::from([
                "release/evidence/benchmarks/cuda-a.json".to_string(),
                "release/evidence/benchmarks/cuda-b.json".to_string(),
            ]),
            "Fix: source_artifact path extraction must expose the same unique usable paths used by release gates."
        );
    }

    #[test]
    fn benchmark_source_artifact_path_rejects_absolute_existing_file() {
        let dir = tempfile::TempDir::new()
            .expect("Fix: create temporary workspace for source artifact path test.");
        let artifact = dir.path().join("release/evidence/benchmarks/source.json");
        std::fs::create_dir_all(
            artifact
                .parent()
                .expect("Fix: source artifact fixture must have a parent directory."),
        )
        .expect("Fix: create temporary source artifact directory.");
        std::fs::write(&artifact, "{}").expect("Fix: write source artifact fixture.");

        assert_eq!(
            benchmark_source_artifact_path_issue(dir.path(), &artifact.display().to_string()),
            Some(BenchmarkArtifactPathIssue::AbsolutePath),
            "Fix: existing absolute source_artifact paths must not pass release evidence validation."
        );
    }

    #[test]
    fn benchmark_source_artifact_path_rejects_parent_traversal() {
        let dir = tempfile::TempDir::new()
            .expect("Fix: create temporary workspace for source artifact traversal test.");

        assert_eq!(
            benchmark_source_artifact_path_issue(
                dir.path(),
                "release/evidence/benchmarks/../../Cargo.toml"
            ),
            Some(BenchmarkArtifactPathIssue::ParentTraversal),
            "Fix: source_artifact validation must reject parent traversal before resolving files."
        );
    }

    #[test]
    fn benchmark_source_artifact_path_rejects_non_release_relative_path() {
        let dir = tempfile::TempDir::new()
            .expect("Fix: create temporary workspace for non-release artifact path test.");

        assert_eq!(
            benchmark_source_artifact_path_issue(dir.path(), "evidence/benchmarks/source.json"),
            Some(BenchmarkArtifactPathIssue::NonReleasePath),
            "Fix: source_artifact validation must keep benchmark evidence references under release/."
        );
    }

    #[test]
    fn benchmark_source_artifact_path_accepts_release_file_inside_workspace() {
        let dir = tempfile::TempDir::new()
            .expect("Fix: create temporary workspace for valid source artifact path test.");
        let artifact = dir.path().join("release/evidence/benchmarks/source.json");
        std::fs::create_dir_all(
            artifact
                .parent()
                .expect("Fix: source artifact fixture must have a parent directory."),
        )
        .expect("Fix: create temporary source artifact directory.");
        std::fs::write(&artifact, "{}").expect("Fix: write source artifact fixture.");

        assert_eq!(
            benchmark_source_artifact_path_issue(
                dir.path(),
                "release/evidence/benchmarks/source.json"
            ),
            None,
            "Fix: release/evidence source artifacts inside the workspace must remain valid."
        );
    }

    #[cfg(unix)]
    #[test]
    fn benchmark_source_artifact_path_rejects_symlink_escape() {
        let workspace = tempfile::TempDir::new()
            .expect("Fix: create temporary workspace for symlink source artifact test.");
        let outside = tempfile::TempDir::new()
            .expect("Fix: create external directory for symlink source artifact test.");
        let outside_artifact = outside.path().join("source.json");
        std::fs::write(&outside_artifact, "{}").expect("Fix: write external source artifact.");
        let link = workspace
            .path()
            .join("release/evidence/benchmarks/source.json");
        std::fs::create_dir_all(
            link.parent()
                .expect("Fix: symlink artifact fixture must have a parent directory."),
        )
        .expect("Fix: create temporary symlink source artifact directory.");
        std::os::unix::fs::symlink(&outside_artifact, &link)
            .expect("Fix: create source artifact symlink.");

        let Some(BenchmarkArtifactPathIssue::OutsideWorkspace { .. }) =
            benchmark_source_artifact_path_issue(
                workspace.path(),
                "release/evidence/benchmarks/source.json",
            )
        else {
            panic!("Fix: source_artifact validation must reject symlink escapes.");
        };
    }

    #[test]
    fn benchmark_duplicate_source_artifact_paths_report_repeated_usable_entries() {
        let report = serde_json::json!({
            "source_artifacts": [
                "",
                null,
                "release/evidence/benchmarks/cuda-a.json",
                "release/evidence/benchmarks/cuda-b.json",
                "release/evidence/benchmarks/cuda-a.json",
                "release/evidence/benchmarks/cuda-b.json",
                "release/evidence/benchmarks/cuda-c.json"
            ]
        });

        assert_eq!(
            benchmark_source_artifact_entry_count(&report),
            5,
            "Fix: raw source_artifact entry counts must ignore blank/non-string entries but preserve duplicate evidence attempts."
        );
        assert_eq!(
            benchmark_duplicate_source_artifact_paths(&report),
            BTreeSet::from([
                "release/evidence/benchmarks/cuda-a.json".to_string(),
                "release/evidence/benchmarks/cuda-b.json".to_string(),
            ]),
            "Fix: aggregate gates must identify duplicated source_artifact paths instead of letting them inflate proof counts."
        );
    }
}
