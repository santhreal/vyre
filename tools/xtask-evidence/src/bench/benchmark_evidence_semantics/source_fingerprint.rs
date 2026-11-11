//! Whether a recorded source fingerprint is precise, and whether it still
//! describes the tree the checker is running on.
//!
//! A measurement only describes the source it was taken on, so evidence names
//! that source twice: a git fingerprint that must be precise enough to
//! distinguish two dirty worktrees, and a source tree fingerprint that is
//! recomputed here and compared against the recorded one. The recomputation is
//! memoized per workspace root because a gate reads dozens of artifacts that
//! all resolve to the same tree.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex, MutexGuard};

use serde_json::Value;

use super::data::SourceFingerprintFreshnessIssue;

static CURRENT_SOURCE_FINGERPRINTS: LazyLock<Mutex<BTreeMap<PathBuf, String>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

static CURRENT_SOURCE_TREE_FINGERPRINTS: LazyLock<Mutex<BTreeMap<PathBuf, String>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// The state name the source fingerprint memo reports on poison.
const SOURCE_MEMO: &str = "the per-root source fingerprint memo";

/// The state name the source tree fingerprint memo reports on poison.
const TREE_MEMO: &str = "the per-root source tree fingerprint memo";

/// Take one fingerprint memo, discarding it after a panic.
///
/// Every entry is recomputed from the tree it names, so a half-written memo is
/// dropped and refilled rather than read. Clearing the poison flag keeps the
/// cost of one panic at one recomputation instead of one per lookup for the
/// life of the process.
fn lock_memo(
    memo: &'static Mutex<BTreeMap<PathBuf, String>>,
    state: &'static str,
) -> MutexGuard<'static, BTreeMap<PathBuf, String>> {
    vyre_foundation::failure_domain::govern_mutex_restartable(
        memo,
        "the evidence source fingerprint checker",
        state,
        BTreeMap::clear,
    )
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

pub(crate) fn report_freshness_fingerprint(report: &Value) -> Option<(&'static str, &str)> {
    for field in ["source_tree_fingerprint", "source_fingerprint"] {
        if let Some(value) = report
            .get(field)
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
        {
            return Some((field, value));
        }
    }
    None
}

pub(crate) fn current_freshness_fingerprint_for_report(
    path: &Path,
    report: &Value,
) -> Option<String> {
    if report
        .get("source_tree_fingerprint")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
    {
        return current_source_tree_fingerprint_for_evidence_path(path);
    }
    current_source_fingerprint_for_evidence_path(path)
}

fn current_source_tree_fingerprint_for_evidence_path(path: &Path) -> Option<String> {
    let workspace_root = workspace_root_for_evidence_path(path)?;
    Some(current_source_tree_fingerprint_at(&workspace_root))
}

fn current_source_fingerprint_for_evidence_path(path: &Path) -> Option<String> {
    let workspace_root = workspace_root_for_evidence_path(path)?;
    Some(current_source_fingerprint_at(&workspace_root))
}

fn current_source_fingerprint_at(workspace_root: &Path) -> String {
    let key = workspace_root
        .canonicalize()
        .unwrap_or_else(|_| workspace_root.to_path_buf());
    let memo = lock_memo(&CURRENT_SOURCE_FINGERPRINTS, SOURCE_MEMO);
    if let Some(fingerprint) = memo.get(&key) {
        return fingerprint.clone();
    }
    drop(memo);

    let git = vyre_bench::probes::capture_git_info_at(workspace_root);
    let fingerprint = vyre_bench::probes::source_fingerprint(&git);
    lock_memo(&CURRENT_SOURCE_FINGERPRINTS, SOURCE_MEMO).insert(key, fingerprint.clone());
    fingerprint
}

fn current_source_tree_fingerprint_at(workspace_root: &Path) -> String {
    let key = workspace_root
        .canonicalize()
        .unwrap_or_else(|_| workspace_root.to_path_buf());
    let memo = lock_memo(&CURRENT_SOURCE_TREE_FINGERPRINTS, TREE_MEMO);
    if let Some(fingerprint) = memo.get(&key) {
        return fingerprint.clone();
    }
    drop(memo);

    let fingerprint = vyre_bench::probes::source_tree_fingerprint_at(workspace_root);
    lock_memo(&CURRENT_SOURCE_TREE_FINGERPRINTS, TREE_MEMO).insert(key, fingerprint.clone());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report_fixture::EvidenceWorkspace;

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
        let workspace = EvidenceWorkspace::new();
        let evidence = workspace
            .path()
            .join("release/evidence/benchmarks/workload.json");

        let fingerprint = current_source_fingerprint_for_evidence_path(&evidence)
            .expect("Fix: resolve workspace source fingerprint from nested release evidence path.");

        assert!(
            fingerprint.starts_with("crate:"),
            "Fix: non-git test workspaces should still produce deterministic crate source provenance, got {fingerprint}."
        );
    }

    #[test]
    fn report_freshness_fingerprint_prefers_source_tree_scope() {
        let report = serde_json::json!({
            "source_fingerprint": "git:abc:dirty=false",
            "source_tree_fingerprint": "source-tree-v1:def",
        });

        assert_eq!(
            report_freshness_fingerprint(&report),
            Some(("source_tree_fingerprint", "source-tree-v1:def")),
            "Fix: current-source gates must prefer evidence-stable source tree provenance over commit-shaped legacy provenance."
        );
    }
}
