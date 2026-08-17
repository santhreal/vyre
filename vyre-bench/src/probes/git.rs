//! Source provenance for benchmark and release evidence.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};
use vyre_foundation::hashing::update_length_delimited_field as update_hash_field;

const MAX_SOURCE_FINGERPRINT_FILE_BYTES: u64 = 64 * 1024 * 1024;

/// Git and source-tree provenance for evidence-producing runs.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SourceProvenance {
    /// Raw git facts captured from the source workspace.
    pub git: BTreeMap<String, String>,
    /// Commit/dirty-state source identity used by release evidence gates.
    pub source_fingerprint: String,
    /// Source-tree content identity used to tolerate evidence-only commit drift.
    pub source_tree_fingerprint: String,
}

impl SourceProvenance {
    /// Capture provenance for the current working directory.
    #[must_use]
    pub fn capture_current() -> Self {
        Self::capture_at(Path::new("."))
    }

    /// Capture provenance for `workspace_root`.
    #[must_use]
    pub fn capture_at(workspace_root: &Path) -> Self {
        let git = capture_git_info_at(workspace_root);
        let source_fingerprint = source_fingerprint(&git);
        let source_tree_fingerprint = source_tree_fingerprint_at(workspace_root);
        Self {
            git,
            source_fingerprint,
            source_tree_fingerprint,
        }
    }
}

/// Capture git facts for the current working directory.
#[must_use]
pub fn capture_git_info() -> BTreeMap<String, String> {
    capture_git_info_at(Path::new("."))
}

/// Capture git facts for `workspace_root`.
///
/// The dirty state and the worktree digest are not measured here. One producer
/// owns the fingerprint every recorded artifact names its tree with
/// (`xtask::source_provenance`), and this probe reads the facts back out of the
/// string it returns: a second implementation of that digest agreed with the
/// first only because a test compared them byte for byte.
#[must_use]
pub fn capture_git_info_at(workspace_root: &Path) -> BTreeMap<String, String> {
    let mut info = BTreeMap::new();

    if let Ok(commit) = shell(workspace_root, &["rev-parse", "HEAD"]) {
        info.insert("commit".to_string(), commit);
    }
    if let Ok(branch) = shell(workspace_root, &["rev-parse", "--abbrev-ref", "HEAD"]) {
        info.insert("branch".to_string(), branch);
    }
    match xtask::source_provenance::capture(workspace_root) {
        Ok(fingerprint) => {
            let dirty = fingerprint.contains(":dirty=true");
            if let Some(worktree) = fingerprint.split(":worktree=").nth(1) {
                info.insert(
                    "dirty_worktree_fingerprint".to_string(),
                    worktree.to_string(),
                );
            }
            info.insert("source_fingerprint".to_string(), fingerprint);
            info.insert("dirty".to_string(), dirty.to_string());
        }
        Err(_) => {
            info.insert("dirty".to_string(), "unknown".to_string());
        }
    }

    if let Ok(parent) = shell(workspace_root, &["rev-parse", "HEAD^"]) {
        info.insert("parent_commit".to_string(), parent);
    }
    if let Ok(timestamp) = shell(workspace_root, &["log", "-1", "--format=%ct"]) {
        info.insert("commit_timestamp".to_string(), timestamp);
    }

    info
}

/// The commit/dirty-state source fingerprint release evidence names a tree with.
///
/// The string is produced once, by the one owner, and carried in the map under
/// `source_fingerprint`. A checkout git cannot identify has no fingerprint, and
/// the crate identity is what a report carries then.
#[must_use]
pub fn source_fingerprint(git: &BTreeMap<String, String>) -> String {
    if let Some(fingerprint) = git
        .get("source_fingerprint")
        .filter(|fingerprint| !fingerprint.is_empty())
    {
        return fingerprint.clone();
    }
    format!(
        "crate:{}:{}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION")
    )
}

/// Capture the runtime source-tree fingerprint for the current working directory.
#[must_use]
pub fn source_tree_fingerprint() -> String {
    source_tree_fingerprint_at(Path::new("."))
}

/// Capture the runtime source-tree fingerprint for `workspace_root`.
///
/// Generated evidence, release tooling, tests, and operator-internal files are
/// excluded because they do not change the benchmarked runtime.
#[must_use]
pub fn source_tree_fingerprint_at(workspace_root: &Path) -> String {
    match shell_bytes(
        workspace_root,
        &[
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
    ) {
        Ok(paths) => format!(
            "source-tree-v1:{}",
            source_tree_fingerprint_from_paths(workspace_root, &paths)
        ),
        Err(_) => format!(
            "crate-source:{}:{}",
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION")
        ),
    }
}

fn source_tree_fingerprint_from_paths(workspace_root: &Path, paths: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    update_hash_field(&mut hasher, b"format", b"vyre-bench-source-tree-v1");
    for path in paths
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .filter(|path| !source_tree_path_is_benchmark_provenance_ignored(path))
    {
        update_hash_field(&mut hasher, b"path", path);
        let path = String::from_utf8_lossy(path);
        match read_source_fingerprint_file_bounded(&workspace_root.join(path.as_ref())) {
            Ok(Some(bytes)) => update_hash_field(&mut hasher, b"content", &bytes),
            Ok(None) => update_hash_field(
                &mut hasher,
                b"content-oversized",
                MAX_SOURCE_FINGERPRINT_FILE_BYTES.to_string().as_bytes(),
            ),
            Err(error) => {
                update_hash_field(&mut hasher, b"read-error", error.to_string().as_bytes())
            }
        }
    }
    hasher.finalize().to_hex().to_string()
}

fn source_tree_path_is_benchmark_provenance_ignored(path: &[u8]) -> bool {
    path == b"cargo_full"
        || path.starts_with(b".github/")
        || path.starts_with(b"release/evidence/")
        || path.starts_with(b"scripts/")
        || path.starts_with(b"xtask/")
        || path.starts_with(b"xtask-")
        || source_tree_path_is_operator_internal(path)
        || source_tree_path_is_test_evidence(path)
}

fn source_tree_path_is_operator_internal(path: &[u8]) -> bool {
    const FILE_NAMES: &[&[u8]] = &[
        b"AGENTS.md",
        b"BACKLOG.md",
        b"CLAUDE.md",
        b"GEMINI.md",
        b"SKILL.md",
    ];

    FILE_NAMES.iter().any(|file_name| {
        path == *file_name
            || path
                .strip_suffix(*file_name)
                .is_some_and(|prefix| prefix.ends_with(b"/"))
    })
}

fn source_tree_path_is_test_evidence(path: &[u8]) -> bool {
    path.starts_with(b"tests/")
        || path_contains(path, b"/tests/")
        || path.ends_with(b"/tests.rs")
        || path.ends_with(b"_tests.rs")
        || path.ends_with(b"_test.rs")
        || path_contains(path, b"_tests_")
        || path_contains(path, b"_test_")
}

fn path_contains(path: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && path.windows(needle.len()).any(|window| window == needle)
}

fn read_source_fingerprint_file_bounded(path: &Path) -> std::io::Result<Option<Vec<u8>>> {
    let mut reader = fs::File::open(path)?;
    let mut bytes = Vec::new();
    let mut total = 0u64;
    let mut chunk = [0u8; 8192];
    loop {
        let read = reader.read(&mut chunk)?;
        if read == 0 {
            return Ok(Some(bytes));
        }
        let read = read as u64;
        total = total.saturating_add(read);
        if total > MAX_SOURCE_FINGERPRINT_FILE_BYTES {
            return Ok(None);
        }
        bytes.extend_from_slice(&chunk[..read as usize]);
    }
}

fn shell(workspace_root: &Path, args: &[&str]) -> Result<String, String> {
    let stdout = shell_bytes(workspace_root, args)?;
    Ok(String::from_utf8_lossy(&stdout).trim().to_string())
}

fn shell_bytes(workspace_root: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(workspace_root)
        .output()
        .map_err(|e| e.to_string())?;
    if output.status.success() {
        Ok(output.stdout)
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}
