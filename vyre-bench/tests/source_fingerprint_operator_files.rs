//! Release source fingerprints exclude operator control files while retaining runtime source identity.

use std::{fs, path::Path, process::Command};

use tempfile::TempDir;
use vyre_bench::probes::source_tree_fingerprint_at;

fn workspace() -> TempDir {
    let workspace = tempfile::tempdir().expect("Fix: create source fingerprint workspace.");
    let output = Command::new("git")
        .args(["init", "--quiet"])
        .current_dir(workspace.path())
        .output()
        .expect("Fix: initialize source fingerprint git workspace.");
    assert!(
        output.status.success(),
        "Fix: initialize source fingerprint git workspace: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    fs::create_dir_all(workspace.path().join("src"))
        .expect("Fix: create source fingerprint fixture directory.");
    fs::write(
        workspace.path().join("src/lib.rs"),
        b"pub fn runtime() {}\n",
    )
    .expect("Fix: write source fingerprint runtime fixture.");
    workspace
}

fn write_fixture(workspace: &Path, relative_path: &str, contents: &[u8]) {
    let path = workspace.join(relative_path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("Fix: create operator-file fixture directory.");
    }
    fs::write(path, contents).expect("Fix: write operator-file fingerprint fixture.");
}

/// Operator guidance, plans, and backlog files do not change executable benchmark identity.
#[test]
fn exact_operator_internal_file_names_do_not_change_runtime_source_identity() {
    let workspace = workspace();
    let base = source_tree_fingerprint_at(workspace.path());
    let operator_files = [
        "AGENTS.md",
        "BACKLOG.md",
        "DEDUP_PLAN.md",
        "policy/CLAUDE.md",
        "nested/review/GEMINI.md",
        "skills/release/SKILL.md",
    ];

    for relative_path in operator_files {
        write_fixture(
            workspace.path(),
            relative_path,
            b"private operator guidance\n",
        );
        assert_eq!(
            source_tree_fingerprint_at(workspace.path()),
            base,
            "Fix: operator-internal file `{relative_path}` must not alter runtime source identity."
        );
        write_fixture(
            workspace.path(),
            relative_path,
            b"changed private guidance\n",
        );
        assert_eq!(
            source_tree_fingerprint_at(workspace.path()),
            base,
            "Fix: operator-internal file content `{relative_path}` must remain outside runtime source identity."
        );
    }
}

/// Platform-specific workspace launchers select the build but do not change runtime code.
#[test]
fn cargo_wrappers_do_not_change_runtime_source_identity() {
    let workspace = workspace();
    let base = source_tree_fingerprint_at(workspace.path());

    for relative_path in ["cargo_full", "cargo_full.cmd"] {
        write_fixture(
            workspace.path(),
            relative_path,
            b"workspace cargo launcher\n",
        );
        assert_eq!(
            source_tree_fingerprint_at(workspace.path()),
            base,
            "Fix: workspace wrapper `{relative_path}` must not alter runtime source identity."
        );
    }
}

/// Filename filtering must match whole basenames so production sources that merely contain an operator filename remain provenance-bearing.
#[test]
fn operator_like_source_names_and_runtime_changes_still_invalidate_identity() {
    let workspace = workspace();
    let base = source_tree_fingerprint_at(workspace.path());

    write_fixture(
        workspace.path(),
        "src/AGENTS.md.rs",
        b"pub fn operator_named_runtime() {}\n",
    );
    let operator_like_source = source_tree_fingerprint_at(workspace.path());
    assert_ne!(
        operator_like_source, base,
        "Fix: suffix lookalikes must remain part of runtime source identity."
    );

    write_fixture(
        workspace.path(),
        "src/lib.rs",
        b"pub fn runtime_changed() {}\n",
    );
    let runtime_changed = source_tree_fingerprint_at(workspace.path());
    assert_ne!(
        runtime_changed, operator_like_source,
        "Fix: real runtime source changes must still invalidate release evidence."
    );

    write_fixture(
        workspace.path(),
        "src/internal/AGENTS.md",
        b"nested private guidance\n",
    );
    assert_eq!(
        source_tree_fingerprint_at(workspace.path()),
        runtime_changed,
        "Fix: a nested exact operator basename must remain excluded after runtime changes."
    );
}
