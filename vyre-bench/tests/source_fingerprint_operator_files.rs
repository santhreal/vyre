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

/// A tree the benchmarked runtime never reads does not change benchmark identity.
///
/// WHY: the fingerprint decides whether a recorded measurement still describes
/// the tree. A generated document, the release paperwork it lands beside, the
/// assurance tooling that writes it, and the tests are each produced from the
/// runtime rather than read by it, so rewriting one invalidated every artifact
/// on disk and forced a whole re-measurement for a file no kernel reads. Each
/// prefix gets a case, so dropping one from the predicate turns this red.
///
/// What this does not catch: a path outside every prefix that the runtime also
/// never reads. The fingerprint counts it, which is the safe direction.
#[test]
fn trees_the_runtime_never_reads_do_not_change_benchmark_identity() {
    let workspace = workspace();
    let base = source_tree_fingerprint_at(workspace.path());
    let excluded = [
        "CHANGELOG.md",
        "docs/generated/op-inventory.toml",
        "release/changes/unreleased/a-fix.toml",
        "release/evidence/benchmarks/workload-01.json",
        ".github/workflows/ci.yml",
        "scripts/release.sh",
        "xtask/src/main.rs",
        "xtask-evidence/src/lib.rs",
        "vyre-crate/tests/all_tests.rs",
        "vyre-crate/src/feature_tests.rs",
    ];

    for relative_path in excluded {
        write_fixture(workspace.path(), relative_path, b"generated content\n");
        assert_eq!(
            source_tree_fingerprint_at(workspace.path()),
            base,
            "Fix: `{relative_path}` must not alter runtime source identity."
        );
    }
}

/// A hand-authored document under `docs/` is still runtime source identity.
///
/// WHY: the generated tree is excluded by prefix, and the prefix one directory
/// up would drop every reference and architecture document with it.
#[test]
fn a_hand_authored_document_still_changes_benchmark_identity() {
    let workspace = workspace();
    let base = source_tree_fingerprint_at(workspace.path());

    write_fixture(
        workspace.path(),
        "docs/reference/values.md",
        b"the contract a caller reads\n",
    );
    assert_ne!(
        source_tree_fingerprint_at(workspace.path()),
        base,
        "Fix: a hand-authored document must remain part of runtime source identity."
    );
}

/// Every document a crate compiles in keys the measurement.
///
/// WHY: the exclusion list is a set of prefixes, and each one added to stop a
/// paperwork rewrite from invalidating a measurement can also drop a file the
/// runtime is built from. A measurement keyed to a fingerprint that ignores a
/// compiled-in table is a number attributed to the wrong binary. The set is
/// read out of the workspace at run time, so a crate that starts including a
/// document under an excluded prefix turns this red without anyone listing it.
///
/// What this does not catch: a file a crate opens at run time by path rather
/// than compiling in. `include_str!` and `include_bytes!` are what the build
/// records.
#[test]
fn every_compiled_in_document_keys_the_measurement() {
    let root = vyre_test_support::monorepo::vyre_workspace_root();
    let included = compiled_in_paths(&root);
    assert!(
        !included.is_empty(),
        "Fix: the workspace compiles in at least one document; the scan found none, so it is \
         reading the wrong tree at {}",
        root.display()
    );

    let workspace = workspace();
    for relative_path in &included {
        let base = source_tree_fingerprint_at(workspace.path());
        write_fixture(workspace.path(), relative_path, b"compiled-in content\n");
        assert_ne!(
            source_tree_fingerprint_at(workspace.path()),
            base,
            "Fix: `{relative_path}` is compiled into a crate, so it must key the measurement. \
             Narrow the prefix that excludes it."
        );
        fs::remove_file(workspace.path().join(relative_path))
            .expect("Fix: remove the compiled-in fixture.");
    }
}

/// Workspace-relative paths a runtime crate compiles in with `include_str!` or
/// `include_bytes!`, excluding a crate's own sources.
///
/// A source under the assurance tooling is skipped: the tooling is not linked
/// into a benchmarked binary, so what it compiles in does not key a
/// measurement either.
fn compiled_in_paths(root: &Path) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for source in runtime_rust_sources(root) {
        let text = fs::read_to_string(&source).unwrap_or_default();
        let directory = source.parent().unwrap_or(root).to_path_buf();
        for macro_name in ["include_str!(", "include_bytes!("] {
            let mut rest = text.as_str();
            while let Some(at) = rest.find(macro_name) {
                rest = &rest[at + macro_name.len()..];
                let Some(open) = rest.find('"') else { break };
                let Some(close) = rest[open + 1..].find('"') else {
                    break;
                };
                let literal = &rest[open + 1..open + 1 + close];
                rest = &rest[open + 1 + close..];
                let Ok(resolved) = directory.join(literal).canonicalize() else {
                    continue;
                };
                let Ok(relative) = resolved.strip_prefix(root) else {
                    continue;
                };
                let relative = relative.to_string_lossy().replace('\\', "/");
                if relative.contains("/src/") || relative.starts_with("src/") {
                    continue;
                }
                if !found.contains(&relative) {
                    found.push(relative);
                }
            }
        }
    }
    found.sort();
    found
}

/// Every tracked `.rs` file that a benchmarked binary can be built from.
fn runtime_rust_sources(root: &Path) -> Vec<std::path::PathBuf> {
    let output = Command::new("git")
        .args(["ls-files", "-z", "--", "*.rs"])
        .current_dir(root)
        .output()
        .expect("Fix: list the workspace Rust sources.");
    assert!(
        output.status.success(),
        "Fix: `git ls-files` failed in {}",
        root.display()
    );
    output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| String::from_utf8_lossy(path).replace('\\', "/"))
        .filter(|relative| !relative.starts_with("xtask"))
        .filter(|relative| !relative.starts_with("scripts/"))
        .filter(|relative| !relative.contains("/tests/") && !relative.starts_with("tests/"))
        .map(|relative| root.join(relative))
        .collect()
}
