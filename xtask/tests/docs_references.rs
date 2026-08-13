//! Published documentation reference contract tests.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use tempfile::TempDir;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("Fix: xtask must remain directly under the workspace root")
        .to_path_buf()
}

fn run_checker(root: &Path) -> Output {
    Command::new("python3")
        .arg(workspace_root().join("scripts/check_docs_references.py"))
        .arg(root)
        .output()
        .expect("Fix: documentation reference checker must launch")
}

fn fixture() -> TempDir {
    let root = tempfile::tempdir().expect("Fix: temporary fixture directory must be writable");
    fs::create_dir_all(root.path().join("docs")).unwrap();
    fs::write(
        root.path().join("Cargo.toml"),
        "[workspace]\nresolver = \"2\"\nmembers = []\n",
    )
    .unwrap();
    fs::write(root.path().join("README.md"), "# Fixture\n").unwrap();
    let status = Command::new("git")
        .args(["init", "-q"])
        .current_dir(root.path())
        .status()
        .expect("Fix: git must launch for ignored-path fixture semantics");
    assert!(status.success());
    root
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// Locks out regressions where the gate passes fixtures but no longer checks the real published corpus.
#[test]
fn current_workspace_references_resolve_to_published_inputs() {
    let output = run_checker(&workspace_root());
    assert!(output.status.success(), "{}", stderr(&output));
    assert!(String::from_utf8_lossy(&output.stdout)
        .contains("path-like code spans and command inputs resolve"));
}

/// Prevents a backticked nonexistent repository path from evading ordinary Markdown link validation.
#[test]
fn missing_path_like_code_span_fails_closed() {
    let root = fixture();
    fs::write(
        root.path().join("docs/guide.md"),
        "Read `docs/missing-input.md` before running the command.\n",
    )
    .unwrap();

    let output = run_checker(root.path());
    assert!(!output.status.success());
    assert!(stderr(&output).contains(
        "MISSING docs/guide.md [code span: docs/missing-input.md] -> docs/missing-input.md"
    ));
}

/// Prevents runnable examples from naming an input file that a clean checkout cannot provide.
#[test]
fn missing_command_input_fails_closed() {
    let root = fixture();
    fs::write(
        root.path().join("docs/guide.md"),
        "```bash\ncat docs/missing.json\n```\n",
    )
    .unwrap();

    let output = run_checker(root.path());
    assert!(!output.status.success());
    assert!(stderr(&output)
        .contains("MISSING docs/guide.md [command: docs/missing.json] -> docs/missing.json"));
}

/// Prevents a locally present but unpublished input from making public instructions appear reproducible.
#[test]
fn gitignored_input_fails_closed() {
    let root = fixture();
    fs::write(root.path().join(".gitignore"), "docs/private.json\n").unwrap();
    fs::write(root.path().join("docs/private.json"), "{}\n").unwrap();
    fs::write(
        root.path().join("docs/guide.md"),
        "Load `docs/private.json`.\n",
    )
    .unwrap();

    let output = run_checker(root.path());
    assert!(!output.status.success());
    assert!(stderr(&output)
        .contains("GITIGNORED docs/guide.md [code span: docs/private.json] -> docs/private.json"));
}

/// Keeps generated destinations usable in examples while still checking the published input beside them.
#[test]
fn output_destinations_do_not_need_to_preexist() {
    let root = fixture();
    fs::create_dir_all(root.path().join("inputs")).unwrap();
    fs::write(root.path().join("inputs/table.json"), "{}\n").unwrap();
    fs::write(
        root.path().join("docs/guide.md"),
        "```bash\ntool emit --out-dir ./generated/ --lr-json inputs/table.json\ntool emit --output docs/generated.json --lr-json inputs/table.json\n```\n",
    )
    .unwrap();

    let output = run_checker(root.path());
    assert!(output.status.success(), "{}", stderr(&output));
}

/// Ensures wildcard evidence references prove that at least one published artifact exists.
#[test]
fn wildcard_input_requires_a_published_match() {
    let root = fixture();
    fs::write(
        root.path().join("docs/guide.md"),
        "Inspect `docs/evidence-*.json`.\n",
    )
    .unwrap();

    let missing = run_checker(root.path());
    assert!(!missing.status.success());
    assert!(stderr(&missing).contains("MISSING docs/guide.md"));

    fs::write(root.path().join("docs/evidence-01.json"), "{}\n").unwrap();
    let present = run_checker(root.path());
    assert!(present.status.success(), "{}", stderr(&present));
}

/// Keeps archived evidence from becoming an executable public-input contract after it is superseded.
#[test]
fn archived_documents_are_not_scanned_as_current_instructions() {
    let root = fixture();
    fs::create_dir_all(root.path().join("docs/archive")).unwrap();
    fs::write(
        root.path().join("docs/archive/old.md"),
        "Historical input: `docs/removed.json`.\n",
    )
    .unwrap();

    let output = run_checker(root.path());
    assert!(output.status.success(), "{}", stderr(&output));
}

/// Prevents generated per-crate guides from bypassing the same clean-checkout input contract as root docs.
#[test]
fn workspace_crate_readmes_are_scanned() {
    let root = fixture();
    fs::create_dir_all(root.path().join("crate-a")).unwrap();
    fs::write(
        root.path().join("Cargo.toml"),
        "[workspace]\nresolver = \"2\"\nmembers = [\"crate-a\"]\n",
    )
    .unwrap();
    fs::write(
        root.path().join("crate-a/README.md"),
        "Run with `scripts/missing-driver.sh`.\n",
    )
    .unwrap();

    let output = run_checker(root.path());
    assert!(!output.status.success());
    assert!(stderr(&output).contains(
        "MISSING crate-a/README.md [code span: scripts/missing-driver.sh] -> scripts/missing-driver.sh"
    ));
}

/// Preserves source line anchors while validating the published file that carries the evidence.
#[test]
fn source_line_selectors_resolve_the_underlying_file() {
    let root = fixture();
    fs::create_dir_all(root.path().join("src")).unwrap();
    fs::write(root.path().join("src/model.rs"), "pub struct Model;\n").unwrap();
    fs::write(
        root.path().join("docs/guide.md"),
        "Inspect `src/model.rs:7-9` for the bounded source claim.\n",
    )
    .unwrap();

    let output = run_checker(root.path());
    assert!(output.status.success(), "{}", stderr(&output));
}

/// Prevents superseded prose from becoming an executable input contract.
#[test]
fn superseded_manifest_documents_are_not_scanned() {
    let root = fixture();
    fs::write(
        root.path().join("docs/old.md"),
        "# Old\n\nUse `docs/removed.json`.\n",
    )
    .unwrap();
    fs::write(
        root.path().join("docs/DOCS.toml"),
        "[[page]]\npath = \"old.md\"\nstatus = \"superseded\"\n",
    )
    .unwrap();

    let output = run_checker(root.path());
    assert!(output.status.success(), "{}", stderr(&output));
}

/// Keeps crate-local source and test paths relative to that crate even when the workspace has the same top-level directory name.
#[test]
fn crate_readme_relative_paths_resolve_inside_the_crate() {
    let root = fixture();
    fs::create_dir_all(root.path().join("crate-a/tests")).unwrap();
    fs::create_dir_all(root.path().join("tests")).unwrap();
    fs::write(
        root.path().join("Cargo.toml"),
        "[workspace]\nresolver = \"2\"\nmembers = [\"crate-a\"]\n",
    )
    .unwrap();
    fs::write(
        root.path().join("crate-a/tests/behavior.rs"),
        "#[test] fn behavior() {}\n",
    )
    .unwrap();
    fs::write(
        root.path().join("crate-a/README.md"),
        "The exact target is `tests/behavior.rs`.\n",
    )
    .unwrap();

    let output = run_checker(root.path());
    assert!(output.status.success(), "{}", stderr(&output));
}
