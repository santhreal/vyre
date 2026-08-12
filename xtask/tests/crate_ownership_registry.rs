//! Workspace crate ownership registry and generated dependency documentation contracts.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("Fix: xtask must remain directly under the workspace root")
        .to_path_buf()
}

fn run_registry(root: &Path, mode: &str) -> Output {
    Command::new("python3")
        .arg(workspace_root().join("scripts/crate_ownership.py"))
        .arg(root)
        .arg(mode)
        .output()
        .expect("Fix: crate ownership generator must launch with python3")
}

fn write_member(root: &Path, path: &str, package: &str, dependencies: &str) {
    let directory = root.join(path);
    fs::create_dir_all(&directory).expect("Fix: fixture crate directory must be creatable");
    fs::write(
        directory.join("Cargo.toml"),
        format!(
            "[package]\nname = \"{package}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n{dependencies}"
        ),
    )
    .expect("Fix: fixture crate manifest must be writable");
}

fn write_workspace(root: &Path, members: &[&str]) {
    fs::create_dir_all(root.join("docs")).expect("Fix: fixture docs directory must be creatable");
    let members = members
        .iter()
        .map(|member| format!("\"{member}\""))
        .collect::<Vec<_>>()
        .join(", ");
    fs::write(
        root.join("Cargo.toml"),
        format!("[workspace]\nresolver = \"2\"\nmembers = [{members}]\n"),
    )
    .expect("Fix: fixture workspace manifest must be writable");
}

fn registry_row(package: &str, path: &str, allowed: &[&str]) -> String {
    let allowed = allowed
        .iter()
        .map(|dependency| format!("\"{dependency}\""))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "[[crate]]\npackage = \"{package}\"\npath = \"{path}\"\nowner = \"fixture-owner\"\nlayer = \"fixture-layer\"\nresponsibility = \"Prove the fixture ownership contract.\"\nallowed_dependencies = [{allowed}]\n\n"
    )
}

/// The checked-in registry and both generated documents must agree with every current manifest edge.
///
/// This is the repository-level regression for stale crate lists, missing ownership
/// sections, and hand-edited dependency diagrams that disagree with Cargo.
#[test]
fn workspace_registry_and_generated_documents_are_current() {
    let output = run_registry(&workspace_root(), "--check");
    assert!(
        output.status.success(),
        "Fix: regenerate or repair crate ownership evidence: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).expect("Fix: generator output must be UTF-8"),
        "crate-ownership: verified docs/CRATE_GRAPH.md and docs/OWNERSHIP.md\n"
    );
}

/// Every workspace member needs one exact registry row, including internal tooling crates.
///
/// A missing row previously let fourteen crates disappear from the public ownership
/// guide even though Cargo still built and linked them.
#[test]
fn missing_workspace_member_ownership_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_workspace(temp.path(), &["a", "b"]);
    write_member(temp.path(), "a", "a", "");
    write_member(temp.path(), "b", "b", "");
    fs::write(
        temp.path().join("docs/CRATE_OWNERSHIP.toml"),
        format!("schema_version = 1\n\n{}", registry_row("a", "a", &[])),
    )
    .expect("Fix: fixture registry must be writable");

    let output = run_registry(temp.path(), "--check");
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        error.contains("missing=['b']") && error.contains("extra=[]"),
        "Fix: missing member diagnostic must identify the exact path; error={error}"
    );
}

/// A new normal dependency must be declared in the ownership registry in the same patch.
///
/// This prevents the generated DAG from silently accepting a cross-layer edge merely
/// because the consumer manifest compiles.
#[test]
fn undeclared_production_dependency_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_workspace(temp.path(), &["a", "b"]);
    write_member(
        temp.path(),
        "a",
        "a",
        "[dependencies]\nb = { version = \"0.1.0\", path = \"../b\" }\n",
    );
    write_member(temp.path(), "b", "b", "");
    fs::write(
        temp.path().join("docs/CRATE_OWNERSHIP.toml"),
        format!(
            "schema_version = 1\n\n{}{}",
            registry_row("a", "a", &[]),
            registry_row("b", "b", &[])
        ),
    )
    .expect("Fix: fixture registry must be writable");

    let output = run_registry(temp.path(), "--check");
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        error.contains("package `a`")
            && error.contains("undeclared=['b']")
            && error.contains("stale=[]"),
        "Fix: undeclared edge diagnostic must name consumer and dependency; error={error}"
    );
}

/// Planned crates must render separately from the current workspace and carry an explicit absence label.
///
/// This protects the planned megakernel compiler boundary from becoming a false
/// support claim before a manifest and implementation actually exist.
#[test]
fn planned_boundary_renders_without_claiming_a_current_crate() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_workspace(temp.path(), &["a"]);
    write_member(temp.path(), "a", "a", "");
    fs::write(
        temp.path().join("docs/CRATE_OWNERSHIP.toml"),
        format!(
            "schema_version = 1\n\n[planned.planned-compiler]\npath = \"planned-compiler\"\npresent = false\nowner = \"compiler\"\nlayer = \"compiler-boundary\"\nresponsibility = \"Compile typed programs.\"\nallowed_dependencies = [\"a\"]\n\n{}",
            registry_row("a", "a", &[])
        ),
    )
    .expect("Fix: fixture registry must be writable");

    let output = run_registry(temp.path(), "--write");
    assert!(
        output.status.success(),
        "Fix: valid planned boundary must generate: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let graph = fs::read_to_string(temp.path().join("docs/CRATE_GRAPH.md"))
        .expect("Fix: generated graph must be readable");
    let ownership = fs::read_to_string(temp.path().join("docs/OWNERSHIP.md"))
        .expect("Fix: generated ownership guide must be readable");
    for document in [&graph, &ownership] {
        assert!(document.contains("`planned-compiler` (planned, not a workspace member)"));
        assert!(!document.contains("| `planned-compiler` |"));
    }
    assert!(graph.contains("The workspace contains 1 crates."));
}

/// A planned entry cannot set `present = true` as a shortcut around workspace discovery.
///
/// The member list and package manifest must become real before documentation may
/// classify the boundary as current.
#[test]
fn planned_present_flag_cannot_claim_implementation() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_workspace(temp.path(), &["a"]);
    write_member(temp.path(), "a", "a", "");
    fs::write(
        temp.path().join("docs/CRATE_OWNERSHIP.toml"),
        format!(
            "schema_version = 1\n\n[planned.future]\npath = \"future\"\npresent = true\nowner = \"compiler\"\nlayer = \"compiler-boundary\"\nresponsibility = \"Compile typed programs.\"\nallowed_dependencies = [\"a\"]\n\n{}",
            registry_row("a", "a", &[])
        ),
    )
    .expect("Fix: fixture registry must be writable");

    let output = run_registry(temp.path(), "--check");
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        error.contains("[planned.future]") && error.contains("present = false"),
        "Fix: false implementation claim must name the planned entry and correction; error={error}"
    );
}
