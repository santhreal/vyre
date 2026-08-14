//! Workspace crate ownership registry and generated dependency documentation contracts.

use std::fs;
use std::path::Path;
use std::process::Output;

use super::common::workspace_root;

fn run_registry(root: &Path, mode: &str) -> Output {
    super::common::run_generator("scripts/crate_ownership.py", root, mode)
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
    let dependencies = allowed
        .iter()
        .map(|dependency| {
            format!(
                "\n[[crate.dependency]]\npackage = \"{dependency}\"\npurpose = \"Use the fixture dependency contract.\"\nfeatures = []\nconditions = [\"always\"]\nkinds = [\"normal\"]\noptional = false\ndefault_features = true\nboundary = \"private\"\nseam = \"fixture-owner\"\n"
            )
        })
        .collect::<Vec<_>>()
        .join("");
    format!(
        "[[crate]]\npackage = \"{package}\"\npath = \"{path}\"\nowner = \"fixture-owner\"\nlayer = \"fixture-layer\"\nresponsibility = \"Prove the fixture ownership contract.\"\n{dependencies}\n"
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
        format!("schema_version = 2\n\n{}", registry_row("a", "a", &[])),
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
            "schema_version = 2\n\n{}{}",
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
            && error.contains("stale=[]")
            && error.contains("owning_boundaries={'b': 'fixture-owner'}")
            && error.contains("declare each required destination"),
        "Fix: undeclared edge diagnostic must name consumer, dependency, and owner; error={error}"
    );
}

/// Every declared edge must carry complete purpose, feature, visibility, and seam metadata.
///
/// This closes the class where a package allowlist was current while the reason and
/// public ownership of each edge remained unaudited.
#[test]
fn incomplete_dependency_metadata_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_workspace(temp.path(), &["a", "b"]);
    write_member(
        temp.path(),
        "a",
        "a",
        "[dependencies]\nb = { version = \"0.1.0\", path = \"../b\" }\n",
    );
    write_member(temp.path(), "b", "b", "");
    let incomplete = registry_row("a", "a", &["b"])
        .replace("purpose = \"Use the fixture dependency contract.\"\n", "");
    fs::write(
        temp.path().join("docs/CRATE_OWNERSHIP.toml"),
        format!(
            "schema_version = 2\n\n{}{}",
            incomplete,
            registry_row("b", "b", &[])
        ),
    )
    .expect("Fix: fixture registry must be writable");

    let output = run_registry(temp.path(), "--check");
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        error.contains("must define non-empty `purpose`"),
        "Fix: incomplete edge metadata must name the missing field; error={error}"
    );
}

/// Each edge names the destination owner, so a reverse or stale seam fails locally.
///
/// This prevents a dependency from compiling after ownership moved while its declared
/// architectural destination still names the old subsystem.
#[test]
fn stale_dependency_seam_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_workspace(temp.path(), &["a", "b"]);
    write_member(
        temp.path(),
        "a",
        "a",
        "[dependencies]\nb = { version = \"0.1.0\", path = \"../b\" }\n",
    );
    write_member(temp.path(), "b", "b", "");
    let stale = registry_row("a", "a", &["b"])
        .replace("seam = \"fixture-owner\"", "seam = \"removed-owner\"");
    fs::write(
        temp.path().join("docs/CRATE_OWNERSHIP.toml"),
        format!(
            "schema_version = 2\n\n{}{}",
            stale,
            registry_row("b", "b", &[])
        ),
    )
    .expect("Fix: fixture registry must be writable");

    let output = run_registry(temp.path(), "--check");
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        error.contains("declares seam `removed-owner`")
            && error.contains("required destination owner is `fixture-owner`"),
        "Fix: stale seam diagnostic must name the declared and required owners; error={error}"
    );
}
