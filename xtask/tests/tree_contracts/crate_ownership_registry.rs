//! Workspace crate ownership registry and generated dependency documentation contracts.

use std::fs;
use std::path::Path;

use xtask::gate::Report;
use xtask::gates::crate_registry::CrateOwnership;

use super::workspace_sources::{run_gate, track_fixture, workspace_root};

/// Run the gate over a fixture checkout.
fn run(root: &Path) -> Report {
    run_gate("crate-ownership", &CrateOwnership, root, false)
}

/// Every message the gate reported, joined for a failure diagnostic.
fn messages(report: &Report) -> String {
    report
        .findings
        .iter()
        .map(|finding| finding.message.clone())
        .collect::<Vec<_>>()
        .join("\n")
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

/// Write the registry and turn the fixture into a checkout the gate can read.
fn seal(root: &Path, registry: String) {
    fs::write(root.join("docs/CRATE_OWNERSHIP.toml"), registry)
        .expect("Fix: fixture registry must be writable");
    track_fixture(root);
}

/// The checked-in registry and both generated documents must agree with every current manifest edge.
///
/// This is the repository-level regression for stale crate lists, missing ownership
/// sections, and hand-edited dependency diagrams that disagree with Cargo.
#[test]
fn workspace_registry_and_generated_documents_are_current() {
    let report = run(&workspace_root());
    assert!(
        report.findings.is_empty(),
        "Fix: regenerate or repair crate ownership evidence:\n{}",
        messages(&report)
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
    seal(
        temp.path(),
        format!("schema_version = 2\n\n{}", registry_row("a", "a", &[])),
    );

    let report = run(temp.path());
    let messages = messages(&report);
    assert!(
        messages.contains("workspace member `b` has no registry row")
            && messages.contains("workspace package `b` has no registry row"),
        "Fix: missing member diagnostic must identify the exact path and package; got\n{messages}"
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
    seal(
        temp.path(),
        format!(
            "schema_version = 2\n\n{}{}",
            registry_row("a", "a", &[]),
            registry_row("b", "b", &[])
        ),
    );

    let report = run(temp.path());
    let messages = messages(&report);
    assert!(
        messages.contains("`a` depends on `b` and declares no record for it"),
        "Fix: undeclared edge diagnostic must name consumer and dependency; got\n{messages}"
    );
}

/// A record for an edge no manifest resolves must fail as loudly as a missing one.
///
/// The reverse direction is the one that survives a dependency removal: the manifest
/// stops naming the crate and the registry keeps advertising the boundary.
#[test]
fn stale_dependency_record_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_workspace(temp.path(), &["a", "b"]);
    write_member(temp.path(), "a", "a", "");
    write_member(temp.path(), "b", "b", "");
    seal(
        temp.path(),
        format!(
            "schema_version = 2\n\n{}{}",
            registry_row("a", "a", &["b"]),
            registry_row("b", "b", &[])
        ),
    );

    let report = run(temp.path());
    let messages = messages(&report);
    assert!(
        messages.contains("`a` declares a record for `b` and no manifest edge resolves to it"),
        "Fix: stale record diagnostic must name consumer and dependency; got\n{messages}"
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
    seal(
        temp.path(),
        format!(
            "schema_version = 2\n\n{}{}",
            incomplete,
            registry_row("b", "b", &[])
        ),
    );

    let report = run(temp.path());
    let messages = messages(&report);
    assert!(
        messages.contains("declares no non-empty `purpose`"),
        "Fix: incomplete edge metadata must name the missing field; got\n{messages}"
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
    seal(
        temp.path(),
        format!(
            "schema_version = 2\n\n{}{}",
            stale,
            registry_row("b", "b", &[])
        ),
    );

    let report = run(temp.path());
    let messages = messages(&report);
    assert!(
        messages.contains(
            "`a` -> `b` declares seam `removed-owner` and the destination owner is `fixture-owner`"
        ),
        "Fix: stale seam diagnostic must name the declared and required owners; got\n{messages}"
    );
}

/// A feature the manifest turns on and the record omits is drift, in either
/// direction.
///
/// The union of an inherited workspace feature list and a local one is what cargo
/// enables, so a record that names only the local half under-reports the edge.
#[test]
fn dependency_feature_drift_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_workspace(temp.path(), &["a", "b"]);
    write_member(
        temp.path(),
        "a",
        "a",
        "[dependencies]\nb = { version = \"0.1.0\", path = \"../b\", features = [\"fast\"] }\n",
    );
    write_member(temp.path(), "b", "b", "\n[features]\nfast = []\n");
    seal(
        temp.path(),
        format!(
            "schema_version = 2\n\n{}{}",
            registry_row("a", "a", &["b"]),
            registry_row("b", "b", &[])
        ),
    );

    let report = run(temp.path());
    let messages = messages(&report);
    assert!(
        messages.contains("`a` -> `b` declares features `` and cargo resolves `fast`"),
        "Fix: feature drift must name both sides; got\n{messages}"
    );
}
