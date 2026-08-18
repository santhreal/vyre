//! Manifest-backed crate README contract tests.

use std::fs;
use std::path::Path;

use xtask::gate::Report;
use xtask::gates::crate_readmes::CrateReadmes;

use super::workspace_sources::{run_gate, track_fixture, workspace_root};

/// Run the gate over a fixture checkout.
fn run(root: &Path, write: bool) -> Report {
    run_gate("crate-readmes", &CrateReadmes, root, write)
}

/// Every message the gate reported, joined for a failure diagnostic.
fn messages(report: &Report) -> String {
    report.finding_messages()
}

fn write_fixture(root: &Path, readme: Option<&str>, include_profile: bool) {
    fs::create_dir_all(root.join("a/src")).expect("Fix: fixture source directory must exist");
    fs::create_dir_all(root.join("a/examples")).expect("Fix: fixture example directory must exist");
    fs::create_dir_all(root.join("docs")).expect("Fix: fixture docs directory must exist");
    fs::create_dir_all(root.join("release")).expect("Fix: fixture release directory must exist");
    fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nresolver = \"2\"\nmembers = [\"a\"]\n",
    )
    .expect("Fix: fixture workspace manifest must be writable");
    fs::write(
        root.join("a/Cargo.toml"),
        "[package]\nname = \"a\"\nversion = \"0.7.9\"\nedition = \"2021\"\n\n[features]\ndefault = [\"safe\"]\nsafe = []\nfast = []\n",
    )
    .expect("Fix: fixture crate manifest must be writable");
    fs::write(root.join("a/src/lib.rs"), "pub fn answer() -> u32 { 42 }\n")
        .expect("Fix: fixture library must be writable");
    fs::write(
        root.join("a/examples/demo.rs"),
        "fn main() { assert_eq!(a::answer(), 42); }\n",
    )
    .expect("Fix: fixture example must be writable");
    if let Some(readme) = readme {
        fs::write(root.join("a/README.md"), readme).expect("Fix: fixture README must be writable");
    }
    fs::write(
        root.join("docs/CRATE_OWNERSHIP.toml"),
        "schema_version = 2\n\n[[crate]]\npackage = \"a\"\npath = \"a\"\nowner = \"fixture-owner\"\nlayer = \"foundation\"\nresponsibility = \"Return the exact fixture answer.\"\n",
    )
    .expect("Fix: fixture ownership registry must be writable");
    let profile = if include_profile {
        "\n[profile.foundation]\nerror_behavior = \"Invalid fixture input returns a structured error.\"\n"
    } else {
        ""
    };
    fs::write(
        root.join("docs/CRATE_GUIDES.toml"),
        format!("schema_version = 1\n{profile}"),
    )
    .expect("Fix: fixture crate guide metadata must be writable");
    fs::write(
        root.join("release/release-train.toml"),
        "[versions]\nvyre = \"0.7.9\"\n",
    )
    .expect("Fix: fixture release train must be writable");
}

/// Every current workspace crate README must contain the exact generated contract and no retired release claim.
///
/// This repository-level regression prevents missing crate guides, stale feature
/// lists, and obsolete dependency versions from returning independently.
#[test]
fn workspace_crate_readme_contracts_are_current() {
    let report = run(&workspace_root(), false);
    assert!(
        report.findings.is_empty(),
        "Fix: regenerate or repair crate README contracts:\n{}",
        messages(&report)
    );
}

/// The managed contract must cover purpose, boundaries, a runnable example, features, errors, testing, release, and ownership.
///
/// These are user decisions. A short package description without this information
/// leaves the crate unusable even when rustdoc exists.
#[test]
fn generated_contract_contains_every_crate_guide_surface() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(
        temp.path(),
        Some("# A crate\n\nThis human-authored introduction must stay.\n"),
        true,
    );
    track_fixture(temp.path());

    let report = run(temp.path(), true);
    assert!(
        report.findings.is_empty(),
        "Fix: valid crate guide must generate:\n{}",
        messages(&report)
    );
    let readme = fs::read_to_string(temp.path().join("a/README.md"))
        .expect("Fix: generated crate README must be readable");
    for exact in [
        "This human-authored introduction must stay.",
        "### Purpose",
        "Return the exact fixture answer.",
        "### Boundaries",
        "The `fixture-owner` owner maintains this `foundation` crate",
        "### Minimal real example",
        "./cargo_full run -p a --example demo",
        "### Features",
        "Manifest features: `default`, `fast`, `safe`",
        "Default feature members: `safe`",
        "### Errors and unsupported behavior",
        "Invalid fixture input returns a structured error.",
        "### Testing",
        "docs/testing/a.md",
        "### Release status",
        "`a@0.7.9` is a publishable crate",
        "### Ownership",
        "docs/CRATE_OWNERSHIP.toml",
    ] {
        assert!(
            readme.contains(exact),
            "Fix: generated crate guide must contain `{exact}`"
        );
    }
}

/// A workspace crate without a README must receive a complete guide instead of a placeholder.
///
/// This locks out the missing Rust frontend guide that exposed the incomplete crate
/// documentation inventory.
#[test]
fn missing_readme_is_created_with_a_real_example() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), None, true);
    track_fixture(temp.path());

    assert!(run(temp.path(), true).findings.is_empty());
    let readme = fs::read_to_string(temp.path().join("a/README.md"))
        .expect("Fix: missing README must be created");
    assert!(readme.starts_with("# `a`\n"));
    assert!(readme.contains("a/examples/demo.rs"));
    assert!(readme.contains("./cargo_full run -p a --example demo"));
}

/// A new manifest feature must invalidate the README until its generated contract is refreshed.
///
/// This prevents hand-maintained feature tables from lagging behind the actual Cargo
/// surface while still looking authoritative.
#[test]
fn manifest_feature_change_invalidates_readme_contract() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), Some("# A\n"), true);
    track_fixture(temp.path());
    assert!(run(temp.path(), true).findings.is_empty());
    fs::write(
        temp.path().join("a/Cargo.toml"),
        "[package]\nname = \"a\"\nversion = \"0.7.9\"\nedition = \"2021\"\n\n[features]\ndefault = [\"safe\"]\nsafe = []\nfast = []\nnew_route = []\n",
    )
    .expect("Fix: changed fixture manifest must be writable");

    let report = run(temp.path(), false);
    assert_eq!(
        report.findings.len(),
        1,
        "Fix: feature drift must be one finding; got\n{}",
        messages(&report)
    );
    assert_eq!(
        report.findings[0].file.as_deref(),
        Some(Path::new("a/README.md")),
        "Fix: feature drift must name the stale crate README"
    );
}

/// Retired release numbers in human-authored prose must fail even when the generated block is current.
///
/// Updating only a generated version field would otherwise leave contradictory install
/// and support claims elsewhere in the same guide.
#[test]
fn retired_release_claim_outside_generated_block_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), Some("# A\n\nInstall a = \"0.4.2\".\n"), true);
    track_fixture(temp.path());

    let report = run(temp.path(), true);
    assert!(
        messages(&report).contains("retired release `0.4.2`"),
        "Fix: retired claim diagnostic must name the version; got\n{}",
        messages(&report)
    );
    assert_eq!(
        report.findings[0].file.as_deref(),
        Some(Path::new("a/README.md"))
    );
    let readme = fs::read_to_string(temp.path().join("a/README.md"))
        .expect("Fix: the README must stay readable");
    assert!(
        readme.contains("### Release status"),
        "Fix: a retired claim in the crate's own prose must not stop the generated \
         region from being written, or --write can never converge"
    );
    assert!(
        readme.contains("Install a = \"0.4.2\"."),
        "Fix: the crate's own text must survive the write that reported it"
    );
}

/// A retired claim inside the generated region blocks the write.
///
/// WHY: the generated region is rendered from the manifests and the release
/// train. A retired version there means one of those authorities is stale, and
/// writing it publishes the break instead of reporting it, which is the failure
/// direction the ownership pair already refuses.
#[test]
fn a_retired_claim_in_the_generated_region_is_not_written() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), Some("# A\n"), true);
    track_fixture(temp.path());
    fs::write(
        temp.path().join("a/Cargo.toml"),
        "[package]\nname = \"a\"\nversion = \"0.4.2\"\nedition = \"2021\"\n\n[features]\ndefault = [\"safe\"]\nsafe = []\nfast = []\n",
    )
    .expect("Fix: retired fixture manifest must be writable");

    let report = run(temp.path(), true);
    assert!(
        messages(&report).contains("the generated contract claims retired release `0.4.2`"),
        "Fix: a retired version in the generated region must be named as generated; got\n{}",
        messages(&report)
    );
    let readme = fs::read_to_string(temp.path().join("a/README.md"))
        .expect("Fix: the README must stay readable");
    assert_eq!(
        readme, "# A\n",
        "Fix: a stale authority must not be published into the README"
    );
}

/// A registry that does not describe the workspace renders nothing.
///
/// WHY: every README is rendered from the ownership registry, so a registry that
/// disagrees with the manifests renders prose from rows that were never read.
/// `crate-ownership` already refuses to write its pair under the same condition.
#[test]
fn a_broken_registry_renders_no_readme() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), None, true);
    fs::write(
        temp.path().join("docs/CRATE_OWNERSHIP.toml"),
        "schema_version = 2\n\n[[crate]]\npackage = \"a\"\npath = \"a\"\nowner = \"fixture-owner\"\nlayer = \"foundation\"\nresponsibility = \"\"\n",
    )
    .expect("Fix: fixture ownership registry must be writable");
    track_fixture(temp.path());

    let report = run(temp.path(), true);
    assert!(
        !report.findings.is_empty(),
        "Fix: an empty responsibility must be a registry finding"
    );
    assert!(
        !temp.path().join("a/README.md").exists(),
        "Fix: a README must not be rendered from a registry that does not hold"
    );
}

/// An empty package override is a finding, not an empty published section.
///
/// WHY: the override supplies the "Errors and unsupported behavior" prose. An
/// empty string rendered a heading with nothing under it, which reads as a crate
/// that documented its error behavior.
#[test]
fn an_empty_error_behavior_override_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), Some("# A\n"), true);
    fs::write(
        temp.path().join("docs/CRATE_GUIDES.toml"),
        "schema_version = 1\n\n[profile.foundation]\nerror_behavior = \"Invalid fixture input returns a structured error.\"\n\n[package.a]\nerror_behavior = \"\"\n",
    )
    .expect("Fix: fixture crate guide metadata must be writable");
    track_fixture(temp.path());

    let report = run(temp.path(), false);
    assert!(
        messages(&report).contains("the override for `a` declares an empty `error_behavior`"),
        "Fix: an empty override must be named; got\n{}",
        messages(&report)
    );
}

/// An override does not stand in for the layer profile.
///
/// WHY: the profile is what every crate in a layer renders and the override is
/// prose for one of them. Accepting an override for a layer that declares no
/// profile left the next crate in that layer with nothing to render, and the
/// missing profile went unreported for as long as one crate carried an override.
#[test]
fn a_package_override_does_not_replace_a_missing_layer_profile() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), Some("# A\n"), false);
    fs::write(
        temp.path().join("docs/CRATE_GUIDES.toml"),
        "schema_version = 1\n\n[package.a]\nerror_behavior = \"The fixture crate returns a structured error.\"\n",
    )
    .expect("Fix: fixture crate guide metadata must be writable");
    track_fixture(temp.path());

    let report = run(temp.path(), false);
    assert!(
        messages(&report).contains("no error profile for layer `foundation`, which `a` occupies"),
        "Fix: a missing layer profile must be reported even when the package overrides it; got\n{}",
        messages(&report)
    );
}

/// Every ownership layer needs an explicit error contract instead of generic fallback prose.
///
/// Error semantics differ across parsers, emitters, drivers, and tooling. Missing
/// metadata must block generation before an inaccurate guide is written.
#[test]
fn missing_layer_error_profile_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), Some("# A\n"), false);
    track_fixture(temp.path());

    let report = run(temp.path(), true);
    assert!(
        messages(&report).contains("no error profile for layer `foundation`, which `a` occupies"),
        "Fix: missing error profile must name layer and crate; got\n{}",
        messages(&report)
    );
}

/// Prevents a local ignored example from becoming the documented minimal example in a clean checkout.
#[test]
fn untracked_examples_are_excluded_from_generated_contracts() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), None, true);
    fs::write(temp.path().join(".gitignore"), "a/examples/demo.rs\n")
        .expect("Fix: ignored example rule must be writable");
    track_fixture(temp.path());

    let report = run(temp.path(), true);
    assert!(
        report.findings.is_empty(),
        "Fix: an ignored example must not stop generation:\n{}",
        messages(&report)
    );
    let readme = fs::read_to_string(temp.path().join("a/README.md"))
        .expect("Fix: generated README must be readable");
    assert!(!readme.contains("a/examples/demo.rs"));
    assert!(readme.contains("./cargo_full test -p a --lib"));
}
