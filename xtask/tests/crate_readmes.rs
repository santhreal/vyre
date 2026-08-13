//! Manifest-backed crate README contract tests.

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

fn run_generator(root: &Path, mode: &str) -> Output {
    Command::new("python3")
        .arg(workspace_root().join("scripts/crate_readmes.py"))
        .arg(root)
        .arg(mode)
        .output()
        .expect("Fix: crate README generator must launch with python3")
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
    let output = run_generator(&workspace_root(), "--check");
    assert!(
        output.status.success(),
        "Fix: regenerate or repair crate README contracts: {}",
        String::from_utf8_lossy(&output.stderr)
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

    let output = run_generator(temp.path(), "--write");
    assert!(
        output.status.success(),
        "Fix: valid crate guide must generate: {}",
        String::from_utf8_lossy(&output.stderr)
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
        "CARGO_BUILD_JOBS=1 ./cargo_full run -p a --example demo",
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

    assert!(run_generator(temp.path(), "--write").status.success());
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
    assert!(run_generator(temp.path(), "--write").status.success());
    fs::write(
        temp.path().join("a/Cargo.toml"),
        "[package]\nname = \"a\"\nversion = \"0.7.9\"\nedition = \"2021\"\n\n[features]\ndefault = [\"safe\"]\nsafe = []\nfast = []\nnew_route = []\n",
    )
    .expect("Fix: changed fixture manifest must be writable");

    let output = run_generator(temp.path(), "--check");
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        error.contains("missing or stale crate README contracts: ['a/README.md']"),
        "Fix: feature drift must name the stale crate README; error={error}"
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

    let output = run_generator(temp.path(), "--write");
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        error.contains("retired 0.4.x release claims: ['a/README.md']"),
        "Fix: retired claim diagnostic must name the README; error={error}"
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

    let output = run_generator(temp.path(), "--write");
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        error.contains("no error profile for layer `foundation` used by `a`"),
        "Fix: missing error profile must name layer and crate; error={error}"
    );
}

/// Prevents a local ignored example from becoming the documented minimal example in a clean checkout.
#[test]
fn gitignored_examples_are_excluded_from_generated_contracts() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), None, true);
    let status = Command::new("git")
        .args(["init", "-q"])
        .current_dir(temp.path())
        .status()
        .expect("Fix: git must launch for ignored example semantics");
    assert!(status.success());
    fs::write(temp.path().join(".gitignore"), "a/examples/demo.rs\n")
        .expect("Fix: ignored example rule must be writable");

    let output = run_generator(temp.path(), "--write");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let readme = fs::read_to_string(temp.path().join("a/README.md"))
        .expect("Fix: generated README must be readable");
    assert!(!readme.contains("a/examples/demo.rs"));
    assert!(readme.contains("./cargo_full test -p a --lib"));
}
