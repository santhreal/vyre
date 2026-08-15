//! Generated per-crate testing guide contracts.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use super::common::workspace_root;

fn run_generator(root: &Path, mode: &str) -> Output {
    super::common::run_generator("scripts/testing_guides.py", root, mode)
}

fn write_fixture(root: &Path, include_profile: bool) {
    fs::create_dir_all(root.join("a/src")).expect("Fix: fixture source directory must exist");
    fs::create_dir_all(root.join("a/tests")).expect("Fix: fixture test directory must exist");
    fs::create_dir_all(root.join("docs/testing"))
        .expect("Fix: fixture testing directory must exist");
    fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nresolver = \"2\"\nmembers = [\"a\"]\n",
    )
    .expect("Fix: fixture workspace manifest must be writable");
    fs::write(
        root.join("a/Cargo.toml"),
        "[package]\nname = \"a\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[features]\ndefault = []\nfast = []\n",
    )
    .expect("Fix: fixture crate manifest must be writable");
    fs::write(root.join("a/src/lib.rs"), "pub fn answer() -> u32 { 42 }\n")
        .expect("Fix: fixture library must be writable");
    fs::write(
        root.join("a/tests/behavior.rs"),
        "#[test] fn behavior() {}\n",
    )
    .expect("Fix: fixture integration test must be writable");
    fs::write(
        root.join("docs/CRATE_OWNERSHIP.toml"),
        "schema_version = 2\n\n[[crate]]\npackage = \"a\"\npath = \"a\"\nowner = \"fixture\"\nlayer = \"foundation\"\nresponsibility = \"Return the exact fixture answer.\"\n",
    )
    .expect("Fix: fixture ownership registry must be writable");
    let profile = if include_profile {
        "\n[profile.foundation]\ntest_classes = [\"Exact fixture behavior\", \"Adversarial fixture inputs\"]\n"
    } else {
        ""
    };
    fs::write(
        root.join("docs/testing/TESTING.toml"),
        format!(
            "schema_version = 1\n\n[defaults]\nhardware = \"No accelerator is required.\"\nexpected_skips = \"No expected skips.\"\nfailure_behavior = \"A wrong answer returns nonzero.\"\nevidence_outputs = [\"Exact assertion output\"]\n{profile}"
        ),
    )
    .expect("Fix: fixture testing metadata must be writable");
}

/// Every current workspace member must have a guide that exactly matches manifests and testing metadata.
///
/// This repository-level check prevents missing guides and cloned placeholder bodies
/// from reappearing when crates or Cargo targets change.
#[test]
fn workspace_testing_guides_are_current() {
    let output = run_generator(&workspace_root(), "--check");
    assert!(
        output.status.success(),
        "Fix: regenerate or repair testing guides: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// A generated guide must expose exact commands, features, targets, hardware, evidence, skips, and failures.
///
/// A path-only or non-empty placeholder guide does not teach a maintainer how to
/// exercise the crate and must not satisfy the documentation contract.
#[test]
fn generated_guide_contains_the_complete_crate_testing_contract() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), true);

    let output = run_generator(temp.path(), "--write");
    assert!(
        output.status.success(),
        "Fix: valid fixture guide must generate: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let guide = fs::read_to_string(temp.path().join("docs/testing/a.md"))
        .expect("Fix: generated fixture guide must be readable");
    for exact in [
        "./cargo_full test -p a",
        "./cargo_full test -p a --all-features",
        "Available manifest features: `default`, `fast`",
        "| `test` | `behavior` | `a/tests/behavior.rs` |",
        "No accelerator is required.",
        "- Exact assertion output",
        "No expected skips.",
        "A wrong answer returns nonzero.",
        "Return the exact fixture answer.",
    ] {
        assert!(
            guide.contains(exact),
            "Fix: generated fixture guide must contain `{exact}`"
        );
    }
}

/// Adding a Cargo test target must make the checked guide stale until regeneration.
///
/// This locks target discovery to the manifest and filesystem instead of a manually
/// copied list that silently omits new integration suites.
#[test]
fn new_cargo_test_target_invalidates_the_guide() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), true);
    assert!(run_generator(temp.path(), "--write").status.success());
    fs::write(
        rooted(temp.path(), "a/tests/adversarial.rs"),
        "#[test] fn bad() {}\n",
    )
    .expect("Fix: added fixture target must be writable");

    let output = run_generator(temp.path(), "--check");
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        error.contains("missing or stale testing guides: ['a.md']"),
        "Fix: target drift must identify the stale guide; error={error}"
    );
}

fn rooted(root: &Path, relative: &str) -> PathBuf {
    root.join(relative)
}

/// Every ownership layer needs maintained test-class metadata before its guides can generate.
///
/// Falling back to a generic body would recreate the twenty-six identical guides that
/// hid crate-specific testing requirements.
#[test]
fn missing_layer_profile_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), false);

    let output = run_generator(temp.path(), "--write");
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        error.contains("no profile for ownership layer `foundation` used by `a`"),
        "Fix: missing profile diagnostic must name layer and crate; error={error}"
    );
}

/// A guide for a package outside workspace.members must be rejected rather than preserved as apparent support.
///
/// This prevents removed crates from retaining current-looking test instructions after
/// their manifest and executable surface disappear.
#[test]
fn orphaned_testing_guide_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), true);
    assert!(run_generator(temp.path(), "--write").status.success());
    fs::write(
        temp.path().join("docs/testing/removed.md"),
        "# Removed crate\n",
    )
    .expect("Fix: orphan guide fixture must be writable");

    let output = run_generator(temp.path(), "--check");
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        error.contains("non-member guides: ['removed.md']"),
        "Fix: orphan guide diagnostic must name the exact file; error={error}"
    );
}

/// Prevents ignored local tests from appearing as published Cargo targets in generated guides.
#[test]
fn gitignored_targets_are_excluded_from_testing_guides() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), true);
    let status = Command::new("git")
        .args(["init", "-q"])
        .current_dir(temp.path())
        .status()
        .expect("Fix: git must launch for ignored target semantics");
    assert!(status.success());
    fs::write(temp.path().join(".gitignore"), "a/tests/behavior.rs\n")
        .expect("Fix: ignored target rule must be writable");

    let output = run_generator(temp.path(), "--write");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let guide = fs::read_to_string(temp.path().join("docs/testing/a.md"))
        .expect("Fix: generated testing guide must be readable");
    assert!(!guide.contains("a/tests/behavior.rs"));
    assert!(guide.contains("| `lib` | `a` | `a/src/lib.rs` |"));
}
