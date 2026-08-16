//! Generated per-crate testing guide contracts.

use std::fs;
use std::path::Path;

use xtask::gate::Report;
use xtask::gates::testing_guides::TestingGuides;

use super::workspace_sources::{run_gate, track_fixture, workspace_root};

/// Run the gate over a fixture checkout.
fn run(root: &Path, write: bool) -> Report {
    run_gate("testing-guides", &TestingGuides, root, write)
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
    let report = run(&workspace_root(), false);
    assert!(
        report.findings.is_empty(),
        "Fix: regenerate or repair testing guides:\n{}",
        messages(&report)
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
    track_fixture(temp.path());

    let report = run(temp.path(), true);
    assert!(
        report.findings.is_empty(),
        "Fix: valid fixture guide must generate:\n{}",
        messages(&report)
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
    track_fixture(temp.path());
    assert!(run(temp.path(), true).findings.is_empty());
    fs::write(
        temp.path().join("a/tests/adversarial.rs"),
        "#[test] fn bad() {}\n",
    )
    .expect("Fix: added fixture target must be writable");
    track_fixture(temp.path());

    let report = run(temp.path(), false);
    assert_eq!(
        report.findings.len(),
        1,
        "Fix: target drift must be one finding; got\n{}",
        messages(&report)
    );
    assert_eq!(
        report.findings[0].file.as_deref(),
        Some(Path::new("docs/testing/a.md")),
        "Fix: target drift must identify the stale guide"
    );
}

/// Every ownership layer needs maintained test-class metadata before its guides can generate.
///
/// Falling back to a generic body would recreate the twenty-six identical guides that
/// hid crate-specific testing requirements.
#[test]
fn missing_layer_profile_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), false);
    track_fixture(temp.path());

    let report = run(temp.path(), true);
    assert!(
        messages(&report).contains("no profile for layer `foundation`, which `a` occupies"),
        "Fix: missing profile diagnostic must name layer and crate; got\n{}",
        messages(&report)
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
    track_fixture(temp.path());
    assert!(run(temp.path(), true).findings.is_empty());
    fs::write(
        temp.path().join("docs/testing/removed.md"),
        "# Removed crate\n",
    )
    .expect("Fix: orphan guide fixture must be writable");
    track_fixture(temp.path());

    let report = run(temp.path(), false);
    assert_eq!(
        report.findings.len(),
        1,
        "Fix: an orphan guide must be one finding; got\n{}",
        messages(&report)
    );
    assert_eq!(
        report.findings[0].file.as_deref(),
        Some(Path::new("docs/testing/removed.md")),
        "Fix: orphan guide diagnostic must name the exact file"
    );
}

/// Prevents ignored local tests from appearing as published Cargo targets in generated guides.
#[test]
fn untracked_targets_are_excluded_from_testing_guides() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), true);
    fs::write(temp.path().join(".gitignore"), "a/tests/behavior.rs\n")
        .expect("Fix: ignored target rule must be writable");
    track_fixture(temp.path());

    let report = run(temp.path(), true);
    assert!(
        report.findings.is_empty(),
        "Fix: an ignored target must not stop generation:\n{}",
        messages(&report)
    );
    let guide = fs::read_to_string(temp.path().join("docs/testing/a.md"))
        .expect("Fix: generated testing guide must be readable");
    assert!(!guide.contains("a/tests/behavior.rs"));
    assert!(guide.contains("| `lib` | `a` | `a/src/lib.rs` |"));
}

/// An unrelated finding must not hide an orphaned guide.
///
/// WHY: the orphan scan used to run only when the report was empty, so any
/// finding anywhere in the gate, including a schema field on the registry
/// itself, made every leftover guide invisible for that run. A reader then
/// fixed the reported problem and only learned about the orphans on the next
/// run. A row that renders leaves its guide in the expected set whatever else
/// was reported, so the two signals belong in one pass.
#[test]
fn an_unrelated_finding_does_not_hide_an_orphaned_guide() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), true);
    track_fixture(temp.path());
    assert!(run(temp.path(), true).findings.is_empty());
    fs::write(
        temp.path().join("docs/testing/removed.md"),
        "# Removed crate\n",
    )
    .expect("Fix: orphan guide fixture must be writable");
    let registry = fs::read_to_string(temp.path().join("docs/CRATE_OWNERSHIP.toml"))
        .expect("Fix: fixture ownership registry must be readable");
    fs::write(
        temp.path().join("docs/CRATE_OWNERSHIP.toml"),
        format!("{registry}\n[planned]\nnext = \"b\"\n"),
    )
    .expect("Fix: fixture ownership registry must be writable");
    track_fixture(temp.path());

    let report = run(temp.path(), false);
    let text = messages(&report);
    assert!(
        report
            .findings
            .iter()
            .any(|finding| finding.file.as_deref() == Some(Path::new("docs/testing/removed.md"))),
        "Fix: the orphan guide must be reported beside the registry finding; got\n{text}"
    );
    assert!(
        text.contains("planned crates"),
        "Fix: the unrelated finding must still be reported; got\n{text}"
    );
}
