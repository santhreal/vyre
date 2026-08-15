//! Release-train-derived documentation contract tests.

#![forbid(unsafe_code)]

mod common;

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

use common::workspace_root;

/// Run the generator the fixture itself carries, not the checkout's copy.
fn run_generator(root: &Path, mode: &str) -> Output {
    Command::new("python3")
        .arg(root.join("scripts/release_docs.py"))
        .arg(mode)
        .output()
        .expect("Fix: release document generator must launch with python3")
}

fn write_fixture(root: &Path, actions: usize, duplicate_package: bool) {
    for directory in ["scripts", "release/changes"] {
        fs::create_dir_all(root.join(directory))
            .expect("Fix: release document fixture directories must be creatable");
    }
    fs::copy(
        workspace_root().join("scripts/release_docs.py"),
        root.join("scripts/release_docs.py"),
    )
    .expect("Fix: fixture must copy the production release document generator");
    fs::copy(
        workspace_root().join("scripts/final-launch.sh"),
        root.join("scripts/final-launch.sh"),
    )
    .expect("Fix: fixture must copy the guarded launch implementation");
    let duplicate_group = if duplicate_package {
        r#"

[release_groups.secondary]
repository = "owner/secondary"
version = "vyre"
packages = ["a"]
"#
    } else {
        ""
    };
    let mut train = format!(
        r#"required_release_note_tokens = ["Vyre 1.2.3", "a@1.2.3", "vyre-v1.2.3"]
required_packaging_steps = ["publish in dependency order"]
package_verify_passed = ["a@1.2.3"]

[versions]
vyre = "1.2.3"

[tags]
vyre_rc = "vyre-v1.2.3-rc.1"
vyre = "vyre-v1.2.3"
policy = "Use product-scoped tags."

[release_groups.vyre]
repository = "owner/vyre"
version = "vyre"
packages = ["a"]
{duplicate_group}"#
    );
    for index in 0..actions {
        train.push_str(&format!(
            "\n[[external_actions]]\nid = \"action-{index}\"\ndescription = \"External action {index}.\"\nevidence = \"evidence-{index}\"\n"
        ));
    }
    fs::write(root.join("release/release-train.toml"), train)
        .expect("Fix: fixture release train must be writable");
    fs::write(
        root.join("release/changes/unreleased.toml"),
        "schema_version = 1\n\n[[fragments]]\nid = \"exact-fix\"\ncategory = \"Fixed\"\ntext = \"The exact release regression is fixed.\"\n",
    )
    .expect("Fix: fixture changelog fragments must be writable");
    fs::write(
        root.join("CHANGELOG.md"),
        "# Changelog\n\n## [Unreleased]\n\n### Fixed\n\n- stale text\n\n## [1.2.2]\n\n- prior release\n",
    )
    .expect("Fix: fixture changelog must be writable");
}

/// Locks the repository release surfaces to the train and fragment authorities.
#[test]
fn workspace_release_documents_are_current() {
    let output = run_generator(&workspace_root(), "--check");
    assert!(
        output.status.success(),
        "Fix: regenerate or repair release documents: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).expect("Fix: generator output must be UTF-8"),
        "release-docs: release train, fragments, changelog, and release notes agree\n"
    );
}

/// Proves one write derives the changelog's release identities and its change
/// list from the train and the fragments, with no hand-maintained copy between.
#[test]
fn write_derives_every_release_metadata_surface() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), 3, false);
    let output = run_generator(temp.path(), "--write");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let changelog = fs::read_to_string(temp.path().join("CHANGELOG.md"))
        .expect("Fix: generated changelog must be readable");
    assert!(changelog.contains("- The exact release regression is fixed."));
    assert!(!changelog.contains("stale text"));
    assert!(changelog.contains(
        "Vyre 1.2.3 releases from candidate tag `vyre-v1.2.3-rc.1` and final tag `vyre-v1.2.3`."
    ));
    assert!(changelog.contains("Backend crates carried at that version: `a@1.2.3`."));
    assert!(changelog.contains("## [1.2.2]"));
}

/// Prevents a hand edit to the generated changelog section from silently
/// changing what a release claims to contain.
#[test]
fn check_rejects_generated_release_document_drift() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), 3, false);
    assert!(run_generator(temp.path(), "--write").status.success());
    let changelog_path = temp.path().join("CHANGELOG.md");
    let generated = fs::read_to_string(&changelog_path)
        .expect("Fix: generated fixture changelog must be readable");
    fs::write(
        &changelog_path,
        generated.replace(
            "- The exact release regression is fixed.",
            "- A hand-written claim no fragment backs.",
        ),
    )
    .expect("Fix: drifted fixture changelog must be writable");

    let output = run_generator(temp.path(), "--check");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("generated release content is stale"));
}

/// Prevents a package from publishing under two versions or repositories in one release train.
#[test]
fn duplicate_release_group_membership_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), 3, true);
    let output = run_generator(temp.path(), "--write");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("package `a` belongs to both `vyre` and `secondary`"));
}

/// Preserves the exact three-action approval boundary required by prepublication evidence.
#[test]
fn incomplete_external_action_boundary_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), 2, false);
    let output = run_generator(temp.path(), "--write");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("exactly three approval-gated external actions"));
}

/// Prevents release notes from passing when the train adds a token that the changelog omits.
///
/// The assertion below named `missing required token`, which the generator has
/// never emitted: it reports `CHANGELOG.md: missing required release token`, so
/// the test was red on arrival and said nothing about the rule it guards. The
/// document name is part of the message and is asserted with it.
#[test]
fn missing_release_note_token_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), 3, false);
    assert!(run_generator(temp.path(), "--write").status.success());
    let train_path = temp.path().join("release/release-train.toml");
    let train = fs::read_to_string(&train_path).expect("Fix: fixture train must be readable");
    fs::write(
        &train_path,
        train.replace(
            "\"vyre-v1.2.3\"]",
            "\"vyre-v1.2.3\", \"required-but-absent\"]",
        ),
    )
    .expect("Fix: modified fixture train must be writable");

    let output = run_generator(temp.path(), "--check");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("CHANGELOG.md: missing required release token `required-but-absent`"));
}

/// Prevents completion evidence from claiming publish or push success before those actions run.
#[test]
fn guarded_launch_order_fails_closed_when_candidate_tags_are_reordered() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), 3, false);
    assert!(run_generator(temp.path(), "--write").status.success());
    let launch_path = temp.path().join("scripts/final-launch.sh");
    let launch =
        fs::read_to_string(&launch_path).expect("Fix: fixture launch script must be readable");
    // Both tokens carry their trailing newline: the launch-complete invocation
    // one screen below has the prepublication command as a strict prefix, so a
    // newline-free token would rewrite that line too and the swap would prove
    // nothing about ordering.
    let candidate = "git tag -a \"$VYRE_RELEASE_TAG_VYRE_RC\" -m \"$VYRE_RELEASE_TAG_VYRE_RC\"\n";
    let prepublish = "\"$CARGO_RUNNER\" run -j1 --manifest-path xtask/Cargo.toml --bin xtask -- vyre-release-gate\n";
    let reordered = launch
        .replace(candidate, "__VYRE_RELEASE_ORDER_SWAP__")
        .replace(prepublish, candidate)
        .replace("__VYRE_RELEASE_ORDER_SWAP__", prepublish);
    fs::write(&launch_path, reordered)
        .expect("Fix: reordered fixture launch script must be writable");

    let output = run_generator(temp.path(), "--check");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("are out of order"));
}
