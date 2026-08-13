//! Release-train-derived documentation contract tests.

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
        .arg(root.join("scripts/release_docs.py"))
        .arg(mode)
        .output()
        .expect("Fix: release document generator must launch with python3")
}

fn write_fixture(root: &Path, actions: usize, duplicate_package: bool) {
    for directory in ["scripts", "release/changes", "docs/release"] {
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
    fs::write(
        root.join("docs/release/v1.2.3.md"),
        "# stale preamble\n\n## What is in this release\n\nThe exact release regression is fixed.\n",
    )
    .expect("Fix: fixture release notes must be writable");
    fs::write(
        root.join("docs/RELEASE.md"),
        "# Release\n\nVyre 1.2.3.\nrelease/release-train.toml\nRELEASE_CHECKLIST.md\nrelease/changes/unreleased.toml\n--prepublish\n\n## Rollback\n",
    )
    .expect("Fix: fixture release runbook must be writable");
}

/// Locks the repository release surfaces to the train, fragment, and generated-view authorities.
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
        "release-docs: release train, fragments, notes, and checklist agree\n"
    );
}

/// Proves one write derives changelog text, package ownership, repository ownership, and tag order from data.
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

    let notes = fs::read_to_string(temp.path().join("docs/release/v1.2.3.md"))
        .expect("Fix: generated release notes must be readable");
    assert!(notes.contains("| `vyre` | `owner/vyre` | `1.2.3` | `a@1.2.3` |"));
    assert!(notes.contains("| Final | Vyre | `vyre-v1.2.3` |"));
    assert!(notes.contains("## Validated changes"));
    assert!(notes.contains("- The exact release regression is fixed."));

    let checklist = fs::read_to_string(temp.path().join("docs/RELEASE_CHECKLIST.md"))
        .expect("Fix: generated release checklist must be readable");
    assert!(checklist.contains("`action-0`: External action 0."));
    assert!(checklist.contains("Push Vyre candidate tag `vyre-v1.2.3-rc.1`"));
}

/// Prevents a hand edit to a generated checklist from silently changing operator release instructions.
#[test]
fn check_rejects_generated_release_document_drift() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), 3, false);
    assert!(run_generator(temp.path(), "--write").status.success());
    fs::write(
        temp.path().join("docs/RELEASE_CHECKLIST.md"),
        "# locally edited release procedure\n",
    )
    .expect("Fix: stale fixture checklist must be writable");

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

/// Prevents release notes from passing when the train adds a token that the published notes omit.
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
        .contains("missing required token `required-but-absent`"));
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
    let candidate = "git tag -a \"$VYRE_RELEASE_TAG_VYRE_RC\" -m \"$VYRE_RELEASE_TAG_VYRE_RC\"";
    let prepublish = "\"$CARGO_RUNNER\" run -j1 --manifest-path xtask/Cargo.toml --bin xtask -- vyre-release-gate --prepublish";
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
