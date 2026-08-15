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
    for directory in ["scripts", "release/changes/unreleased"] {
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
        root.join("release/changes/unreleased/exact-fix.toml"),
        "category = \"Fixed\"\ntext = \"The exact release regression is fixed.\"\n",
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
fn write_derives_the_changelog_from_fragments() {
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
fn check_rejects_generated_changelog_drift() {
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

/// Runs `git` in `dir` and returns its stdout, refusing to continue on failure.
fn git(dir: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .expect("Fix: the merge demonstration needs git on PATH");
    assert!(
        output.status.success(),
        "Fix: git {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("Fix: git output must be UTF-8")
}

/// A checkout with one committed fragment, and the sha to branch two ways from.
fn seed_fragment_repository(root: &Path) -> String {
    write_fixture(root, 3, false);
    git(root, &["init", "--quiet", "."]);
    git(root, &["config", "user.email", "release@example.invalid"]);
    git(root, &["config", "user.name", "release"]);
    git(root, &["add", "--all"]);
    git(root, &["commit", "--quiet", "-m", "seed"]);
    git(root, &["rev-parse", "HEAD"]).trim().to_string()
}

/// Commit `content` at `path` on a new branch cut from `base`.
fn append_fragment_on_branch(root: &Path, base: &str, branch: &str, path: &str, content: &str) {
    git(root, &["checkout", "--quiet", "-b", branch, base]);
    let target = root.join(path);
    fs::create_dir_all(target.parent().expect("Fix: fragment paths have a parent"))
        .expect("Fix: the fragment directory must be creatable");
    let mut existing = fs::read_to_string(&target).unwrap_or_default();
    existing.push_str(content);
    fs::write(&target, existing).expect("Fix: the fragment must be writable");
    git(root, &["add", "--all"]);
    git(root, &["commit", "--quiet", "-m", branch]);
}

/// The merge that used to eat a `[[fragments]]` header now keeps both fragments.
///
/// Two branches append a fragment, so both insertions begin with the same blank
/// line and the same `[[fragments]]` line. A three-way merge matches that shared
/// prefix between the two sides and leaves it out of the conflicting region, so
/// the region holds only the differing tails and the header is common context.
/// `merge=union` then resolves that region by concatenation and one header ends
/// up carrying two ids, which is not valid TOML. The attribute did not fail to
/// prevent the corruption, it produced it: without the attribute the same merge
/// stops on a conflict a person resolves.
///
/// The control below performs the exact append on a shared file under the exact
/// attribute and asserts the fusion, so this test fails if it stops
/// demonstrating the defect the per-file layout exists to prevent.
#[test]
fn two_branches_appending_a_fragment_merge_with_both_fragments_intact() {
    let shared = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    let one_file = "release/changes/shared.toml";
    let _ = seed_fragment_repository(shared.path());
    fs::write(
        shared.path().join(".gitattributes"),
        format!("{one_file} merge=union\n"),
    )
    .expect("Fix: the control must carry the attribute the tree carried");
    fs::write(
        shared.path().join(one_file),
        "schema_version = 1\n\n[[fragments]]\nid = \"exact-fix\"\ncategory = \"Fixed\"\ntext = \"The exact release regression is fixed.\"\n",
    )
    .expect("Fix: the shared-file control must be writable");
    git(shared.path(), &["add", "--all"]);
    git(shared.path(), &["commit", "--quiet", "-m", "shared base"]);
    let shared_base = git(shared.path(), &["rev-parse", "HEAD"]).trim().to_string();
    append_fragment_on_branch(
        shared.path(),
        &shared_base,
        "shared-alpha",
        one_file,
        "\n[[fragments]]\nid = \"alpha\"\ncategory = \"Added\"\ntext = \"Alpha.\"\n",
    );
    append_fragment_on_branch(
        shared.path(),
        &shared_base,
        "shared-beta",
        one_file,
        "\n[[fragments]]\nid = \"beta\"\ncategory = \"Added\"\ntext = \"Beta.\"\n",
    );
    git(shared.path(), &["merge", "shared-alpha", "-m", "merge"]);
    let fused = fs::read_to_string(shared.path().join(one_file))
        .expect("Fix: the merged control must be readable");
    assert_eq!(
        (
            fused.matches("[[fragments]]").count(),
            fused.matches("\nid = ").count()
        ),
        (2, 3),
        "Fix: the shared-file control must still fuse three fragments under two headers, or it proves nothing: {fused}"
    );
    assert!(
        fused.contains("id = \"alpha\"") && fused.contains("id = \"beta\""),
        "Fix: the control must keep both ids under one header: {fused}"
    );

    let split = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    let base = seed_fragment_repository(split.path());
    append_fragment_on_branch(
        split.path(),
        &base,
        "alpha",
        "release/changes/unreleased/alpha-fragment.toml",
        "category = \"Added\"\ntext = \"Alpha landed on its own branch.\"\n",
    );
    append_fragment_on_branch(
        split.path(),
        &base,
        "beta",
        "release/changes/unreleased/beta-fragment.toml",
        "category = \"Added\"\ntext = \"Beta landed on its own branch.\"\n",
    );
    git(split.path(), &["merge", "alpha", "-m", "merge"]);

    let output = run_generator(split.path(), "--write");
    assert!(
        output.status.success(),
        "Fix: the merged fragment set must still parse: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let changelog = fs::read_to_string(split.path().join("CHANGELOG.md"))
        .expect("Fix: generated changelog must be readable");
    assert!(
        changelog.contains("- Alpha landed on its own branch.")
            && changelog.contains("- Beta landed on its own branch."),
        "Fix: both merged fragments must reach the changelog: {changelog}"
    );
}
