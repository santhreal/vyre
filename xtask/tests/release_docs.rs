//! Release-train-derived documentation contract tests.

#![forbid(unsafe_code)]

mod workspace_sources;

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

use workspace_sources::workspace_root;

/// Run the gate against one checkout, from inside it.
///
/// The gate resolves its root by walking up from the working directory, so a
/// fixture is judged by running there. `mode` is `--check` or `--write`, and
/// `--check` is the absence of `--write`.
fn run_gate(root: &Path, mode: &str) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command.current_dir(root).arg("release-docs");
    if mode == "--write" {
        command.arg("--write");
    }
    command
        .output()
        .expect("Fix: the xtask binary must be runnable")
}

/// Everything the gate reported, so an assertion names the finding text.
fn reported(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn write_fixture(root: &Path, actions: usize, duplicate_package: bool) {
    for directory in ["scripts", "release/changes/unreleased"] {
        fs::create_dir_all(root.join(directory))
            .expect("Fix: release document fixture directories must be creatable");
    }
    // The gate resolves the checkout root by walking up for a `[workspace]`, so
    // a fixture without one would be judged against whichever ancestor has one.
    fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = []\nresolver = \"2\"\n",
    )
    .expect("Fix: fixture workspace manifest must be writable");
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
    let output = run_gate(&workspace_root(), "--check");
    assert!(
        output.status.success(),
        "Fix: regenerate or repair release documents: {}",
        reported(&output)
    );
    assert!(
        reported(&output).ends_with("release-docs: 0 finding(s)\n"),
        "{}",
        reported(&output)
    );
}

/// Proves one write derives the changelog's release identities and its change
/// list from the train and the fragments, with no hand-maintained copy between.
#[test]
fn write_derives_the_changelog_from_fragments() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), 3, false);
    let output = run_gate(temp.path(), "--write");
    assert!(output.status.success(), "{}", reported(&output));

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

/// WHY: a fragment used to be a `[[fragments]]` table in one shared file, and a
/// merge ate the header ten times in one week: every fragment opens with that
/// identical line, so diff3 treats it as common context between two appended
/// blocks and only one copy survives, leaving the second fragment's keys inside
/// the first. The id is the file name now, so a file carrying the old shape is
/// not a fragment that merged badly, it is a file nobody migrated, and it is
/// rejected by name rather than folded into its neighbour.
#[test]
fn a_fragment_file_in_the_retired_shape_is_rejected_by_name() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), 3, false);
    fs::write(
        temp.path().join("release/changes/unreleased/legacy-shape.toml"),
        "[[fragments]]\nid = \"legacy-shape\"\ncategory = \"Fixed\"\ntext = \"Carried the retired table shape.\"\n",
    )
    .expect("Fix: retired-shape fixture fragment must be writable");

    let output = run_gate(temp.path(), "--check");
    assert!(!output.status.success());
    let reported = reported(&output);
    assert!(
        reported
            .contains("release/changes/unreleased/legacy-shape.toml: unexpected key(s) fragments"),
        "the rejection must name the file and the key: {reported}"
    );
}

/// WHY: the reason a fragment is its own file is that two of them written
/// independently must both survive being brought together. Two files, two
/// entries, no shared line to lose.
#[test]
fn two_independently_written_fragments_both_reach_the_changelog() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), 3, false);
    fs::write(
        temp.path().join("release/changes/unreleased/second-change.toml"),
        "category = \"Changed\"\ntext = \"The second author's change is recorded too.\"\n",
    )
    .expect("Fix: second fixture fragment must be writable");
    assert!(run_gate(temp.path(), "--write").status.success());

    let changelog = fs::read_to_string(temp.path().join("CHANGELOG.md"))
        .expect("Fix: generated changelog must be readable");
    assert!(changelog.contains("- The exact release regression is fixed."));
    assert!(changelog.contains("- The second author's change is recorded too."));
}

/// A fragment written but not yet staged still reaches the changelog.
///
/// WHY: the fragment is written in the same change it describes, and the
/// documents are regenerated before anything is staged. Listing the git index
/// instead of the directory would drop exactly the newest fragment, and the
/// release would ship describing every change but the last one.
#[test]
fn an_unstaged_fragment_reaches_the_changelog() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), 3, false);
    assert!(
        Command::new("git")
            .arg("-C")
            .arg(temp.path())
            .arg("init")
            .arg("--quiet")
            .status()
            .expect("Fix: git must be runnable")
            .success(),
        "Fix: the fixture must initialise as a repository"
    );
    fs::write(
        temp.path().join("release/changes/unreleased/never-staged.toml"),
        "category = \"Added\"\ntext = \"The unstaged fragment is still a change.\"\n",
    )
    .expect("Fix: unstaged fixture fragment must be writable");
    assert!(run_gate(temp.path(), "--write").status.success());

    let changelog = fs::read_to_string(temp.path().join("CHANGELOG.md"))
        .expect("Fix: generated changelog must be readable");
    assert!(changelog.contains("- The unstaged fragment is still a change."));
}

/// Prevents a hand edit to the generated changelog section from silently
/// changing what a release claims to contain.
#[test]
fn check_rejects_generated_changelog_drift() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), 3, false);
    assert!(run_gate(temp.path(), "--write").status.success());
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

    let output = run_gate(temp.path(), "--check");
    assert!(!output.status.success());
    assert!(reported(&output).contains(
        "the generated release content disagrees with the fragments and the train"
    ));
}

/// Prevents a package from publishing under two versions or repositories in one release train.
#[test]
fn duplicate_release_group_membership_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), 3, true);
    let output = run_gate(temp.path(), "--write");
    assert!(!output.status.success());
    assert!(reported(&output).contains("package `a` belongs to both `secondary` and `vyre`"));
}

/// Preserves the exact approval boundary required by prepublication evidence.
///
/// Every external action carries an id, and no two share one: the launch record
/// cites an action by id, so a missing or repeated id records approval for an
/// action nobody can identify.
#[test]
fn an_external_action_without_an_id_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), 3, false);
    let train_path = temp.path().join("release/release-train.toml");
    let train = fs::read_to_string(&train_path).expect("Fix: fixture train must be readable");
    fs::write(&train_path, train.replace("id = \"action-2\"\n", ""))
        .expect("Fix: modified fixture train must be writable");

    let output = run_gate(temp.path(), "--write");
    assert!(!output.status.success());
    assert!(reported(&output)
        .contains("an approval-gated external action has no id, or two share one"));
}

/// The train declares one entry per action the launch contract needs.
///
/// WHY: the count used to be the literal three, restated here and in the launch
/// contract. A fourth required action would have left this gate green while the
/// release recorded approval for three.
#[test]
fn a_missing_external_action_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), 2, false);
    let output = run_gate(temp.path(), "--write");
    assert!(!output.status.success());
    assert!(reported(&output).contains(
        "the train declares 2 approval-gated external action(s) and the launch contract needs 3"
    ));
}

/// Prevents release notes from passing when the train adds a token that the changelog omits.
#[test]
fn missing_release_note_token_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), 3, false);
    assert!(run_gate(temp.path(), "--write").status.success());
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

    let output = run_gate(temp.path(), "--check");
    assert!(!output.status.success());
    assert!(reported(&output).contains(
        "the release train requires the token `required-but-absent`, which the changelog does not carry"
    ));
}

/// Prevents completion evidence from claiming publish or push success before those actions run.
#[test]
fn guarded_launch_order_fails_closed_when_candidate_tags_are_reordered() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_fixture(temp.path(), 3, false);
    assert!(run_gate(temp.path(), "--write").status.success());
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

    let output = run_gate(temp.path(), "--check");
    assert!(!output.status.success());
    assert!(reported(&output).contains("are out of order"));
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
