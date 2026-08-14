//! Workspace-level platform documentation boundary contract.

use std::path::Path;

use std::process::Command;

#[test]
fn platform_crate_docs_and_comments_do_not_name_consumers() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest
        .parent()
        .expect("vyre-pass-engine should live directly under the workspace root");
    let script = workspace.join("scripts/check_platform_consumer_docs.sh");

    let output = Command::new("bash")
        .arg(&script)
        .current_dir(workspace)
        .output()
        .expect("platform consumer-doc boundary script should execute");

    assert!(
        output.status.success(),
        "platform consumer-doc boundary failed.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stderr).as_ref(),
        "",
        "platform consumer-doc boundary must be quiet on success"
    );
}

/// docs/INDEX.md must list every markdown document that exists under `docs/`
/// and is not gitignored, and must link nothing else: not a file that does not
/// exist, and not a gitignored file.
///
/// The oracle is the FILESYSTEM, not git's index. A documentation index is a
/// set of links, and "this link resolves" is answered by whether the file is
/// there, which is a stat(2) question. Tracking state answers a different
/// question, namely whether a change has been staged yet, and using it here
/// inverted the gate in both directions at once. It called
/// docs/archive/README.md (2172 bytes on disk) and docs/legacy/README.md (1199
/// bytes on disk) "missing files" purely because they had been written that day
/// and not committed: a present-but-uncommitted file is NOT a missing file.
/// Symmetrically it demanded rows for docs/catalog/anonymous.md,
/// docs/catalogs/coverage-matrix.md and docs/catalogs/op-id-catalog.md, three
/// documents already deleted from disk, because the deletions had not been
/// committed: a deleted-but-still-tracked file is NOT a present file. A gate
/// wrong in both directions is worse than no gate, and while red it hides the
/// next real INDEX.md drift.
///
/// Ignore status is the one thing still read from git, deliberately.
/// `git check-ignore` answers "will this file ever reach another reader",
/// which is not the same question as "is it committed right now". .gitignore
/// excludes `**/*PLAN*.md`, `**/*STATUS*.md`, `**/*ROADMAP*.md`,
/// `**/*AUDIT*.md`, `**/*BACKLOG*.md` and `**/AGENT_*.md` as private working
/// notes, so a row pointing at one is a link every reader outside the authoring
/// working copy finds broken. Listing it is worse than omitting it. The check
/// is index-aware, so a file matching an ignore pattern that is already tracked
/// is already in public history and stays indexed.
///
/// `scripts/check_docs_index.sh` owns the rule; this test is the cargo-visible
/// entry point. An uncommitted new document fails CI, a deleted document cannot
/// retain a dead index row, and private plans cannot enter public navigation.
#[test]
fn docs_index_covers_every_public_markdown_document() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest
        .parent()
        .expect("vyre-pass-engine should live directly under the workspace root");
    let script = workspace.join("scripts/check_docs_index.sh");

    let output = Command::new("bash")
        .arg(&script)
        .current_dir(workspace)
        .output()
        .expect("documentation index contract script should execute");

    assert!(
        output.status.success(),
        "documentation index contract failed.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stderr).as_ref(),
        "",
        "documentation index contract must be quiet on success"
    );
}

/// No public document may link a target a reader cannot open. Three classes, in
/// descending severity: the link escapes the repository root, so it resolves for
/// nobody; the target does not exist on disk, so it resolves for nobody; the
/// target exists here but is gitignored, so it resolves for the author and fails
/// for every other reader.
///
/// This exists because docs_index_covers_every_public_markdown_document only
/// guards docs/INDEX.md, and the same defect was sitting one directory down:
/// docs/archive/README.md listed fourteen sibling documents that .gitignore keeps
/// out of public history, so it read as a complete directory listing to us and as
/// fourteen dead links to everyone else. docs/TESTING_PROGRAM.md was worse: it
/// linked `../../../../../STANDARD.md`, five levels above the repository root into
/// a private monorepo, which is both unresolvable for every clone and a disclosure
/// of internal layout inside a crate we intend to publish. That link is gone and
/// the rule it pointed at is now stated inline in that document, which removes the
/// dependency rather than renaming it.
///
/// SCOPE IS A RULE, NOT A FILE LIST. An allowlist of exempt paths is a deferral
/// that rots when someone adds the next file. The rule: a HISTORICAL RECORD is not
/// gated, because a document under docs/archive/ or docs/legacy/ is a snapshot of
/// what was true on its date, and rewriting its links to resolve against today's
/// tree would falsify the record. A snapshot with a stale link is worth more than
/// a quietly corrected one, because the stale link is honest about when it was
/// written. The carve-out from the carve-out is the directory's own README.md:
/// that is not a snapshot, it is the current signpost telling a reader the
/// directory is stale and where to go instead, so its links are live claims and
/// are gated. This mirrors the exemption check_docs_index.sh makes for the
/// per-directory status rule, for the same reason, which is what makes it a rule
/// and not two coincidences.
///
/// Anchor fragments are deliberately out of scope: whether #some-heading still
/// exists is a far weaker signal than whether the FILE exists, and folding it in
/// would bury the two classes that matter under heading churn.
///
/// Breaks if it regresses: published docs accumulate links only the author can
/// follow, internal paths leak into shipped crates, and a reader who trusts a
/// signpost is sent to a file that is not there.
#[test]
fn public_docs_never_link_unreachable_targets() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest
        .parent()
        .expect("vyre-pass-engine should live directly under the workspace root");
    let script = workspace.join("scripts/check_docs_links.sh");

    let output = Command::new("bash")
        .arg(&script)
        .current_dir(workspace)
        .output()
        .expect("documentation link contract script should execute");

    assert!(
        output.status.success(),
        "documentation link contract failed.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stderr).as_ref(),
        "",
        "documentation link contract must be quiet on success"
    );
}
