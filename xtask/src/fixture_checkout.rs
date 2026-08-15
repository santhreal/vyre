//! Building a throwaway git checkout for a contract to judge.
//!
//! Every contract that measures what a checkout says about itself needs a real
//! repository to measure: a fingerprint, a dirty-worktree digest, and a
//! duplication scan all read `git` rather than a directory listing. Each such
//! test grew its own seeded checkout, so a fixture that stopped configuring an
//! identity or stopped committing broke one suite and not its twin.
//!
//! `xtask` is unpublished build tooling, so this is `pub` without a feature:
//! `xtask-evidence` builds the same fixture in an integration test, and it
//! reaches this crate as an ordinary dependency.

use std::path::Path;
use std::process::Command;

/// A checkout with one commit and one tracked file, and nothing else.
///
/// The identity is configured in the repository rather than read from the
/// environment, because a workstation with no global git identity would
/// otherwise fail the commit and report it as the contract failing.
///
/// # Panics
///
/// When the directory cannot be created, the tracked file cannot be written,
/// or a `git` invocation fails. A fixture that half exists measures the
/// fixture, not the contract.
pub fn seeded(directory: &Path) {
    create(directory);
    std::fs::write(directory.join("tracked.txt"), "original\n")
        .expect("Fix: write the tracked file.");
    for arguments in [
        vec!["init", "--quiet"],
        vec!["config", "user.email", "gate@example.invalid"],
        vec!["config", "user.name", "gate"],
        vec!["add", "tracked.txt"],
        vec!["commit", "--quiet", "-m", "seed"],
    ] {
        run(directory, &arguments);
    }
}

/// An initialized repository with no commit and no tracked file.
///
/// A scan that measures tracked and untracked files needs a repository, and
/// nothing else: seeding a commit would put a file in every measurement.
///
/// # Panics
///
/// When the directory cannot be created or `git init` fails, for the reason
/// [`seeded`] documents.
pub fn empty(directory: &Path) {
    create(directory);
    run(directory, &["init", "--quiet"]);
}

/// Create the fixture directory, because a caller may name one that a
/// temporary root does not hold yet.
fn create(directory: &Path) {
    std::fs::create_dir_all(directory).expect("Fix: create the fixture directory.");
}

/// Run one `git` invocation in `directory`, or fail naming it.
fn run(directory: &Path, arguments: &[&str]) {
    let status = Command::new("git")
        .args(arguments)
        .current_dir(directory)
        .status()
        .expect("Fix: run git to build the fixture checkout.");
    assert!(
        status.success(),
        "Fix: git {arguments:?} failed in the fixture."
    );
}
