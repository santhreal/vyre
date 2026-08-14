//! `vyre_workspace_root` answers for the checkout the command ran in.
//!
//! # The class this closes
//!
//! A helper that resolves the checkout root at compile time answers for
//! whichever checkout built the binary. Cargo computes the same unit hash for a
//! member in two checkouts that share a target directory, so it hands one
//! checkout a binary the other one built, and every contract test reading that
//! root then reads the wrong tree: a gate reports pins, fixtures, and workflow
//! files that belong to somebody else's worktree, and it passes or fails on
//! them. Nothing in the failure message says which tree it read.
//!
//! This asserts the property that makes that impossible, rather than the
//! spelling that caused it: put the process in a directory under a different
//! workspace manifest and the answer changes to that one. A compiled-in path
//! cannot satisfy it, whatever it is spelled with.
//!
//! # What it does not catch
//!
//! It proves the resolution follows the working directory. It does not prove
//! every caller asks this function rather than computing a root of its own.

use std::fs;

use vyre_test_support::monorepo::vyre_workspace_root;

/// Sole test in this binary on purpose: it moves the process working directory,
/// which is process-global state, so a second test in the same binary could
/// observe it mid-flight.
#[test]
fn workspace_root_is_the_workspace_the_process_stands_in() {
    let outside = tempfile::tempdir().expect("Fix: the temporary directory must be writable");
    let root = outside
        .path()
        .canonicalize()
        .expect("Fix: temp root must canonicalize");
    fs::write(root.join("Cargo.toml"), "[workspace]\nmembers = []\n")
        .expect("Fix: the synthetic workspace manifest must be writable");
    let nested = root.join("some-member/src");
    fs::create_dir_all(&nested).expect("Fix: the synthetic member directory must be creatable");

    let real_checkout = vyre_workspace_root();
    assert!(
        real_checkout.join("Cargo.toml").is_file(),
        "before moving, the answer must be this checkout, but {} holds no manifest",
        real_checkout.display()
    );
    assert_ne!(
        real_checkout, root,
        "the synthetic workspace must not be this checkout, or the test proves nothing"
    );

    std::env::set_current_dir(&nested)
        .expect("Fix: the synthetic member directory must be enterable");
    let moved = vyre_workspace_root();
    std::env::set_current_dir(&real_checkout).expect("Fix: the checkout must be re-enterable");

    assert_eq!(
        moved,
        root,
        "the workspace root must be resolved from the working directory at run time. It answered \
         with {} while the process stood in {}, which means it was decided when the binary was \
         compiled and now names whichever checkout built it.",
        moved.display(),
        nested.display()
    );
}
