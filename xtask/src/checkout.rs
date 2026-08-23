//! The checkout these tools report on.
//!
//! Every gate here reads the working tree and prints a number about it, so the
//! tree it names has to be the tree it ran in. Two earlier answers were wrong.
//!
//! Deriving the root from `CARGO_MANIFEST_DIR` gave the right answer in one
//! checkout and the wrong one across two: cargo hashes a workspace member by its
//! path relative to the workspace root and checks freshness by mtime, so a target
//! directory shared by several checkouts computes the same unit hash for all of
//! them and hands one checkout a binary compiled by another.
//!
//! Reading `VYRE_CHECKOUT_ROOT` with `env!` was supposed to fix that by making
//! the value a fingerprint input. It does not. Measured 2026-08-12: the shared
//! `xtask` binary baked a worktree's path, `dup-scan` read that worktree's
//! `xtask/dup-baseline.toml`, and it reported this tree's `xtask` at 473 lines
//! against a pin of 465 while this tree's own file said 411. Cargo does not
//! export a `relative = true` config variable to the process it runs either, so
//! the run-time lookup fell through to the compiled-in value every time.
//!
//! The root is therefore resolved from the working directory at run time, which
//! is correct whichever binary cargo decided to reuse. The walk itself lives in
//! `structure-gate`, the one crate here that depends on no vyre crate, so a
//! single owner answers this question for every gate.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Absolute root of the checkout this tool was invoked in.
///
/// # Panics
///
/// Panics when no ancestor of the working directory declares a `[workspace]`,
/// because every gate here reports on a workspace and has nothing to measure
/// outside one.
#[must_use]
pub fn checkout_root() -> PathBuf {
    structure_gate::workspace_root()
}

/// Whether this checkout resolves a git reference to a commit.
#[must_use]
pub fn git_ref_exists(root: &Path, reference: &str) -> bool {
    Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "--verify", "--quiet"])
        .arg(format!("{reference}^{{commit}}"))
        .output()
        .is_ok_and(|output| output.status.success())
}

/// The environment variable a pull request carries its base branch in.
pub const BASE_REF_VARIABLE: &str = "GITHUB_BASE_REF";

/// The ref a run compares against, from the flag then the event.
///
/// `--base` wins, then the pull-request variable, which is set on that event and
/// empty on every other. The event arrives as an argument so that the gate that
/// reads a diff answers from its arguments alone: asking it for the worktree
/// diff means the worktree diff, on a runner whose event names a base branch
/// too.
#[must_use]
pub fn requested_base_ref(flag: Option<&str>, event: Option<String>) -> Option<String> {
    flag.map(str::to_string)
        .or(event)
        .filter(|reference| !reference.is_empty())
}

/// The ref this checkout resolves for the name a caller supplied.
///
/// [`BASE_REF_VARIABLE`] carries a bare branch name that often exists only as a
/// remote ref, so a bare name falls back to `origin/<name>`. Prefixing before
/// looking tells a caller who named a revision this checkout already holds to
/// fetch a ref that is in front of it.
#[must_use]
pub fn resolvable_base_ref(root: &Path, named: &str) -> Option<String> {
    if git_ref_exists(root, named) {
        return Some(named.to_string());
    }
    let remote = format!("origin/{named}");
    git_ref_exists(root, &remote).then_some(remote)
}

#[cfg(test)]
mod tests {
    use super::{checkout_root, requested_base_ref, resolvable_base_ref};

    /// The gates answer for the tree they run in, not for the tree that built them.
    #[test]
    fn the_root_contains_the_directory_this_test_runs_in() {
        let root = checkout_root();
        let invoked_in =
            std::env::current_dir().expect("Fix: the working directory must be readable");

        assert!(
            invoked_in.starts_with(&root),
            "resolved `{}`, which does not contain `{}`",
            root.display(),
            invoked_in.display()
        );
        assert!(root.join("xtask/Cargo.toml").is_file());
    }

    /// WHY: a gate that resolved this out of the process environment inside its
    /// diff reader answered a pull-request question to a caller who asked for
    /// the worktree. Every combination of flag and event is stated here, so a
    /// second reader cannot appear without a decision.
    #[test]
    fn the_base_ref_comes_from_the_flag_then_the_event_and_never_from_nothing() {
        assert_eq!(
            requested_base_ref(Some("release"), Some("main".to_string())),
            Some("release".to_string()),
            "Fix: an explicit --base must win over the event"
        );
        assert_eq!(
            requested_base_ref(None, Some("main".to_string())),
            Some("main".to_string()),
            "Fix: a pull request compares against the branch it targets"
        );
        assert_eq!(
            requested_base_ref(None, Some(String::new())),
            None,
            "Fix: the variable is empty on every event that is not a pull request"
        );
        assert_eq!(
            requested_base_ref(Some(""), None),
            None,
            "Fix: an empty flag names no ref"
        );
        assert_eq!(requested_base_ref(None, None), None);
    }

    /// WHY: prefixing a name with `origin/` before looking for it told a caller
    /// who named a revision this checkout holds to fetch a ref in front of it,
    /// and a name that resolves nowhere has to stay unresolved rather than
    /// become a remote ref nobody can fetch.
    #[test]
    fn a_base_ref_resolves_locally_first_and_falls_back_to_the_remote() {
        let root = checkout_root();

        assert_eq!(
            resolvable_base_ref(&root, "HEAD"),
            Some("HEAD".to_string()),
            "Fix: a revision this checkout resolves is the base, unprefixed"
        );
        assert_eq!(
            resolvable_base_ref(&root, "a-branch-no-checkout-has-6f1c2a"),
            None,
            "Fix: a name that resolves neither locally nor on origin has no base"
        );
    }
}
