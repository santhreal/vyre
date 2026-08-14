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

use std::path::PathBuf;

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

#[cfg(test)]
mod tests {
    use super::checkout_root;

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
}
