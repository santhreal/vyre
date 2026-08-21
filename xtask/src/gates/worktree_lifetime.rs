//! A worktree outlives its branch only until that branch merges.
//!
//! Worktrees share one cargo target directory, and cargo reuses a compiled
//! artifact when package name, version and features match. A crate built from
//! one worktree's source is therefore served to another whose source differs,
//! and the error names the wrong thing: a trait method the building tree's own
//! source already declares, or a macro argument its source already accepts.
//! Proc-macro crates are the worst case, since a stale dylib reports errors
//! naming the trait rather than the macro that produced it.
//!
//! `CONTRIBUTING.md` records why the target directory stays shared: splitting
//! it per worktree multiplies a terabyte-scale `debug/deps` by the worktree
//! count. Worktree lifetime is the bound instead. That bound was prose, and
//! prose does not delete anything, so a merged branch kept a whole second
//! source tree alive with a live claim on the shared artifacts, and every grep,
//! gate and duplication scan walked it for nothing.
//!
//! A worktree whose branch has merged has no unmerged work left to protect, so
//! it is reported. One on an unmerged branch is the supported lane and is not.

use std::path::Path;
use std::process::Command;

use crate::gate::{Finding, GateError};

/// Refs a branch is considered merged into.
///
/// Both are named because a lane merges into the integration ref first and
/// reaches `main` later; a worktree is spent as soon as either has its work.
/// A ref this checkout does not have is skipped rather than assumed unmerged,
/// so a fresh clone with no local `integration` does not report every lane.
const MERGE_TARGETS: &[&str] = &["integration", "main"];

/// One worktree as `git worktree list --porcelain` describes it.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct Worktree {
    /// Absolute path of the checkout.
    pub(crate) path: String,
    /// Branch it has checked out, absent when detached.
    pub(crate) branch: Option<String>,
}

/// Worktrees whose branch has already merged.
pub(crate) fn findings(root: &Path) -> Result<Vec<Finding>, GateError> {
    let listing = git(root, &["worktree", "list", "--porcelain"])?;
    let worktrees = parse_worktrees(&listing);
    let mut merged = Vec::new();
    for worktree in worktrees.iter().skip(1) {
        let Some(branch) = worktree.branch.as_deref() else {
            continue;
        };
        if let Some(target) = merged_into(root, branch)? {
            merged.push((worktree, target));
        }
    }
    Ok(merged
        .into_iter()
        .map(|(worktree, target)| {
            let branch = worktree.branch.as_deref().unwrap_or("");
            Finding::new(
                format!(
                    "the worktree at `{}` is on `{branch}`, which has already merged into `{target}`",
                    worktree.path
                ),
                "run `git worktree remove` on it and delete the branch; a merged worktree protects no unmerged work and keeps a second source of the same package versions against the shared cargo target directory",
            )
        })
        .collect())
}

/// The first ref in [`MERGE_TARGETS`] that already contains `branch`.
fn merged_into(root: &Path, branch: &str) -> Result<Option<&'static str>, GateError> {
    for target in MERGE_TARGETS {
        if branch == *target || !ref_exists(root, target) {
            continue;
        }
        if is_ancestor(root, branch, target)? {
            return Ok(Some(target));
        }
    }
    Ok(None)
}

/// Whether this checkout has `reference` at all.
fn ref_exists(root: &Path, reference: &str) -> bool {
    Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "--verify", "--quiet", reference])
        .output()
        .is_ok_and(|output| output.status.success())
}

/// Whether every commit on `branch` is already on `target`.
fn is_ancestor(root: &Path, branch: &str, target: &str) -> Result<bool, GateError> {
    let status = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["merge-base", "--is-ancestor", branch, target])
        .status()
        .map_err(|error| {
            GateError::new(
                format!("cannot ask git whether `{branch}` merged into `{target}`: {error}"),
                "install git, or run this gate inside a git checkout",
            )
        })?;
    Ok(status.success())
}

/// Run a git command and return its stdout.
fn git(root: &Path, args: &[&str]) -> Result<String, GateError> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .map_err(|error| {
            GateError::new(
                format!("cannot run `git {}`: {error}", args.join(" ")),
                "install git, or run this gate inside a git checkout",
            )
        })?;
    if !output.status.success() {
        return Err(GateError::new(
            format!("`git {}` failed", args.join(" ")),
            "run this gate inside a git checkout",
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Parse `git worktree list --porcelain`, in listing order.
///
/// The format is stanzas separated by a blank line: `worktree <path>`, then
/// optionally `branch refs/heads/<name>` or `detached`. Parsing it here rather
/// than the human listing avoids guessing where a path with a space ends.
pub(crate) fn parse_worktrees(listing: &str) -> Vec<Worktree> {
    let mut worktrees: Vec<Worktree> = Vec::new();
    for line in listing.lines() {
        if let Some(path) = line.strip_prefix("worktree ") {
            worktrees.push(Worktree {
                path: path.to_string(),
                branch: None,
            });
        } else if let Some(reference) = line.strip_prefix("branch ") {
            if let Some(last) = worktrees.last_mut() {
                last.branch = Some(
                    reference
                        .strip_prefix("refs/heads/")
                        .unwrap_or(reference)
                        .to_string(),
                );
            }
        }
    }
    worktrees
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the porcelain format is stanzas, and a detached worktree carries no
    /// `branch` line at all. Reading the human listing instead would have to
    /// guess where a path containing a space ends.
    #[test]
    fn the_porcelain_listing_parses_paths_branches_and_detachment() {
        let parsed = parse_worktrees(
            "worktree /repo/main\nHEAD abc\nbranch refs/heads/integration\n\n\
             worktree /repo/work trees/lane\nHEAD def\nbranch refs/heads/lane/one\n\n\
             worktree /repo/detached\nHEAD 0f0\ndetached\n",
        );
        assert_eq!(
            parsed,
            vec![
                Worktree {
                    path: "/repo/main".to_string(),
                    branch: Some("integration".to_string()),
                },
                Worktree {
                    path: "/repo/work trees/lane".to_string(),
                    branch: Some("lane/one".to_string()),
                },
                Worktree {
                    path: "/repo/detached".to_string(),
                    branch: None,
                },
            ]
        );
    }

    /// WHY: an empty listing must not panic or index a first element that is
    /// not there. `git worktree list` always prints the main checkout, but a
    /// gate that trusts that is one failed command away from an unwrap.
    #[test]
    fn an_empty_listing_yields_no_worktrees() {
        assert!(parse_worktrees("").is_empty());
    }

    /// WHY: this checkout is the one the gate runs against, and it must be
    /// clean. A finding here means a merged worktree is live right now.
    #[test]
    fn this_checkout_has_no_merged_worktree() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the xtask manifest sits under the workspace root");
        let found = findings(root).expect("git is available in a checkout");
        assert!(found.is_empty(), "{found:?}");
    }

    /// Build a repository with `main`, a merged branch and an unmerged one.
    ///
    /// Real git rather than a fake listing, because the part that can be wrong
    /// is the plumbing: ancestry direction, ref existence, and which stanza a
    /// branch line belongs to.
    fn scratch_repository() -> tempfile::TempDir {
        let directory = tempfile::tempdir().expect("a temp directory");
        let root = directory.path();
        let run = |args: &[&str]| {
            let status = Command::new("git")
                .arg("-C")
                .arg(root)
                .args(args)
                .status()
                .expect("git runs");
            assert!(status.success(), "git {args:?}");
        };
        run(&["init", "--quiet", "--initial-branch", "main"]);
        run(&["config", "user.name", "gate"]);
        run(&["config", "user.email", "gate@example.invalid"]);
        run(&["config", "commit.gpgsign", "false"]);
        run(&["commit", "--quiet", "--allow-empty", "-m", "base"]);
        run(&["branch", "merged"]);
        run(&["checkout", "--quiet", "-b", "unmerged"]);
        run(&["commit", "--quiet", "--allow-empty", "-m", "ahead"]);
        run(&["checkout", "--quiet", "main"]);
        run(&["commit", "--quiet", "--allow-empty", "-m", "later"]);
        directory
    }

    /// WHY: the clean-tree assertion above passes just as well against a gate
    /// that always returns nothing. This one proves the rule fires, and names
    /// the ref that already has the work.
    #[test]
    fn a_worktree_on_a_merged_branch_is_reported() {
        let repository = scratch_repository();
        let root = repository.path();
        let checkout = root.join("spent");
        let status = Command::new("git")
            .arg("-C")
            .arg(root)
            .arg("worktree")
            .arg("add")
            .arg("--quiet")
            .arg(&checkout)
            .arg("merged")
            .status()
            .expect("git runs");
        assert!(status.success());

        let found = findings(root).expect("git is available");
        assert_eq!(found.len(), 1, "{found:?}");
        let message = format!("{found:?}");
        assert!(message.contains("merged"), "{message}");
        assert!(message.contains("main"), "{message}");
    }

    /// WHY: the supported lane must stay silent. A rule that convicts the
    /// worktree someone is working in right now is a rule that gets deleted.
    #[test]
    fn a_worktree_on_an_unmerged_branch_is_left_alone() {
        let repository = scratch_repository();
        let root = repository.path();
        let status = Command::new("git")
            .arg("-C")
            .arg(root)
            .arg("worktree")
            .arg("add")
            .arg("--quiet")
            .arg(root.join("live"))
            .arg("unmerged")
            .status()
            .expect("git runs");
        assert!(status.success());

        let found = findings(root).expect("git is available");
        assert!(found.is_empty(), "{found:?}");
    }

    /// WHY: `merge-base --is-ancestor` is directional, and the natural way to
    /// get it backwards is to ask whether the target is an ancestor of the
    /// branch. That mistake reports every live lane and clears every spent one,
    /// so both directions are pinned from one repository.
    #[test]
    fn ancestry_is_asked_in_the_direction_that_means_merged() {
        let repository = scratch_repository();
        let root = repository.path();
        assert_eq!(
            merged_into(root, "merged").expect("git is available"),
            Some("main")
        );
        assert_eq!(
            merged_into(root, "unmerged").expect("git is available"),
            None
        );
    }
}
