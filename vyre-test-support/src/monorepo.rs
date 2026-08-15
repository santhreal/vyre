//! Locating the sibling repositories a few cross-crate contracts read.
//!
//! # ONE definitional home for "where do the sibling products live"
//!
//! A handful of contract tests do not read vyre source at all. They read the
//! compiler CLI at `tools/vyrec` and the dataflow crates at `libs/dataflow`,
//! both of which live outside the vyre workspace,
//! next to it inside a larger monorepo. Those tests used to find them by
//! counting parent directories: `ancestors().nth(5)` from `CARGO_MANIFEST_DIR`,
//! or a literal `../../../../tools/vyrec/src`. A fixed count encodes one exact
//! checkout layout. Move the workspace anywhere else, including a git worktree
//! or a standalone clone of the published crate, and the walk runs off the
//! filesystem root and the test dies with a raw missing-path panic instead of
//! saying what it wanted.
//!
//! This module owns the resolution instead, and it answers with an [`Option`]
//! so callers can say "not here" out loud rather than panic. Nothing else in
//! the workspace should compute a sibling path by hand.
//!
//! # How resolution works
//!
//! [`vyre_workspace_root`] delegates to [`structure_gate::workspace_root`],
//! which walks up from the working directory to the manifest that declares the
//! workspace. It is deliberately not this crate's `CARGO_MANIFEST_DIR`: a
//! target directory shared by several checkouts computes the same unit hash for
//! this crate, so cargo hands one checkout a binary another one built, and a
//! compiled-in path then names the wrong tree.
//!
//! [`santh_root`] takes the first of these that works:
//!
//! 1. `VYRE_SANTH_ROOT`, if set. An explicit declaration always wins, and it is
//!    how you point the contracts at a monorepo checkout from a workspace that
//!    is not inside one.
//! 2. The nearest ancestor of the vyre workspace root that contains
//!    `tools/vyrec/src`. This searches for the marker rather than counting
//!    hops, so depth stops mattering.
//!
//! When neither works there is no monorepo here, which is the normal case for a
//! standalone clone. Call [`skip_without_santh_root`] to report the skip and
//! name the escape hatch, then return. A silent skip would be a coverage hole
//! nobody can see, so the message is mandatory and goes to stderr.

use std::path::{Path, PathBuf};

/// Environment variable that declares the monorepo root explicitly.
///
/// Set it when the vyre workspace is checked out on its own and you still want
/// the cross-repository contracts to run against a monorepo elsewhere on disk.
pub const SANTH_ROOT_ENV: &str = "VYRE_SANTH_ROOT";

/// Path, relative to the monorepo root, that marks a monorepo root.
///
/// The compiler CLI is the sibling every cross-repository contract needs, so
/// its presence is the same question as "is this the root".
const SANTH_ROOT_MARKER: &str = "tools/vyrec/src";

/// Path, relative to the monorepo root, holding the dataflow crates.
const DATAFLOW_RELATIVE: &str = "libs/dataflow";

/// Root of the vyre cargo workspace, resolved from the working directory.
///
/// Delegates to [`structure_gate::workspace_root`], the one owner of "which
/// checkout am I reporting on". Never compiled in: every checkout of this
/// repository shares one cargo target directory, cargo hashes a member by its
/// path relative to the workspace root and checks freshness by mtime, so two
/// checkouts compute the same unit hash and hand each other compiled binaries.
/// A path fixed at compile time then names whichever tree built last, and a
/// test that reads docs, pins, fixtures or golden files through it audits that
/// tree while claiming to describe this one.
///
/// # Panics
///
/// Panics when no ancestor of the working directory declares a `[workspace]`.
#[must_use]
pub fn vyre_workspace_root() -> PathBuf {
    structure_gate::workspace_root()
}

/// Directory of the workspace member that declares `package`, in this checkout.
///
/// A gate needing its own crate directory calls this with `CARGO_PKG_NAME`
/// rather than joining a compiled-in manifest path: the package name is stable
/// across checkouts, the directory is not, and a member's directory is not
/// always its package name.
///
/// # Panics
///
/// Panics when no workspace member declares `package`.
#[must_use]
pub fn vyre_crate_directory(package: &str) -> PathBuf {
    structure_gate::member_directory(&vyre_workspace_root(), package)
}

/// Root of the monorepo hosting vyre and its sibling products, if there is one.
///
/// Returns `None` for a standalone vyre checkout. See the module docs for the
/// resolution order.
#[must_use]
pub fn santh_root() -> Option<PathBuf> {
    if let Some(declared) = std::env::var_os(SANTH_ROOT_ENV) {
        let declared = PathBuf::from(declared);
        assert!(
            declared.join(SANTH_ROOT_MARKER).is_dir(),
            "Fix: {SANTH_ROOT_ENV} is set to {} but that directory does not contain \
             {SANTH_ROOT_MARKER}; point it at the monorepo root that holds the vyrec \
             compiler CLI, or unset it to let the contracts skip.",
            declared.display()
        );
        return Some(declared);
    }

    vyre_workspace_root()
        .ancestors()
        .find(|candidate| candidate.join(SANTH_ROOT_MARKER).is_dir())
        .map(Path::to_path_buf)
}

/// Directory holding the dataflow crates, if the monorepo is present.
///
/// Resolved from [`santh_root`] rather than by walking up from the vyre
/// workspace, so both siblings answer to the same root.
#[must_use]
pub fn dataflow_root() -> Option<PathBuf> {
    let root = santh_root()?.join(DATAFLOW_RELATIVE);
    root.is_dir().then_some(root)
}

/// Reports that a cross-repository contract is not running here, and why.
///
/// Use it at the top of any test that needs [`santh_root`]:
///
/// ```ignore
/// let Some(root) = santh_root() else {
///     skip_without_santh_root("vyrec backend linkage");
///     return;
/// };
/// ```
///
/// The message names the contract and the environment variable, so a run that
/// skipped is distinguishable in the log from a run that passed. Skipping
/// quietly would let a whole class of contracts disappear from CI unnoticed.
pub fn skip_without_santh_root(contract: &str) {
    eprintln!("{}", skip_notice(contract));
}

/// The text [`skip_without_santh_root`] prints, as a value.
///
/// Kept separate only so the wording has one definition that both the printer
/// and its test read.
#[must_use]
pub fn skip_notice(contract: &str) -> String {
    format!(
        "SKIP {contract}: no monorepo root found above {}. This contract reads sibling \
         repositories ({SANTH_ROOT_MARKER}, {DATAFLOW_RELATIVE}) that a standalone vyre \
         checkout does not have. Set {SANTH_ROOT_ENV} to a monorepo root to run it.",
        vyre_workspace_root().display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The workspace root must be the directory holding the workspace manifest.
    ///
    /// This is the anchor every other resolution hangs off. If it drifted (say
    /// this crate were moved a level deeper) the sibling search would start
    /// from the wrong place and silently find nothing.
    #[test]
    fn vyre_workspace_root_is_the_directory_holding_the_workspace_manifest() {
        let root = vyre_workspace_root();
        assert!(
            root.join("Cargo.toml").is_file(),
            "expected a workspace manifest at {}",
            root.join("Cargo.toml").display()
        );
        assert!(
            root.join("vyre-test-support").join("Cargo.toml").is_file(),
            "expected this crate to be a direct child of {}",
            root.display()
        );
    }

    /// A resolved monorepo root must actually contain the marker.
    ///
    /// The old fixed-depth walk could land on any directory at all and only
    /// failed later, inside an unrelated `read`, with a missing-path panic. The
    /// resolver decides the question up front: whatever it returns holds the
    /// sibling it was looking for.
    #[test]
    fn a_resolved_santh_root_contains_the_marker_it_searched_for() {
        if let Some(root) = santh_root() {
            assert!(
                root.join(SANTH_ROOT_MARKER).is_dir(),
                "santh_root() returned {} which lacks {SANTH_ROOT_MARKER}",
                root.display()
            );
        }
    }

    /// Resolution must not depend on how deep the workspace sits.
    ///
    /// This is the regression the module exists for: verifying 0.6.6 from a git
    /// worktree one directory shallower than the canonical checkout made five
    /// contracts fail on `ancestors().nth(5)` walking past the intended root.
    /// Searching for the marker gives the same answer at any depth, so the
    /// resolved root is an ancestor of the workspace, never a fixed hop count.
    #[test]
    fn santh_root_is_an_ancestor_of_the_workspace_rather_than_a_fixed_hop_count() {
        let Some(root) = santh_root() else {
            skip_without_santh_root("santh_root ancestry");
            return;
        };
        let workspace = vyre_workspace_root();
        assert!(
            workspace.ancestors().any(|candidate| candidate == root),
            "santh_root() returned {}, which is not an ancestor of {}",
            root.display(),
            workspace.display()
        );
    }

    /// The dataflow root must hang off the same monorepo root as the CLI.
    ///
    /// It used to be reached by a separate relative walk
    /// (`workspace/../../../dataflow`), which could resolve against a different
    /// tree than the one `santh_root()` found. Deriving both from one root
    /// keeps the two siblings consistent by construction.
    #[test]
    fn dataflow_root_is_derived_from_the_same_root_as_the_compiler_cli() {
        let Some(root) = santh_root() else {
            skip_without_santh_root("dataflow root derivation");
            return;
        };
        if let Some(dataflow) = dataflow_root() {
            assert_eq!(
                dataflow,
                root.join(DATAFLOW_RELATIVE),
                "dataflow_root() must be {DATAFLOW_RELATIVE} under the resolved monorepo root"
            );
            assert!(
                dataflow.is_dir(),
                "{} must be a directory",
                dataflow.display()
            );
        }
    }

    /// A skip has to say something.
    ///
    /// The whole point of returning `Option` instead of panicking is that a
    /// standalone checkout runs green. That is only acceptable if the skip is
    /// visible, so this pins that the message names both the contract and the
    /// escape hatch.
    #[test]
    fn the_skip_notice_names_the_contract_and_the_environment_variable() {
        let rendered = skip_notice("example contract");
        assert!(rendered.starts_with("SKIP example contract:"), "{rendered}");
        assert!(rendered.contains(SANTH_ROOT_ENV), "{rendered}");
        assert!(rendered.contains(SANTH_ROOT_MARKER), "{rendered}");
    }
}
