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

/// The one source file in this checkout that declares `marker`.
///
/// A roster test derives its member set by reading the file that publishes the
/// family, and naming that file by a crate plus a relative path pins the layout
/// the test was written against. The file then moves, the read fails with
/// `NotFound`, and the failure describes a missing path rather than the roster
/// the test claims to check. Searching for the declaration instead follows the
/// file wherever it goes, across crates as well as directories.
///
/// The search covers every source file the workspace holds, from
/// [`structure_gate::scan`], which is the same roster the structure gate walks.
/// `marker` must be the opening text of the declaration, such as
/// `pub fn fnv1a64_program`, and a file matches when some line begins with it
/// once indentation is trimmed. Matching a line prefix rather than any
/// occurrence keeps the caller's own copy of the marker, which is a string
/// literal in the middle of a line, from answering as a second home.
///
/// Exactly one file must match. Zero means the declaration was renamed or
/// deleted, and a roster derived from nothing proves nothing. Two or more means
/// the family has a second home, which is the duplication these contracts exist
/// to catch, so it is reported rather than silently resolved to the first hit.
///
/// # Panics
///
/// Panics when the number of matching files is not exactly one.
#[must_use]
pub fn declaring_source_file(marker: &str) -> PathBuf {
    let root = vyre_workspace_root();
    let matches: Vec<String> = structure_gate::scan(&root)
        .source_files
        .into_iter()
        .filter(|relative| {
            std::fs::read_to_string(root.join(relative)).is_ok_and(|source| {
                source
                    .lines()
                    .any(|line| line.trim_start().starts_with(marker))
            })
        })
        .collect();
    match matches.as_slice() {
        [only] => root.join(only),
        [] => panic!(
            "no source file in this workspace declares `{marker}`. Fix: the declaration was \
             renamed or deleted, so update the marker to the name the family publishes now."
        ),
        several => panic!(
            "`{marker}` is declared in {} files: {}. Fix: the family has more than one home, \
             so collapse it onto one owner.",
            several.len(),
            several.join(", ")
        ),
    }
}

/// Workspace member paths and excluded paths from the root manifest, in this
/// checkout.
///
/// Delegates to [`structure_gate`], which owns the root-manifest parse. A
/// contract that reads `Cargo.toml` itself to answer "is this directory in the
/// workspace" grows a second roster, and the two disagree the first time a
/// path is written in a shape only one of them recognizes.
///
/// # Panics
///
/// Panics when the root manifest cannot be read or parsed.
#[must_use]
pub fn vyre_workspace_rosters() -> WorkspaceRosters {
    let root = vyre_workspace_root();
    WorkspaceRosters {
        members: structure_gate::workspace_members(&root)
            .into_iter()
            .collect(),
        excluded: structure_gate::workspace_excludes(&root)
            .into_iter()
            .collect(),
    }
}

/// What the root manifest says the workspace holds and keeps out.
pub struct WorkspaceRosters {
    /// Paths declared as workspace members.
    pub members: std::collections::BTreeSet<String>,
    /// Paths declared in `workspace.exclude`.
    pub excluded: std::collections::BTreeSet<String>,
}

/// Directory cargo is writing this run's build artifacts into.
///
/// A test that has to build a scratch crate of its own puts it under here
/// rather than under [`std::env::temp_dir`]. A temp filesystem is small, shared
/// and capped: one fixture that compiled a dependency graph into it filled the
/// filesystem and failed unrelated builds with no space left on device. Build
/// artifacts belong in the build directory, which every host of this workspace
/// already points at a disk sized for them.
///
/// Resolved from the running test binary, which cargo placed inside the target
/// directory, by finding the ancestor cargo tagged as a cache directory. A test
/// therefore never reads or sets `CARGO_TARGET_DIR`: the location is declared
/// once, in the checkout's cargo configuration, and this reports where that
/// declaration actually put the artifacts.
///
/// # Panics
///
/// Panics when the test binary sits outside any cargo target directory.
#[must_use]
pub fn cargo_target_directory() -> PathBuf {
    let executable = std::env::current_exe().expect("Fix: the test binary path must be readable");
    executable
        .ancestors()
        .find(|directory| directory.join("CACHEDIR.TAG").is_file())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| {
            panic!(
                "Fix: no ancestor of `{}` is a cargo target directory; run this test through cargo",
                executable.display()
            )
        })
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

    /// A resolved declaring file must actually declare what was asked for.
    ///
    /// The failure this closes: a roster test named its source by a crate plus
    /// a relative path, the file moved crates, and the read failed with
    /// `NotFound`. The resolver answers with a file that exists and holds the
    /// declaration, so a rehome is followed instead of reported as missing.
    /// The marker is this function's own signature, which is present exactly
    /// once and moves with this file, so the test cannot go stale against a
    /// path written into it.
    #[test]
    fn a_resolved_declaring_file_holds_the_declaration_it_searched_for() {
        let marker = "pub fn declaring_source_file";
        let resolved = declaring_source_file(marker);
        assert!(
            resolved.is_absolute(),
            "{} must be absolute so a caller can read it from any directory",
            resolved.display()
        );
        let source = std::fs::read_to_string(&resolved)
            .unwrap_or_else(|error| panic!("{} must be readable: {error}", resolved.display()));
        assert!(
            source
                .lines()
                .any(|line| line.trim_start().starts_with(marker)),
            "{} does not declare `{marker}`",
            resolved.display()
        );
    }

    /// Only a declaration counts, never a mention.
    ///
    /// A caller stores its marker as a string literal, so a substring search
    /// finds the caller too and reports the family as having two homes. That
    /// turns a working resolver into a false duplication report, so the match
    /// is anchored at the start of a line.
    #[test]
    fn a_marker_quoted_inside_a_line_is_not_a_declaration() {
        // This very line mentions the marker mid-line, and this file still
        // resolves as the single declaring home above.
        let quoted = "pub fn declaring_source_file";
        assert_eq!(
            declaring_source_file(quoted),
            vyre_workspace_root()
                .join("vyre-test-support")
                .join("src")
                .join("monorepo.rs")
        );
    }

    /// A marker nothing declares must say so rather than resolve to anything.
    ///
    /// Returning a first hit or an empty path would let a roster test derive
    /// its member set from the wrong file, or from nothing, and still pass.
    #[test]
    #[should_panic(expected = "no source file in this workspace declares")]
    fn a_marker_no_file_declares_is_reported_rather_than_guessed() {
        let _ = declaring_source_file("pub fn no_such_declaration_exists_anywhere");
    }
}
