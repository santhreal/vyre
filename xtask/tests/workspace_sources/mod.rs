//! Harness shared by this crate's integration test targets.
//!
//! Every target here resolves the checkout root and most of them run a
//! repository generator over a fixture workspace. Each target compiles this
//! module separately and uses the subset it needs, so an item unused by one
//! target is not dead code.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

use proc_macro2::LineColumn;
use xtask::gate::{Gate, GateCtx, Report};

/// The checkout root, resolved from the working directory at run time.
///
/// Delegates to the one owner of that answer. A root fixed at compile time names
/// whichever checkout built the binary, and every checkout here shares one cargo
/// target directory, so these contracts would judge another tree's files.
pub(crate) fn workspace_root() -> PathBuf {
    structure_gate::workspace_root()
}

/// Every workspace member's `src` directory, read from the root manifest at run
/// time.
///
/// A gate that judges production sources across the workspace has to know which
/// directories those are, and a hardcoded list goes stale the day a crate is
/// added, which is the same failure as having no gate. Members that ship no
/// `src` directory are skipped rather than reported: a conform harness or
/// fixture crate is not production source.
pub(crate) fn workspace_member_src_dirs(root: &Path) -> Vec<PathBuf> {
    structure_gate::workspace_members(root)
        .into_iter()
        .map(|member| root.join(member).join("src"))
        .filter(|path| path.is_dir())
        .collect()
}

/// Every file under `dir` whose extension is in `extensions`, at any depth.
///
/// One walk for every contract that reads the tree. A contract that wanted a
/// different extension used to copy the walk rather than widen it, and a copy
/// of a walk is a second answer to "which files does this gate cover".
pub(crate) fn sources_under(dir: &Path, extensions: &[&str]) -> Vec<PathBuf> {
    let mut sources = Vec::new();
    for entry in walkdir::WalkDir::new(dir) {
        let Ok(entry) = entry else {
            continue;
        };
        let path = entry.path();
        let matches = path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extensions.contains(&extension));
        if entry.file_type().is_file() && matches {
            sources.push(path.to_path_buf());
        }
    }
    sources
}

/// Every Rust source file under `dir`, at any depth.
pub(crate) fn rust_sources_under(dir: &Path) -> Vec<PathBuf> {
    sources_under(dir, &["rs"])
}

/// Every Rust source file under every workspace member's `src` directory.
pub(crate) fn workspace_member_sources(root: &Path) -> Vec<PathBuf> {
    workspace_member_src_dirs(root)
        .iter()
        .flat_map(|dir| rust_sources_under(dir))
        .collect()
}

/// Run one gate over a fixture checkout, in check or write mode.
///
/// The fixture has to be a real checkout: every gate here reads the tree
/// through `git ls-files`, so a directory of untracked files reads as an empty
/// workspace and the gate would report nothing at all.
pub(crate) fn run_gate(gate: &dyn Gate, root: &Path, write: bool) -> Report {
    let args = if write {
        vec!["--write".to_string()]
    } else {
        Vec::new()
    };
    gate.run(&GateCtx::new(root.to_path_buf(), args))
        .unwrap_or_else(|error| panic!("Fix: {} must run: {error:?}", gate.name()))
}

/// Track everything currently in a fixture directory, making it a checkout the
/// gates can read.
pub(crate) fn track_fixture(root: &Path) {
    for arguments in [
        vec!["init", "--quiet"],
        vec!["config", "user.email", "gate@example.invalid"],
        vec!["config", "user.name", "gate"],
        vec!["add", "--all", "--"],
        vec!["commit", "--quiet", "-m", "fixture"],
    ] {
        let status = Command::new("git")
            .args(&arguments)
            .current_dir(root)
            .status()
            .expect("Fix: git must launch to build the fixture checkout");
        assert!(
            status.success(),
            "Fix: git {arguments:?} failed in the fixture"
        );
    }
}

/// A `path:line:column` violation, with the path relative to `root`.
///
/// Column is one-based here and zero-based in `proc_macro2`, because a reader
/// pastes this into an editor. Two structural gates formatted it identically
/// and a third would have had to guess which convention they used.
pub(crate) fn violation_location(root: &Path, path: &Path, location: LineColumn) -> String {
    format!(
        "{}:{}:{}",
        path.strip_prefix(root).unwrap_or(path).display(),
        location.line,
        location.column + 1
    )
}
