//! Harness shared by this crate's integration test targets.
//!
//! Every target here resolves the checkout root, and each of the items below is
//! reached from both of the crate's test binaries. A helper reached from one of
//! them is declared in that binary instead, because an item unused by a target
//! that compiles this module is dead code and the allowance that would hide it
//! also hides a helper that has lost its last caller.

use std::path::{Path, PathBuf};
use std::process::Command;

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
