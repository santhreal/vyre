//! Harness shared by this crate's integration test targets.
//!
//! Every target here resolves the checkout root and most of them run a
//! repository generator over a fixture workspace. Each target compiles this
//! module separately and uses the subset it needs, so an item unused by one
//! target is not dead code.
#![allow(dead_code)]

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// The checkout root, resolved from the working directory at run time.
///
/// Delegates to the one owner of that answer. A root fixed at compile time names
/// whichever checkout built the binary, and every checkout here shares one cargo
/// target directory, so these contracts would judge another tree's files.
pub(crate) fn workspace_root() -> PathBuf {
    structure_gate::workspace_root()
}

/// Run a repository script under `python3` and capture its output.
pub(crate) fn run_python(script: &str, args: &[&OsStr]) -> Output {
    Command::new("python3")
        .arg(workspace_root().join(script))
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("Fix: {script} must launch with python3: {error}"))
}

/// Run a generator over a fixture `root` in `mode`, the shape four gates share.
pub(crate) fn run_generator(script: &str, root: &Path, mode: &str) -> Output {
    run_python(script, &[root.as_os_str(), OsStr::new(mode)])
}
