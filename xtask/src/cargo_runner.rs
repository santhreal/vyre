//! The cargo a gate runs.
//!
//! Two answers to one question lived in this workspace: a `CARGO`-then-`PATH`
//! lookup copied into four gates, and a wrapper lookup in `output_arg` that ten
//! more called. A copy is where a default diverges, and these two diverged on
//! the thing that decides what gets compiled. The lookup is here now, once.
//!
//! Job count and target directory are declared outside this module, so nothing
//! here sets either.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Environment variable the workspace wrapper exports, naming itself.
const RUNNER_VARIABLE: &str = "VYRE_CARGO_RUNNER";

/// Environment variable cargo exports to every process it starts.
const CARGO_VARIABLE: &str = "CARGO";

/// The wrapper's file name at the workspace root.
const WRAPPER: &str = "cargo_full";

/// Pick the cargo to start from the environment and the workspace root.
///
/// The order is the order of authority. `VYRE_CARGO_RUNNER` is set by the
/// wrapper to its own path, so it names the checkout being judged and wins. A
/// wrapper beside `root` comes next, because the target directory is derived
/// from the wrapper's own location: a child started through a bare cargo with a
/// scrubbed environment compiles a member into a directory another checkout
/// already owns, and the two builds produce the same unit hash for different
/// source. `CARGO` follows, carrying the toolchain that started this process, so
/// a child of a `+nightly` run is not silently built by whatever is first on
/// `PATH`. The bare name is last.
fn resolve(runner: Option<OsString>, root: &Path, cargo: Option<OsString>) -> PathBuf {
    if let Some(runner) = runner.filter(|value| !value.is_empty()) {
        return PathBuf::from(runner);
    }
    let wrapper = root.join(WRAPPER);
    if wrapper.is_file() {
        return wrapper;
    }
    if let Some(cargo) = cargo.filter(|value| !value.is_empty()) {
        return PathBuf::from(cargo);
    }
    PathBuf::from("cargo")
}

/// The cargo binary this process should start for a build rooted at `root`.
#[must_use]
pub fn binary(root: &Path) -> PathBuf {
    resolve(
        std::env::var_os(RUNNER_VARIABLE),
        root,
        std::env::var_os(CARGO_VARIABLE),
    )
}

/// A cargo command rooted at `root`.
#[must_use]
pub fn command(root: &Path) -> Command {
    let mut command = Command::new(binary(root));
    command.current_dir(root);
    command
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: precedence is the whole content of this module, and it cannot be
    /// asserted through `binary` without writing process environment that every
    /// other test in the binary shares. `resolve` is crate-private, so no
    /// integration test reaches it.
    #[test]
    fn the_exported_runner_outranks_every_other_answer() {
        let root = Path::new("/does/not/exist");
        let chosen = resolve(
            Some(OsString::from("/wrapper/cargo_full")),
            root,
            Some(OsString::from("/toolchain/cargo")),
        );

        assert_eq!(chosen, PathBuf::from("/wrapper/cargo_full"));
    }

    #[test]
    fn a_wrapper_beside_the_root_outranks_the_parent_toolchain() {
        let root = tempfile::tempdir().expect("temporary directory");
        std::fs::write(root.path().join(WRAPPER), "#!/bin/sh\nexec cargo \"$@\"\n")
            .expect("wrapper");

        let chosen = resolve(None, root.path(), Some(OsString::from("/toolchain/cargo")));

        assert_eq!(chosen, root.path().join(WRAPPER));
    }

    #[test]
    fn the_parent_toolchain_answers_when_no_wrapper_is_beside_the_root() {
        let root = tempfile::tempdir().expect("temporary directory");

        let chosen = resolve(None, root.path(), Some(OsString::from("/toolchain/cargo")));

        assert_eq!(chosen, PathBuf::from("/toolchain/cargo"));
    }

    #[test]
    fn an_empty_variable_is_not_an_answer() {
        let root = tempfile::tempdir().expect("temporary directory");

        let chosen = resolve(
            Some(OsString::new()),
            root.path(),
            Some(OsString::new()),
        );

        assert_eq!(chosen, PathBuf::from("cargo"));
    }

    #[test]
    fn a_directory_named_like_the_wrapper_is_not_the_wrapper() {
        let root = tempfile::tempdir().expect("temporary directory");
        std::fs::create_dir(root.path().join(WRAPPER)).expect("directory");

        let chosen = resolve(None, root.path(), Some(OsString::from("/toolchain/cargo")));

        assert_eq!(chosen, PathBuf::from("/toolchain/cargo"));
    }
}
