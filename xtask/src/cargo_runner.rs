//! Which cargo this tooling starts.
//!
//! One owner, because a second copy of this rule is a second answer. The
//! workspace wrapper declares the job count and the target directory once in
//! `.cargo/config.toml`; a caller that resolved its own cargo, or exported its
//! own job count, built a different build in the same checkout and defeated the
//! shared compilation cache.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// The bounded workspace cargo wrapper this tooling spawns.
///
/// `VYRE_CARGO_RUNNER` wins, then a `cargo_full` beside the workspace root,
/// then the name alone for a wrapper on `PATH`.
#[must_use]
pub fn runner(workspace_root: &Path) -> PathBuf {
    resolve(std::env::var_os("VYRE_CARGO_RUNNER"), workspace_root)
}

/// The precedence itself, with the environment passed in.
///
/// The wrapper exports `VYRE_CARGO_RUNNER` into every process it starts, this
/// tooling's own tests included, so a test that read the live environment would
/// observe the host wrapper instead of the rule.
fn resolve(override_value: Option<OsString>, workspace_root: &Path) -> PathBuf {
    if let Some(runner) = override_value {
        return PathBuf::from(runner);
    }
    let local = workspace_root.join("cargo_full");
    if local.is_file() {
        return local;
    }
    PathBuf::from("cargo_full")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: three call sites resolved cargo three ways, so which binary built a
    /// child depended on which module spawned it: two read `CARGO`, which cargo
    /// sets to the plain binary and never to the wrapper, and one read the
    /// wrapper. The precedence is the contract, and the fallback is a name
    /// rather than a path so a checkout without the wrapper still resolves.
    #[test]
    fn the_override_wins_over_the_wrapper_and_the_wrapper_over_the_name() {
        let root = std::env::temp_dir().join(format!("vyre-runner-{}", std::process::id()));
        std::fs::create_dir_all(&root).expect("the fixture tree is created");
        let absent = resolve(None, &root);

        let wrapper = root.join("cargo_full");
        std::fs::write(&wrapper, "#!/bin/sh\n").expect("the wrapper is written");
        let present = resolve(None, &root);
        let overridden = resolve(Some(OsString::from("/opt/cargo-wrapper")), &root);
        std::fs::remove_dir_all(&root).ok();

        assert_eq!(absent, PathBuf::from("cargo_full"));
        assert_eq!(present, wrapper);
        assert_eq!(overridden, PathBuf::from("/opt/cargo-wrapper"));
    }
}
