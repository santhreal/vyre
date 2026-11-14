//! Invokes the workspace cargo wrapper.

use std::path::Path;
use std::process::Command;

/// Command configured to invoke `./cargo_full` in `root` (or `cargo` if absent).
#[must_use]
pub fn command(root: &Path) -> Command {
    let wrapper = root.join("cargo_full");
    let mut cmd = if wrapper.is_file() {
        Command::new(wrapper)
    } else {
        Command::new("cargo")
    };
    cmd.current_dir(root);
    cmd
}
