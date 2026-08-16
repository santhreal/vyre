//! The cargo a gate runs.
//!
//! Cargo exports `CARGO` to anything it starts, and that value is the toolchain
//! the caller invoked, so a child build made through it matches the parent's
//! toolchain instead of whatever is first on `PATH`. Four gates each carried
//! their own copy of the lookup, and a copy is where a default diverges: one of
//! them reaching for a bare `cargo` inside a `+nightly` run compiles a different
//! workspace than the one being judged.
//!
//! Job count and target directory are declared in `.cargo/config.toml`, so
//! nothing here sets either.

use std::ffi::OsString;
use std::path::Path;
use std::process::Command;

/// The cargo binary this process should start, falling back to `PATH`.
#[must_use]
pub fn binary() -> OsString {
    std::env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
}

/// A cargo command rooted at `root`.
#[must_use]
pub fn command(root: &Path) -> Command {
    let mut command = Command::new(binary());
    command.current_dir(root);
    command
}
