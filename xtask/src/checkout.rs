//! The checkout these tools compile from and report on.
//!
//! Every gate here reads the working tree and prints a number about it, so the
//! tree it names has to be the tree it ran in. Deriving that root from
//! `CARGO_MANIFEST_DIR` at each call site gave the same answer in one checkout
//! and the wrong answer across two: cargo hashes a workspace member by its path
//! relative to the workspace root, checks freshness by mtime, and a target
//! directory shared by several checkouts therefore handed one checkout a binary
//! compiled by another. `dup-scan` read 11811 duplicate lines for `vyre-libs`
//! when its own tree held 13406.
//!
//! `VYRE_CHECKOUT_ROOT` comes from `.cargo/config.toml` with `relative = true`,
//! so its value is this checkout's absolute path. Reading it with `env!`
//! records that value in this crate's dep-info, which makes the artifact
//! unshareable between checkouts, and `xtask-registry` and `xtask-evidence`
//! inherit the same protection through their dependency on this crate.

use std::path::PathBuf;

/// Absolute root of the checkout that compiled this binary.
#[must_use]
pub fn compiled_checkout_root() -> PathBuf {
    PathBuf::from(env!(
        "VYRE_CHECKOUT_ROOT",
        "Fix: run cargo from inside the vyre checkout so its .cargo/config.toml applies."
    ))
}

/// Absolute root of the checkout this tool was invoked in.
///
/// Read from the environment at run time, with the compiled-in value as the
/// fallback for a binary invoked outside cargo. Both name the same checkout,
/// because the compiled-in value is a fingerprint input.
#[must_use]
pub fn checkout_root() -> PathBuf {
    std::env::var_os("VYRE_CHECKOUT_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(compiled_checkout_root)
}
