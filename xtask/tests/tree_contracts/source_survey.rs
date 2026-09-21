//! Rust-source enumeration and violation formatting for the tree contracts.
//!
//! These three helpers are used by this harness and by no other target in the
//! crate, so they are declared here rather than in `tests/workspace_sources`.
//! A helper that sits in the shared module but is reached from one target is
//! dead code in every other target that compiles the module, and the crate-wide
//! allowance that hides it also hides a helper that has genuinely lost its last
//! caller. Every item in the shared module is reached from every target that
//! includes it, and every item here is reached from this one.

use std::path::{Path, PathBuf};

use proc_macro2::LineColumn;

use super::workspace_sources::{sources_under, workspace_member_src_dirs};

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
