//! One owner for the question every lint asks first: which files do I see, and
//! what is each one called?
//!
//! Five lints each carried their own `walkdir` loop with the same extension
//! filter and the same workspace-relative naming call. One of them
//! (`raw_ir_in_libs`) also carried a second, disagreeing implementation of the
//! naming rule that only recognised paths under `vyre-libs/`, so the same file
//! was reported under two different names depending on which lint found it and
//! an allowlist entry written against one spelling did not match the other.

use crate::paths::workspace_relative;
use crate::Violation;
use anyhow::Result;
use std::path::Path;

/// Extensions a lint that reads Rust source accepts.
pub(crate) const RUST_SOURCE: &[&str] = &["rs"];

/// Extensions a lint that also reads prose accepts.
pub(crate) const RUST_AND_MARKDOWN: &[&str] = &["rs", "md"];

/// Call `visit` for every file under `root` whose extension is in
/// `extensions`, with its path and its workspace-relative name.
///
/// Unreadable directory entries are skipped rather than failing the scan: a
/// lint reports on the tree it can read, and a permission error on one entry
/// must not hide violations in the rest.
pub(crate) fn visit_sources(
    root: &Path,
    extensions: &[&str],
    mut visit: impl FnMut(&Path, &str) -> Result<()>,
) -> Result<()> {
    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|entry| entry.ok())
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let matches = path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| extensions.contains(&ext));
        if !matches {
            continue;
        }
        let workspace_rel = workspace_relative(path);
        visit(path, &workspace_rel)?;
    }
    Ok(())
}

/// Collect violations from every accepted file under `root`.
///
/// `accept` decides from the workspace-relative name alone whether a file is
/// in scope, so an exempt file is never read.
pub(crate) fn collect_violations(
    root: &Path,
    extensions: &[&str],
    accept: impl Fn(&str) -> bool,
    scan_file: impl Fn(&Path, &str) -> Result<Vec<Violation>>,
) -> Result<Vec<Violation>> {
    let mut all = Vec::new();
    visit_sources(root, extensions, |path, workspace_rel| {
        if accept(workspace_rel) {
            all.extend(scan_file(path, workspace_rel)?);
        }
        Ok(())
    })?;
    Ok(all)
}

/// Every file is in scope.
pub(crate) fn accept_all(_workspace_rel: &str) -> bool {
    true
}
