//! Walk a source tree for `Cargo.toml` files and parse each one.
//!
//! Parse failures are collected as blockers rather than aborting the walk, so a
//! gate reports every bad manifest in one run.

use std::path::Path;

use walkdir::WalkDir;

pub(crate) fn collect_manifests<T>(
    root: &Path,
    surface: &str,
    items: &mut Vec<T>,
    blockers: &mut Vec<String>,
    mut parse: impl FnMut(&Path) -> Result<Option<T>, String>,
) {
    for entry in WalkDir::new(root).into_iter().filter_entry(|entry| {
        !matches!(
            entry.file_name().to_string_lossy().as_ref(),
            "target"
                | "target-codex"
                | "target_tests"
                | ".git"
                | ".cargo-target"
                | "release"
                | "examples"
        )
    }) {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                blockers.push(format!(
                    "failed to walk {surface} root `{}`: {error}",
                    error
                        .path()
                        .map(|path| path.display().to_string())
                        .unwrap_or_else(|| root.display().to_string())
                ));
                continue;
            }
        };
        let path = entry.path();
        if path.file_name().and_then(|name| name.to_str()) != Some("Cargo.toml") {
            continue;
        }
        match parse(path) {
            Ok(Some(item)) => items.push(item),
            Ok(None) => {}
            Err(error) => blockers.push(error),
        }
    }
}
