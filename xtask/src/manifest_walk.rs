//! Walk a source tree for `Cargo.toml` files and parse each one.
//!
//! Parse failures are collected as blockers rather than aborting the walk, so a
//! gate reports every bad manifest in one run.
//!
//! This module also owns the read bound and the workspace-inheritance lookup,
//! because four release generators each carried their own copy of both.

use std::path::Path;

use walkdir::WalkDir;

/// Largest `Cargo.toml` this tooling will read.
pub(crate) const MAX_MANIFEST_BYTES: u64 = 1_048_576;

/// The `[workspace.package]` table of the workspace root manifest.
///
/// Two generators read this identically, down to the blocker text, one to
/// resolve every inherited field and one to resolve `version` alone. Both now
/// read a manifest under the manifest bound rather than one of them reaching for
/// an unrelated eight-megabyte evidence-text bound.
pub(crate) fn workspace_package(
    root: &Path,
    surface: &str,
    blockers: &mut Vec<String>,
) -> Option<toml::value::Table> {
    let manifest = root.join("Cargo.toml");
    let text = match crate::output_arg::read_text_bounded(&manifest, MAX_MANIFEST_BYTES, surface) {
        Ok(text) => text,
        Err(error) => {
            blockers.push(format!(
                "failed to read workspace package manifest `{}`: {error}",
                manifest.display()
            ));
            return None;
        }
    };
    let value = match toml::from_str::<toml::Value>(&text) {
        Ok(value) => value,
        Err(error) => {
            blockers.push(format!(
                "failed to parse workspace package manifest `{}`: {error}",
                manifest.display()
            ));
            return None;
        }
    };
    value
        .get("workspace")
        .and_then(|workspace| workspace.get("package"))
        .and_then(toml::Value::as_table)
        .cloned()
}

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
