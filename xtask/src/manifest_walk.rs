//! Walk a source tree for `Cargo.toml` files and parse each one.
//!
//! Parse failures are collected as blockers rather than aborting the walk, so a
//! gate reports every bad manifest in one run.
//!
//! This module also owns the read bound, the workspace-inheritance lookup and
//! the read-parse-name sequence every package manifest goes through, because
//! four release generators each carried their own copy of all three.

use std::path::Path;

use crate::tree_walk::{self, BUILD_OUTPUT_AND_VCS};

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
    for entry in tree_walk::pruned_by(root, |name| {
        !BUILD_OUTPUT_AND_VCS.contains(&name) && !matches!(name, "release" | "examples")
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

/// Read a package manifest and hand its document and `[package]` table to
/// `build`.
///
/// `read_context` names the read cap in an over-size error and `noun` names the
/// manifest in a read or parse failure, because two generators read the same
/// file for different reasons and each says which reason it was serving.
///
/// `Ok(None)` for a manifest that declares no `[package]` table, which is what
/// the workspace root manifest is. A manifest that declares one without a name
/// is an error: an unnamed package cannot be reported against.
pub(crate) fn parse_package_manifest<T>(
    path: &Path,
    read_context: &str,
    noun: &str,
    build: impl FnOnce(&str, &toml::Value, &toml::value::Table) -> Result<Option<T>, String>,
) -> Result<Option<T>, String> {
    let text = crate::output_arg::read_text_bounded(path, MAX_MANIFEST_BYTES, read_context)
        .map_err(|error| format!("failed to read {noun} `{}`: {error}", path.display()))?;
    let document = toml::from_str::<toml::Value>(&text)
        .map_err(|error| format!("failed to parse {noun} `{}`: {error}", path.display()))?;
    let Some(package) = document.get("package").and_then(toml::Value::as_table) else {
        return Ok(None);
    };
    let Some(name) = package.get("name").and_then(toml::Value::as_str) else {
        return Err(format!(
            "package manifest `{}` is missing package.name",
            path.display()
        ));
    };
    build(name, &document, package)
}
