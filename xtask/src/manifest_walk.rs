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
pub const MAX_MANIFEST_BYTES: u64 = 1_048_576;

/// One parsed package manifest: the whole document, and the package name.
pub struct PackageManifest {
    /// The parsed manifest.
    pub document: toml::Table,
    /// The value of `package.name`.
    pub name: String,
}

/// Read and parse `path`, returning its document with the package name.
///
/// Three release generators carried this same bounded read, parse and
/// name-lookup with three different error sentences, so one malformed manifest
/// was reported three different ways depending on which gate reached it first.
///
/// `Ok(None)` means the manifest declares no `[package]`. A workspace root
/// manifest is not a defect, so the caller skips it rather than reporting it.
///
/// # Errors
///
/// One sentence naming the manifest when it cannot be read, cannot be parsed,
/// or declares a `[package]` without a `name`.
pub fn parse_package_manifest(
    path: &Path,
    surface: &str,
) -> Result<Option<PackageManifest>, String> {
    let text = crate::output_arg::read_text_bounded(path, MAX_MANIFEST_BYTES, surface)
        .map_err(|error| format!("failed to read manifest `{}`: {error}", path.display()))?;
    let document = toml::from_str::<toml::Table>(&text)
        .map_err(|error| format!("failed to parse manifest `{}`: {error}", path.display()))?;
    let Some(package) = document.get("package").and_then(toml::Value::as_table) else {
        return Ok(None);
    };
    let Some(name) = package.get("name").and_then(toml::Value::as_str) else {
        return Err(format!(
            "package manifest `{}` is missing package.name",
            path.display()
        ));
    };
    let name = name.to_string();
    Ok(Some(PackageManifest { document, name }))
}

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

