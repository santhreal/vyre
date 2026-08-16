//! Where each registered operation is defined, read from the checkout.
//!
//! An operation id names the crate that minted it and is frozen from then on.
//! Eighteen composition domains moved to `vyre-libs` keeping their
//! `vyre-primitives::` ids, so the prefix stopped being a placement fact for
//! 130 operations. Every placement answer here comes from the source tree: the
//! module tree walked down from each crate root, the file that registers the
//! id, and the `cfg` attributes on the declarations that reach that file.
//!
//! Three things this reader must not do, each of which produced a wrong answer
//! before. It must not decide a module exists because a directory exists: git
//! tracks files, so a domain that moved away leaves its directory behind in
//! every checkout that pulled the deletion. It must not keep a table of domain
//! names; a table is a snapshot of one move and goes stale in silence on the
//! next one. It must not read a bare mention of an id as a definition: a gate
//! that lists ids, a doc comment that quotes one, and a spec fixture all spell
//! the literal, and reading those as definition sites reported 132 operations
//! as defined in two crates at once and failed the whole document.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use structure_gate::source_scan::{gating_features, module_routes, string_literals};
use structure_gate::{parse_registrations, strip_cfg_test_items};

/// Largest source file this reader will open.
const MAX_SOURCE_BYTES: u64 = 4_194_304;

/// Where one operation is defined and what enables it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    /// Workspace member that holds the definition.
    pub crate_name: String,
    /// Features that reach the registration. Empty means the defining crate
    /// links it unconditionally.
    pub features: Vec<String>,
}

/// Every operation placement read from one checkout.
pub struct Placements {
    by_id: BTreeMap<String, Placement>,
    directories: BTreeMap<String, String>,
}

impl Placements {
    /// Placement of one operation, or `None` when no source file defines it.
    pub fn get(&self, id: &str) -> Option<&Placement> {
        self.by_id.get(id)
    }

    /// Directory one package is built from, relative to the workspace root.
    ///
    /// A package name is not a path: `vyre-conform` lives in `conform`. The
    /// reader that already parsed each member manifest is the one place that
    /// knows both, so it answers rather than leaving a caller to guess.
    pub fn directory(&self, package: &str) -> Option<&str> {
        self.directories.get(package).map(String::as_str)
    }
}

/// Read the placement of every id in `ids` from the checkout at `root`.
///
/// An id no source file defines is an error naming the id. Silently routing it
/// to a default crate is how a moved operation kept reporting the crate it left.
pub fn read(root: &Path, ids: &BTreeSet<&str>, errors: &mut Vec<String>) -> Placements {
    let mut registered: BTreeMap<&str, Vec<Site>> = BTreeMap::new();
    let mut mentioned: BTreeMap<&str, Vec<Site>> = BTreeMap::new();
    // An operation is defined where it is registered, so a crate that registers
    // none defines none. Every registration counts, not only one whose id was
    // asked about: a macro takes the id at its invocation, so the file that
    // spells it names no registration of its own, and reading the set from the
    // matched ids alone left that crate registering nothing.
    let mut registering: BTreeSet<String> = BTreeSet::new();
    let mut directories: BTreeMap<String, String> = BTreeMap::new();
    for member in members(root, errors) {
        directories.insert(member.name.clone(), member.path.clone());
        for module in module_routes(&root.join(&member.path).join("src")) {
            let raw = match read_capped(&module.path) {
                Ok(raw) => raw,
                Err(reason) => {
                    errors.push(reason);
                    continue;
                }
            };
            let text = strip_cfg_test_items(&raw);
            let mut features = module.features;
            for feature in submission_features(&text) {
                if !features.contains(&feature) {
                    features.push(feature);
                }
            }
            let site = Site {
                crate_name: member.name.clone(),
                path: module
                    .path
                    .strip_prefix(root)
                    .unwrap_or(&module.path)
                    .to_string_lossy()
                    .replace('\\', "/"),
                features,
            };
            let declared: BTreeSet<String> = if text.contains("OperationRegistration") {
                parse_registrations(&text)
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect()
            } else {
                BTreeSet::new()
            };
            if !declared.is_empty() {
                registering.insert(member.name.clone());
            }
            for literal in string_literals(&text).into_iter().collect::<BTreeSet<&str>>() {
                let Some(id) = ids.get(literal) else {
                    continue;
                };
                if declared.contains(literal) {
                    registered.entry(*id).or_default().push(site.clone());
                } else {
                    mentioned.entry(*id).or_default().push(site.clone());
                }
            }
        }
    }

    let mut by_id = BTreeMap::new();
    for id in ids {
        let found: Vec<&Site> = match registered.get(*id) {
            Some(sites) => sites.iter().collect(),
            None => mentioned
                .get(*id)
                .map(|sites| {
                    sites
                        .iter()
                        .filter(|site| registering.contains(site.crate_name.as_str()))
                        .collect()
                })
                .unwrap_or_default(),
        };
        if found.is_empty() {
            errors.push(format!(
                "operation `{id}` has no definition site: no module compiled into a member that registers operations declares it. Fix: register it in the crate whose builder produces it, or drop the registration"
            ));
            continue;
        }
        let crates: BTreeSet<&str> = found.iter().map(|site| site.crate_name.as_str()).collect();
        if crates.len() > 1 {
            let named = crates.into_iter().collect::<Vec<_>>().join(", ");
            errors.push(format!(
                "operation `{id}` is defined in {named}; one operation has one defining crate"
            ));
            continue;
        }
        let routes: BTreeSet<&[String]> = found.iter().map(|site| site.features.as_slice()).collect();
        if routes.len() > 1 {
            let named = found
                .iter()
                .map(|site| format!("{} behind [{}]", site.path, site.features.join(", ")))
                .collect::<Vec<_>>()
                .join("; ");
            errors.push(format!(
                "operation `{id}` is reached through disagreeing feature routes: {named}"
            ));
            continue;
        }
        by_id.insert(
            (*id).to_string(),
            Placement {
                crate_name: found[0].crate_name.clone(),
                features: found[0].features.clone(),
            },
        );
    }
    Placements { by_id, directories }
}

/// One compiled module file that names an operation id.
#[derive(Clone)]
struct Site {
    crate_name: String,
    path: String,
    features: Vec<String>,
}

/// One workspace member that carries a crate root.
struct Member {
    /// Directory of the member, relative to the workspace root.
    path: String,
    /// Declared `package.name`. A diagnostic names the package, because that is
    /// what a reader passes to `cargo` and what the registry records; the
    /// directory is a layout detail and `conform/vyre-conform` is not a crate.
    name: String,
}

/// Workspace members that carry a crate root.
///
/// Read from the root manifest, not from a directory listing: a member deleted
/// upstream leaves its directory behind in every checkout that pulled the
/// deletion rather than cloning fresh.
///
/// Every failure here is an error rather than an empty roster. An empty roster
/// answered `no definition site` for all 327 registered operations, which reads
/// as a broken registry instead of one unreadable file, and it was reached by a
/// parse that could not succeed: `toml::Value` parses a single TOML value, so a
/// document opening with `[workspace]` was read as an array and rejected as
/// trailing content.
fn members(root: &Path, errors: &mut Vec<String>) -> Vec<Member> {
    let manifest_path = root.join("Cargo.toml");
    let text = match fs::read_to_string(&manifest_path) {
        Ok(text) => text,
        Err(error) => {
            errors.push(format!(
                "cannot read `{}`: {error}. Fix: run this from a checkout that carries the workspace manifest",
                manifest_path.display()
            ));
            return Vec::new();
        }
    };
    let manifest = match toml::from_str::<toml::Table>(&text) {
        Ok(manifest) => manifest,
        Err(error) => {
            errors.push(format!(
                "cannot parse `{}`: {error}. Fix: repair the workspace manifest so it parses",
                manifest_path.display()
            ));
            return Vec::new();
        }
    };
    let declared = manifest
        .get("workspace")
        .and_then(|workspace| workspace.get("members"))
        .and_then(toml::Value::as_array);
    let Some(declared) = declared else {
        errors.push(format!(
            "`{}` declares no `[workspace] members` array. Fix: declare the workspace members that carry operation registrations",
            manifest_path.display()
        ));
        return Vec::new();
    };
    let mut carrying = Vec::new();
    for path in declared.iter().filter_map(toml::Value::as_str) {
        if !root.join(path).join("src/lib.rs").is_file() {
            continue;
        }
        match package_name(root, path) {
            Ok(name) => carrying.push(Member {
                path: path.to_string(),
                name,
            }),
            Err(reason) => errors.push(reason),
        }
    }
    if carrying.is_empty() {
        errors.push(format!(
            "`{}` names {} member(s) and none carries a readable `src/lib.rs` and `package.name`. Fix: declare the members that hold the crate roots",
            manifest_path.display(),
            declared.len()
        ));
    }
    carrying
}

/// The `package.name` one member declares.
fn package_name(root: &Path, member: &str) -> Result<String, String> {
    let manifest_path = root.join(member).join("Cargo.toml");
    let text = fs::read_to_string(&manifest_path).map_err(|error| {
        format!(
            "cannot read `{}`: {error}. Fix: give the member a manifest or drop it from the workspace members",
            manifest_path.display()
        )
    })?;
    let manifest = toml::from_str::<toml::Table>(&text).map_err(|error| {
        format!(
            "cannot parse `{}`: {error}. Fix: repair the member manifest so it parses",
            manifest_path.display()
        )
    })?;
    manifest
        .get("package")
        .and_then(|package| package.get("name"))
        .and_then(toml::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| {
            format!(
                "`{}` declares no `package.name`. Fix: name the package the member builds",
                manifest_path.display()
            )
        })
}

/// Features gating an `inventory::submit!` block in one file.
///
/// `vyre-primitives` gates its submissions on `inventory-registry` rather than
/// on the module, so the file is the only place that route is written.
fn submission_features(text: &str) -> Vec<String> {
    let mut features = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if !line.trim_start().starts_with("inventory::submit!") {
            continue;
        }
        for feature in gating_features(text, index).unwrap_or_default() {
            if !features.contains(&feature) {
                features.push(feature);
            }
        }
    }
    features
}

/// Read one source file, or say why it was not read.
///
/// A file over the cap is skipped, and a skipped file registers nothing, so a
/// silent skip answered `no definition site` for every operation it defined.
/// The cap stays; the skip is named.
fn read_capped(path: &Path) -> Result<String, String> {
    let meta = fs::metadata(path).map_err(|error| {
        format!(
            "cannot read `{}`: {error}. Fix: make the module file readable, or drop the `mod` declaration that names it",
            path.display()
        )
    })?;
    if meta.len() > MAX_SOURCE_BYTES {
        return Err(format!(
            "`{}` is {} bytes, over the {MAX_SOURCE_BYTES} byte read cap, so the registrations it holds are unread. Fix: split the file",
            path.display(),
            meta.len()
        ));
    }
    fs::read_to_string(path).map_err(|error| {
        format!(
            "cannot read `{}`: {error}. Fix: make the module file readable",
            path.display()
        )
    })
}

/// These readers are crate-private, and no integration test can reach
/// `submission_features` to pin the case that produced wrong placements. What
/// `read` answers over a fixture checkout is asserted in
/// `tests/registry_contracts/operation_schema_placement.rs`.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_submission_attribute_adds_its_feature_to_the_module_route() {
        assert_eq!(
            submission_features(
                "#[cfg(feature = \"inventory-registry\")]\ninventory::submit! {\n}\n"
            ),
            vec!["inventory-registry".to_string()]
        );
        assert_eq!(
            submission_features("inventory::submit! {\n}\n"),
            Vec::<String>::new()
        );
    }
}
