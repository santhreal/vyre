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
use std::path::{Path, PathBuf};

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
}

impl Placements {
    /// Placement of one operation, or `None` when no source file defines it.
    pub fn get(&self, id: &str) -> Option<&Placement> {
        self.by_id.get(id)
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
    for member in source_members(root, errors) {
        for module in module_routes(&root.join(&member).join("src")) {
            let Ok(raw) = read_capped(&module.path) else {
                continue;
            };
            let text = strip_cfg_test_items(&raw);
            let mut features = module.features;
            for feature in submission_features(&text) {
                if !features.contains(&feature) {
                    features.push(feature);
                }
            }
            let site = Site {
                crate_name: member.clone(),
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
                registering.insert(member.clone());
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
    Placements { by_id }
}

/// One compiled module file that names an operation id.
#[derive(Clone)]
struct Site {
    crate_name: String,
    path: String,
    features: Vec<String>,
}

/// One module file the crate root reaches, with the features that reach it.
struct Module {
    path: PathBuf,
    features: Vec<String>,
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
fn source_members(root: &Path, errors: &mut Vec<String>) -> Vec<String> {
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
    let members = manifest
        .get("workspace")
        .and_then(|workspace| workspace.get("members"))
        .and_then(toml::Value::as_array);
    let Some(members) = members else {
        errors.push(format!(
            "`{}` declares no `[workspace] members` array. Fix: declare the workspace members that carry operation registrations",
            manifest_path.display()
        ));
        return Vec::new();
    };
    let carrying: Vec<String> = members
        .iter()
        .filter_map(toml::Value::as_str)
        .filter(|member| root.join(member).join("src/lib.rs").is_file())
        .map(str::to_string)
        .collect();
    if carrying.is_empty() {
        errors.push(format!(
            "`{}` names {} member(s) and none carries a `src/lib.rs`. Fix: declare the members that hold the crate roots",
            manifest_path.display(),
            members.len()
        ));
    }
    carrying
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

fn read_capped(path: &Path) -> Result<String, ()> {
    let Ok(meta) = fs::metadata(path) else {
        return Err(());
    };
    if meta.len() > MAX_SOURCE_BYTES {
        return Err(());
    }
    fs::read_to_string(path).map_err(|_| ())
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
