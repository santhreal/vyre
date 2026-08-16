//! Where each registered operation is defined, read from the checkout.
//!
//! An operation id names the crate that minted it and is frozen from then on.
//! Eighteen composition domains moved to `vyre-libs` keeping their
//! `vyre-primitives::` ids, so the prefix stopped being a placement fact for
//! 130 operations. Every placement answer here comes from the source tree: the
//! file that spells the id literal, the module directory that file sits in, and
//! the `cfg` that gates that module in its crate's `lib.rs`.
//!
//! Two things this reader must not do, both of which produced a wrong answer
//! before. It must not decide a module exists because a directory exists: git
//! tracks files, so a domain that moved away leaves its directory behind in
//! every checkout that pulled the deletion. It must not keep a table of domain
//! names; a table is a snapshot of one move and goes stale in silence on the
//! next one.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Largest source file this reader will open.
const MAX_SOURCE_BYTES: u64 = 4_194_304;

/// Where one operation is defined and what enables it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Placement {
    /// Workspace member that holds the definition.
    pub(super) crate_name: String,
    /// Features that have to be enabled for the registration to link.
    pub(super) features: Vec<String>,
}

/// Every operation placement read from one checkout.
pub(super) struct Placements {
    by_id: BTreeMap<String, Placement>,
}

impl Placements {
    /// Placement of one operation, or `None` when no source file spells its id.
    pub(super) fn get(&self, id: &str) -> Option<&Placement> {
        self.by_id.get(id)
    }
}

/// Read the placement of every id in `ids` from the checkout at `root`.
///
/// An id no source file spells is an error naming the id. Silently routing it
/// to a default crate is how a moved operation kept reporting the crate it left.
pub(super) fn read(root: &Path, ids: &BTreeSet<&str>, errors: &mut Vec<String>) -> Placements {
    let members = source_members(root);
    let mut sites: BTreeMap<&str, Vec<Site>> = BTreeMap::new();
    for member in &members {
        let gates = module_gates(&root.join(member).join("src/lib.rs"));
        for file in rust_sources(&root.join(member).join("src")) {
            let Ok(text) = read_capped(&file) else {
                continue;
            };
            let spelled = spelled_ids(&text, ids);
            if spelled.is_empty() {
                continue;
            }
            let relative = file
                .strip_prefix(root.join(member).join("src"))
                .unwrap_or(&file)
                .to_string_lossy()
                .replace('\\', "/");
            let module = relative.split('/').next().unwrap_or_default();
            let module = module.strip_suffix(".rs").unwrap_or(module).to_string();
            let submits = text.contains("inventory::submit!");
            let mut features = Vec::new();
            if let Some(feature) = gates.get(&module) {
                features.push(feature.clone());
            }
            if let Some(feature) = submission_gate(&text) {
                if !features.contains(&feature) {
                    features.push(feature);
                }
            }
            for id in spelled {
                sites.entry(id).or_default().push(Site {
                    crate_name: member.clone(),
                    features: features.clone(),
                    submits,
                });
            }
        }
    }

    let mut by_id = BTreeMap::new();
    for id in ids {
        let Some(found) = sites.get(*id) else {
            errors.push(format!(
                "operation `{id}` is spelled by no source file under any workspace member; an id with no definition site has no placement to record"
            ));
            continue;
        };
        let crates: BTreeSet<&str> = found.iter().map(|site| site.crate_name.as_str()).collect();
        if crates.len() > 1 {
            let named = crates.into_iter().collect::<Vec<_>>().join(", ");
            errors.push(format!(
                "operation `{id}` is spelled in {named}; one operation has one defining crate"
            ));
            continue;
        }
        // The registration site is the authority when several files in the crate
        // spell the id, because a caller spells it too.
        let site = found
            .iter()
            .find(|site| site.submits)
            .unwrap_or_else(|| &found[0]);
        by_id.insert(
            (*id).to_string(),
            Placement {
                crate_name: site.crate_name.clone(),
                features: site.features.clone(),
            },
        );
    }
    Placements { by_id }
}

/// One file that spells an operation id.
struct Site {
    crate_name: String,
    features: Vec<String>,
    submits: bool,
}

/// Workspace members that carry a `src` directory with Rust in it.
///
/// Read from the root manifest, not from a directory listing: a member deleted
/// upstream leaves its directory behind in every checkout that pulled the
/// deletion rather than cloning fresh.
fn source_members(root: &Path) -> Vec<String> {
    let Ok(text) = fs::read_to_string(root.join("Cargo.toml")) else {
        return Vec::new();
    };
    let Ok(manifest) = text.parse::<toml::Value>() else {
        return Vec::new();
    };
    manifest
        .get("workspace")
        .and_then(|workspace| workspace.get("members"))
        .and_then(toml::Value::as_array)
        .map(|members| {
            members
                .iter()
                .filter_map(toml::Value::as_str)
                .filter(|member| root.join(member).join("src/lib.rs").is_file())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Every `.rs` file under `dir`, at any depth.
fn rust_sources(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
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

/// Ids from `ids` that appear as a string literal in `text`.
fn spelled_ids<'a>(text: &str, ids: &BTreeSet<&'a str>) -> Vec<&'a str> {
    let mut found = Vec::new();
    for id in ids {
        let mut quoted = String::with_capacity(id.len() + 2);
        quoted.push('"');
        quoted.push_str(id);
        quoted.push('"');
        if text.contains(&quoted) {
            found.push(*id);
        }
    }
    found
}

/// `module directory -> feature` for every `pub mod` in one `lib.rs`.
///
/// Reads the declarations rather than the directory listing, so a module whose
/// files are gone is not a module however many empty directories survive.
fn module_gates(lib: &Path) -> BTreeMap<String, String> {
    let mut gates = BTreeMap::new();
    let Ok(text) = read_capped(lib) else {
        return gates;
    };
    let mut pending: Option<String> = None;
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(feature) = cfg_feature(trimmed) {
            pending = Some(feature);
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("pub mod ") {
            if let Some(name) = rest.strip_suffix(';') {
                if let Some(feature) = pending.take() {
                    gates.insert(name.to_string(), feature);
                }
            }
            continue;
        }
        if !trimmed.is_empty() && !trimmed.starts_with("///") && !trimmed.starts_with("#[") {
            pending = None;
        }
    }
    gates
}

/// Feature named by a single-feature `#[cfg(feature = "...")]` attribute.
fn cfg_feature(line: &str) -> Option<String> {
    let rest = line.strip_prefix("#[cfg(feature = \"")?;
    let feature = rest.strip_suffix("\")]")?;
    (!feature.is_empty()).then(|| feature.to_string())
}

/// Feature gating the `inventory::submit!` block in one source file.
fn submission_gate(text: &str) -> Option<String> {
    let mut previous: Option<String> = None;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("inventory::submit!") {
            return previous;
        }
        previous = cfg_feature(trimmed);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_gates_read_the_declaration_not_the_directory() {
        let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
        let lib = dir.path().join("lib.rs");
        fs::write(
            &lib,
            "/// Doc.\n#[cfg(feature = \"bitset\")]\npub mod bitset;\n\npub mod prelude;\n",
        )
        .expect("Fix: fixture must be writable");
        let gates = module_gates(&lib);
        assert_eq!(gates.get("bitset").map(String::as_str), Some("bitset"));
        assert_eq!(gates.get("prelude"), None);
    }

    /// A module the crate no longer declares carries no feature.
    ///
    /// This is the case an existence test gets wrong: deleting a domain leaves
    /// its directory in every checkout that pulled the deletion, so the only
    /// honest source is the declaration.
    #[test]
    fn a_module_that_is_no_longer_declared_has_no_gate() {
        let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
        let lib = dir.path().join("lib.rs");
        fs::write(&lib, "#[cfg(feature = \"reduce\")]\npub mod reduce;\n")
            .expect("Fix: fixture must be writable");
        fs::create_dir_all(dir.path().join("matching/ops"))
            .expect("Fix: fixture directory must be creatable");
        let gates = module_gates(&lib);
        assert_eq!(gates.get("matching"), None);
    }

    #[test]
    fn submission_gate_reads_the_attribute_above_the_block() {
        assert_eq!(
            submission_gate("#[cfg(feature = \"inventory-registry\")]\ninventory::submit! {\n}\n"),
            Some("inventory-registry".to_string())
        );
        assert_eq!(submission_gate("inventory::submit! {\n}\n"), None);
    }

    #[test]
    fn spelled_ids_match_whole_literals() {
        let ids = BTreeSet::from(["vyre-libs::math::matmul"]);
        assert_eq!(
            spelled_ids("const OP: &str = \"vyre-libs::math::matmul\";", &ids),
            vec!["vyre-libs::math::matmul"]
        );
        assert!(spelled_ids("vyre-libs::math::matmul", &ids).is_empty());
        assert!(spelled_ids("\"vyre-libs::math::matmul_bias\"", &ids).is_empty());
    }
}
