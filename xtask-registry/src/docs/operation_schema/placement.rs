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

use structure_gate::{parse_registrations, strip_cfg_test_items};

/// Largest source file this reader will open.
const MAX_SOURCE_BYTES: u64 = 4_194_304;

/// Where one operation is defined and what enables it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Placement {
    /// Workspace member that holds the definition.
    pub(super) crate_name: String,
    /// Features that reach the registration. Empty means the defining crate
    /// links it unconditionally.
    pub(super) features: Vec<String>,
}

/// Every operation placement read from one checkout.
pub(super) struct Placements {
    by_id: BTreeMap<String, Placement>,
}

impl Placements {
    /// Placement of one operation, or `None` when no source file defines it.
    pub(super) fn get(&self, id: &str) -> Option<&Placement> {
        self.by_id.get(id)
    }
}

/// Read the placement of every id in `ids` from the checkout at `root`.
///
/// An id no source file defines is an error naming the id. Silently routing it
/// to a default crate is how a moved operation kept reporting the crate it left.
pub(super) fn read(root: &Path, ids: &BTreeSet<&str>, errors: &mut Vec<String>) -> Placements {
    let mut registered: BTreeMap<&str, Vec<Site>> = BTreeMap::new();
    let mut mentioned: BTreeMap<&str, Vec<Site>> = BTreeMap::new();
    for member in source_members(root) {
        for module in compiled_modules(&root.join(&member).join("src")) {
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
            for literal in quoted_literals(&text) {
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

    // An operation is defined where it is registered, so a crate that registers
    // none defines none. Deriving that set from the registrations rather than
    // naming the two crates keeps the answer right through the next move.
    let registering: BTreeSet<&str> = registered
        .values()
        .flatten()
        .map(|site| site.crate_name.as_str())
        .collect();

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

/// Every module file `src/lib.rs` reaches, with the features on the way.
///
/// Walks the `mod` declarations rather than the directory listing, so a
/// directory whose declaration is gone is not a module and a file no
/// declaration names is not compiled. Modules reachable only under `cfg(test)`
/// are left out: a fixture registration is not a definition.
fn compiled_modules(source: &Path) -> Vec<Module> {
    let mut found = Vec::new();
    let mut pending = vec![Module {
        path: source.join("lib.rs"),
        features: Vec::new(),
    }];
    while let Some(module) = pending.pop() {
        let Ok(text) = read_capped(&module.path) else {
            continue;
        };
        let directory = module_directory(&module.path);
        for (name, attributes) in module_declarations(&text) {
            let Some(gates) = reachable_features(&attributes) else {
                continue;
            };
            let mut features = module.features.clone();
            for gate in gates {
                if !features.contains(&gate) {
                    features.push(gate);
                }
            }
            let file = directory.join(format!("{name}.rs"));
            let path = if file.is_file() {
                file
            } else {
                directory.join(&name).join("mod.rs")
            };
            if path.is_file() {
                pending.push(Module { path, features });
            }
        }
        found.push(module);
    }
    found
}

/// Directory the modules a file declares live in.
fn module_directory(file: &Path) -> PathBuf {
    let parent = file.parent().unwrap_or(Path::new("")).to_path_buf();
    match file.file_name().and_then(|name| name.to_str()) {
        Some("lib.rs" | "mod.rs") | None => parent,
        Some(name) => parent.join(name.trim_end_matches(".rs")),
    }
}

/// `(module name, attributes above it)` for every out-of-line `mod` in a file.
fn module_declarations(text: &str) -> Vec<(String, Vec<String>)> {
    let lines: Vec<&str> = text.lines().collect();
    let attributes = attribute_blocks(&lines);
    let mut declarations = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let Some(name) = module_name(line) else {
            continue;
        };
        declarations.push((name, gating_attributes(&lines, &attributes, index)));
    }
    declarations
}

/// Module name declared by an out-of-line `mod` statement on one line.
fn module_name(line: &str) -> Option<String> {
    let rest = line.trim();
    let rest = rest.strip_prefix("pub ").unwrap_or(rest).trim_start();
    let rest = match rest.strip_prefix("pub(") {
        Some(tail) => tail.split_once(')')?.1.trim_start(),
        None => rest,
    };
    let name = rest.strip_prefix("mod ")?.strip_suffix(';')?.trim();
    (!name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_'))
        .then(|| name.to_string())
}

/// `(last line, (first line, joined text))` for every attribute in a file.
///
/// An attribute is joined across lines because the tree writes
/// `#[cfg(any(\n    feature = "a",\n    feature = "b"\n))]`, and reading only
/// the last line of that spelling records no feature at all.
fn attribute_blocks(lines: &[&str]) -> BTreeMap<usize, (usize, String)> {
    let mut blocks = BTreeMap::new();
    let mut index = 0;
    while index < lines.len() {
        let trimmed = lines[index].trim();
        if !trimmed.starts_with("#[") {
            index += 1;
            continue;
        }
        let mut joined = trimmed.to_string();
        let mut last = index;
        while joined.matches('(').count() > joined.matches(')').count() && last + 1 < lines.len() {
            last += 1;
            joined.push(' ');
            joined.push_str(lines[last].trim());
        }
        blocks.insert(last, (index, joined));
        index = last + 1;
    }
    blocks
}

/// Attribute texts that gate the item on line `at`.
fn gating_attributes(
    lines: &[&str],
    blocks: &BTreeMap<usize, (usize, String)>,
    at: usize,
) -> Vec<String> {
    let mut found = Vec::new();
    let mut index = at;
    while index > 0 {
        index -= 1;
        let trimmed = lines[index].trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        let Some((first, text)) = blocks.get(&index) else {
            break;
        };
        if text.starts_with("#[cfg") {
            found.push(text.clone());
        }
        index = *first;
    }
    found
}

/// Features that reach an item, or `None` when only a test build reaches it.
///
/// A `cfg` naming `test` beside a feature, such as
/// `any(test, feature = "cpu-parity")`, still compiles in a feature build, so
/// only a `cfg` that names `test` and no feature at all is test-only.
fn reachable_features(attributes: &[String]) -> Option<Vec<String>> {
    let mut features = Vec::new();
    for attribute in attributes {
        let named = cfg_features(attribute);
        if named.is_empty() && names_test(attribute) {
            return None;
        }
        for feature in named {
            if !features.contains(&feature) {
                features.push(feature);
            }
        }
    }
    Some(features)
}

/// Every feature named by one `cfg` attribute.
fn cfg_features(attribute: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut rest = attribute;
    while let Some(start) = rest.find("feature = \"") {
        rest = &rest[start + "feature = \"".len()..];
        let Some(end) = rest.find('"') else {
            break;
        };
        let feature = rest[..end].to_string();
        if !feature.is_empty() && !found.contains(&feature) {
            found.push(feature);
        }
        rest = &rest[end + 1..];
    }
    found
}

/// Whether one `cfg` attribute names the `test` predicate.
fn names_test(attribute: &str) -> bool {
    let bytes = attribute.as_bytes();
    let mut at = 0;
    while let Some(found) = attribute[at..].find("test") {
        let start = at + found;
        let end = start + "test".len();
        let before = start
            .checked_sub(1)
            .is_none_or(|index| !is_word_byte(bytes[index]));
        let after = end >= bytes.len() || !is_word_byte(bytes[end]);
        if before && after {
            return true;
        }
        at = end;
    }
    false
}

fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Features gating an `inventory::submit!` block in one file.
///
/// `vyre-primitives` gates its submissions on `inventory-registry` rather than
/// on the module, so the file is the only place that route is written.
fn submission_features(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    let blocks = attribute_blocks(&lines);
    let mut features = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        if !line.trim_start().starts_with("inventory::submit!") {
            continue;
        }
        for attribute in gating_attributes(&lines, &blocks, index) {
            for feature in cfg_features(&attribute) {
                if !features.contains(&feature) {
                    features.push(feature);
                }
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

/// Contents of every double-quoted literal in one file.
///
/// One pass over the text, rather than one substring search per candidate id:
/// the checkout carries 327 ids and some 2000 compiled modules.
fn quoted_literals(text: &str) -> BTreeSet<&str> {
    let mut found = BTreeSet::new();
    let bytes = text.as_bytes();
    let mut at = 0;
    while let Some(open) = text[at..].find('"') {
        let start = at + open + 1;
        let mut end = start;
        while end < bytes.len() && bytes[end] != b'"' {
            end += if bytes[end] == b'\\' { 2 } else { 1 };
        }
        if end >= bytes.len() {
            break;
        }
        if let Some(literal) = text.get(start..end) {
            found.insert(literal);
        }
        at = end + 1;
    }
    found
}

/// These readers are crate-private: `read` is called by `assemble`, and no
/// integration test can reach `compiled_modules`, `module_name`,
/// `reachable_features` or `quoted_literals` to pin the cases that produced
/// wrong placements.
#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, text: &str) {
        fs::create_dir_all(path.parent().expect("Fix: fixture path must have a parent"))
            .expect("Fix: fixture directory must be creatable");
        fs::write(path, text).expect("Fix: fixture must be writable");
    }

    fn workspace(root: &Path, members: &[&str]) {
        let named = members
            .iter()
            .map(|member| format!("\"{member}\""))
            .collect::<Vec<_>>()
            .join(", ");
        write(
            &root.join("Cargo.toml"),
            &format!("[workspace]\nmembers = [{named}]\n"),
        );
    }

    #[test]
    fn a_registration_places_the_operation_in_its_crate_behind_its_module_features() {
        let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
        let root = dir.path();
        workspace(root, &["libs"]);
        write(
            &root.join("libs/src/lib.rs"),
            "#[cfg(any(\n    feature = \"math-dialect\",\n    feature = \"math-kernels\"\n))]\npub mod math;\n",
        );
        write(&root.join("libs/src/math/mod.rs"), "pub mod square;\n");
        write(
            &root.join("libs/src/math/square.rs"),
            "const OP_ID: &str = \"libs::math::square\";\ninventory::submit! {\n    OperationRegistration::library(OP_ID, builder)\n}\n",
        );

        let ids = BTreeSet::from(["libs::math::square"]);
        let mut errors = Vec::new();
        let placements = read(root, &ids, &mut errors);

        assert_eq!(errors, Vec::<String>::new());
        assert_eq!(
            placements.get("libs::math::square"),
            Some(&Placement {
                crate_name: "libs".to_string(),
                features: vec!["math-dialect".to_string(), "math-kernels".to_string()],
            })
        );
    }

    /// A gate that lists ids is not a definition site.
    ///
    /// Reading every spelling as a definition reported 132 operations as
    /// defined in two crates at once and failed the whole schema document.
    #[test]
    fn a_crate_that_only_names_the_id_is_not_the_defining_crate() {
        let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
        let root = dir.path();
        workspace(root, &["libs", "tooling"]);
        write(&root.join("libs/src/lib.rs"), "pub mod square;\n");
        write(
            &root.join("libs/src/square.rs"),
            "inventory::submit! {\n    OperationRegistration::library(\"libs::square\", builder)\n}\n",
        );
        write(&root.join("tooling/src/lib.rs"), "pub mod audit;\n");
        write(
            &root.join("tooling/src/audit.rs"),
            "const WAIVED: [&str; 1] = [\"libs::square\"];\n",
        );

        let ids = BTreeSet::from(["libs::square"]);
        let mut errors = Vec::new();
        let placements = read(root, &ids, &mut errors);

        assert_eq!(errors, Vec::<String>::new());
        assert_eq!(
            placements.get("libs::square").map(|found| &found.crate_name),
            Some(&"libs".to_string())
        );
    }

    /// A macro that registers takes the id at the invocation, so the file that
    /// spells it is the definition even though it names no registration.
    #[test]
    fn an_id_passed_to_a_registering_macro_places_in_the_invoking_module() {
        let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
        let root = dir.path();
        workspace(root, &["libs"]);
        write(
            &root.join("libs/src/lib.rs"),
            "#[cfg(feature = \"bitset\")]\npub mod bitset;\n",
        );
        write(
            &root.join("libs/src/bitset/mod.rs"),
            "pub mod word;\n\ndefine_op! {\n    op_id: \"libs::bitset::or_into\",\n}\n",
        );
        write(
            &root.join("libs/src/bitset/word.rs"),
            "inventory::submit! {\n    OperationRegistration::library(\"libs::bitset::and\", builder)\n}\n",
        );

        let ids = BTreeSet::from(["libs::bitset::or_into"]);
        let mut errors = Vec::new();
        let placements = read(root, &ids, &mut errors);

        assert_eq!(errors, Vec::<String>::new());
        assert_eq!(
            placements.get("libs::bitset::or_into"),
            Some(&Placement {
                crate_name: "libs".to_string(),
                features: vec!["bitset".to_string()],
            })
        );
    }

    /// A registration reachable only under `cfg(test)` is a fixture.
    #[test]
    fn a_test_only_module_is_not_a_definition_site() {
        let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
        let root = dir.path();
        workspace(root, &["libs"]);
        write(
            &root.join("libs/src/lib.rs"),
            "pub mod real;\n#[cfg(test)]\nmod fixtures;\n",
        );
        write(
            &root.join("libs/src/real.rs"),
            "inventory::submit! {\n    OperationRegistration::library(\"libs::real\", builder)\n}\n",
        );
        write(
            &root.join("libs/src/fixtures.rs"),
            "inventory::submit! {\n    OperationRegistration::library(\"libs::ghost\", builder)\n}\n",
        );

        let ids = BTreeSet::from(["libs::real", "libs::ghost"]);
        let mut errors = Vec::new();
        let placements = read(root, &ids, &mut errors);

        assert!(placements.get("libs::real").is_some());
        assert_eq!(placements.get("libs::ghost"), None);
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(
            errors[0].contains("operation `libs::ghost` has no definition site"),
            "{errors:?}"
        );
    }

    /// A domain that moved leaves its directory behind in every checkout that
    /// pulled the deletion, so only the declaration answers.
    #[test]
    fn a_directory_no_declaration_names_carries_no_module() {
        let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
        let root = dir.path();
        workspace(root, &["libs"]);
        write(
            &root.join("libs/src/lib.rs"),
            "#[cfg(feature = \"reduce\")]\npub mod reduce;\n",
        );
        write(&root.join("libs/src/reduce.rs"), "pub fn sum() {}\n");
        write(
            &root.join("libs/src/matching/mod.rs"),
            "inventory::submit! {\n    OperationRegistration::library(\"libs::matching::dfa\", builder)\n}\n",
        );

        let ids = BTreeSet::from(["libs::matching::dfa"]);
        let mut errors = Vec::new();
        let placements = read(root, &ids, &mut errors);

        assert_eq!(placements.get("libs::matching::dfa"), None);
        assert_eq!(errors.len(), 1, "{errors:?}");
    }

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

    #[test]
    fn a_test_predicate_beside_a_feature_still_compiles() {
        assert_eq!(
            reachable_features(&["#[cfg(any(test, feature = \"cpu-parity\"))]".to_string()]),
            Some(vec!["cpu-parity".to_string()])
        );
        assert_eq!(
            reachable_features(&["#[cfg(test)]".to_string()]),
            None::<Vec<String>>
        );
        assert_eq!(
            reachable_features(&["#[cfg(feature = \"latest\")]".to_string()]),
            Some(vec!["latest".to_string()]),
            "a feature whose name contains `test` is not the test predicate"
        );
    }

    #[test]
    fn a_module_declaration_is_read_through_its_visibility() {
        assert_eq!(module_name("pub(crate) mod builder;"), Some("builder".to_string()));
        assert_eq!(module_name("    mod inner;"), Some("inner".to_string()));
        assert_eq!(module_name("pub mod math;"), Some("math".to_string()));
        assert_eq!(module_name("mod tests {"), None);
        assert_eq!(module_name("// mod commented;"), None);
    }

    #[test]
    fn quoted_literals_read_whole_strings() {
        let found = quoted_literals("const A: &str = \"libs::math::matmul\";\nlet b = \"x\\\"y\";\n");
        assert!(found.contains("libs::math::matmul"));
        assert!(!found.contains("libs::math::matmul_bias"));
    }
}
