//! Every registration an operation-owning crate submits is visible to this
//! walker.
//!
//! WHY: this crate's gates generate `docs/generated/op-inventory.toml`,
//! `docs/generated/OP_SCHEMA.json`, `docs/generated/catalog.toml` and the backend matrix
//! check from the live registry. A registration behind a Cargo feature this
//! crate's dependency edge does not enable is not absent, it is invisible: the
//! walker reports a smaller registry, every generated document agrees with that
//! smaller registry, and every rule that compares two of those documents still
//! passes. The only thing that goes red is a matrix row somebody wrote while the
//! edge was wide, which is how this was found: three rows had "no live
//! registration" because `geom`, `opt`, `decode` and `visual` had been dropped
//! from the feature list here long after the documents were generated with them
//! on.
//!
//! Nothing below is a list. The submitting files come from the tree, the feature
//! that gates each one comes from the module declarations above it, and the
//! enabled set comes from this crate's own manifest resolved through the source
//! crate's feature table. A new domain feature that gates a registration turns
//! this red until the edge names it.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    structure_gate::workspace_root()
}

fn read_manifest(path: &Path) -> toml::Value {
    let text = fs::read_to_string(path).unwrap_or_else(|error| {
        panic!("Fix: manifest {} must be readable: {error}", path.display())
    });
    toml::from_str(&text).unwrap_or_else(|error| {
        panic!(
            "Fix: manifest {} must be valid TOML: {error}",
            path.display()
        )
    })
}

/// Feature selections this crate declares on its dependencies, keyed by crate
/// name. A dependency that names no features contributes an empty selection,
/// which is still an answer: it means the default set only.
fn declared_selections() -> BTreeMap<String, (BTreeSet<String>, bool)> {
    let manifest = read_manifest(&workspace_root().join("xtask-registry/Cargo.toml"));
    let table = manifest
        .get("dependencies")
        .and_then(toml::Value::as_table)
        .expect("Fix: xtask-registry must declare [dependencies]");
    table
        .iter()
        .map(|(name, spec)| {
            let features: BTreeSet<String> = xtask::toml_text::string_array(spec.get("features"))
                .into_iter()
                .collect();
            let default_features = spec
                .get("default-features")
                .and_then(toml::Value::as_bool)
                .unwrap_or(true);
            (name.clone(), (features, default_features))
        })
        .collect()
}

/// Rust sources under `dir`, ignoring nothing: a registration in a test module
/// is still a registration.
fn rust_sources(dir: &Path) -> Vec<PathBuf> {
    walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .map(walkdir::DirEntry::into_path)
        .filter(|path| path.extension().is_some_and(|extension| extension == "rs"))
        .collect()
}

/// Features gating each module path in a crate, read from the `#[cfg(feature =
/// ...)]` attributes that sit directly above `mod` declarations.
///
/// The module path is spelled the way the file tree spells it, so a submitting
/// file resolves its gates by walking its own ancestors. A `#[cfg]` naming
/// several features gates the module on any one of them, so the resolved set is
/// a requirement of "at least one of these", not all.
fn module_gates(src: &Path) -> BTreeMap<String, BTreeSet<String>> {
    let files = rust_sources(src);
    let mut gates: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for file in files {
        let text = match fs::read_to_string(&file) {
            Ok(text) => text,
            Err(_) => continue,
        };
        let owner = if matches!(
            file.file_name().and_then(|n| n.to_str()),
            Some("lib.rs" | "mod.rs")
        ) {
            file.parent()
                .expect("Fix: a source file has a parent")
                .to_path_buf()
        } else {
            file.with_extension("")
        };
        let mut pending: BTreeSet<String> = BTreeSet::new();
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with("#[cfg(") {
                pending = feature_names(line);
            } else if let Some(rest) = module_name(line) {
                let path = owner.join(rest);
                let relative = path
                    .strip_prefix(src)
                    .expect("Fix: a module path lives under the crate source root")
                    .to_string_lossy()
                    .to_string();
                gates
                    .entry(relative)
                    .or_default()
                    .extend(pending.iter().cloned());
                pending.clear();
            } else if !line.is_empty() && !line.starts_with("//") && !line.starts_with('#') {
                pending.clear();
            }
        }
    }
    gates
}

fn feature_names(attribute: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut rest = attribute;
    while let Some(start) = rest.find("feature = \"") {
        rest = &rest[start + "feature = \"".len()..];
        let Some(end) = rest.find('"') else { break };
        names.insert(rest[..end].to_string());
        rest = &rest[end..];
    }
    names
}

/// The declared module name, for a `mod x;` or `pub mod x;` declaration only. An
/// inline `mod x {` block shares its parent's gates, so it needs no entry.
fn module_name(line: &str) -> Option<&str> {
    let rest = line
        .strip_prefix("pub mod ")
        .or_else(|| line.strip_prefix("pub(crate) mod "))
        .or_else(|| line.strip_prefix("mod "))?;
    rest.strip_suffix(';')
}

/// Files that submit into an inventory registry, with the gates a build has to
/// satisfy to compile each one.
///
/// A gate is one `#[cfg]`, and the features inside it are alternatives: `any(A,
/// B)` is satisfied by either. Separate gates are not alternatives, they stack:
/// a submission inside a module gated on `geom` and itself gated on
/// `inventory-registry` needs both, and reading the two as one set of
/// alternatives is what made this rule pass against the narrowed edge it was
/// written to catch. So the requirement is a list of gates, each satisfied
/// independently. A file with no gate compiles in every build.
fn submitting_files(crate_dir: &Path) -> BTreeMap<String, Vec<BTreeSet<String>>> {
    let src = crate_dir.join("src");
    let gates = module_gates(&src);
    let files = rust_sources(&src);
    let mut out = BTreeMap::new();
    for file in files {
        let Ok(raw) = fs::read_to_string(&file) else {
            continue;
        };
        // A fixture inside a `#[cfg(test)]` module is not a registration this
        // walker has to link, and neither is one quoted in a string. Both
        // questions are `structure_gate`'s, which is where the scan that reads
        // the same files answers them.
        let text = structure_gate::cfg_test::strip_cfg_test_items(&raw);
        if !structure_gate::submits_registrations(&text) {
            continue;
        }
        let relative = file
            .strip_prefix(&src)
            .expect("Fix: a submitting file lives under the crate source root")
            .with_extension("")
            .to_string_lossy()
            .to_string();
        let mut required: Vec<BTreeSet<String>> = Vec::new();
        let components: Vec<&str> = relative.split('/').collect();
        for depth in (1..=components.len()).rev() {
            let ancestor = components[..depth].join("/");
            match gates.get(&ancestor) {
                Some(features) if !features.is_empty() => required.push(features.clone()),
                _ => {}
            }
        }
        // A submission can also be gated in place. Only the attribute directly
        // above `inventory::submit!` gates it: a `#[cfg(feature = "cpu-parity")]`
        // on the CPU reference function further up the same file gates that
        // function, and reading it here would report a feature the registration
        // does not need.
        let mut attribute: Option<&str> = None;
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with("inventory::submit!")
                || line.starts_with("submit_hardware_intrinsic!")
                || line.starts_with("submit_intrinsic_operation!")
                || line.starts_with("define_unary_u32_hardware_intrinsic!")
                || line.starts_with("define_barrier_u32_hardware_intrinsic!")
            {
                if let Some(gate) = attribute.filter(|gate| gate.contains("feature")) {
                    let features = feature_names(gate);
                    if !features.is_empty() {
                        required.push(features);
                    }
                }
                attribute = None;
            } else if line.starts_with("#[") {
                attribute = Some(line);
            } else if !line.is_empty() && !line.starts_with("//") {
                attribute = None;
            }
        }
        out.insert(relative, required);
    }
    out
}

/// Transitive closure of a feature selection through a crate's own feature
/// table, collecting the `dependency/feature` entries it turns on for other
/// crates as it goes.
fn resolve(
    manifest: &toml::Value,
    selection: &BTreeSet<String>,
    with_default: bool,
) -> (BTreeSet<String>, BTreeMap<String, BTreeSet<String>>) {
    let table = manifest.get("features").and_then(toml::Value::as_table);
    let mut enabled: BTreeSet<String> = BTreeSet::new();
    let mut downstream: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut queue: VecDeque<String> = selection.iter().cloned().collect();
    if with_default {
        queue.push_back("default".to_string());
    }
    // An `optional = true` dependency is itself a feature name, so a selection
    // naming one is not a typo and contributes no further features.
    while let Some(feature) = queue.pop_front() {
        if let Some((dependency, downstream_feature)) = feature.split_once('/') {
            let dependency = dependency.trim_end_matches('?');
            downstream
                .entry(dependency.to_string())
                .or_default()
                .insert(downstream_feature.to_string());
            continue;
        }
        let feature = feature.strip_prefix("dep:").unwrap_or(&feature).to_string();
        if !enabled.insert(feature.clone()) {
            continue;
        }
        let Some(entries) = table
            .and_then(|table| table.get(&feature))
            .and_then(toml::Value::as_array)
        else {
            continue;
        };
        for entry in entries.iter().filter_map(toml::Value::as_str) {
            queue.push_back(entry.to_string());
        }
    }
    (enabled, downstream)
}

/// Crates whose registrations this walker must see: the operation-owning crates
/// it depends on directly, discovered by asking which of its dependencies submit
/// registrations at all.
fn operation_sources() -> Vec<String> {
    let root = workspace_root();
    declared_selections()
        .into_keys()
        .filter(|name| {
            let dir = root.join(name);
            dir.join("Cargo.toml").is_file() && !submitting_files(&dir).is_empty()
        })
        .collect()
}

/// The union of features reaching each source crate: what this crate names on it
/// directly, plus what the other dependencies turn on through
/// `source/feature` entries.
fn reaching_features(source: &str) -> BTreeSet<String> {
    let root = workspace_root();
    let selections = declared_selections();
    let mut reaching = BTreeSet::new();
    for (dependency, (selection, with_default)) in &selections {
        let manifest_path = root.join(dependency).join("Cargo.toml");
        if !manifest_path.is_file() {
            continue;
        }
        let manifest = read_manifest(&manifest_path);
        let (enabled, downstream) = resolve(&manifest, selection, *with_default);
        if dependency == source {
            reaching.extend(enabled);
        }
        if let Some(requested) = downstream.get(source) {
            let source_manifest = read_manifest(&root.join(source).join("Cargo.toml"));
            let (enabled, _) = resolve(&source_manifest, requested, false);
            reaching.extend(enabled);
        }
    }
    reaching
}

/// The walker's dependency edge is wide enough that no registration is
/// invisible to it.
#[test]
fn every_submitted_registration_is_reachable_from_this_walker() {
    let root = workspace_root();
    let sources = operation_sources();
    assert!(
        sources.len() >= 2,
        "Fix: the workspace has at least two operation-owning crates, found {sources:?}. \
         A source that stopped submitting registrations is the defect this counts."
    );
    let mut invisible: Vec<String> = Vec::new();
    for source in &sources {
        let reaching = reaching_features(source);
        for (file, required) in submitting_files(&root.join(source)) {
            let unsatisfied: Vec<&BTreeSet<String>> = required
                .iter()
                .filter(|gate| !gate.iter().any(|feature| reaching.contains(feature)))
                .collect();
            if unsatisfied.is_empty() {
                continue;
            }
            invisible.push(format!(
                "{source}/src/{file}.rs submits a registration behind {unsatisfied:?}, \
                 and this walker's dependency edge enables nothing in those gates"
            ));
        }
    }
    assert!(
        invisible.is_empty(),
        "Fix: xtask-registry generates every operation document from the live registry, \
         so a registration it cannot link is reported as absent rather than as an error. \
         Widen the feature selection in xtask-registry/Cargo.toml, using the crate's own \
         aggregate feature rather than a hand-kept list.\n{}",
        invisible.join("\n")
    );
}

/// Every feature that gates a registration in a source crate is a feature that
/// crate's widest aggregate turns on.
///
/// WHY: the previous rule is satisfied by naming features one at a time here,
/// which is the maintenance shape that rotted. A source crate that offers an
/// aggregate has one place to add a new domain, and this proves the aggregate is
/// actually complete, so naming it here is enough.
#[test]
fn each_source_aggregate_covers_every_feature_that_gates_a_registration() {
    let root = workspace_root();
    let mut gaps: Vec<String> = Vec::new();
    for source in operation_sources() {
        let manifest = read_manifest(&root.join(&source).join("Cargo.toml"));
        let Some(features) = manifest.get("features").and_then(toml::Value::as_table) else {
            continue;
        };
        let gating: BTreeSet<String> = submitting_files(&root.join(&source))
            .into_values()
            .flatten()
            .flatten()
            .filter(|feature| features.contains_key(feature))
            .collect();
        if gating.is_empty() {
            continue;
        }
        // The widest aggregate is the declared feature whose closure turns on
        // the most of this crate's own features. Derived, so a renamed
        // aggregate is still found.
        let widest = features
            .keys()
            .map(|name| {
                let selection = BTreeSet::from([name.clone()]);
                let (enabled, _) = resolve(&manifest, &selection, false);
                (enabled.len(), name.clone(), enabled)
            })
            .max_by_key(|(count, _, _)| *count);
        let Some((_, name, enabled)) = widest else {
            continue;
        };
        let missing: Vec<String> = gating.difference(&enabled).cloned().collect();
        if !missing.is_empty() {
            gaps.push(format!(
                "{source} feature `{name}` is its widest aggregate but does not enable {missing:?}, \
                 each of which gates a registration"
            ));
        }
    }
    assert!(
        gaps.is_empty(),
        "Fix: a crate that owns registrations offers one aggregate feature that turns on every \
         domain submitting them, so a consumer naming the aggregate sees the whole registry.\n{}",
        gaps.join("\n")
    );
}
