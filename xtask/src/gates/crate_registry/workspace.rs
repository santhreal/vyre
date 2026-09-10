//! Reading the workspace as cargo resolves it.
//!
//! The edge set is resolved under the union of every feature, because an
//! optional dependency a feature activates is an edge the default resolution
//! does not show.

use std::collections::{BTreeMap, BTreeSet};

use toml::Value;

use crate::gate::GateError;
use crate::gates::scan::Tree;

use super::*;

/// The dependency tables of one manifest, with the kind and condition each is
/// declared under.
pub(super) fn dependency_tables(manifest: &toml::Table) -> Vec<(&toml::Table, &'static str, String)> {
    let mut tables = Vec::new();
    for (key, kind) in [
        ("dependencies", "normal"),
        ("build-dependencies", "build"),
        ("dev-dependencies", "dev"),
    ] {
        if let Some(table) = manifest.get(key).and_then(Value::as_table) {
            tables.push((table, kind, "always".to_string()));
        }
    }
    if let Some(targets) = manifest.get("target").and_then(Value::as_table) {
        for (condition, target) in targets {
            let Some(target) = target.as_table() else {
                continue;
            };
            for (key, kind) in [
                ("dependencies", "normal"),
                ("build-dependencies", "build"),
                ("dev-dependencies", "dev"),
            ] {
                if let Some(table) = target.get(key).and_then(Value::as_table) {
                    tables.push((table, kind, condition.clone()));
                }
            }
        }
    }
    tables
}

/// One dependency specification with anything it inherits from the workspace
/// table folded in.
///
/// A `workspace = true` entry takes the workspace declaration and then its own
/// keys on top, and the feature lists are unioned rather than replaced: cargo
/// enables both sets, so reading only the local list under-reports the edge.
pub(super) fn merged_specification(
    alias: &str,
    specification: &Value,
    workspace: &toml::Table,
) -> toml::Table {
    let mut merged = toml::Table::new();
    match specification {
        Value::Table(table) if table.get("workspace").and_then(Value::as_bool) == Some(true) => {
            match workspace.get(alias) {
                Some(Value::Table(inherited)) => merged.extend(inherited.clone()),
                Some(Value::String(version)) => {
                    merged.insert("version".to_string(), Value::String(version.clone()));
                }
                _ => {}
            }
        }
        Value::Table(table) => merged.extend(table.clone()),
        Value::String(version) => {
            merged.insert("version".to_string(), Value::String(version.clone()));
        }
        _ => {}
    }
    let Value::Table(table) = specification else {
        return merged;
    };
    let inherited: Vec<String> = feature_list(&merged);
    let local: Vec<String> = crate::toml_text::string_array(table.get("features"));
    for (key, value) in table {
        if key != "workspace" {
            merged.insert(key.clone(), value.clone());
        }
    }
    let mut union: Vec<String> = inherited.into_iter().chain(local).collect();
    union.sort();
    union.dedup();
    merged.insert(
        "features".to_string(),
        Value::Array(union.into_iter().map(Value::String).collect()),
    );
    merged
}

/// The feature list a merged specification carries.
pub(super) fn feature_list(table: &toml::Table) -> Vec<String> {
    crate::toml_text::string_array(table.get("features"))
}

/// What one member's feature table does to its dependency edges.
#[derive(Debug, Default, Eq, PartialEq)]
pub(super) struct FeatureEffect {
    /// Dependency alias to the features of this package that activate it.
    pub(super) activated_by: BTreeMap<String, BTreeSet<String>>,
    /// Dependency alias to the destination features this package turns on.
    pub(super) enabled: BTreeMap<String, BTreeSet<String>>,
    /// Aliases some feature names with `dep:`, so cargo derives no implicit
    /// feature for them.
    pub(super) explicit: BTreeSet<String>,
}

/// Every edge activation and destination feature the `[features]` table
/// implies, transitively.
///
/// The default resolution shows neither. `libs-compositions = ["dep:vyre-libs",
/// "vyre-libs/encoding"]` is one edge and one destination feature that appear
/// only when the feature is on, and a feature that names another feature
/// reaches everything that one reaches. Reading only each dependency table's
/// own `features` key reports the edge as carrying no features and no
/// activation, which is the graph cargo resolves with `--no-default-features`
/// and not the one it resolves with `--all-features`.
pub(super) fn feature_effect(manifest: &toml::Table, optional: &BTreeSet<String>) -> FeatureEffect {
    let mut effect = FeatureEffect::default();
    let Some(features) = manifest.get("features").and_then(Value::as_table) else {
        return effect;
    };
    let items: BTreeMap<&str, Vec<&str>> = features
        .iter()
        .map(|(name, list)| {
            (
                name.as_str(),
                list.as_array()
                    .map(|array| array.iter().filter_map(Value::as_str).collect())
                    .unwrap_or_default(),
            )
        })
        .collect();
    for list in items.values() {
        for item in list {
            if let Some(alias) = item.strip_prefix("dep:") {
                effect.explicit.insert(alias.to_string());
            }
        }
    }
    for name in items.keys() {
        // Everything this feature reaches, including itself: a feature that
        // names another feature activates whatever that one activates.
        let mut reached: BTreeSet<&str> = BTreeSet::from([*name]);
        let mut pending: Vec<&str> = vec![name];
        while let Some(current) = pending.pop() {
            for item in items.get(current).into_iter().flatten() {
                if items.contains_key(*item) && reached.insert(item) {
                    pending.push(item);
                }
            }
        }
        for feature in reached {
            for item in items.get(feature).into_iter().flatten() {
                if let Some(alias) = item.strip_prefix("dep:") {
                    effect
                        .activated_by
                        .entry(alias.to_string())
                        .or_default()
                        .insert((*name).to_string());
                    continue;
                }
                if let Some((alias, enabled)) = item.split_once('/') {
                    let weak = alias.ends_with('?');
                    let alias = alias.trim_end_matches('?');
                    effect
                        .enabled
                        .entry(alias.to_string())
                        .or_default()
                        .insert(enabled.to_string());
                    if !weak {
                        effect
                            .activated_by
                            .entry(alias.to_string())
                            .or_default()
                            .insert((*name).to_string());
                    }
                    continue;
                }
                if !items.contains_key(*item) && optional.contains(*item) {
                    effect
                        .activated_by
                        .entry((*item).to_string())
                        .or_default()
                        .insert((*name).to_string());
                }
            }
        }
    }
    effect
}

/// The publication class a member's own manifest declares under
/// `[package.metadata.vyre]`.
pub(super) fn manifest_publication_class(manifest: &toml::Table) -> Option<String> {
    manifest
        .get("package")?
        .get("metadata")?
        .get("vyre")?
        .get("publication_class")?
        .as_str()
        .map(str::to_string)
}

/// The workspace as cargo declares it: members, their packages, and the
/// internal edges each one resolves under the union of every feature.
pub fn workspace_state(tree: &Tree) -> Result<WorkspaceState, GateError> {
    let root_manifest = tree.read_toml("Cargo.toml")?;
    let workspace = root_manifest
        .get("workspace")
        .and_then(Value::as_table)
        .ok_or_else(|| {
            GateError::new(
                "the root Cargo.toml declares no [workspace] table",
                "declare the workspace at the repository root",
            )
        })?;
    let members: Vec<String> = workspace
        .get("members")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            GateError::new(
                "the root Cargo.toml declares no workspace.members array",
                "declare workspace.members as an array of member directories",
            )
        })?
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect();
    let workspace_dependencies = workspace
        .get("dependencies")
        .and_then(Value::as_table)
        .cloned()
        .unwrap_or_default();

    // A duplicate is fatal rather than a finding: the state every contract is
    // judged against maps one package name to one manifest, so a second member
    // under the same name overwrites the first and the surviving row decides
    // what the whole gate reports.
    let mut paths = BTreeMap::new();
    let mut manifests = BTreeMap::new();
    let mut listed: BTreeSet<&str> = BTreeSet::new();
    for member in &members {
        if !listed.insert(member.as_str()) {
            return Err(GateError::new(
                format!("the root Cargo.toml lists workspace member `{member}` twice"),
                "list every workspace member once",
            ));
        }
        let manifest = tree.read_toml(format!("{member}/Cargo.toml"))?;
        let name = manifest
            .get("package")
            .and_then(|package| package.get("name"))
            .and_then(Value::as_str)
            .ok_or_else(|| {
                GateError::new(
                    format!("`{member}/Cargo.toml` declares no package.name"),
                    "declare package.name in the member manifest",
                )
            })?
            .to_string();
        if let Some(first) = paths.insert(name.clone(), member.clone()) {
            return Err(GateError::new(
                format!("`{first}` and `{member}` both declare package `{name}`"),
                "give each workspace member a distinct package.name",
            ));
        }
        manifests.insert(name, manifest);
    }

    let package_names: BTreeSet<String> = manifests.keys().cloned().collect();
    let mut dependencies = BTreeMap::new();
    let mut development = BTreeMap::new();
    for (package, manifest) in &manifests {
        // Two passes over the tables. The first records the alias every entry
        // is written under, because the feature table names the alias and not
        // the package. The second folds the feature table in.
        let mut optional_aliases: BTreeSet<String> = BTreeSet::new();
        let mut resolved: Vec<(String, String, &'static str, String, toml::Table)> = Vec::new();
        for (table, kind, condition) in dependency_tables(manifest) {
            for (alias, specification) in table {
                let merged = merged_specification(alias, specification, &workspace_dependencies);
                let destination = merged
                    .get("package")
                    .and_then(Value::as_str)
                    .unwrap_or(alias)
                    .to_string();
                if !package_names.contains(&destination) {
                    continue;
                }
                if merged.get("optional").and_then(Value::as_bool) == Some(true) {
                    optional_aliases.insert(alias.clone());
                }
                resolved.push((alias.clone(), destination, kind, condition.clone(), merged));
            }
        }
        let effect = feature_effect(manifest, &optional_aliases);

        let mut production: BTreeMap<String, DependencyUse> = BTreeMap::new();
        let mut dev: BTreeMap<String, DependencyUse> = BTreeMap::new();
        for (alias, destination, kind, condition, merged) in resolved {
            let optional = merged.get("optional").and_then(Value::as_bool) == Some(true);
            let bag = if kind == "dev" {
                &mut dev
            } else {
                &mut production
            };
            let entry = bag.entry(destination).or_insert(DependencyUse {
                default_features: true,
                ..DependencyUse::default()
            });
            entry.features.extend(feature_list(&merged));
            entry
                .features
                .extend(effect.enabled.get(&alias).into_iter().flatten().cloned());
            entry.conditions.push(condition);
            entry.kinds.push(kind.to_string());
            entry.optional = entry.optional || optional;
            entry.default_features = entry.default_features
                && merged
                    .get("default-features")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
            if optional {
                let named = effect
                    .activated_by
                    .get(&alias)
                    .is_some_and(|features| !features.is_empty());
                entry.named_activation = entry.named_activation || named;
                entry.activating_features.extend(
                    effect
                        .activated_by
                        .get(&alias)
                        .into_iter()
                        .flatten()
                        .cloned(),
                );
                // Cargo derives a feature named after an optional dependency
                // unless some feature spells it `dep:`, so an entry nothing
                // names explicitly still has one way in.
                if !effect.explicit.contains(&alias) {
                    entry.activating_features.push(alias.clone());
                }
            }
        }
        for bag in [&mut production, &mut dev] {
            for edge in bag.values_mut() {
                for list in [
                    &mut edge.features,
                    &mut edge.conditions,
                    &mut edge.kinds,
                    &mut edge.activating_features,
                ] {
                    list.sort();
                    list.dedup();
                }
            }
        }
        dependencies.insert(package.clone(), production);
        development.insert(package.clone(), dev);
    }
    Ok(WorkspaceState {
        members,
        paths,
        dependencies,
        development,
    })
}
