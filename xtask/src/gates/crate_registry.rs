//! The workspace ownership contract, and the two documents generated from it.
//!
//! `docs/CRATE_OWNERSHIP.toml` declares one row per workspace member: who owns
//! it, which layer it sits in, and one complete record per internal production
//! edge. Cargo declares the same edges a second time, in the manifests. This
//! gate holds the two to each other and renders the dependency graph and the
//! per-crate ownership page from the result, so the boundary a reviewer reads
//! is the boundary cargo resolves. [`GRAPH`] and [`OWNERSHIP`] name the two.
//!
//! It was a Python generator under `scripts/`, invoked by `check-tier-deps`
//! through `python3 --check` and by two integration tests. That put the
//! ownership contract in a second language with its own error handling, its own
//! exit codes, and no baseline: the whole contract reported one violation at a
//! time, because it raised on the first, and a tree with ten drifted edges
//! looked like a tree with one. Every rule below is a finding now, so the pinned
//! count moves when any of them does.
//!
//! Nothing here is a list. The member set comes from `workspace.members`, the
//! edges from each member's own manifest, and the seam an edge must name from
//! the owner of its destination row.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use toml::Value;

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// The authority every row is read from.
pub const REGISTRY: &str = "docs/CRATE_OWNERSHIP.toml";
/// The rendered dependency graph.
pub const GRAPH: &str = "docs/CRATE_GRAPH.md";
/// The rendered per-crate ownership document.
pub const OWNERSHIP: &str = "docs/OWNERSHIP.md";
/// The command that rewrites both documents.
pub const WRITE_COMMAND: &str = "xtask crate-ownership --write";
/// Schema the registry must declare.
pub const SCHEMA_VERSION: i64 = 3;

/// What a caller does about any disagreement this gate reports.
const FIX: &str = "change the manifest and its `[[crate.dependency]]` record together, then run `xtask crate-ownership --write`";

/// Valid publication classes for workspace crates.
pub const VALID_PUBLICATION_CLASSES: &[&str] = &[
    "stable-consumer-sdk",
    "extension-sdk",
    "concrete-backend",
    "internal-engine",
    "conformance-tooling",
    "private-test-support",
];
/// One declared layer, and its position in the dependency DAG.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerRecord {
    /// Layer name, as a `[[crate]]` row spells it.
    pub name: String,
    /// Dependency depth. A production edge is legal only when the source rank
    /// is strictly greater than the destination rank.
    pub rank: i64,
    /// What the layer is for, in the registry's own words.
    pub purpose: String,
}

/// One declared internal production edge.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DependencyRecord {
    /// Destination package.
    pub package: String,
    /// Why the edge exists.
    pub purpose: String,
    /// Cargo features the edge turns on, sorted.
    pub features: Vec<String>,
    /// Target conditions the edge is declared under, sorted.
    pub conditions: Vec<String>,
    /// Dependency kinds the edge appears as, sorted.
    pub kinds: Vec<String>,
    /// Whether cargo declares it optional.
    pub optional: bool,
    /// Whether cargo leaves default features on.
    pub default_features: bool,
    /// Whether the edge crosses the public API.
    pub boundary: String,
    /// The owner of the destination, which is the seam that owns the contract.
    pub seam: String,
}

/// One declared workspace member.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CrateRecord {
    /// Package name.
    pub package: String,
    /// Member directory, relative to the workspace root.
    pub path: String,
    /// Owning seam.
    pub owner: String,
    /// Layer the crate sits in.
    pub layer: String,
    /// Explicit publication classification.
    pub publication_class: String,
    /// What the crate is for, in the registry's own words.
    pub responsibility: String,
    /// Declared edges, sorted by destination package.
    pub dependencies: Vec<DependencyRecord>,
}

impl CrateRecord {
    /// Destination package names, sorted.
    #[must_use]
    pub fn allowed_dependencies(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .dependencies
            .iter()
            .map(|dependency| dependency.package.clone())
            .collect();
        names.sort();
        names
    }
}

/// One internal edge as cargo resolves it.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DependencyUse {
    /// Features, sorted and deduplicated across every declaration of the edge.
    pub features: Vec<String>,
    /// Target conditions the edge appears under, sorted.
    pub conditions: Vec<String>,
    /// Kinds the edge appears as, sorted.
    pub kinds: Vec<String>,
    /// True when any declaration is optional.
    pub optional: bool,
    /// True only when every declaration leaves default features on.
    pub default_features: bool,
}

/// The workspace as cargo declares it.
pub struct WorkspaceState {
    /// Member directories, in declaration order.
    pub members: Vec<String>,
    /// Package name to member directory.
    pub paths: BTreeMap<String, String>,
    /// Package name to its internal edges, keyed by destination package.
    pub dependencies: BTreeMap<String, BTreeMap<String, DependencyUse>>,
}

/// The ownership registry and the two documents rendered from it.
pub struct CrateOwnership;

impl crate::gate::GateBehavior for CrateOwnership {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        report.produced(GRAPH);
        report.produced(OWNERSHIP);
        let records = load_registry(&tree, &mut report)?;
        let layers = load_layers(&tree, &mut report)?;
        report.cover_complete("workspace crates", records.len());
        report.cover_complete("architecture layers", layers.len());
        let state = workspace_state(&tree)?;
        report.findings.extend(contract_findings(&state, &records));
        report
            .findings
            .extend(direction_findings(&state, &records, &layers));
        report.findings.extend(cycle_findings(&state));
        report.note(format!(
            "{} registry row(s) across {} workspace member(s) in {} layer(s)",
            records.len(),
            state.members.len(),
            layers.len()
        ));

        // A registry that does not describe this workspace cannot render a
        // document about it, and writing one from a broken registry publishes
        // the break. The rendered pair is judged only once the contract holds.
        if !report.findings.is_empty() {
            return Ok(report);
        }
        for (path, rendered) in [
            (GRAPH, render_graph(&records, &layers)?),
            (OWNERSHIP, render_ownership(&records)),
        ] {
            report
                .findings
                .extend(document_findings(&ctx.root, path, &rendered, ctx.write)?);
        }
        Ok(report)
    }
}

/// Hold one rendered document to the tree, or rewrite it.
fn document_findings(
    root: &Path,
    relative: &str,
    rendered: &str,
    write: bool,
) -> Result<Vec<Finding>, GateError> {
    let path = root.join(relative);
    if write {
        fs::write(&path, rendered).map_err(|error| {
            GateError::new(
                format!("cannot write `{relative}`: {error}"),
                "make the documentation directory writable",
            )
        })?;
        return Ok(Vec::new());
    }
    let actual = fs::read_to_string(&path).unwrap_or_default();
    // Line comparison: a checkout that materializes the document with CRLF
    // endings renders the same rows, and a byte comparison would report a
    // divergence the finding text cannot name.
    if actual.lines().eq(rendered.lines()) {
        return Ok(Vec::new());
    }
    Ok(vec![Finding::in_file(
        relative,
        format!("`{relative}` does not match what the registry renders"),
        format!("run `{WRITE_COMMAND}`"),
    )])
}

/// Read one required string field.
fn text(row: &Value, field: &str, context: &str, report: &mut Report) -> String {
    match row.get(field).and_then(Value::as_str) {
        Some(value) if !value.trim().is_empty() => value.trim().to_string(),
        _ => {
            report.find(Finding::in_file(
                REGISTRY,
                format!("{context} declares no non-empty `{field}`"),
                FIX,
            ));
            String::new()
        }
    }
}

/// Read one required string-array field, sorted and duplicate-free.
fn strings(row: &Value, field: &str, context: &str, report: &mut Report) -> Vec<String> {
    let Some(array) = row.get(field).and_then(Value::as_array) else {
        report.find(Finding::in_file(
            REGISTRY,
            format!("{context} declares no string array `{field}`"),
            FIX,
        ));
        return Vec::new();
    };
    let mut values = Vec::new();
    for item in array {
        match item.as_str() {
            Some(value) if !value.trim().is_empty() => values.push(value.trim().to_string()),
            _ => {
                report.find(Finding::in_file(
                    REGISTRY,
                    format!("{context} `{field}` holds an entry that is not non-empty text"),
                    FIX,
                ));
                return Vec::new();
            }
        }
    }
    values.sort();
    let before = values.len();
    values.dedup();
    if before != values.len() {
        report.find(Finding::in_file(
            REGISTRY,
            format!("{context} `{field}` repeats a value"),
            FIX,
        ));
    }
    values
}

/// Read one required boolean field.
fn boolean(row: &Value, field: &str, context: &str, report: &mut Report) -> bool {
    match row.get(field).and_then(Value::as_bool) {
        Some(value) => value,
        None => {
            report.find(Finding::in_file(
                REGISTRY,
                format!("{context} declares no boolean `{field}`"),
                FIX,
            ));
            false
        }
    }
}

/// Read one `[[crate.dependency]]` record.
fn load_dependency(row: &Value, context: &str, report: &mut Report) -> DependencyRecord {
    let boundary = text(row, "boundary", context, report);
    if !boundary.is_empty() && boundary != "public" && boundary != "private" {
        report.find(Finding::in_file(
            REGISTRY,
            format!(
                "{context} declares boundary `{boundary}`, which is neither public nor private"
            ),
            FIX,
        ));
    }
    let kinds = strings(row, "kinds", context, report);
    if kinds.is_empty() || kinds.iter().any(|kind| kind != "normal" && kind != "build") {
        report.find(Finding::in_file(
            REGISTRY,
            format!("{context} `kinds` must hold only `normal` or `build`"),
            FIX,
        ));
    }
    let conditions = strings(row, "conditions", context, report);
    if conditions.is_empty() {
        report.find(Finding::in_file(
            REGISTRY,
            format!("{context} declares no dependency condition"),
            FIX,
        ));
    }
    DependencyRecord {
        package: text(row, "package", context, report),
        purpose: text(row, "purpose", context, report),
        features: strings(row, "features", context, report),
        conditions,
        kinds,
        optional: boolean(row, "optional", context, report),
        default_features: boolean(row, "default_features", context, report),
        boundary,
        seam: text(row, "seam", context, report),
    }
}

/// Every `[[crate]]` row the registry declares.
pub fn load_registry(tree: &Tree, report: &mut Report) -> Result<Vec<CrateRecord>, GateError> {
    let registry = tree.read_toml(REGISTRY)?;
    if registry.get("schema_version").and_then(Value::as_integer) != Some(SCHEMA_VERSION) {
        report.find(Finding::in_file(
            REGISTRY,
            format!("the registry does not declare schema_version = {SCHEMA_VERSION}"),
            FIX,
        ));
    }
    if registry.contains_key("planned") {
        report.find(Finding::in_file(
            REGISTRY,
            "the registry describes planned crates",
            "record only current workspace owners; a planned row describes an architecture nothing resolves",
        ));
    }
    let Some(rows) = registry.get("crate").and_then(Value::as_array) else {
        report.find(Finding::in_file(
            REGISTRY,
            "the registry declares no [[crate]] rows",
            FIX,
        ));
        return Ok(Vec::new());
    };
    let mut records = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        let context = format!("{REGISTRY} [[crate]] row {}", index + 1);
        if row.get("allowed_dependencies").is_some() {
            report.find(Finding::in_file(
                REGISTRY,
                format!("{context} uses the removed `allowed_dependencies` key"),
                "declare one complete [[crate.dependency]] record per internal edge",
            ));
        }
        let mut dependencies = Vec::new();
        if let Some(rows) = row.get("dependency") {
            match rows.as_array() {
                Some(rows) => {
                    for (at, dependency) in rows.iter().enumerate() {
                        dependencies.push(load_dependency(
                            dependency,
                            &format!("{context} dependency {}", at + 1),
                            report,
                        ));
                    }
                }
                None => report.find(Finding::in_file(
                    REGISTRY,
                    format!("{context} `dependency` is not an array of tables"),
                    FIX,
                )),
            }
        }
        dependencies.sort_by(|left, right| left.package.cmp(&right.package));
        let names: BTreeSet<&str> = dependencies
            .iter()
            .map(|dependency| dependency.package.as_str())
            .collect();
        if names.len() != dependencies.len() {
            report.find(Finding::in_file(
                REGISTRY,
                format!("{context} declares the same dependency package twice"),
                FIX,
            ));
        }
        let pub_class = text(row, "publication_class", &context, report);
        if !pub_class.is_empty() && !VALID_PUBLICATION_CLASSES.contains(&pub_class.as_str()) {
            report.find(Finding::in_file(
                REGISTRY,
                format!("{context} declares invalid publication_class `{pub_class}`"),
                "declare one of: stable-consumer-sdk, extension-sdk, concrete-backend, internal-engine, conformance-tooling, private-test-support",
            ));
        }
        records.push(CrateRecord {
            package: text(row, "package", &context, report),
            path: text(row, "path", &context, report),
            owner: text(row, "owner", &context, report),
            layer: text(row, "layer", &context, report),
            publication_class: pub_class,
            responsibility: text(row, "responsibility", &context, report),
            dependencies,
        });
    }
    Ok(records)
}

/// What a caller does about a direction or layer disagreement.
const DIRECTION_FIX: &str = "put the dependency's destination in a lower-ranked layer, or correct the two `[[layer]]` ranks, then run `xtask crate-ownership --write`";

/// Every `[[layer]]` row the registry declares.
pub fn load_layers(tree: &Tree, report: &mut Report) -> Result<Vec<LayerRecord>, GateError> {
    let registry = tree.read_toml(REGISTRY)?;
    let Some(rows) = registry.get("layer").and_then(Value::as_array) else {
        report.find(Finding::in_file(
            REGISTRY,
            "the registry declares no [[layer]] rows",
            DIRECTION_FIX,
        ));
        return Ok(Vec::new());
    };
    let mut layers: Vec<LayerRecord> = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        let context = format!("{REGISTRY} [[layer]] row {}", index + 1);
        let name = text(row, "name", &context, report);
        let purpose = text(row, "purpose", &context, report);
        let rank = match row.get("rank").and_then(Value::as_integer) {
            Some(rank) if rank >= 0 => rank,
            _ => {
                report.find(Finding::in_file(
                    REGISTRY,
                    format!("{context} declares no non-negative integer `rank`"),
                    DIRECTION_FIX,
                ));
                -1
            }
        };
        if layers.iter().any(|earlier| earlier.name == name) {
            report.find(Finding::in_file(
                REGISTRY,
                format!("{context} declares layer `{name}` a second time"),
                DIRECTION_FIX,
            ));
        }
        layers.push(LayerRecord {
            name,
            rank,
            purpose,
        });
    }
    Ok(layers)
}

/// Every way the resolved graph disagrees with the declared layer DAG.
///
/// The rule is one comparison: a production edge is legal when the source layer
/// outranks the destination layer, and an edge inside one layer is always
/// legal. A reversal fails it directly, and a layer cycle cannot be written
/// down at all, because a cycle needs at least one edge whose source does not
/// outrank its destination. Nothing here enumerates permitted layer pairs, so
/// the registry carries no second roster to drift from the manifests.
///
/// Only `normal` edges are judged. A dev-dependency on a higher layer is how a
/// crate tests against the facade that consumes it, and cargo builds it in a
/// separate graph that cannot form a production cycle.
fn direction_findings(
    state: &WorkspaceState,
    records: &[CrateRecord],
    layers: &[LayerRecord],
) -> Vec<Finding> {
    let mut findings = Vec::new();
    let rank: BTreeMap<&str, i64> = layers
        .iter()
        .map(|layer| (layer.name.as_str(), layer.rank))
        .collect();
    let layer_of: BTreeMap<&str, &str> = records
        .iter()
        .map(|record| (record.package.as_str(), record.layer.as_str()))
        .collect();

    for record in records {
        if !rank.contains_key(record.layer.as_str()) {
            findings.push(Finding::in_file(
                REGISTRY,
                format!(
                    "`{}` sits in layer `{}` and no [[layer]] row declares it",
                    record.package, record.layer
                ),
                DIRECTION_FIX,
            ));
        }
    }
    let occupied: BTreeSet<&str> = records.iter().map(|record| record.layer.as_str()).collect();
    for layer in layers {
        if !occupied.contains(layer.name.as_str()) {
            findings.push(Finding::in_file(
                REGISTRY,
                format!("layer `{}` holds no workspace member", layer.name),
                "delete the [[layer]] row, or move a member into it",
            ));
        }
    }

    for (package, destinations) in &state.dependencies {
        let Some(source_layer) = layer_of.get(package.as_str()) else {
            continue;
        };
        let Some(source_rank) = rank.get(*source_layer) else {
            continue;
        };
        for (destination, use_) in destinations {
            if !use_.kinds.iter().any(|kind| kind == "normal") {
                continue;
            }
            let Some(destination_layer) = layer_of.get(destination.as_str()) else {
                continue;
            };
            if source_layer == destination_layer {
                continue;
            }
            let Some(destination_rank) = rank.get(*destination_layer) else {
                continue;
            };
            if source_rank <= destination_rank {
                findings.push(Finding::in_file(
                    REGISTRY,
                    format!(
                        "`{package}` in layer `{source_layer}` (rank {source_rank}) depends on `{destination}` in layer `{destination_layer}` (rank {destination_rank})"
                    ),
                    DIRECTION_FIX,
                ));
            }
        }
    }
    findings
}

/// What a caller does about a circular dependency.
const CYCLE_FIX: &str = "remove one of the circular dependencies so the internal production dependency graph is acyclic";

/// Every cycle in the resolved internal production dependency graph.
///
/// Internal normal and build dependencies must form a DAG. Cross-layer edges
/// are held to rank ordering, which rejects cross-layer cycles directly. This
/// check additionally rejects intra-layer cycles and reports the exact path of
/// every cycle deterministically.
fn cycle_findings(state: &WorkspaceState) -> Vec<Finding> {
    let mut findings = Vec::new();
    let mut adjacency: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for package in state.paths.keys() {
        adjacency.insert(package.as_str(), Vec::new());
    }
    for (package, destinations) in &state.dependencies {
        let entry = adjacency.entry(package.as_str()).or_default();
        for (destination, use_) in destinations {
            if use_.kinds.iter().any(|kind| kind == "normal" || kind == "build") {
                entry.push(destination.as_str());
            }
        }
        entry.sort();
        entry.dedup();
    }

    let mut visit_state: BTreeMap<&str, u8> = BTreeMap::new();
    let mut path: Vec<&str> = Vec::new();
    let mut reported: BTreeSet<Vec<String>> = BTreeSet::new();

    fn dfs<'a>(
        u: &'a str,
        adjacency: &BTreeMap<&'a str, Vec<&'a str>>,
        visit_state: &mut BTreeMap<&'a str, u8>,
        path: &mut Vec<&'a str>,
        reported: &mut BTreeSet<Vec<String>>,
        findings: &mut Vec<Finding>,
    ) {
        visit_state.insert(u, 1);
        path.push(u);
        if let Some(neighbors) = adjacency.get(u) {
            for &v in neighbors {
                match visit_state.get(v).copied().unwrap_or(0) {
                    1 => {
                        if let Some(start_idx) = path.iter().position(|&x| x == v) {
                            let cycle_slice = &path[start_idx..];
                            let min_pos = cycle_slice
                                .iter()
                                .enumerate()
                                .min_by_key(|(_, &x)| x)
                                .map(|(idx, _)| idx)
                                .unwrap_or(0);
                            let mut canonical: Vec<String> = Vec::with_capacity(cycle_slice.len());
                            for &item in &cycle_slice[min_pos..] {
                                canonical.push(item.to_string());
                            }
                            for &item in &cycle_slice[..min_pos] {
                                canonical.push(item.to_string());
                            }
                            if reported.insert(canonical.clone()) {
                                let cycle_str = format!("{} -> {}", canonical.join(" -> "), canonical[0]);
                                findings.push(Finding::in_file(
                                    REGISTRY,
                                    format!("dependency cycle detected: {cycle_str}"),
                                    CYCLE_FIX,
                                ));
                            }
                        }
                    }
                    0 => {
                        dfs(v, adjacency, visit_state, path, reported, findings);
                    }
                    _ => {}
                }
            }
        }
        path.pop();
        visit_state.insert(u, 2);
    }

    for package in state.paths.keys() {
        if visit_state.get(package.as_str()).copied().unwrap_or(0) == 0 {
            dfs(package.as_str(), &adjacency, &mut visit_state, &mut path, &mut reported, &mut findings);
        }
    }

    findings
}

/// One `[[crate]]` row as a gate that judges something else reads it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeclaredCrate {
    /// Package name.
    pub package: String,
    /// Member directory, relative to the workspace root.
    pub path: String,
    /// Layer the crate sits in.
    pub layer: String,
}

/// Every crate the ownership registry declares, by package, directory and layer.
///
/// [`load_registry`] judges the registry's own contract and reports every way
/// it disagrees with the manifests. A gate that needs a crate's layer or
/// directory to decide something else must not report those defects a second
/// time under its own name, so it reads the rows here. A row missing any of the
/// three keys is skipped: the finding for it belongs to the gate that owns the
/// registry.
pub fn declared_crates(tree: &Tree) -> Result<Vec<DeclaredCrate>, GateError> {
    let table = tree.read_toml(REGISTRY)?;
    let rows = table
        .get("crate")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            GateError::new(
                format!("{REGISTRY} declares no [[crate]] entries"),
                "declare every workspace member with its path and layer",
            )
        })?;
    let mut declared = Vec::new();
    for row in rows {
        let read = |key: &str| row.get(key).and_then(Value::as_str);
        let (Some(package), Some(path), Some(layer)) =
            (read("package"), read("path"), read("layer"))
        else {
            continue;
        };
        declared.push(DeclaredCrate {
            package: package.to_owned(),
            path: path.to_owned(),
            layer: layer.to_owned(),
        });
    }
    Ok(declared)
}

/// The dependency tables of one manifest, with the kind and condition each is
/// declared under.
fn dependency_tables(manifest: &toml::Table) -> Vec<(&toml::Table, &'static str, String)> {
    let mut tables = Vec::new();
    for (key, kind) in [("dependencies", "normal"), ("build-dependencies", "build")] {
        if let Some(table) = manifest.get(key).and_then(Value::as_table) {
            tables.push((table, kind, "always".to_string()));
        }
    }
    if let Some(targets) = manifest.get("target").and_then(Value::as_table) {
        for (condition, target) in targets {
            let Some(target) = target.as_table() else {
                continue;
            };
            for (key, kind) in [("dependencies", "normal"), ("build-dependencies", "build")] {
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
fn merged_specification(
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
fn feature_list(table: &toml::Table) -> Vec<String> {
    crate::toml_text::string_array(table.get("features"))
}

/// The workspace as cargo declares it: members, their packages, and the
/// internal edges each one resolves.
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
    for (package, manifest) in &manifests {
        let mut edges: BTreeMap<String, DependencyUse> = BTreeMap::new();
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
                let entry = edges.entry(destination).or_insert(DependencyUse {
                    default_features: true,
                    ..DependencyUse::default()
                });
                entry.features.extend(feature_list(&merged));
                entry.conditions.push(condition.clone());
                entry.kinds.push(kind.to_string());
                entry.optional =
                    entry.optional || merged.get("optional").and_then(Value::as_bool) == Some(true);
                entry.default_features = entry.default_features
                    && merged
                        .get("default-features")
                        .and_then(Value::as_bool)
                        .unwrap_or(true);
            }
        }
        for edge in edges.values_mut() {
            for list in [&mut edge.features, &mut edge.conditions, &mut edge.kinds] {
                list.sort();
                list.dedup();
            }
        }
        dependencies.insert(package.clone(), edges);
    }
    Ok(WorkspaceState {
        members,
        paths,
        dependencies,
    })
}

/// Every disagreement between the registry and the manifests.
fn contract_findings(state: &WorkspaceState, records: &[CrateRecord]) -> Vec<Finding> {
    let mut findings = Vec::new();
    let mut by_package: BTreeMap<&str, &CrateRecord> = BTreeMap::new();
    let mut by_path: BTreeMap<&str, &CrateRecord> = BTreeMap::new();
    for record in records {
        if by_package.insert(record.package.as_str(), record).is_some() {
            findings.push(Finding::in_file(
                REGISTRY,
                format!("the registry declares package `{}` twice", record.package),
                FIX,
            ));
        }
        if by_path.insert(record.path.as_str(), record).is_some() {
            findings.push(Finding::in_file(
                REGISTRY,
                format!("the registry declares path `{}` twice", record.path),
                FIX,
            ));
        }
    }

    let member_set: BTreeSet<&str> = state.members.iter().map(String::as_str).collect();
    for path in member_set.difference(&by_path.keys().copied().collect()) {
        findings.push(Finding::in_file(
            REGISTRY,
            format!("workspace member `{path}` has no registry row"),
            FIX,
        ));
    }
    for path in by_path
        .keys()
        .copied()
        .collect::<BTreeSet<&str>>()
        .difference(&member_set)
    {
        findings.push(Finding::in_file(
            REGISTRY,
            format!("registry row `{path}` is not a workspace member"),
            FIX,
        ));
    }

    for (package, path) in &state.paths {
        let Some(record) = by_package.get(package.as_str()) else {
            findings.push(Finding::in_file(
                REGISTRY,
                format!("workspace package `{package}` has no registry row"),
                FIX,
            ));
            continue;
        };
        if record.path != *path {
            findings.push(Finding::in_file(
                REGISTRY,
                format!(
                    "package `{package}` is registered at `{}` and lives at `{path}`",
                    record.path
                ),
                FIX,
            ));
        }
        let actual = state.dependencies.get(package).cloned().unwrap_or_default();
        let declared: BTreeMap<&str, &DependencyRecord> = record
            .dependencies
            .iter()
            .map(|dependency| (dependency.package.as_str(), dependency))
            .collect();
        for destination in actual.keys() {
            if !declared.contains_key(destination.as_str()) {
                findings.push(Finding::in_file(
                    REGISTRY,
                    format!("`{package}` depends on `{destination}` and declares no record for it"),
                    FIX,
                ));
            }
        }
        for destination in declared.keys() {
            if !actual.contains_key(*destination) {
                findings.push(Finding::in_file(
                    REGISTRY,
                    format!("`{package}` declares a record for `{destination}` and no manifest edge resolves to it"),
                    FIX,
                ));
            }
        }
        for (destination, expected) in &declared {
            let Some(observed) = actual.get(*destination) else {
                continue;
            };
            for (field, declared_value, actual_value) in [
                (
                    "features",
                    expected.features.join(","),
                    observed.features.join(","),
                ),
                (
                    "conditions",
                    expected.conditions.join(","),
                    observed.conditions.join(","),
                ),
                ("kinds", expected.kinds.join(","), observed.kinds.join(",")),
                (
                    "optional",
                    expected.optional.to_string(),
                    observed.optional.to_string(),
                ),
                (
                    "default_features",
                    expected.default_features.to_string(),
                    observed.default_features.to_string(),
                ),
            ] {
                if declared_value != actual_value {
                    findings.push(Finding::in_file(
                        REGISTRY,
                        format!(
                            "`{package}` -> `{destination}` declares {field} `{declared_value}` and cargo resolves `{actual_value}`"
                        ),
                        FIX,
                    ));
                }
            }
            let required = by_package
                .get(*destination)
                .map_or("", |record| record.owner.as_str());
            if expected.seam != required {
                findings.push(Finding::in_file(
                    REGISTRY,
                    format!(
                        "`{package}` -> `{destination}` declares seam `{}` and the destination owner is `{required}`",
                        expected.seam
                    ),
                    FIX,
                ));
            }
        }
    }
    findings
}

/// A backtick-joined list, or `None` when the list is empty.
fn format_list(values: &[String]) -> String {
    crate::toml_text::joined_backticked(values)
}

/// Extract a table of override tables from `config`, reporting entries that are not tables.
pub(crate) fn extract_table_overrides(
    metadata_file: &str,
    config: &toml::Table,
    key: &str,
    report: &mut Report,
) -> BTreeMap<String, toml::Table> {
    let mut overrides = BTreeMap::new();
    if let Some(table) = config.get(key).and_then(Value::as_table) {
        for (name, value) in table {
            match value.as_table() {
                Some(entry) => {
                    overrides.insert(name.clone(), entry.clone());
                }
                None => report.find(Finding::in_file(
                    metadata_file,
                    format!("the override for `{name}` is not a table"),
                    format!("declare [{key}.<name>] as a table"),
                )),
            }
        }
    }
    overrides
}

/// Reject an override or profile describing no crate or layer in the tree.
pub(crate) fn validate_metadata_membership<'a>(
    metadata_file: &str,
    override_packages: impl IntoIterator<Item = &'a String>,
    profile_layers: impl IntoIterator<Item = &'a String>,
    records: &[CrateRecord],
    profile_finding_message: impl Fn(&str) -> String,
    report: &mut Report,
) {
    let packages: BTreeSet<&str> = records.iter().map(|r| r.package.as_str()).collect();
    let layers: BTreeSet<&str> = records.iter().map(|r| r.layer.as_str()).collect();
    for package in override_packages {
        if !packages.contains(package.as_str()) {
            report.find(Finding::in_file(
                metadata_file,
                format!("`{package}` has an override and is not a workspace crate"),
                "delete the override, or restore the crate",
            ));
        }
    }
    for layer in profile_layers {
        if !layers.contains(layer.as_str()) {
            report.find(Finding::in_file(
                metadata_file,
                profile_finding_message(layer.as_str()),
                "delete the profile, or record the crate that occupies the layer",
            ));
        }
    }
}

/// Records in package order, which is the order both documents render in.
fn ordered(records: &[CrateRecord]) -> Vec<&CrateRecord> {
    let mut ordered: Vec<&CrateRecord> = records.iter().collect();
    ordered.sort_by(|left, right| left.package.cmp(&right.package));
    ordered
}

/// The dependency graph document.
///
/// A dependency whose package carries no record is an error rather than a
/// panic. The gate renders only after the contract holds, so the set is
/// complete on that path, but this is a public renderer and a caller that hands
/// it a partial record set gets the package name back instead of an index out
/// of a map.
pub fn render_graph(records: &[CrateRecord], layers: &[LayerRecord]) -> Result<String, GateError> {
    let ordered = ordered(records);
    let ids: BTreeMap<&str, String> = ordered
        .iter()
        .enumerate()
        .map(|(index, record)| (record.package.as_str(), format!("C{index}")))
        .collect();
    let node = |package: &str| -> Result<String, GateError> {
        ids.get(package).cloned().ok_or_else(|| {
            GateError::new(
                format!("`{package}` is named as a dependency and carries no registry record"),
                "declare the crate in docs/CRATE_OWNERSHIP.toml before rendering the graph; a node with no record has no place in the document",
            )
        })
    };
    let mut lines = vec![
        "# Vyre Crate Graph".to_string(),
        String::new(),
        format!("This file is generated by `{WRITE_COMMAND}` from"),
        "the workspace manifests and `docs/CRATE_OWNERSHIP.toml`. Edit those authorities"
            .to_string(),
        "together, then regenerate this file.".to_string(),
        String::new(),
        "## Layer ranks".to_string(),
        String::new(),
        "A production dependency is legal only when the consumer's layer outranks the".to_string(),
        "dependency's layer. Two layers share a rank when neither depends on the other."
            .to_string(),
        String::new(),
        "| Rank | Layer | Purpose |".to_string(),
        "| --- | --- | --- |".to_string(),
    ];
    let mut ranked: Vec<&LayerRecord> = layers.iter().collect();
    ranked.sort_by(|left, right| {
        left.rank
            .cmp(&right.rank)
            .then_with(|| left.name.cmp(&right.name))
    });
    for layer in &ranked {
        lines.push(format!(
            "| `{}` | `{}` | {} |",
            layer.rank, layer.name, layer.purpose
        ));
    }
    lines.extend([
        String::new(),
        "## Workspace dependency graph".to_string(),
        String::new(),
        format!(
            "The workspace contains {} crates. An arrow points from a crate to",
            ordered.len()
        ),
        "an internal normal or build dependency. Development dependencies are excluded."
            .to_string(),
        String::new(),
        "```mermaid".to_string(),
        "graph TD".to_string(),
    ]);
    for record in &ordered {
        lines.push(format!(
            "  {}[\"{}\"]",
            node(&record.package)?,
            record.package
        ));
    }
    for record in &ordered {
        for dependency in &record.dependencies {
            lines.push(format!(
                "  {} --> {}",
                node(&record.package)?,
                node(&dependency.package)?
            ));
        }
    }
    lines.extend([
        "```".to_string(),
        String::new(),
        "## Dependency contracts".to_string(),
        String::new(),
        "| Consumer | Dependency | Purpose | Features | Conditions | Kinds | Optional | Default features | Boundary | Owning seam |".to_string(),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |".to_string(),
    ]);
    for record in &ordered {
        for dependency in &record.dependencies {
            lines.push(format!(
                "| `{}` | `{}` | {} | {} | {} | {} | `{}` | `{}` | `{}` | `{}` |",
                record.package,
                dependency.package,
                dependency.purpose,
                format_list(&dependency.features),
                format_list(&dependency.conditions),
                format_list(&dependency.kinds),
                dependency.optional,
                dependency.default_features,
                dependency.boundary,
                dependency.seam
            ));
        }
    }
    lines.extend([
        String::new(),
        "## Changing a dependency".to_string(),
        String::new(),
        "Change the Cargo manifest and its complete `[[crate.dependency]]` record in".to_string(),
        "the same patch. The registry rejects undeclared packages, feature drift, target"
            .to_string(),
        "condition drift, stale seams, and missing visibility declarations.".to_string(),
        String::new(),
    ]);
    Ok(lines.join("\n"))
}

/// The per-crate ownership document.
#[must_use]
pub fn render_ownership(records: &[CrateRecord]) -> String {
    let ordered = ordered(records);
    let mut lines = vec![
        "# Vyre Crate Ownership".to_string(),
        String::new(),
        format!("This file is generated by `{WRITE_COMMAND}` from"),
        "`docs/CRATE_OWNERSHIP.toml` and the workspace manifests.".to_string(),
        String::new(),
        "## Boundary rule".to_string(),
        String::new(),
        "Each workspace crate has one owner and responsibility. Each internal production"
            .to_string(),
        "edge declares why it exists, its Cargo feature and target conditions, whether it"
            .to_string(),
        "crosses the public API, and the destination seam that owns the contract.".to_string(),
        String::new(),
        "## Per-crate ownership".to_string(),
        String::new(),
    ];
    for record in &ordered {
        lines.extend([
            format!("### `{}`", record.package),
            String::new(),
            record.responsibility.clone(),
            String::new(),
            format!("- Path: `{}`", record.path),
            format!("- Owner: `{}`", record.owner),
            format!("- Layer: `{}`", record.layer),
            format!("- Publication class: `{}`", record.publication_class),
            format!(
                "- Internal production dependencies: {}",
                format_list(&record.allowed_dependencies())
            ),
            String::new(),
        ]);
        if record.dependencies.is_empty() {
            continue;
        }
        lines.extend([
            "| Dependency | Purpose | Boundary | Owning seam |".to_string(),
            "| --- | --- | --- | --- |".to_string(),
        ]);
        for dependency in &record.dependencies {
            lines.push(format!(
                "| `{}` | {} | `{}` | `{}` |",
                dependency.package, dependency.purpose, dependency.boundary, dependency.seam
            ));
        }
        lines.push(String::new());
    }
    lines.extend([
        "## Changing a boundary".to_string(),
        String::new(),
        "1. Change the manifest and `docs/CRATE_OWNERSHIP.toml` together.".to_string(),
        format!("2. Run `{WRITE_COMMAND}`."),
        "3. Add a public import migration test when a public edge changes.".to_string(),
        "4. Run `./cargo_full run --bin xtask -- check-tier-deps` and `lego-audit`.".to_string(),
        String::new(),
    ]);
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: a `workspace = true` entry that also names features enables both
    /// sets, so reading only the local list under-reports the edge and the
    /// registry row that records the union would be reported as drifted.
    #[test]
    fn inherited_and_local_features_are_unioned() {
        let workspace: toml::Table =
            toml::from_str("[serde]\nversion = \"1\"\nfeatures = [\"std\"]\n")
                .expect("the workspace table parses");
        let specification: Value =
            toml::from_str::<toml::Table>("workspace = true\nfeatures = [\"derive\"]\n")
                .expect("the specification parses")
                .into();
        let merged = merged_specification("serde", &specification, &workspace);
        assert_eq!(feature_list(&merged), vec!["derive", "std"]);
    }

    /// WHY: an edge declared under two target conditions is one edge with two
    /// conditions, and an edge that is optional anywhere is optional. Cargo
    /// unions the first and disjoins the second, so a gate that overwrote
    /// either would report drift on a correct manifest.
    #[test]
    fn a_renamed_package_key_is_the_destination() {
        let workspace = toml::Table::new();
        let specification: Value =
            toml::from_str::<toml::Table>("package = \"vyre-libs\"\nversion = \"0.7\"\n")
                .expect("the specification parses")
                .into();
        let merged = merged_specification("libs", &specification, &workspace);
        assert_eq!(
            merged.get("package").and_then(Value::as_str),
            Some("vyre-libs")
        );
    }

    /// WHY: the two documents are the reviewable form of the registry, so an
    /// empty list has to render as a word rather than as nothing: a blank cell
    /// reads as an unfilled table rather than as an edge with no features.
    #[test]
    fn an_empty_list_renders_as_none() {
        assert_eq!(format_list(&[]), "None");
        assert_eq!(
            format_list(&["gpu".to_string(), "std".to_string()]),
            "`gpu`, `std`"
        );
    }

    /// One consumer, one dependency, and the layer rows to judge them by.
    fn direction_case(
        source_layer: &str,
        source_rank: i64,
        destination_layer: &str,
        destination_rank: i64,
        kinds: &[&str],
    ) -> Vec<Finding> {
        let records = vec![
            CrateRecord {
                package: "consumer".to_string(),
                path: "consumer".to_string(),
                owner: "consumer-seam".to_string(),
                layer: source_layer.to_string(),
                publication_class: "internal-engine".to_string(),
                responsibility: "consume".to_string(),
                dependencies: Vec::new(),
            },
            CrateRecord {
                package: "dependency".to_string(),
                path: "dependency".to_string(),
                owner: "dependency-seam".to_string(),
                layer: destination_layer.to_string(),
                publication_class: "internal-engine".to_string(),
                responsibility: "be consumed".to_string(),
                dependencies: Vec::new(),
            },
        ];
        let layers = vec![
            LayerRecord {
                name: source_layer.to_string(),
                rank: source_rank,
                purpose: "consume".to_string(),
            },
            LayerRecord {
                name: destination_layer.to_string(),
                rank: destination_rank,
                purpose: "be consumed".to_string(),
            },
        ];
        let state = WorkspaceState {
            members: vec!["consumer".to_string(), "dependency".to_string()],
            paths: BTreeMap::from([
                ("consumer".to_string(), "consumer".to_string()),
                ("dependency".to_string(), "dependency".to_string()),
            ]),
            dependencies: BTreeMap::from([(
                "consumer".to_string(),
                BTreeMap::from([(
                    "dependency".to_string(),
                    DependencyUse {
                        kinds: kinds.iter().map(|kind| (*kind).to_string()).collect(),
                        ..DependencyUse::default()
                    },
                )]),
            )]),
        };
        direction_findings(&state, &records, &layers)
    }

    /// WHY: the rank comparison is the whole direction contract, so the case it
    /// exists for has to fail. A consumer in a layer the dependency's layer
    /// outranks is a reversal, and an equal rank is one too: two layers share a
    /// rank only when neither depends on the other.
    #[test]
    fn a_layer_reversal_is_a_finding() {
        let reversed = direction_case("low", 1, "high", 4, &["normal"]);
        assert_eq!(reversed.len(), 1, "{reversed:?}");
        let equal = direction_case("left", 3, "right", 3, &["normal"]);
        assert_eq!(equal.len(), 1, "{equal:?}");
        assert!(direction_case("high", 4, "low", 1, &["normal"]).is_empty());
    }

    /// WHY: a dev-dependency on a higher layer is how a crate tests against the
    /// facade that consumes it. Cargo resolves it in a separate graph that
    /// cannot form a production cycle, so judging it would reject the intended
    /// shape.
    #[test]
    fn a_development_edge_carries_no_direction() {
        assert!(direction_case("low", 1, "high", 4, &["dev"]).is_empty());
        assert_eq!(
            direction_case("low", 1, "high", 4, &["dev", "normal"]).len(),
            1
        );
    }

    /// WHY: a layer a crate row names and no `[[layer]]` row declares has no
    /// rank, so every edge touching it would be skipped rather than judged. The
    /// unranked layer itself is the finding, and so is a declared layer no
    /// member occupies: it is a rank nothing is held to.
    #[test]
    fn an_unmatched_layer_is_a_finding() {
        let records = vec![CrateRecord {
            package: "consumer".to_string(),
            path: "consumer".to_string(),
            owner: "consumer-seam".to_string(),
            layer: "undeclared".to_string(),
            publication_class: "internal-engine".to_string(),
            responsibility: "consume".to_string(),
            dependencies: Vec::new(),
        }];
        let layers = vec![LayerRecord {
            name: "empty".to_string(),
            rank: 0,
            purpose: "nothing".to_string(),
        }];
        let state = WorkspaceState {
            members: vec!["consumer".to_string()],
            paths: BTreeMap::from([("consumer".to_string(), "consumer".to_string())]),
            dependencies: BTreeMap::new(),
        };
        let findings = direction_findings(&state, &records, &layers);
        assert_eq!(findings.len(), 2, "{findings:?}");
    }

    fn base_records_and_state() -> (Vec<CrateRecord>, WorkspaceState) {
        let records = vec![
            CrateRecord {
                package: "vyre-bench".to_string(),
                path: "vyre-bench".to_string(),
                owner: "benchmarks".to_string(),
                layer: "tooling".to_string(),
                publication_class: "conformance-tooling".to_string(),
                responsibility: "benchmarks".to_string(),
                dependencies: vec![DependencyRecord {
                    package: "vyre-driver-cuda".to_string(),
                    purpose: "cuda execution".to_string(),
                    features: vec![],
                    conditions: vec!["cfg(not(target_os = \"macos\"))".to_string()],
                    kinds: vec!["normal".to_string()],
                    optional: false,
                    default_features: true,
                    boundary: "private".to_string(),
                    seam: "cuda-driver".to_string(),
                }],
            },
            CrateRecord {
                package: "vyre-driver-cuda".to_string(),
                path: "vyre-driver-cuda".to_string(),
                owner: "cuda-driver".to_string(),
                layer: "concrete-backend".to_string(),
                publication_class: "concrete-backend".to_string(),
                responsibility: "cuda driver".to_string(),
                dependencies: vec![],
            },
        ];
        let state = WorkspaceState {
            members: vec!["vyre-bench".to_string(), "vyre-driver-cuda".to_string()],
            paths: BTreeMap::from([
                ("vyre-bench".to_string(), "vyre-bench".to_string()),
                ("vyre-driver-cuda".to_string(), "vyre-driver-cuda".to_string()),
            ]),
            dependencies: BTreeMap::from([
                (
                    "vyre-bench".to_string(),
                    BTreeMap::from([(
                        "vyre-driver-cuda".to_string(),
                        DependencyUse {
                            features: vec![],
                            conditions: vec!["cfg(not(target_os = \"macos\"))".to_string()],
                            kinds: vec!["normal".to_string()],
                            optional: false,
                            default_features: true,
                        },
                    )]),
                ),
                ("vyre-driver-cuda".to_string(), BTreeMap::new()),
            ]),
        };
        (records, state)
    }

    /// WHY: matching manifests and registry produce 0 contract findings.
    #[test]
    fn matching_manifest_and_registry_has_no_findings() {
        let (records, state) = base_records_and_state();
        let findings = contract_findings(&state, &records);
        assert!(findings.is_empty(), "{findings:?}");
    }

    /// WHY: when an edge is in the registry but no manifest resolves it,
    /// the gate must reject the stale record.
    #[test]
    fn stale_dependency_record_in_registry_is_a_finding() {
        let (records, mut state) = base_records_and_state();
        state.dependencies.get_mut("vyre-bench").unwrap().clear();
        let findings = contract_findings(&state, &records);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0].message.contains("declares a record for `vyre-driver-cuda` and no manifest edge resolves to it"));
    }

    /// WHY: when a manifest adds an internal dependency not in the registry,
    /// the gate must reject the undeclared edge.
    #[test]
    fn undeclared_manifest_dependency_is_a_finding() {
        let (mut records, state) = base_records_and_state();
        records[0].dependencies.clear();
        let findings = contract_findings(&state, &records);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0].message.contains("depends on `vyre-driver-cuda` and declares no record for it"));
    }

    /// WHY: mismatched dependency attributes (features, conditions, kinds, optional, default_features)
    /// must each produce a finding.
    #[test]
    fn mismatched_dependency_attributes_are_findings() {
        let (mut records, state) = base_records_and_state();
        records[0].dependencies[0].conditions = vec!["always".to_string()];
        records[0].dependencies[0].optional = true;
        let findings = contract_findings(&state, &records);
        assert_eq!(findings.len(), 2, "{findings:?}");
        assert!(findings.iter().any(|f| f.message.contains("declares conditions `always`")));
        assert!(findings.iter().any(|f| f.message.contains("declares optional `true`")));
    }

    /// WHY: declaring a seam that does not match the destination crate's owner
    /// must produce a finding.
    #[test]
    fn mismatched_seam_owner_is_a_finding() {
        let (mut records, state) = base_records_and_state();
        records[0].dependencies[0].seam = "wrong-seam".to_string();
        let findings = contract_findings(&state, &records);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0].message.contains("declares seam `wrong-seam` and the destination owner is `cuda-driver`"));
    }
    /// WHY: internal production dependency cycles (including intra-layer cycles)
    /// must fail closed with a finding naming the exact cycle path.
    #[test]
    fn dependency_cycle_is_a_finding() {
        let state = WorkspaceState {
            members: vec!["a".to_string(), "b".to_string()],
            paths: BTreeMap::from([("a".to_string(), "a".to_string()), ("b".to_string(), "b".to_string())]),
            dependencies: BTreeMap::from([
                (
                    "a".to_string(),
                    BTreeMap::from([(
                        "b".to_string(),
                        DependencyUse {
                            kinds: vec!["normal".to_string()],
                            ..DependencyUse::default()
                        },
                    )]),
                ),
                (
                    "b".to_string(),
                    BTreeMap::from([(
                        "a".to_string(),
                        DependencyUse {
                            kinds: vec!["normal".to_string()],
                            ..DependencyUse::default()
                        },
                    )]),
                ),
            ]),
        };
        let findings = cycle_findings(&state);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0].message.contains("dependency cycle detected: a -> b -> a"));
    }

    /// WHY: a feature inherited from workspace dependencies or feature unification
    /// that the registry declaration omits is drift and must be rejected.
    #[test]
    fn hidden_feature_unified_edge_is_a_finding() {
        let (mut records, state) = base_records_and_state();
        records[0].dependencies[0].features = vec!["unregistered-feature".to_string()];
        let findings = contract_findings(&state, &records);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0].message.contains("declares features `unregistered-feature` and cargo resolves ``"));
    }

    /// WHY: declaring normal dependency kind when cargo resolves build (or vice-versa)
    /// must be rejected.
    #[test]
    fn wrong_dependency_kind_is_a_finding() {
        let (mut records, state) = base_records_and_state();
        records[0].dependencies[0].kinds = vec!["build".to_string()];
        let findings = contract_findings(&state, &records);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0].message.contains("declares kinds `build` and cargo resolves `normal`"));
    }

    /// WHY: a lower layer (e.g. foundation, rank 0) depending on facade (rank 6)
    /// is a layer reversal and must be rejected.
    #[test]
    fn facade_imported_from_lower_layer_is_a_layer_reversal_finding() {
        let reversed = direction_case("foundation", 0, "facade", 6, &["normal"]);
        assert_eq!(reversed.len(), 1, "{reversed:?}");
        assert!(reversed[0].message.contains("`consumer` in layer `foundation` (rank 0) depends on `dependency` in layer `facade` (rank 6)"));
    }
}
