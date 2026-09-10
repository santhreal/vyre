//! The architecture manifest, joined to cargo, and the two documents rendered
//! from the join.
//!
//! Two authorities, no overlap. Cargo owns the actual package and feature
//! graph: which internal edge exists, its kind, whether it is optional, which
//! feature activates it, which target condition it is declared under, and the
//! publication class each member states beside its own `publish` key.
//! `docs/CRATE_OWNERSHIP.toml` owns intent: one row per workspace member
//! stating its layer, the seam it exposes, the one documented interface every
//! production edge into it crosses, and what it is responsible for.
//!
//! Schema 3 carried a `[[crate.dependency]]` row per internal production edge
//! and copied the destination's features, target conditions, kinds,
//! optionality and default-feature setting out of the manifests. That is a
//! second graph to keep exact, and keeping it exact bought a description per
//! consumer rather than per seam: 27 destinations carried more than one
//! interface text and 10 were declared public by one consumer and private by
//! another. The roster is gone. What survives is per member, so a new edge to
//! an already-declared seam needs no manifest edit and cannot be described
//! twice.
//!
//! The edge set is resolved under the union of every feature, because an
//! optional dependency a feature activates is an edge the default resolution
//! does not show. A gate that read only the default features would certify a
//! graph that does not exist under `--all-features`, so every rule below is
//! applied to the unified graph and every finding names the feature that
//! activated the edge.
//!
//! Nothing here is a list. The member set comes from `workspace.members`, the
//! edges from each member's own manifest with its feature table expanded, the
//! layer ranks and the seam names from the manifest rows.

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
/// Schema the manifest must declare.
pub const SCHEMA_VERSION: i64 = 4;
/// Where each member states its publication class. One home, beside the
/// `publish` key the class qualifies.
const MANIFEST_PUBLICATION_KEY: &str = "package.metadata.vyre.publication_class";

/// What a caller does about a disagreement between a row and the workspace.
const FIX: &str =
    "correct the `[[crate]]` row or the manifest, then run `xtask crate-ownership --write`";

/// Valid publication classes for workspace crates.
pub const VALID_PUBLICATION_CLASSES: &[&str] = &[
    "stable-consumer-sdk",
    "extension-sdk",
    "concrete-backend",
    "internal-engine",
    "conformance-tooling",
    "private-test-support",
];

/// Dependency kinds cargo builds into a shipped artifact.
const PRODUCTION_KINDS: [&str; 2] = ["normal", "build"];

/// One declared layer, and its position in the dependency DAG.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LayerRecord {
    /// Layer name, as a `[[crate]]` row spells it.
    pub name: String,
    /// Dependency depth. A production edge is legal only when the source rank
    /// is strictly greater than the destination rank.
    pub rank: i64,
    /// What the layer is for, in the manifest's own words.
    pub purpose: String,
    /// The closed set of layers whose members may cross a seam in this layer
    /// with a production edge. `None` admits every layer the rank rule allows.
    pub consumed_by: Option<Vec<String>>,
    /// Whether a member of this layer may cross only a seam whose row sets
    /// `facade_exported`.
    pub exports_declared_seams: bool,
}

/// One declared workspace member.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CrateRecord {
    /// Package name.
    pub package: String,
    /// Member directory, relative to the workspace root.
    pub path: String,
    /// Layer the crate sits in.
    pub layer: String,
    /// Explicit publication classification.
    pub publication_class: String,
    /// The named narrow interface this member exposes. Unique across members,
    /// so one concern has one package owner.
    pub seam: String,
    /// What crossing that seam provides, stated once for every consumer.
    pub interface: String,
    /// What the crate is for, in the manifest's own words.
    pub responsibility: String,
    /// Whether a member of an exporting layer may cross this seam.
    pub facade_exported: bool,
}

/// One internal edge as cargo resolves it under the union of every feature.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DependencyUse {
    /// Destination features this edge turns on: the dependency table's own
    /// `features` list plus everything the source's feature table enables on
    /// the destination. Sorted and deduplicated.
    pub features: Vec<String>,
    /// Target conditions the edge appears under, sorted.
    pub conditions: Vec<String>,
    /// Kinds the edge appears as, sorted.
    pub kinds: Vec<String>,
    /// True when any declaration is optional.
    pub optional: bool,
    /// True only when every declaration leaves default features on.
    pub default_features: bool,
    /// Features of the source package that activate an optional edge, sorted.
    /// Empty for a required edge, which the default resolution already shows.
    pub activating_features: Vec<String>,
}

impl DependencyUse {
    /// Whether cargo builds this edge into a shipped artifact.
    #[must_use]
    pub fn is_production(&self) -> bool {
        self.kinds
            .iter()
            .any(|kind| PRODUCTION_KINDS.contains(&kind.as_str()))
    }

    /// How a finding names the edge's activation.
    fn activation(&self) -> String {
        if self.activating_features.is_empty() {
            return "always".to_string();
        }
        format!(
            "under feature {}",
            crate::toml_text::joined_backticked(&self.activating_features)
        )
    }
}

/// The workspace as cargo declares it.
pub struct WorkspaceState {
    /// Member directories, in declaration order.
    pub members: Vec<String>,
    /// Package name to member directory.
    pub paths: BTreeMap<String, String>,
    /// Package name to its internal production edges, keyed by destination.
    pub dependencies: BTreeMap<String, BTreeMap<String, DependencyUse>>,
    /// Package name to its internal development edges, keyed by destination.
    pub development: BTreeMap<String, BTreeMap<String, DependencyUse>>,
}

/// The architecture manifest and the two documents rendered from it.
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
        report.findings.extend(activation_findings(&state));
        let edges: usize = state.dependencies.values().map(BTreeMap::len).sum();
        let hidden: usize = state
            .dependencies
            .values()
            .flat_map(BTreeMap::values)
            .filter(|use_| !use_.activating_features.is_empty())
            .count();
        report.note(format!(
            "{} manifest row(s) across {} workspace member(s) in {} layer(s); {edges} production edge(s), {hidden} activated by a feature",
            records.len(),
            state.members.len(),
            layers.len()
        ));

        // A manifest that does not describe this workspace cannot render a
        // document about it, and writing one from a broken manifest publishes
        // the break. The rendered pair is judged only once the contract holds.
        if !report.findings.is_empty() {
            return Ok(report);
        }
        for (path, rendered) in [
            (GRAPH, render_graph(&records, &layers, &state)?),
            (OWNERSHIP, render_ownership(&records, &layers, &state)?),
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
        format!("`{relative}` does not match what the manifest renders"),
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

/// Read one optional string-array field, sorted and duplicate-free.
fn optional_strings(
    row: &Value,
    field: &str,
    context: &str,
    report: &mut Report,
) -> Option<Vec<String>> {
    let value = row.get(field)?;
    let Some(array) = value.as_array() else {
        report.find(Finding::in_file(
            REGISTRY,
            format!("{context} `{field}` is not an array of layer names"),
            FIX,
        ));
        return Some(Vec::new());
    };
    let mut values = Vec::new();
    for item in array {
        match item.as_str() {
            Some(entry) if !entry.trim().is_empty() => values.push(entry.trim().to_string()),
            _ => {
                report.find(Finding::in_file(
                    REGISTRY,
                    format!("{context} `{field}` holds an entry that is not non-empty text"),
                    FIX,
                ));
                return Some(Vec::new());
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
    Some(values)
}

/// Read one optional boolean field, defaulting to false.
fn flag(row: &Value, field: &str, context: &str, report: &mut Report) -> bool {
    match row.get(field) {
        None => false,
        Some(Value::Boolean(value)) => *value,
        Some(_) => {
            report.find(Finding::in_file(
                REGISTRY,
                format!("{context} `{field}` is not a boolean"),
                FIX,
            ));
            false
        }
    }
}

/// Keys schema 3 carried and schema 4 does not.
///
/// Each named a cargo fact the manifest copied out of a member manifest, or a
/// per-edge description the destination row now states once. A row that still
/// carries one is a rebase artifact, and reporting it by name is how the
/// migration finishes rather than half-applies.
const RETIRED_CRATE_KEYS: [(&str, &str); 4] = [
    (
        "dependency",
        "delete the per-edge roster; cargo owns the edge set and the destination row owns the seam",
    ),
    (
        "allowed_dependencies",
        "delete the roster; cargo owns the edge set",
    ),
    (
        "owner",
        "rename the key to `seam` and add the `interface` the seam provides",
    ),
    (
        "publication_class",
        "delete the key; the class lives beside `publish` in the member's own `[package.metadata.vyre]` table",
    ),
];

/// The publication class a member's own manifest declares.
///
/// One home. The class sits beside `publish` in `[package.metadata.vyre]`,
/// which is the fact it qualifies and the only place cargo can read it, so
/// the architecture manifest states no class of its own and nothing has to
/// hold two copies to each other. A member that declares none is a finding
/// against the member manifest, because that is the file that has to change.
fn declared_publication_class(
    tree: &Tree,
    path: &str,
    context: &str,
    report: &mut Report,
) -> String {
    if path.is_empty() {
        return String::new();
    }
    let relative = format!("{path}/Cargo.toml");
    let Ok(manifest) = tree.read_toml(&relative) else {
        // The row names a directory the workspace does not have. The member
        // join reports that as a stale row; adding a second finding here would
        // charge one defect twice.
        return String::new();
    };
    let Some(class) = manifest_publication_class(&manifest) else {
        report.find(Finding::in_file(
            &relative,
            format!("`{path}` declares no `{MANIFEST_PUBLICATION_KEY}`"),
            "declare the publication class beside `publish` in `[package.metadata.vyre]`",
        ));
        return String::new();
    };
    if !VALID_PUBLICATION_CLASSES.contains(&class.as_str()) {
        report.find(Finding::in_file(
            &relative,
            format!("{context} resolves to invalid publication class `{class}`"),
            "declare one of: stable-consumer-sdk, extension-sdk, concrete-backend, internal-engine, conformance-tooling, private-test-support",
        ));
        return String::new();
    }
    class
}

/// Every `[[crate]]` row the manifest declares.
pub fn load_registry(tree: &Tree, report: &mut Report) -> Result<Vec<CrateRecord>, GateError> {
    let registry = tree.read_toml(REGISTRY)?;
    if registry.get("schema_version").and_then(Value::as_integer) != Some(SCHEMA_VERSION) {
        report.find(Finding::in_file(
            REGISTRY,
            format!("the manifest does not declare schema_version = {SCHEMA_VERSION}"),
            FIX,
        ));
    }
    if registry.contains_key("planned") {
        report.find(Finding::in_file(
            REGISTRY,
            "the manifest describes planned crates",
            "record only current workspace owners; a planned row describes an architecture nothing resolves",
        ));
    }
    let Some(rows) = registry.get("crate").and_then(Value::as_array) else {
        report.find(Finding::in_file(
            REGISTRY,
            "the manifest declares no [[crate]] rows",
            FIX,
        ));
        return Ok(Vec::new());
    };
    let mut records = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        let context = format!("{REGISTRY} [[crate]] row {}", index + 1);
        for (key, fix) in RETIRED_CRATE_KEYS {
            if row.get(key).is_some() {
                report.find(Finding::in_file(
                    REGISTRY,
                    format!("{context} uses the retired `{key}` key"),
                    fix,
                ));
            }
        }
        let path = text(row, "path", &context, report);
        let publication_class = declared_publication_class(tree, &path, &context, report);
        records.push(CrateRecord {
            package: text(row, "package", &context, report),
            path,
            layer: text(row, "layer", &context, report),
            publication_class,
            seam: text(row, "seam", &context, report),
            interface: text(row, "interface", &context, report),
            responsibility: text(row, "responsibility", &context, report),
            facade_exported: flag(row, "facade_exported", &context, report),
        });
    }
    Ok(records)
}

/// What a caller does about a direction or layer disagreement.
const DIRECTION_FIX: &str = "put the dependency's destination in a lower-ranked layer, or correct the two `[[layer]]` ranks, then run `xtask crate-ownership --write`";

/// Every `[[layer]]` row the manifest declares.
pub fn load_layers(tree: &Tree, report: &mut Report) -> Result<Vec<LayerRecord>, GateError> {
    let registry = tree.read_toml(REGISTRY)?;
    let Some(rows) = registry.get("layer").and_then(Value::as_array) else {
        report.find(Finding::in_file(
            REGISTRY,
            "the manifest declares no [[layer]] rows",
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
            consumed_by: optional_strings(row, "consumed_by", &context, report),
            exports_declared_seams: flag(row, "exports_declared_seams", &context, report),
        });
    }
    Ok(layers)
}

/// Every way the resolved graph disagrees with the declared layer DAG.
///
/// Four rules over one graph, and each rejects something the others cannot.
///
/// Rank is the direction contract: a production edge is legal when the source
/// layer outranks the destination layer, and an edge inside one layer is
/// always legal. A reversal fails it directly, and a cross-layer cycle cannot
/// be written down at all, because a cycle needs at least one edge whose
/// source does not outrank its destination. Nothing enumerates permitted layer
/// pairs, so there is no second roster to drift from the manifests.
///
/// `consumed_by` narrows a destination layer to a closed set of consumer
/// layers, which is the one statement rank cannot make: rank says a runtime
/// crate may reach a composition library, and `consumed_by` says which layers
/// actually may. An entry no edge crosses is a stale declaration and is
/// reported as one.
///
/// `exports_declared_seams` holds an exporting layer to the seams that admit
/// it. The facade outranks everything, so rank permits it to reach any crate
/// in the workspace; the export flag on the destination is what keeps it from
/// re-exporting an emitter or a pass engine. A flag no facade edge crosses is
/// stale too.
///
/// Every edge is judged under the union of every feature, so an edge an
/// optional feature activates is held to all four rules and the finding names
/// the feature.
///
/// Only production edges are judged. A dev-dependency on a higher layer is how
/// a crate tests against the facade that consumes it, and cargo builds it in a
/// separate graph that cannot form a production cycle.
fn direction_findings(
    state: &WorkspaceState,
    records: &[CrateRecord],
    layers: &[LayerRecord],
) -> Vec<Finding> {
    let mut findings = Vec::new();
    let by_name: BTreeMap<&str, &LayerRecord> = layers
        .iter()
        .map(|layer| (layer.name.as_str(), layer))
        .collect();
    let record_of: BTreeMap<&str, &CrateRecord> = records
        .iter()
        .map(|record| (record.package.as_str(), record))
        .collect();

    for record in records {
        if !by_name.contains_key(record.layer.as_str()) {
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
        for admitted in layer.consumed_by.iter().flatten() {
            if !by_name.contains_key(admitted.as_str()) {
                findings.push(Finding::in_file(
                    REGISTRY,
                    format!(
                        "layer `{}` admits consumer layer `{admitted}` and no [[layer]] row declares it",
                        layer.name
                    ),
                    DIRECTION_FIX,
                ));
            }
        }
    }

    // Which admissions and which export flags an edge actually crosses. An
    // unused one is a declaration the workspace stopped needing, which is the
    // stale half of the join.
    let mut crossed: BTreeSet<(&str, &str)> = BTreeSet::new();
    let mut exported: BTreeSet<&str> = BTreeSet::new();

    for (package, destinations) in &state.dependencies {
        let Some(source) = record_of.get(package.as_str()) else {
            continue;
        };
        let Some(source_layer) = by_name.get(source.layer.as_str()) else {
            continue;
        };
        for (destination, use_) in destinations {
            if !use_.is_production() {
                continue;
            }
            let Some(target) = record_of.get(destination.as_str()) else {
                continue;
            };
            let Some(target_layer) = by_name.get(target.layer.as_str()) else {
                continue;
            };
            let activation = use_.activation();
            if !edge_points_down(
                &source_layer.name,
                source_layer.rank,
                &target_layer.name,
                target_layer.rank,
            ) {
                findings.push(Finding::in_file(
                    REGISTRY,
                    format!(
                        "`{package}` in layer `{}` (rank {}) depends {activation} on `{destination}` in layer `{}` (rank {})",
                        source_layer.name, source_layer.rank, target_layer.name, target_layer.rank
                    ),
                    DIRECTION_FIX,
                ));
            }
            if let Some(admitted) = target_layer.consumed_by.as_deref() {
                if admitted.iter().any(|name| name == &source_layer.name) {
                    crossed.insert((target_layer.name.as_str(), source_layer.name.as_str()));
                } else {
                    findings.push(Finding::in_file(
                        REGISTRY,
                        format!(
                            "`{package}` in layer `{}` depends {activation} on `{destination}` over the `{}` seam, and layer `{}` admits no consumer in `{}`",
                            source_layer.name, target.seam, target_layer.name, source_layer.name
                        ),
                        "move the dependency behind a layer the destination admits, or record the consumer layer in the destination layer's `consumed_by`",
                    ));
                }
            }
            if source_layer.exports_declared_seams {
                if target.facade_exported {
                    exported.insert(target.package.as_str());
                } else {
                    findings.push(Finding::in_file(
                        REGISTRY,
                        format!(
                            "`{package}` exports declared seams and depends {activation} on `{destination}`, whose row does not set `facade_exported`"
                        ),
                        "re-export the destination through a seam that declares itself exported, or set `facade_exported = true` on its row and state why the curated surface carries it",
                    ));
                }
            }
        }
    }

    for layer in layers {
        for admitted in layer.consumed_by.iter().flatten() {
            if by_name.contains_key(admitted.as_str())
                && !crossed.contains(&(layer.name.as_str(), admitted.as_str()))
            {
                findings.push(Finding::in_file(
                    REGISTRY,
                    format!(
                        "layer `{}` admits consumer layer `{admitted}` and no production edge crosses it",
                        layer.name
                    ),
                    "delete the stale `consumed_by` entry",
                ));
            }
        }
    }
    for record in records {
        if record.facade_exported && !exported.contains(record.package.as_str()) {
            findings.push(Finding::in_file(
                REGISTRY,
                format!(
                    "`{}` declares `facade_exported` and no exporting layer depends on it",
                    record.package
                ),
                "delete the stale `facade_exported` flag",
            ));
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
/// every cycle deterministically. The graph is the feature-unified one, so a
/// cycle that closes only when a feature is on is a cycle here.
fn cycle_findings(state: &WorkspaceState) -> Vec<Finding> {
    let mut findings = Vec::new();
    let mut adjacency: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for package in state.paths.keys() {
        adjacency.insert(package.as_str(), Vec::new());
    }
    for (package, destinations) in &state.dependencies {
        let entry = adjacency.entry(package.as_str()).or_default();
        for (destination, use_) in destinations {
            if use_.is_production() {
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
                                let cycle_str =
                                    format!("{} -> {}", canonical.join(" -> "), canonical[0]);
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
            dfs(
                package.as_str(),
                &adjacency,
                &mut visit_state,
                &mut path,
                &mut reported,
                &mut findings,
            );
        }
    }

    findings
}

/// Every optional edge no feature of its consumer can turn on.
///
/// An optional dependency is reachable only through a feature that names it,
/// either as `dep:name`, as `name/feature`, or as the implicit feature cargo
/// derives when nothing writes `dep:name`. One that no feature reaches is an
/// edge no build resolves, and it reads as a live dependency to anyone
/// counting rebuild fan-out.
fn activation_findings(state: &WorkspaceState) -> Vec<Finding> {
    let mut findings = Vec::new();
    for (package, destinations) in &state.dependencies {
        for (destination, use_) in destinations {
            if use_.optional && use_.activating_features.is_empty() {
                findings.push(Finding::in_file(
                    format!("{}/Cargo.toml", state.paths.get(package).map_or("", String::as_str)),
                    format!(
                        "`{package}` declares `{destination}` optional and no feature activates it"
                    ),
                    "name the dependency in the feature that turns it on as `dep:name`, or make the edge required",
                ));
            }
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

/// Every crate the architecture manifest declares, by package, directory and
/// layer.
///
/// [`load_registry`] judges the manifest's own contract and reports every way
/// it disagrees with the workspace. A gate that needs a crate's layer or
/// directory to decide something else must not report those defects a second
/// time under its own name, so it reads the rows here. A row missing any of the
/// three keys is skipped: the finding for it belongs to the gate that owns the
/// manifest.
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

/// The rank each declared layer carries, for a gate that judges direction
/// without re-reading the whole manifest.
///
/// One statement of the ordering. A gate that kept its own list of layers,
/// ordered its own way, answered a different question about the same edge than
/// the manifest did, and both kept passing.
pub fn declared_layer_ranks(tree: &Tree) -> Result<BTreeMap<String, i64>, GateError> {
    let table = tree.read_toml(REGISTRY)?;
    let rows = table
        .get("layer")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            GateError::new(
                format!("{REGISTRY} declares no [[layer]] entries"),
                "declare every architecture layer with its rank",
            )
        })?;
    let mut ranks = BTreeMap::new();
    for row in rows {
        let (Some(name), Some(rank)) = (
            row.get("name").and_then(Value::as_str),
            row.get("rank").and_then(Value::as_integer),
        ) else {
            continue;
        };
        ranks.insert(name.to_owned(), rank);
    }
    Ok(ranks)
}

/// Whether a production edge from one layer to another points down the DAG.
///
/// One statement of the direction rule. An edge inside a layer is always
/// legal, because a layer is a set of crates that may reach each other. A
/// cross-layer edge is legal only when the source strictly outranks the
/// destination, so two layers at equal rank are mutually unreachable and a
/// cross-layer cycle cannot be written down.
///
/// A gate that judged direction with its own comparison answered a different
/// question about the same edge than the manifest did, and both kept passing.
#[must_use]
pub fn edge_points_down(
    source_layer: &str,
    source_rank: i64,
    target_layer: &str,
    target_rank: i64,
) -> bool {
    source_layer == target_layer || source_rank > target_rank
}

/// The dependency tables of one manifest, with the kind and condition each is
/// declared under.
fn dependency_tables(manifest: &toml::Table) -> Vec<(&toml::Table, &'static str, String)> {
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

/// What one member's feature table does to its dependency edges.
#[derive(Debug, Default, Eq, PartialEq)]
struct FeatureEffect {
    /// Dependency alias to the features of this package that activate it.
    activated_by: BTreeMap<String, BTreeSet<String>>,
    /// Dependency alias to the destination features this package turns on.
    enabled: BTreeMap<String, BTreeSet<String>>,
    /// Aliases some feature names with `dep:`, so cargo derives no implicit
    /// feature for them.
    explicit: BTreeSet<String>,
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
fn feature_effect(manifest: &toml::Table, optional: &BTreeSet<String>) -> FeatureEffect {
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
fn manifest_publication_class(manifest: &toml::Table) -> Option<String> {
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

/// Every disagreement between the architecture manifest and the workspace.
///
/// The join key is the member. A member cargo resolves and the manifest does
/// not describe is an undeclared member, and a row describing no member is a
/// stale declaration. Seam uniqueness is the one-owner-per-concern rule: two
/// rows claiming one seam name leave every edge into it crossing an interface
/// with two owners.
fn contract_findings(state: &WorkspaceState, records: &[CrateRecord]) -> Vec<Finding> {
    let mut findings = Vec::new();
    let mut by_package: BTreeMap<&str, &CrateRecord> = BTreeMap::new();
    let mut by_path: BTreeMap<&str, &CrateRecord> = BTreeMap::new();
    let mut by_seam: BTreeMap<&str, &CrateRecord> = BTreeMap::new();
    for record in records {
        if let Some(first) = by_package.insert(record.package.as_str(), record) {
            findings.push(Finding::in_file(
                REGISTRY,
                format!("the manifest declares package `{}` twice", first.package),
                FIX,
            ));
        }
        if let Some(first) = by_path.insert(record.path.as_str(), record) {
            findings.push(Finding::in_file(
                REGISTRY,
                format!("the manifest declares path `{}` twice", first.path),
                FIX,
            ));
        }
        if record.seam.is_empty() {
            continue;
        }
        if let Some(first) = by_seam.insert(record.seam.as_str(), record) {
            findings.push(Finding::in_file(
                REGISTRY,
                format!(
                    "`{}` and `{}` both own the `{}` seam",
                    first.package, record.package, record.seam
                ),
                "give each package its own seam name; one concern has one package owner",
            ));
        }
    }

    let member_set: BTreeSet<&str> = state.members.iter().map(String::as_str).collect();
    for path in member_set.difference(&by_path.keys().copied().collect()) {
        findings.push(Finding::in_file(
            REGISTRY,
            format!("workspace member `{path}` has no manifest row"),
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
            format!("manifest row `{path}` is not a workspace member"),
            FIX,
        ));
    }

    for (package, path) in &state.paths {
        let Some(record) = by_package.get(package.as_str()) else {
            findings.push(Finding::in_file(
                REGISTRY,
                format!("workspace package `{package}` has no manifest row"),
                FIX,
            ));
            continue;
        };
        if record.path != *path {
            findings.push(Finding::in_file(
                REGISTRY,
                format!(
                    "package `{package}` is declared at `{}` and lives at `{path}`",
                    record.path
                ),
                FIX,
            ));
        }
    }

    // A production edge to a package with no row crosses no declared seam, so
    // nothing states what it is allowed to reach through it.
    for (package, destinations) in &state.dependencies {
        for (destination, use_) in destinations {
            if !use_.is_production() || by_package.contains_key(destination.as_str()) {
                continue;
            }
            findings.push(Finding::in_file(
                REGISTRY,
                format!(
                    "`{package}` depends {} on `{destination}`, which owns no declared seam",
                    use_.activation()
                ),
                FIX,
            ));
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
    let Some(table) = config.get(key).and_then(Value::as_table) else {
        return overrides;
    };
    for (name, value) in table {
        match value.as_table() {
            Some(entry) => {
                overrides.insert(name.clone(), entry.clone());
            }
            None => report.find(Finding::in_file(
                metadata_file,
                format!("`{key}.{name}` is not a table"),
                "declare it as a table",
            )),
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
    let packages: BTreeSet<&str> = records
        .iter()
        .map(|record| record.package.as_str())
        .collect();
    let layers: BTreeSet<&str> = records.iter().map(|record| record.layer.as_str()).collect();
    for package in override_packages {
        if !packages.contains(package.as_str()) {
            report.find(Finding::in_file(
                metadata_file,
                format!("`{package}` has an override and is not a workspace crate"),
                "delete the override, or name a crate the workspace has",
            ));
        }
    }
    for layer in profile_layers {
        if !layers.contains(layer.as_str()) {
            report.find(Finding::in_file(
                metadata_file,
                profile_finding_message(layer),
                "delete the profile, or move a crate into the layer",
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

/// Layers by rank, then by name, which is how both documents order them.
fn ranked(layers: &[LayerRecord]) -> Vec<&LayerRecord> {
    let mut ranked: Vec<&LayerRecord> = layers.iter().collect();
    ranked.sort_by(|left, right| {
        left.rank
            .cmp(&right.rank)
            .then_with(|| left.name.cmp(&right.name))
    });
    ranked
}

/// Every production edge, source first, as one flat sorted list.
fn production_edges<'a>(state: &'a WorkspaceState) -> Vec<(&'a str, &'a str, &'a DependencyUse)> {
    let mut edges = Vec::new();
    for (package, destinations) in &state.dependencies {
        for (destination, use_) in destinations {
            if use_.is_production() {
                edges.push((package.as_str(), destination.as_str(), use_));
            }
        }
    }
    edges
}

/// The seam a destination row names, or an error when it has no row.
fn seam_of<'a>(
    records: &BTreeMap<&str, &'a CrateRecord>,
    package: &str,
) -> Result<&'a str, GateError> {
    records
        .get(package)
        .map(|record| record.seam.as_str())
        .ok_or_else(|| {
            GateError::new(
                format!("`{package}` is named as a dependency and carries no manifest row"),
                "declare the crate in docs/CRATE_OWNERSHIP.toml before rendering the graph; a node with no row has no place in the document",
            )
        })
}

/// The dependency graph document.
///
/// Every edge here comes from cargo. The document is a rendering of the graph
/// the build resolves, annotated with the layer and seam each edge crosses, so
/// it cannot disagree with the manifests: `--write` rewrites it and the gate
/// rejects a checked-in copy that differs.
pub fn render_graph(
    records: &[CrateRecord],
    layers: &[LayerRecord],
    state: &WorkspaceState,
) -> Result<String, GateError> {
    let ordered = ordered(records);
    let by_package: BTreeMap<&str, &CrateRecord> = ordered
        .iter()
        .map(|record| (record.package.as_str(), *record))
        .collect();
    let ids: BTreeMap<&str, String> = ordered
        .iter()
        .enumerate()
        .map(|(index, record)| (record.package.as_str(), format!("C{index}")))
        .collect();
    let node = |package: &str| -> Result<String, GateError> {
        ids.get(package).cloned().ok_or_else(|| {
            GateError::new(
                format!("`{package}` is named as a dependency and carries no manifest row"),
                "declare the crate in docs/CRATE_OWNERSHIP.toml before rendering the graph; a node with no row has no place in the document",
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
        "A layer that admits a closed set of consumer layers names them; one that admits"
            .to_string(),
        "every layer the rank rule allows names none.".to_string(),
        String::new(),
        "| Rank | Layer | Admitted consumer layers | Purpose |".to_string(),
        "| --- | --- | --- | --- |".to_string(),
    ];
    for layer in ranked(layers) {
        lines.push(format!(
            "| `{}` | `{}` | {} | {} |",
            layer.rank,
            layer.name,
            layer
                .consumed_by
                .as_deref()
                .map_or_else(|| "every outranking layer".to_string(), format_list),
            layer.purpose
        ));
    }
    let edges = production_edges(state);
    lines.extend([
        String::new(),
        "## Workspace dependency graph".to_string(),
        String::new(),
        format!(
            "The workspace contains {} crates and {} internal production edges, resolved",
            ordered.len(),
            edges.len()
        ),
        "under the union of every feature. An arrow points from a crate to an internal".to_string(),
        "normal or build dependency. Development dependencies are excluded.".to_string(),
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
    for (source, destination, _) in &edges {
        lines.push(format!("  {} --> {}", node(source)?, node(destination)?));
    }
    lines.extend([
        "```".to_string(),
        String::new(),
        "## Resolved production edges".to_string(),
        String::new(),
        "| Consumer | Dependency | Seam crossed | Kinds | Conditions | Destination features | Optional | Default features | Activated by |".to_string(),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |".to_string(),
    ]);
    for (source, destination, use_) in &edges {
        lines.push(format!(
            "| `{source}` | `{destination}` | `{}` | {} | {} | {} | `{}` | `{}` | {} |",
            seam_of(&by_package, destination)?,
            format_list(&use_.kinds),
            format_list(&use_.conditions),
            format_list(&use_.features),
            use_.optional,
            use_.default_features,
            format_list(&use_.activating_features)
        ));
    }
    lines.extend([
        String::new(),
        "## Changing a dependency".to_string(),
        String::new(),
        "Change the Cargo manifest. The architecture manifest needs an edit only when the"
            .to_string(),
        "edge crosses a seam the destination's layer does not admit, when the curated".to_string(),
        "surface starts carrying a new seam, or when a member is added or moved.".to_string(),
        String::new(),
    ]);
    Ok(lines.join("\n"))
}

/// The per-crate ownership document.
pub fn render_ownership(
    records: &[CrateRecord],
    layers: &[LayerRecord],
    state: &WorkspaceState,
) -> Result<String, GateError> {
    let ordered = ordered(records);
    let by_package: BTreeMap<&str, &CrateRecord> = ordered
        .iter()
        .map(|record| (record.package.as_str(), *record))
        .collect();
    let rank: BTreeMap<&str, i64> = layers
        .iter()
        .map(|layer| (layer.name.as_str(), layer.rank))
        .collect();
    let mut lines = vec![
        "# Vyre Crate Ownership".to_string(),
        String::new(),
        format!("This file is generated by `{WRITE_COMMAND}` from"),
        "`docs/CRATE_OWNERSHIP.toml` and the workspace manifests.".to_string(),
        String::new(),
        "## Boundary rule".to_string(),
        String::new(),
        "Each workspace crate owns one concern and exposes one named seam. Every internal"
            .to_string(),
        "production dependency crosses the destination's seam, and the interface stated"
            .to_string(),
        "on that destination's row is what crossing it provides, for every consumer.".to_string(),
        String::new(),
        "## Per-crate ownership".to_string(),
        String::new(),
    ];
    for record in &ordered {
        let empty = BTreeMap::new();
        let outbound = state.dependencies.get(&record.package).unwrap_or(&empty);
        let mut inbound: Vec<(&str, &DependencyUse)> = Vec::new();
        for (source, destinations) in &state.dependencies {
            if let Some(use_) = destinations.get(&record.package) {
                if use_.is_production() {
                    inbound.push((source.as_str(), use_));
                }
            }
        }
        lines.extend([
            format!("### `{}`", record.package),
            String::new(),
            record.responsibility.clone(),
            String::new(),
            format!("- Path: `{}`", record.path),
            format!(
                "- Layer: `{}` (rank {})",
                record.layer,
                rank.get(record.layer.as_str())
                    .map_or_else(|| "unranked".to_string(), i64::to_string)
            ),
            format!("- Publication class: `{}`", record.publication_class),
            format!("- Seam: `{}`", record.seam),
            format!("- Interface: {}", record.interface),
            format!(
                "- Carried by the curated surface: `{}`",
                record.facade_exported
            ),
            format!("- Consumed by {} production edge(s)", inbound.len()),
            String::new(),
        ]);
        let production: Vec<(&String, &DependencyUse)> = outbound
            .iter()
            .filter(|(_, use_)| use_.is_production())
            .collect();
        if production.is_empty() {
            continue;
        }
        lines.extend([
            "| Dependency | Seam crossed | Interface | Activated by |".to_string(),
            "| --- | --- | --- | --- |".to_string(),
        ]);
        for (destination, use_) in production {
            let target = by_package.get(destination.as_str()).ok_or_else(|| {
                GateError::new(
                    format!("`{destination}` is named as a dependency and carries no manifest row"),
                    "declare the crate in docs/CRATE_OWNERSHIP.toml before rendering the document",
                )
            })?;
            lines.push(format!(
                "| `{destination}` | `{}` | {} | {} |",
                target.seam,
                target.interface,
                format_list(&use_.activating_features)
            ));
        }
        lines.push(String::new());
    }
    lines.extend([
        "## Changing a boundary".to_string(),
        String::new(),
        "1. Change the Cargo manifest.".to_string(),
        "2. Change `docs/CRATE_OWNERSHIP.toml` when the member set, a layer, a seam, a".to_string(),
        "   publication class or the curated surface changes.".to_string(),
        format!("3. Run `{WRITE_COMMAND}`."),
        "4. Add a public import migration test when the curated surface changes.".to_string(),
        String::new(),
    ]);
    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: a `workspace = true` entry that also names features enables both
    /// sets, so reading only the local list under-reports the edge.
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

    /// WHY: a renamed dependency key is the alias, not the destination, and
    /// every rule is keyed on the destination package.
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

    /// WHY: the two documents are the reviewable form of the manifest, so an
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

    fn manifest(text: &str) -> toml::Table {
        toml::from_str(text).expect("the manifest parses")
    }

    /// WHY: this is the hidden-edge class. An optional dependency a feature
    /// turns on is absent from the default resolution, and the destination
    /// features that feature enables are absent from the dependency table's
    /// own `features` key. A gate that read only the table would record the
    /// edge as carrying no features and no activation, which is the graph
    /// cargo resolves with `--no-default-features`, not the one it resolves
    /// with `--all-features`.
    #[test]
    fn a_feature_activated_edge_carries_its_activation_and_its_features() {
        let table = manifest(
            "[dependencies]\nvyre-libs = { path = \"../vyre-libs\", optional = true }\n\
             [features]\ndefault = []\nlibs-compositions = [\"dep:vyre-libs\", \"vyre-libs/encoding\"]\n\
             everything = [\"libs-compositions\"]\n",
        );
        let effect = feature_effect(&table, &BTreeSet::from(["vyre-libs".to_string()]));
        assert_eq!(
            effect.activated_by.get("vyre-libs"),
            Some(&BTreeSet::from([
                "everything".to_string(),
                "libs-compositions".to_string()
            ])),
            "a feature that names another feature activates what that one activates"
        );
        assert_eq!(
            effect.enabled.get("vyre-libs"),
            Some(&BTreeSet::from(["encoding".to_string()]))
        );
        assert!(effect.explicit.contains("vyre-libs"));
    }

    /// WHY: a weak `dep?/feature` enables a destination feature without
    /// activating the edge, so recording it as an activation would report an
    /// edge the build never resolves.
    #[test]
    fn a_weak_feature_reference_is_not_an_activation() {
        let table = manifest(
            "[dependencies]\nserde = { version = \"1\", optional = true }\n\
             [features]\njson = [\"serde?/derive\"]\n",
        );
        let effect = feature_effect(&table, &BTreeSet::from(["serde".to_string()]));
        assert!(effect.activated_by.get("serde").is_none());
        assert_eq!(
            effect.enabled.get("serde"),
            Some(&BTreeSet::from(["derive".to_string()]))
        );
    }

    /// WHY: cargo derives a feature named after an optional dependency unless
    /// some feature spells it `dep:`, so treating an unnamed optional entry as
    /// unreachable would report a live edge as dead.
    #[test]
    fn an_implicit_feature_activates_its_optional_dependency() {
        let table = manifest("[dependencies]\nureq = { version = \"2\", optional = true }\n");
        let effect = feature_effect(&table, &BTreeSet::from(["ureq".to_string()]));
        assert!(effect.explicit.is_empty());
        assert!(effect.activated_by.is_empty());
    }

    /// One consumer, one dependency, and the rows to judge them by.
    struct Case {
        records: Vec<CrateRecord>,
        layers: Vec<LayerRecord>,
        state: WorkspaceState,
    }

    impl Case {
        fn new(source_layer: &str, source_rank: i64, target_layer: &str, target_rank: i64) -> Self {
            let records = vec![
                CrateRecord {
                    package: "consumer".to_string(),
                    path: "consumer".to_string(),
                    layer: source_layer.to_string(),
                    publication_class: "internal-engine".to_string(),
                    seam: "consumer-seam".to_string(),
                    interface: "consume".to_string(),
                    responsibility: "consume".to_string(),
                    facade_exported: false,
                },
                CrateRecord {
                    package: "dependency".to_string(),
                    path: "dependency".to_string(),
                    layer: target_layer.to_string(),
                    publication_class: "internal-engine".to_string(),
                    seam: "dependency-seam".to_string(),
                    interface: "be consumed".to_string(),
                    responsibility: "be consumed".to_string(),
                    facade_exported: false,
                },
            ];
            let mut layers = vec![LayerRecord {
                name: source_layer.to_string(),
                rank: source_rank,
                purpose: "consume".to_string(),
                ..LayerRecord::default()
            }];
            if target_layer != source_layer {
                layers.push(LayerRecord {
                    name: target_layer.to_string(),
                    rank: target_rank,
                    purpose: "be consumed".to_string(),
                    ..LayerRecord::default()
                });
            }
            let state = WorkspaceState {
                members: vec!["consumer".to_string(), "dependency".to_string()],
                paths: BTreeMap::from([
                    ("consumer".to_string(), "consumer".to_string()),
                    ("dependency".to_string(), "dependency".to_string()),
                ]),
                dependencies: BTreeMap::from([
                    ("consumer".to_string(), BTreeMap::new()),
                    ("dependency".to_string(), BTreeMap::new()),
                ]),
                development: BTreeMap::from([
                    ("consumer".to_string(), BTreeMap::new()),
                    ("dependency".to_string(), BTreeMap::new()),
                ]),
            };
            Self {
                records,
                layers,
                state,
            }
        }

        fn edge(mut self, use_: DependencyUse) -> Self {
            self.state.dependencies.insert(
                "consumer".to_string(),
                BTreeMap::from([("dependency".to_string(), use_)]),
            );
            self
        }

        fn direction(&self) -> Vec<Finding> {
            direction_findings(&self.state, &self.records, &self.layers)
        }

        fn contract(&self) -> Vec<Finding> {
            contract_findings(&self.state, &self.records)
        }
    }

    fn production(kinds: &[&str]) -> DependencyUse {
        DependencyUse {
            kinds: kinds.iter().map(|kind| (*kind).to_string()).collect(),
            conditions: vec!["always".to_string()],
            default_features: true,
            ..DependencyUse::default()
        }
    }

    /// WHY: the rank comparison is the whole direction contract, so the case it
    /// exists for has to fail. A consumer in a layer the dependency's layer
    /// outranks is a reversal, and an equal rank across two layers is one too:
    /// two layers share a rank only when neither depends on the other.
    #[test]
    fn a_layer_reversal_is_a_finding() {
        let reversed = Case::new("low", 1, "high", 4)
            .edge(production(&["normal"]))
            .direction();
        assert_eq!(reversed.len(), 1, "{reversed:?}");
        assert!(
            reversed[0]
                .message
                .contains("`consumer` in layer `low` (rank 1) depends always on `dependency` in layer `high` (rank 4)"),
            "{reversed:?}"
        );
        let equal = Case::new("left", 3, "right", 3)
            .edge(production(&["normal"]))
            .direction();
        assert_eq!(equal.len(), 1, "{equal:?}");
        assert!(Case::new("high", 4, "low", 1)
            .edge(production(&["normal"]))
            .direction()
            .is_empty());
    }

    /// WHY: an edge that exists only when a feature is on is still an edge the
    /// build resolves, and it is the one a gate reading the default resolution
    /// never sees. It is held to the same rank rule, and the finding names the
    /// feature so a reader knows which build has it.
    #[test]
    fn a_feature_activated_reversal_is_a_finding_naming_the_feature() {
        let findings = Case::new("low", 1, "high", 4)
            .edge(DependencyUse {
                optional: true,
                activating_features: vec!["extra".to_string()],
                ..production(&["normal"])
            })
            .direction();
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(
            findings[0]
                .message
                .contains("depends under feature `extra` on `dependency`"),
            "{findings:?}"
        );
    }

    /// WHY: a dev-dependency on a higher layer is how a crate tests against the
    /// facade that consumes it. Cargo resolves it in a separate graph that
    /// cannot form a production cycle, so judging it would reject the intended
    /// shape. A build-dependency is not exempt: cargo links it.
    #[test]
    fn a_development_edge_carries_no_direction() {
        assert!(Case::new("low", 1, "high", 4)
            .edge(production(&["dev"]))
            .direction()
            .is_empty());
        assert_eq!(
            Case::new("low", 1, "high", 4)
                .edge(production(&["dev", "normal"]))
                .direction()
                .len(),
            1
        );
        assert_eq!(
            Case::new("low", 1, "high", 4)
                .edge(production(&["build"]))
                .direction()
                .len(),
            1
        );
    }

    /// WHY: a layer a crate row names and no `[[layer]]` row declares has no
    /// rank, so every edge touching it would be skipped rather than judged. The
    /// unranked layer itself is the finding, and so is a declared layer no
    /// member occupies: it is a rank nothing is held to.
    #[test]
    fn an_unmatched_layer_is_a_finding() {
        let mut case = Case::new("undeclared", 0, "undeclared", 0);
        case.layers = vec![LayerRecord {
            name: "empty".to_string(),
            rank: 0,
            purpose: "nothing".to_string(),
            ..LayerRecord::default()
        }];
        let findings = case.direction();
        assert_eq!(findings.len(), 3, "{findings:?}");
    }

    /// WHY: `consumed_by` is the one statement rank cannot make. Rank permits a
    /// runtime crate to reach a composition library; the admitted set is what
    /// decides whether it may. An edge from a layer the destination does not
    /// admit is the undeclared half, and an admitted layer no edge crosses is
    /// the stale half.
    #[test]
    fn an_unadmitted_consumer_layer_is_a_finding_and_an_unused_admission_is_too() {
        let mut case = Case::new("high", 4, "low", 1).edge(production(&["normal"]));
        case.layers[1].consumed_by = Some(vec!["other".to_string()]);
        case.layers.push(LayerRecord {
            name: "other".to_string(),
            rank: 3,
            purpose: "elsewhere".to_string(),
            ..LayerRecord::default()
        });
        case.records.push(CrateRecord {
            package: "other-member".to_string(),
            path: "other-member".to_string(),
            layer: "other".to_string(),
            publication_class: "internal-engine".to_string(),
            seam: "other-seam".to_string(),
            interface: "elsewhere".to_string(),
            responsibility: "elsewhere".to_string(),
            facade_exported: false,
        });
        let findings = case.direction();
        assert!(
            findings.iter().any(|finding| finding
                .message
                .contains("layer `low` admits no consumer in `high`")),
            "{findings:?}"
        );
        assert!(
            findings.iter().any(|finding| finding.message.contains(
                "layer `low` admits consumer layer `other` and no production edge crosses it"
            )),
            "{findings:?}"
        );
    }

    /// WHY: this is the facade rule, and rank cannot express it. The exporting
    /// layer outranks everything, so rank permits it to reach any crate in the
    /// workspace; the export flag on the destination is what keeps the curated
    /// surface from re-exporting an emitter. A flag no exporting edge crosses
    /// is a stale declaration.
    #[test]
    fn an_unexported_seam_reached_from_an_exporting_layer_is_a_finding() {
        let mut case = Case::new("facade", 6, "emitter", 2).edge(production(&["normal"]));
        case.layers[0].exports_declared_seams = true;
        let findings = case.direction();
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(
            findings[0].message.contains(
                "`consumer` exports declared seams and depends always on `dependency`, whose row does not set `facade_exported`"
            ),
            "{findings:?}"
        );

        case.records[1].facade_exported = true;
        assert!(case.direction().is_empty());

        let mut stale = Case::new("facade", 6, "emitter", 2);
        stale.layers[0].exports_declared_seams = true;
        stale.records[1].facade_exported = true;
        let findings = stale.direction();
        assert!(
            findings.iter().any(|finding| finding.message.contains(
                "`dependency` declares `facade_exported` and no exporting layer depends on it"
            )),
            "{findings:?}"
        );
    }

    /// WHY: a member cargo resolves and no row describes has no layer and no
    /// seam, so every rule silently skips it. Adding a workspace member has to
    /// turn this red.
    #[test]
    fn an_undeclared_member_is_a_finding() {
        let mut case = Case::new("high", 4, "low", 1);
        case.state.members.push("newcomer".to_string());
        case.state
            .paths
            .insert("newcomer".to_string(), "newcomer".to_string());
        let findings = case.contract();
        assert!(
            findings
                .iter()
                .any(|finding| finding.message == "workspace member `newcomer` has no manifest row"),
            "{findings:?}"
        );
        assert!(
            findings.iter().any(
                |finding| finding.message == "workspace package `newcomer` has no manifest row"
            ),
            "{findings:?}"
        );
    }

    /// WHY: a row describing no member is the surviving stale-declaration
    /// class. The per-edge roster it replaced could go stale one edge at a
    /// time; a member row can only go stale as a whole, and it does so loudly.
    #[test]
    fn a_row_that_is_not_a_member_is_a_finding() {
        let mut case = Case::new("high", 4, "low", 1);
        case.records.push(CrateRecord {
            package: "vyre-ghost".to_string(),
            path: "vyre-ghost".to_string(),
            layer: "low".to_string(),
            publication_class: "internal-engine".to_string(),
            seam: "ghost".to_string(),
            interface: "nothing".to_string(),
            responsibility: "nothing".to_string(),
            facade_exported: false,
        });
        let findings = case.contract();
        assert!(
            findings
                .iter()
                .any(|finding| finding.message
                    == "manifest row `vyre-ghost` is not a workspace member"),
            "{findings:?}"
        );
    }

    /// WHY: two rows claiming one seam name leave every edge into that seam
    /// crossing an interface with two owners, which is what the 22-package
    /// split produced: 23 packages owned `product-libraries`.
    #[test]
    fn two_rows_owning_one_seam_is_a_finding() {
        let mut case = Case::new("high", 4, "low", 1);
        case.records[1].seam = "consumer-seam".to_string();
        let findings = case.contract();
        assert!(
            findings.iter().any(|finding| finding
                .message
                .contains("`consumer` and `dependency` both own the `consumer-seam` seam")),
            "{findings:?}"
        );
    }

    /// WHY: the publication class decides whether a crate is published and
    /// whether a consumer may depend on it. It has one home, beside the
    /// `publish` key it qualifies, and this reader is the only thing that finds
    /// it: a class written into the architecture manifest instead is a retired
    /// key, not a second source, so nothing here may fall back to one.
    #[test]
    fn the_publication_class_is_read_only_from_the_member_manifest() {
        let declared: toml::Table = toml::from_str(
            "[package]\nname = \"a\"\n[package.metadata.vyre]\npublication_class = \"extension-sdk\"\n",
        )
        .expect("the fixture manifest parses");
        assert_eq!(
            manifest_publication_class(&declared).as_deref(),
            Some("extension-sdk")
        );

        for elsewhere in [
            "[package]\nname = \"a\"\n",
            "[package]\nname = \"a\"\npublication_class = \"extension-sdk\"\n",
            "[package]\nname = \"a\"\n[package.metadata]\npublication_class = \"extension-sdk\"\n",
            "[package]\nname = \"a\"\n[metadata.vyre]\npublication_class = \"extension-sdk\"\n",
        ] {
            let manifest: toml::Table =
                toml::from_str(elsewhere).expect("the fixture manifest parses");
            assert_eq!(
                manifest_publication_class(&manifest),
                None,
                "`{MANIFEST_PUBLICATION_KEY}` is the only home; `{elsewhere}` declares no class"
            );
        }
    }

    /// WHY: internal production dependency cycles, including intra-layer ones
    /// that the rank rule cannot see, must fail closed with a finding naming
    /// the exact cycle path.
    #[test]
    fn dependency_cycle_is_a_finding() {
        let state = WorkspaceState {
            members: vec!["a".to_string(), "b".to_string()],
            paths: BTreeMap::from([
                ("a".to_string(), "a".to_string()),
                ("b".to_string(), "b".to_string()),
            ]),
            development: BTreeMap::new(),
            dependencies: BTreeMap::from([
                (
                    "a".to_string(),
                    BTreeMap::from([("b".to_string(), production(&["normal"]))]),
                ),
                (
                    "b".to_string(),
                    BTreeMap::from([("a".to_string(), production(&["normal"]))]),
                ),
            ]),
        };
        let findings = cycle_findings(&state);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0]
            .message
            .contains("dependency cycle detected: a -> b -> a"));
    }

    /// WHY: an optional dependency no feature names is an edge no build
    /// resolves. It reads as a live dependency to anyone counting rebuild
    /// fan-out, and it is the shape a half-finished feature rename leaves.
    #[test]
    fn an_unactivatable_optional_edge_is_a_finding() {
        let state = WorkspaceState {
            members: vec!["a".to_string()],
            paths: BTreeMap::from([("a".to_string(), "a".to_string())]),
            development: BTreeMap::new(),
            dependencies: BTreeMap::from([(
                "a".to_string(),
                BTreeMap::from([(
                    "b".to_string(),
                    DependencyUse {
                        optional: true,
                        ..production(&["normal"])
                    },
                )]),
            )]),
        };
        let findings = activation_findings(&state);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0]
            .message
            .contains("`a` declares `b` optional and no feature activates it"));
    }
}
