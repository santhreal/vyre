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

mod render;
mod workspace;

#[cfg(test)]
mod tests;

pub use render::{render_graph, render_ownership};
use workspace::manifest_publication_class;
pub use workspace::workspace_state;

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
    /// with a production edge. Required on every row, including the layer's
    /// own name when its members reach each other, so no layer pair is
    /// admitted without a row recording the decision.
    pub consumed_by: Vec<String>,
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
    /// True when some feature of the source names the optional edge, as
    /// `dep:alias`, as `alias/feature`, or as a bare alias. False for a
    /// required edge and for one reachable only through the feature cargo
    /// derives from the dependency key.
    pub named_activation: bool,
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

/// Read one required string-array field, sorted and duplicate-free.
///
/// An absent key is a finding rather than a permissive default. A layer that
/// stated nothing used to admit every consumer the rank rule allowed, so
/// fifteen of eighteen layers accepted a new edge from anywhere above them
/// with no row recording the decision. An empty array still states that
/// nothing may reach the layer in production, and it has to be written down.
fn required_strings(row: &Value, field: &str, context: &str, report: &mut Report) -> Vec<String> {
    let Some(value) = row.get(field) else {
        report.find(Finding::in_file(
            REGISTRY,
            format!("{context} declares no `{field}` array"),
            "record every layer whose members reach this one with a production edge, or `[]` when none may",
        ));
        return Vec::new();
    };
    let Some(array) = value.as_array() else {
        report.find(Finding::in_file(
            REGISTRY,
            format!("{context} `{field}` is not an array of layer names"),
            FIX,
        ));
        return Vec::new();
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
            consumed_by: required_strings(row, "consumed_by", &context, report),
            exports_declared_seams: flag(row, "exports_declared_seams", &context, report),
        });
    }
    Ok(layers)
}

/// Every way the resolved graph disagrees with the declared layer DAG.
///
/// Four rules over one graph, and each rejects something the others cannot.
///
/// Rank is the direction rule: a production edge points down when the source
/// layer outranks the destination layer, and an edge inside one layer points
/// nowhere. A reversal fails it directly, and a cross-layer cycle cannot be
/// written down at all, because a cycle needs at least one edge whose source
/// does not outrank its destination. Nothing enumerates permitted layer
/// pairs, so there is no second roster to drift from the manifests.
///
/// `consumed_by` is the admission rule, and it closes what rank leaves open.
/// Rank says a runtime crate may reach a composition library and says nothing
/// about whether it should; a same-layer edge it does not judge at all. Every
/// layer states the closed set of consumer layers admitted to it, its own name
/// included when its members reach each other, so a production edge is legal
/// only when both rules pass. A pair no row records fails, which is how a new
/// edge between two existing crates stops for a decision instead of expanding
/// rebuild fan-out silently. An entry no edge crosses is a stale declaration
/// and fails as loudly.
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
        for admitted in &layer.consumed_by {
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
            if target_layer
                .consumed_by
                .iter()
                .any(|name| name == &source_layer.name)
            {
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
        for admitted in &layer.consumed_by {
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

/// Every optional edge no feature of its consumer names.
///
/// An optional dependency is reachable through a feature that names it as
/// `dep:alias`, as `alias/feature`, or as a bare alias, and otherwise only
/// through the feature cargo derives from the dependency key. The derived one
/// is a public feature of the crate that no line of the manifest declares, so
/// nothing states what turning it on means and the `--all-features` graph
/// carries an edge the feature table never mentions.
///
/// The rule read `activating_features` before, which holds the derived feature
/// too, so the emptiness it tested was unreachable: cargo derives that feature
/// for exactly the edges no feature names. It certified a shape it could not
/// observe.
fn activation_findings(state: &WorkspaceState) -> Vec<Finding> {
    let mut findings = Vec::new();
    for (package, destinations) in &state.dependencies {
        for (destination, use_) in destinations {
            if use_.optional && !use_.named_activation {
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
