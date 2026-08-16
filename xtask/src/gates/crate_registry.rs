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

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
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
const SCHEMA_VERSION: i64 = 2;

/// What a caller does about any disagreement this gate reports.
const FIX: &str = "change the manifest and its `[[crate.dependency]]` record together, then run `xtask crate-ownership --write`";

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

impl Gate for CrateOwnership {
    fn name(&self) -> &'static str {
        "crate-ownership"
    }

    fn help(&self) -> &'static str {
        "Hold the ownership registry to the workspace manifests and docs/CRATE_GRAPH.md and docs/OWNERSHIP.md to the registry; --write regenerates both"
    }

    fn generates(&self) -> bool {
        true
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        let records = load_registry(&tree, &mut report)?;
        let state = workspace_state(&tree)?;
        report
            .findings
            .extend(contract_findings(&state, &records));
        report.note(format!(
            "{} registry row(s) across {} workspace member(s)",
            records.len(),
            state.members.len()
        ));

        // A registry that does not describe this workspace cannot render a
        // document about it, and writing one from a broken registry publishes
        // the break. The rendered pair is judged only once the contract holds.
        if !report.findings.is_empty() {
            return Ok(report);
        }
        for (path, rendered) in [
            (GRAPH, render_graph(&records)?),
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
    if actual == rendered {
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
            format!("{context} declares boundary `{boundary}`, which is neither public nor private"),
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
        records.push(CrateRecord {
            package: text(row, "package", &context, report),
            path: text(row, "path", &context, report),
            owner: text(row, "owner", &context, report),
            layer: text(row, "layer", &context, report),
            responsibility: text(row, "responsibility", &context, report),
            dependencies,
        });
    }
    Ok(records)
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
    let local: Vec<String> = table
        .get("features")
        .and_then(Value::as_array)
        .map(|array| {
            array
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
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
    table
        .get("features")
        .and_then(Value::as_array)
        .map(|array| {
            array
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
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
                entry.optional = entry.optional
                    || merged.get("optional").and_then(Value::as_bool) == Some(true);
                entry.default_features = entry.default_features
                    && merged
                        .get("default-features")
                        .and_then(Value::as_bool)
                        .unwrap_or(true);
            }
        }
        for edge in edges.values_mut() {
            for list in [
                &mut edge.features,
                &mut edge.conditions,
                &mut edge.kinds,
            ] {
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
        let actual = state
            .dependencies
            .get(package)
            .cloned()
            .unwrap_or_default();
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
    if values.is_empty() {
        return "None".to_string();
    }
    values
        .iter()
        .map(|value| format!("`{value}`"))
        .collect::<Vec<_>>()
        .join(", ")
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
pub fn render_graph(records: &[CrateRecord]) -> Result<String, GateError> {
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
        "the workspace manifests and `docs/CRATE_OWNERSHIP.toml`. Edit those authorities".to_string(),
        "together, then regenerate this file.".to_string(),
        String::new(),
        "## Workspace dependency graph".to_string(),
        String::new(),
        format!(
            "The workspace contains {} crates. An arrow points from a crate to",
            ordered.len()
        ),
        "an internal normal or build dependency. Development dependencies are excluded.".to_string(),
        String::new(),
        "```mermaid".to_string(),
        "graph TD".to_string(),
    ];
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
        "the same patch. The registry rejects undeclared packages, feature drift, target".to_string(),
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
        "Each workspace crate has one owner and responsibility. Each internal production".to_string(),
        "edge declares why it exists, its Cargo feature and target conditions, whether it".to_string(),
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
}
