//! `cargo xtask check-tier-deps` - reject upward layer dependencies in workspace manifests.
//!
//! A crate may depend on its own architectural layer or on any layer below it,
//! never on one above. `docs/CRATE_OWNERSHIP.toml` states both halves of that:
//! a `[[crate]]` row gives each member its layer and a `[[layer]]` row gives
//! each layer its rank. This gate states neither and reads both, so adding a
//! crate or a layer is one edit in the manifest.
//!
//! `crate-ownership` judges the same direction rule over the feature-unified
//! graph it resolves, and both call [`crate_registry::edge_points_down`] so one
//! comparison answers for both. What this gate adds is the location: it reports
//! the manifest table and the dependency key that declares the upward edge,
//! which is the line a fix edits, and it reads the target-conditional tables
//! under the same rule so a `[target.'cfg(...)'.dependencies]` entry is not a
//! way around it.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use toml::Value;

use crate::gate::{GateCtx, GateError, Report};
use crate::gates::crate_registry;
use crate::gates::scan::Tree;
use crate::manifest_walk::MAX_MANIFEST_BYTES;

/// What an upward edge or an unclaimed layer costs the reader.
const FIX: &str = "remove the upward dependency, or move the crate to the layer that matches it in docs/CRATE_OWNERSHIP.toml, then regenerate the ownership docs";

/// One member's declared layer, carrying the rank that orders it.
///
/// A declared layer string is resolved once, where the manifest is read, and
/// every later comparison uses this value. Carrying the rank instead of the
/// string is what makes an unranked layer unrepresentable downstream.
#[derive(Clone, Copy)]
struct Layer<'a> {
    /// Rank the `[[layer]]` row declares. A destination must rank lower.
    rank: i64,
    /// Layer name, as the manifest spells it.
    name: &'a str,
}

/// Holds every crate to the layer it declares and to the layers below it.
pub struct CheckTierDeps;

impl crate::gate::GateBehavior for CheckTierDeps {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let root = &ctx.root;
        let tree = Tree::open(root)?;
        let members = workspace_members(root);
        let mut failures = Vec::new();

        let ranks = crate_registry::declared_layer_ranks(&tree)?;
        let layers = declared_layers(&tree, &ranks, &mut failures)?;
        let workspace_deps = workspace_dependency_packages(root);
        let mut packages = BTreeMap::new();
        let mut manifests = Vec::new();
        for member in &members {
            let manifest = root.join(member).join("Cargo.toml");
            let text = read_bounded(&manifest);
            let table = parse_toml(&manifest, &text);
            let package = package_name(&manifest, &table);
            packages.insert(package.clone(), member.clone());
            manifests.push((package, table));
        }
        let members_by_package: BTreeSet<&str> = packages.keys().map(String::as_str).collect();
        for (package, table) in &manifests {
            let Some(&layer) = layers.get(package) else {
                failures.push(format!(
                    "`{package}` is a workspace member with no entry in docs/CRATE_OWNERSHIP.toml; declare its layer there"
                ));
                continue;
            };
            scan_manifest(
                package,
                layer,
                &layers,
                &members_by_package,
                &workspace_deps,
                table,
                &mut failures,
            );
        }
        validate_cross_crate_promotion_contract(root, &mut failures);

        let mut report = Report::from_messages(failures, FIX);
        report.cover_complete("workspace members", members.len());
        report.note(format!(
            "{} workspace members across {} declared layers",
            members.len(),
            ranks.len()
        ));
        Ok(report)
    }
}

fn workspace_members(root: &Path) -> Vec<String> {
    let text = read_bounded(&root.join("Cargo.toml"));
    let table = parse_toml(&root.join("Cargo.toml"), &text);
    table
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Each member's declared layer, resolved to the rank that orders it.
///
/// Both halves come from `docs/CRATE_OWNERSHIP.toml` through
/// [`crate_registry`], which owns that file. A row whose layer carries no
/// `[[layer]]` rank is a failure rather than a default, so a new layer cannot
/// be introduced without recording where it sits.
fn declared_layers<'a>(
    tree: &Tree,
    ranks: &'a BTreeMap<String, i64>,
    failures: &mut Vec<String>,
) -> Result<BTreeMap<String, Layer<'a>>, GateError> {
    let mut layers = BTreeMap::new();
    for declared in crate_registry::declared_crates(tree)? {
        let Some((name, &rank)) = ranks.get_key_value(&declared.layer) else {
            failures.push(format!(
                "`{}` declares layer `{}`, which no `[[layer]]` row in docs/CRATE_OWNERSHIP.toml ranks; record where it sits relative to the existing layers",
                declared.package, declared.layer
            ));
            continue;
        };
        layers.insert(
            declared.package,
            Layer {
                rank,
                name: name.as_str(),
            },
        );
    }
    Ok(layers)
}

/// Package name a workspace member publishes, which is what a dependency names.
///
/// # Panics
///
/// Panics when the manifest table does not contain a `[package]` name.
fn package_name(manifest: &Path, table: &Value) -> String {
    table
        .get("package")
        .and_then(|package| package.get("name"))
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("Fix: {} declares no [package] name", manifest.display()))
        .to_string()
}

/// Package each `[workspace.dependencies]` key resolves to, so a member written
/// as `dep.workspace = true` is checked like any other edge.
fn workspace_dependency_packages(root: &Path) -> BTreeMap<String, String> {
    let path = root.join("Cargo.toml");
    let text = read_bounded(&path);
    let table = parse_toml(&path, &text);
    let mut packages = BTreeMap::new();
    let Some(deps) = table
        .get("workspace")
        .and_then(|workspace| workspace.get("dependencies"))
        .and_then(Value::as_table)
    else {
        return packages;
    };
    for (key, value) in deps {
        let package = value
            .get("package")
            .and_then(Value::as_str)
            .unwrap_or(key.as_str());
        packages.insert(key.clone(), package.to_string());
    }
    packages
}

/// Package a dependency entry names, resolving renames and workspace inheritance.
fn dep_package(key: &str, value: &Value, workspace_deps: &BTreeMap<String, String>) -> String {
    if let Some(package) = value.get("package").and_then(Value::as_str) {
        return package.to_string();
    }
    if value.get("workspace").and_then(Value::as_bool) == Some(true) {
        if let Some(package) = workspace_deps.get(key) {
            return package.clone();
        }
    }
    key.to_string()
}

/// Report every production dependency that climbs to a layer the manifest
/// ranks at or above the consumer's own.
///
/// Dev-dependencies are exempt: a contract test legitimately drives its own
/// crate through a backend or the facade, and that edge is absent from anything
/// a consumer builds. Every other dependency table is in scope, including the
/// target-conditional forms, because an edge declared under a `cfg` still ships
/// on the target it names.
fn scan_manifest(
    package: &str,
    layer: Layer<'_>,
    layers: &BTreeMap<String, Layer<'_>>,
    members: &BTreeSet<&str>,
    workspace_deps: &BTreeMap<String, String>,
    table: &Value,
    failures: &mut Vec<String>,
) {
    for (deps, dep_kind) in production_tables(table) {
        for (key, value) in deps {
            let dep = dep_package(key, value, workspace_deps);
            if !members.contains(dep.as_str()) {
                continue;
            }
            let Some(dep_layer) = layers.get(&dep) else {
                continue;
            };
            if crate_registry::edge_points_down(
                layer.name,
                layer.rank,
                dep_layer.name,
                dep_layer.rank,
            ) {
                continue;
            }
            failures.push(format!(
                "{package} ({}, rank {}) must not depend on {dep} ({}, rank {}) via `{key}` in {dep_kind}",
                layer.name, layer.rank, dep_layer.name, dep_layer.rank
            ));
        }
    }
}

/// Every production dependency table of one manifest, named as a reader finds
/// it.
///
/// A `[target.'cfg(...)'.dependencies]` table is reported under its own name so
/// the failure points at the table that declares the edge rather than at the
/// plain one that does not.
fn production_tables(table: &Value) -> Vec<(&toml::map::Map<String, Value>, String)> {
    let mut tables = Vec::new();
    for kind in ["dependencies", "build-dependencies"] {
        if let Some(deps) = table.get(kind).and_then(Value::as_table) {
            tables.push((deps, kind.to_string()));
        }
    }
    if let Some(targets) = table.get("target").and_then(Value::as_table) {
        for (condition, entry) in targets {
            for kind in ["dependencies", "build-dependencies"] {
                if let Some(deps) = entry.get(kind).and_then(Value::as_table) {
                    tables.push((deps, format!("target.'{condition}'.{kind}")));
                }
            }
        }
    }
    tables
}

fn validate_cross_crate_promotion_contract(root: &Path, failures: &mut Vec<String>) {
    let crate_graph = read_contract_doc(root, "docs/CRATE_GRAPH.md", failures);
    let lego_rule = read_contract_doc(root, "docs/lego-block-rule.md", failures);
    failures.extend(cross_crate_promotion_contract_text_failures(
        crate_graph.as_deref().unwrap_or(""),
        lego_rule.as_deref().unwrap_or(""),
    ));
}

fn read_contract_doc(root: &Path, rel: &str, failures: &mut Vec<String>) -> Option<String> {
    let path = root.join(rel);
    match fs::read_to_string(&path) {
        Ok(text) => Some(text),
        Err(error) => {
            failures.push(format!(
                "cross-crate promotion contract could not read `{rel}`: {error}"
            ));
            None
        }
    }
}

fn cross_crate_promotion_contract_text_failures(crate_graph: &str, lego_rule: &str) -> Vec<String> {
    let mut failures = Vec::new();
    // The generated crate graph proves the dependency surface exists; the
    // `crate-ownership` gate owns its freshness, and the LEGO rule owns the
    // promotion contract text, so the marker requirement applies to the rule doc.
    if crate_graph.is_empty() {
        failures.push("docs/CRATE_GRAPH.md is empty or unreadable".to_string());
    }
    for marker in [
        "Cross-crate promotion patch contract",
        "import-path migration test",
        "check-tier-deps",
        "lego-audit",
    ] {
        if !lego_rule.contains(marker) {
            failures.push(format!(
                "docs/lego-block-rule.md is missing `{marker}` for cross-crate promotion ownership"
            ));
        }
    }
    failures
}

/// Read a manifest file within bounded size.
///
/// # Panics
///
/// Panics when the manifest file cannot be read or exceeds `MAX_MANIFEST_BYTES`.
fn read_bounded(path: &Path) -> String {
    crate::output_arg::read_text_bounded(path, MAX_MANIFEST_BYTES, "tier dependency manifest")
        .unwrap_or_else(|error| {
            panic!("Fix: cannot read {}: {error}", path.display());
        })
}

/// Parse a manifest TOML string into a Value table.
///
/// # Panics
///
/// Panics when `text` is not valid TOML.
fn parse_toml(path: &Path, text: &str) -> Value {
    let table: toml::Table = toml::from_str(text).unwrap_or_else(|e| {
        panic!("Fix: parse {}: {e}", path.display());
    });
    Value::Table(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_DOC: &str = "Cross-crate promotion patch contract\nimport-path migration test\ncheck-tier-deps\nlego-audit\n";

    #[test]
    fn cross_crate_promotion_contract_accepts_complete_docs() {
        assert!(cross_crate_promotion_contract_text_failures("graph", VALID_DOC).is_empty());
    }

    #[test]
    fn cross_crate_promotion_contract_rejects_missing_markers() {
        let failures =
            cross_crate_promotion_contract_text_failures("graph", "check-tier-deps\nlego-audit\n");

        assert!(failures
            .iter()
            .any(|failure| failure.contains("import-path migration test")));
        assert!(failures
            .iter()
            .any(|failure| failure.contains("Cross-crate promotion patch contract")));
    }

    #[test]
    fn cross_crate_promotion_contract_rejects_missing_graph() {
        let failures = cross_crate_promotion_contract_text_failures("", VALID_DOC);

        assert!(failures
            .iter()
            .any(|failure| failure.contains("docs/CRATE_GRAPH.md")));
    }
}

#[cfg(test)]
mod dependency_kind_tests {
    use super::*;

    /// Ranks the fixtures are judged against, mirroring the shape the manifest
    /// declares: a lower number is a lower layer.
    fn fixture_ranks() -> BTreeMap<String, i64> {
        BTreeMap::from([
            ("primitives".to_string(), 1),
            ("backend-neutral".to_string(), 4),
        ])
    }

    fn fixture_layer<'a>(ranks: &'a BTreeMap<String, i64>, name: &str) -> Layer<'a> {
        let (name, &rank) = ranks
            .get_key_value(name)
            .unwrap_or_else(|| panic!("`{name}` must carry a fixture rank"));
        Layer {
            rank,
            name: name.as_str(),
        }
    }

    fn fixture_layers<'a>(ranks: &'a BTreeMap<String, i64>) -> BTreeMap<String, Layer<'a>> {
        BTreeMap::from([
            (
                "vyre-primitives".to_string(),
                fixture_layer(ranks, "primitives"),
            ),
            (
                "vyre-driver".to_string(),
                fixture_layer(ranks, "backend-neutral"),
            ),
        ])
    }

    fn fixture_members() -> BTreeSet<&'static str> {
        BTreeSet::from(["vyre-primitives", "vyre-driver"])
    }

    fn scan(manifest: &str) -> Vec<String> {
        let table = parse_toml(Path::new("fixture/Cargo.toml"), manifest);
        let ranks = fixture_ranks();
        let workspace_deps =
            BTreeMap::from([("vyre-driver".to_string(), "vyre-driver".to_string())]);
        let mut failures = Vec::new();
        scan_manifest(
            "vyre-primitives",
            fixture_layer(&ranks, "primitives"),
            &fixture_layers(&ranks),
            &fixture_members(),
            &workspace_deps,
            &table,
            &mut failures,
        );
        failures
    }

    #[test]
    fn production_upward_dependency_fails() {
        let failures = scan("[dependencies]\nvyre-driver = { path = \"../vyre-driver\" }\n");

        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(
            failures[0].contains(
                "vyre-primitives (primitives, rank 1) must not depend on vyre-driver (backend-neutral, rank 4)"
            ),
            "{failures:?}"
        );
    }

    /// A `dep.workspace = true` edge carries no path, and reading only inline
    /// `path` entries left almost every real dependency unjudged.
    #[test]
    fn production_upward_workspace_inherited_dependency_fails() {
        let failures = scan("[dependencies]\nvyre-driver.workspace = true\n");

        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(failures[0].contains("in dependencies"), "{failures:?}");
    }

    /// WHY: an edge declared under a `cfg` still links on the target it names,
    /// so a target table is not a way around the rule. Only the plain tables
    /// were read before, and the one non-`always` production edge in this tree
    /// is declared exactly this way.
    #[test]
    fn a_target_conditional_upward_dependency_fails_and_names_its_table() {
        let failures = scan(
            "[target.'cfg(not(target_os = \"macos\"))'.dependencies]\nvyre-driver.workspace = true\n",
        );

        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(
            failures[0]
                .contains("in target.'cfg(not(target_os = \"macos\"))'.dependencies"),
            "{failures:?}"
        );
    }

    #[test]
    fn dev_upward_dependency_is_allowed_for_contract_tests() {
        let failures = scan("[dev-dependencies]\nvyre-driver.workspace = true\n");

        assert!(failures.is_empty(), "{failures:?}");
    }

    #[test]
    fn downward_dependency_is_allowed() {
        let table = parse_toml(
            Path::new("fixture/Cargo.toml"),
            "[dependencies]\nvyre-primitives.workspace = true\n",
        );
        let ranks = fixture_ranks();
        let mut failures = Vec::new();
        scan_manifest(
            "vyre-driver",
            fixture_layer(&ranks, "backend-neutral"),
            &fixture_layers(&ranks),
            &fixture_members(),
            &BTreeMap::from([("vyre-primitives".to_string(), "vyre-primitives".to_string())]),
            &table,
            &mut failures,
        );

        assert!(failures.is_empty(), "{failures:?}");
    }

    /// WHY: two layers sharing a rank is the manifest stating that neither
    /// depends on the other, so an edge between them is upward. An equal rank
    /// read as permission while a second, total ordering existed beside the
    /// ranks; there is one ordering now and equality is a refusal.
    #[test]
    fn an_edge_between_two_layers_of_equal_rank_is_reported() {
        let table = parse_toml(
            Path::new("fixture/Cargo.toml"),
            "[dependencies]\nvyre-megakernel.workspace = true\n",
        );
        let ranks = BTreeMap::from([
            ("emitter".to_string(), 2),
            ("compiler-boundary".to_string(), 2),
        ]);
        let layers = BTreeMap::from([
            ("vyre-emit-ptx".to_string(), fixture_layer(&ranks, "emitter")),
            (
                "vyre-megakernel".to_string(),
                fixture_layer(&ranks, "compiler-boundary"),
            ),
        ]);
        let mut failures = Vec::new();
        scan_manifest(
            "vyre-emit-ptx",
            fixture_layer(&ranks, "emitter"),
            &layers,
            &BTreeSet::from(["vyre-emit-ptx", "vyre-megakernel"]),
            &BTreeMap::from([("vyre-megakernel".to_string(), "vyre-megakernel".to_string())]),
            &table,
            &mut failures,
        );

        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(
            failures[0].contains("(emitter, rank 2) must not depend on vyre-megakernel (compiler-boundary, rank 2)"),
            "{failures:?}"
        );
    }

    /// WHY: an edge inside one layer is legal, because a layer is a set of
    /// crates that may reach each other. The rank comparison alone would refuse
    /// it, and this gate and `crate-ownership` share the comparison, so the
    /// same-layer allowance has to hold in one place for both.
    #[test]
    fn an_edge_inside_one_layer_is_allowed() {
        let ranks = BTreeMap::from([("libraries".to_string(), 3)]);
        let table = parse_toml(
            Path::new("fixture/Cargo.toml"),
            "[dependencies]\nvyre-libs-nn.workspace = true\n",
        );
        let layers = BTreeMap::from([
            (
                "vyre-libs-math".to_string(),
                fixture_layer(&ranks, "libraries"),
            ),
            (
                "vyre-libs-nn".to_string(),
                fixture_layer(&ranks, "libraries"),
            ),
        ]);
        let mut failures = Vec::new();
        scan_manifest(
            "vyre-libs-math",
            fixture_layer(&ranks, "libraries"),
            &layers,
            &BTreeSet::from(["vyre-libs-math", "vyre-libs-nn"]),
            &BTreeMap::from([("vyre-libs-nn".to_string(), "vyre-libs-nn".to_string())]),
            &table,
            &mut failures,
        );

        assert!(failures.is_empty(), "{failures:?}");
    }

    /// WHY: this gate states no layer roster of its own, so every layer a
    /// `[[crate]]` row names has to be ranked by a `[[layer]]` row in the same
    /// file. Derived from the checkout at run time: adding a member in a layer
    /// nothing ranks turns this red without a test edit, and so does deleting a
    /// `[[layer]]` row that members still declare.
    #[test]
    fn every_declared_member_layer_carries_a_declared_rank() {
        let tree = Tree::open(&crate::checkout::checkout_root())
            .expect("Fix: the checkout must be readable");
        let ranks = crate_registry::declared_layer_ranks(&tree)
            .expect("Fix: docs/CRATE_OWNERSHIP.toml must declare [[layer]] rows");
        let mut failures = Vec::new();
        let layers = declared_layers(&tree, &ranks, &mut failures)
            .expect("Fix: docs/CRATE_OWNERSHIP.toml must declare [[crate]] rows");

        assert!(failures.is_empty(), "{failures:?}");
        assert_eq!(
            layers.len(),
            crate_registry::declared_crates(&tree)
                .expect("Fix: the manifest must declare [[crate]] rows")
                .len(),
            "every declared member must resolve to a ranked layer"
        );
        for (package, layer) in &layers {
            assert_eq!(
                ranks.get(layer.name).copied(),
                Some(layer.rank),
                "`{package}` resolved to a rank the manifest does not declare"
            );
        }

        let mut unranked = Vec::new();
        let none = BTreeMap::new();
        let unknown = declared_layers(&tree, &none, &mut unranked)
            .expect("Fix: the manifest must declare [[crate]] rows");
        assert!(
            unknown.is_empty() && unranked.len() == layers.len(),
            "an unranked layer must arrive as a failure rather than a default rank"
        );
    }
}
