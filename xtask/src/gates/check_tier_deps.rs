//! `cargo xtask check-tier-deps` - reject upward layer dependencies in workspace manifests.
//!
//! A crate may depend on its own architectural layer or on any layer below it,
//! never on one above. Each crate declares its layer in
//! `docs/CRATE_OWNERSHIP.toml`; this gate owns only the ordering between layers,
//! so adding a crate states its layer once, in the registry, and adding a layer
//! turns the gate red until its position is recorded here.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::process::{self, Command};

use toml::Value;

use crate::manifest_walk::MAX_MANIFEST_BYTES;

/// Architectural layers, most fundamental first. A crate may depend on its own
/// layer or on any earlier layer, never on a later one.
///
/// This is the only statement of the ordering. Which layer a crate belongs to is
/// read from `docs/CRATE_OWNERSHIP.toml`, so a rename or a new crate cannot make
/// this list stale.
///
/// `standalone-tooling` sits below `foundation` because it depends on no crate in
/// the workspace and must keep answering while the workspace does not compile,
/// which is what lets a test-support crate resolve the checkout root through it.
///
/// `registry-link` sits above `facade` and below `conformance` because the crate
/// that owns the inventory link anchors has to name every registration source,
/// including the concrete drivers, while still being callable from the
/// conformance and tooling crates that read those registries.
const LAYER_ORDER: &[&str] = &[
    "standalone-tooling",
    "foundation",
    "test-tooling",
    "primitives",
    "frontend",
    "lowering",
    "semantics",
    "libraries",
    "pass-engine",
    "compiler-boundary",
    "emitter",
    "backend-neutral",
    "concrete-backend",
    "runtime",
    "packaging",
    "facade",
    "registry-link",
    "conformance",
    "tooling",
];

/// Run the tier-dependency gate.
pub(crate) fn run(args: &[String]) {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!(
            "USAGE:\n  cargo xtask check-tier-deps\n\n\
             Fails on upward layer dependencies, undeclared production edges, incomplete crate ownership, or generated crate-documentation drift."
        );
        return;
    }
    if args.len() > 2 {
        eprintln!("Fix: check-tier-deps takes no arguments.");
        process::exit(2);
    }

    let root = crate::checkout::checkout_root();
    let members = workspace_members(&root);
    let mut failures = Vec::new();

    let layers = declared_layers(&root, &mut failures);
    let workspace_deps = workspace_dependency_packages(&root);
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
    let mut claimed: BTreeSet<&str> = BTreeSet::new();
    for (package, table) in &manifests {
        let Some(layer) = layers.get(package) else {
            failures.push(format!(
                "`{package}` is a workspace member with no entry in docs/CRATE_OWNERSHIP.toml; declare its layer there"
            ));
            continue;
        };
        claimed.insert(
            LAYER_ORDER[layer_rank(layer).expect("declared_layers rejects unknown layers")],
        );
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
    for layer in LAYER_ORDER {
        if !claimed.contains(layer) {
            failures.push(format!(
                "layer `{layer}` holds a position in the layer order and no crate declares it; remove the position or record the crate"
            ));
        }
    }
    validate_cross_crate_promotion_contract(&root, &mut failures);
    validate_crate_ownership_registry(&root, &mut failures);

    if failures.is_empty() {
        println!(
            "check-tier-deps: {} workspace members; layer, ownership, and generated graph contracts agree",
            members.len()
        );
    } else {
        eprintln!("check-tier-deps: {} violation(s):", failures.len());
        for f in &failures {
            eprintln!("  - {f}");
        }
        eprintln!(
            "Fix: remove the upward dependency, or move the crate to the layer that matches it in docs/CRATE_OWNERSHIP.toml, then regenerate the ownership docs."
        );
        process::exit(1);
    }
}

fn validate_crate_ownership_registry(root: &Path, failures: &mut Vec<String>) {
    let script = root.join("scripts/crate_ownership.py");
    let output = match Command::new("python3")
        .arg(&script)
        .arg("--check")
        .current_dir(root)
        .output()
    {
        Ok(output) => output,
        Err(error) => {
            failures.push(format!(
                "could not launch `{}` with python3: {error}",
                script.display()
            ));
            return;
        }
    };
    if output.status.success() {
        return;
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let message = stderr.trim();
    failures.push(if message.is_empty() {
        format!(
            "`{}` failed with status {} and no diagnostic",
            script.display(),
            output.status
        )
    } else {
        message.to_string()
    });
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

/// Position of a layer in [`LAYER_ORDER`], or `None` when the layer is unknown.
fn layer_rank(layer: &str) -> Option<usize> {
    LAYER_ORDER.iter().position(|known| *known == layer)
}

/// Each crate's declared architectural layer, read from the ownership registry.
///
/// A layer the registry names and [`LAYER_ORDER`] does not is a failure rather
/// than a default, so a new layer cannot be introduced without recording where
/// it sits.
fn declared_layers(root: &Path, failures: &mut Vec<String>) -> BTreeMap<String, String> {
    let path = root.join("docs/CRATE_OWNERSHIP.toml");
    let text = read_bounded(&path);
    let table = parse_toml(&path, &text);
    let mut layers = BTreeMap::new();
    let Some(entries) = table.get("crate").and_then(Value::as_array) else {
        failures.push("docs/CRATE_OWNERSHIP.toml declares no `[[crate]]` entries".to_string());
        return layers;
    };
    for entry in entries {
        let Some(package) = entry.get("package").and_then(Value::as_str) else {
            failures.push("a docs/CRATE_OWNERSHIP.toml entry declares no `package`".to_string());
            continue;
        };
        let Some(layer) = entry.get("layer").and_then(Value::as_str) else {
            failures.push(format!(
                "`{package}` declares no `layer` in docs/CRATE_OWNERSHIP.toml"
            ));
            continue;
        };
        if layer_rank(layer).is_none() {
            failures.push(format!(
                "`{package}` declares layer `{layer}`, which holds no position in the layer order; record where it sits relative to the existing layers"
            ));
            continue;
        }
        layers.insert(package.to_string(), layer.to_string());
    }
    layers
}

/// Package name a workspace member publishes, which is what a dependency names.
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

/// Report every production dependency that climbs to a later layer.
///
/// Dev-dependencies are exempt: a contract test legitimately drives its own
/// crate through a backend or the facade, and that edge is absent from anything
/// a consumer builds.
fn scan_manifest(
    package: &str,
    layer: &str,
    layers: &BTreeMap<String, String>,
    members: &BTreeSet<&str>,
    workspace_deps: &BTreeMap<String, String>,
    table: &Value,
    failures: &mut Vec<String>,
) {
    let rank = layer_rank(layer).expect("declared_layers rejects unknown layers");
    for dep_kind in ["dependencies", "build-dependencies"] {
        let Some(deps) = table.get(dep_kind).and_then(Value::as_table) else {
            continue;
        };
        for (key, value) in deps {
            let dep = dep_package(key, value, workspace_deps);
            if !members.contains(dep.as_str()) {
                continue;
            }
            let Some(dep_layer) = layers.get(&dep) else {
                continue;
            };
            let dep_rank = layer_rank(dep_layer).expect("declared_layers rejects unknown layers");
            if dep_rank > rank {
                failures.push(format!(
                    "{package} ({layer}) must not depend on {dep} ({dep_layer}) via `{key}` in {dep_kind}"
                ));
            }
        }
    }
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
    // The generated crate graph proves the dependency surface exists and is
    // fresh (crate_ownership.py --check); the LEGO rule owns the promotion
    // contract text, so the marker requirement applies to the rule doc.
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

fn read_bounded(path: &Path) -> String {
    crate::output_arg::read_text_bounded(path, MAX_MANIFEST_BYTES, "tier dependency manifest")
        .unwrap_or_else(|error| {
            panic!("Fix: cannot read {}: {error}", path.display());
        })
}

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

    fn fixture_layers() -> BTreeMap<String, String> {
        BTreeMap::from([
            ("vyre-primitives".to_string(), "primitives".to_string()),
            ("vyre-driver".to_string(), "backend-neutral".to_string()),
        ])
    }

    fn fixture_members() -> BTreeSet<&'static str> {
        BTreeSet::from(["vyre-primitives", "vyre-driver"])
    }

    fn scan(manifest: &str) -> Vec<String> {
        let table = parse_toml(Path::new("fixture/Cargo.toml"), manifest);
        let layers = fixture_layers();
        let members = fixture_members();
        let workspace_deps =
            BTreeMap::from([("vyre-driver".to_string(), "vyre-driver".to_string())]);
        let mut failures = Vec::new();
        scan_manifest(
            "vyre-primitives",
            "primitives",
            &layers,
            &members,
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
                "vyre-primitives (primitives) must not depend on vyre-driver (backend-neutral)"
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
        let mut failures = Vec::new();
        scan_manifest(
            "vyre-driver",
            "backend-neutral",
            &fixture_layers(),
            &fixture_members(),
            &BTreeMap::from([("vyre-primitives".to_string(), "vyre-primitives".to_string())]),
            &table,
            &mut failures,
        );

        assert!(failures.is_empty(), "{failures:?}");
    }

    /// Every layer named in the registry must hold a position, so a new layer
    /// cannot arrive with an implicit default rank.
    #[test]
    fn every_declared_layer_holds_a_position_in_the_order() {
        let root = crate::checkout::checkout_root();
        let mut failures = Vec::new();
        let layers = declared_layers(&root, &mut failures);

        assert!(failures.is_empty(), "{failures:?}");
        assert!(!layers.is_empty());
        for (package, layer) in &layers {
            assert!(
                layer_rank(layer).is_some(),
                "`{package}` declares unranked layer `{layer}`"
            );
        }
    }
}
