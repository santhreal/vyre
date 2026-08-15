//! Whether the resolved crate graph stays inside the layering the ownership
//! registry declares.
//!
//! Two rules over one graph. A member may reach another member only if its
//! `docs/CRATE_OWNERSHIP.toml` entry allows it, directly or through a declared
//! edge. A member in a substrate-neutral layer may not reach a backend API crate
//! at all, whatever the intermediate was.
//!
//! The graph comes from the manifests and the lockfile, never from cargo. The
//! shell form ran `cargo tree` once per member and once more per violation, so a
//! workspace that did not resolve produced no verdict, and the `--edges=normal`
//! trees were the only statement of which edges counted. Manifest edges are read
//! with the member's own default features activated, which is the same edge set
//! `cargo tree` prints, and third-party edges come from `Cargo.lock`, so a
//! neutral crate that reaches a backend API through another third-party crate is
//! still caught.

use std::collections::{BTreeMap, BTreeSet};

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::manifest_contract::{dependency_hosts, entries, target_package};
use crate::gates::scan::Tree;

/// Production dependency tables. Development edges are excluded because a test
/// may depend upward deliberately, which is the same allowance `check-tier-deps`
/// makes.
const PRODUCTION_TABLES: &[&str] = &["dependencies", "build-dependencies"];

/// Whether each layer is substrate-neutral.
///
/// Every layer a member declares needs a decision here. A member whose layer is
/// missing is an unreviewed crate, and a decision no member uses is an allowance
/// nothing needs; both are fatal rather than reported, because either one makes
/// the neutrality half of the gate answer for a roster nobody checked.
const NEUTRAL_LAYERS: &[(&str, bool)] = &[
    ("backend-neutral", true),
    ("compiler-boundary", true),
    ("concrete-backend", false),
    ("conformance", false),
    ("emitter", false),
    ("facade", true),
    ("foundation", true),
    ("libraries", true),
    ("lowering", true),
    ("packaging", true),
    // Optimizer passes expressed as Vyre programs, dispatched through the
    // `ProgramDispatcher` seam, so the crate names no backend API.
    ("pass-engine", true),
    ("primitives", true),
    // Substrate-bound by function, not by accident: the link crate must name
    // every source whose registrations a build links, and the concrete drivers
    // are sources, so it reaches each backend API through them.
    ("registry-link", false),
    ("runtime", true),
    ("semantics", true),
    // `structure-gate` depends on no vyre crate, so it keeps running while the
    // workspace does not compile. Nothing it reads is substrate-bound.
    ("standalone-tooling", true),
    ("test-tooling", false),
    ("tooling", false),
];

/// Third-party crates that are the substrate boundary. A neutral crate reaching
/// one of these has crossed it whatever the intermediate was.
const BACKEND_APIS: &[&str] = &["ash", "cudarc", "metal", "naga", "wgpu"];

/// Every internal edge stays inside its declared closure, and no substrate-neutral
/// crate reaches a backend API.
pub struct Layering;

impl Gate for Layering {
    fn name(&self) -> &'static str {
        "layering"
    }

    fn help(&self) -> &'static str {
        "hold every member inside the dependency closure its ownership entry declares, and keep substrate-neutral layers away from backend API crates"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let graph = Graph::read(&tree)?;
        let registry = Registry::read(&tree, &graph.members)?;
        let mut report = Report::clean();
        let mut edges = 0usize;

        for member in &graph.members {
            let allowed = registry.closure(member);
            let internal = graph.reachable_members(member);
            edges += internal.len();
            for reached in &internal {
                if allowed.contains(reached) {
                    continue;
                }
                report.find(Finding::in_file(
                    format!("{}/Cargo.toml", graph.directory(member)),
                    format!(
                        "`{member}` reaches `{reached}`, which its ownership entry does not \
                         allow directly or through a declared edge: {}",
                        graph.path_to(member, reached)
                    ),
                    "remove the edge, or declare it in the docs/CRATE_OWNERSHIP.toml entry \
                     for the crate that owns it and regenerate the ownership docs",
                ));
            }
            if !registry.neutral(member) {
                continue;
            }
            for api in graph.reachable_backend_apis(member) {
                report.find(Finding::in_file(
                    format!("{}/Cargo.toml", graph.directory(member)),
                    format!(
                        "`{member}` is in the substrate-neutral layer `{}` and reaches the \
                         backend API crate `{api}`: {}",
                        registry.layer(member),
                        graph.path_to(member, &api)
                    ),
                    "move the code that needs the backend API into the crate that owns that \
                     backend, or move this crate out of the neutral layer",
                ));
            }
        }

        report.note(format!(
            "{edges} resolved internal edge(s) across {} workspace member(s)",
            graph.members.len()
        ));
        report.note(format!(
            "{} member(s) in a substrate-neutral layer, checked against {}",
            graph
                .members
                .iter()
                .filter(|member| registry.neutral(member))
                .count(),
            BACKEND_APIS.join(", ")
        ));
        Ok(report)
    }
}

/// The resolved dependency graph: member edges from the manifests, third-party
/// edges from the lockfile.
struct Graph {
    /// Member package names, in manifest order.
    members: BTreeSet<String>,
    /// Repository-relative directory per member package.
    directories: BTreeMap<String, String>,
    /// Direct production edges of each member, with default features activated.
    edges: BTreeMap<String, BTreeSet<String>>,
    /// Direct edges of each locked package, which is every crate the build can
    /// reach including the ones no manifest here names.
    locked: BTreeMap<String, BTreeSet<String>>,
}

impl Graph {
    /// Read the graph from the root manifest, the member manifests and the lockfile.
    fn read(tree: &Tree) -> Result<Self, GateError> {
        let root = tree.read_toml("Cargo.toml")?;
        let workspace = root
            .get("workspace")
            .and_then(toml::Value::as_table)
            .ok_or_else(|| {
                GateError::new(
                    "the root Cargo.toml declares no [workspace] table",
                    "restore the workspace table; a layering scan over no members reports \
                     success forever",
                )
            })?;
        let workspace_deps = workspace
            .get("dependencies")
            .and_then(toml::Value::as_table)
            .cloned()
            .unwrap_or_default();
        for api in BACKEND_APIS {
            if !workspace_deps.contains_key(*api) {
                return Err(GateError::new(
                    format!("backend API crate `{api}` is not in [workspace.dependencies]"),
                    "name the crate in [workspace.dependencies] or drop it from BACKEND_APIS \
                     in xtask/src/gates/layering.rs; a boundary named after a crate the \
                     workspace never resolves cannot be crossed",
                ));
            }
        }

        let mut members = BTreeSet::new();
        let mut directories = BTreeMap::new();
        let mut edges = BTreeMap::new();
        let listed = tree.members()?;
        if listed.is_empty() {
            return Err(GateError::new(
                "the root manifest declares no workspace members",
                "declare workspace.members; a layering scan over an empty roster reports \
                 success forever",
            ));
        }
        for directory in listed {
            let manifest = tree.read_toml(format!("{directory}/Cargo.toml"))?;
            let name = manifest
                .get("package")
                .and_then(toml::Value::as_table)
                .and_then(|package| package.get("name"))
                .and_then(toml::Value::as_str)
                .ok_or_else(|| {
                    GateError::new(
                        format!("{directory}/Cargo.toml declares no [package] name"),
                        "give the member a package name, or remove it from workspace.members",
                    )
                })?
                .to_string();
            edges.insert(name.clone(), direct_edges(&manifest, &workspace_deps));
            directories.insert(name.clone(), directory);
            members.insert(name);
        }

        Ok(Self {
            members,
            directories,
            edges,
            locked: locked_edges(&tree.read_toml("Cargo.lock")?),
        })
    }

    /// The directory a member's manifest sits in.
    fn directory(&self, member: &str) -> &str {
        self.directories
            .get(member)
            .map_or("Cargo.toml", String::as_str)
    }

    /// Every member the given member reaches through production edges, itself
    /// excluded.
    fn reachable_members(&self, member: &str) -> BTreeSet<String> {
        let mut reached = self.reach(member, false);
        reached.remove(member);
        reached.retain(|name| self.members.contains(name));
        reached
    }

    /// Every backend API crate the given member reaches, through members or
    /// through third-party crates.
    fn reachable_backend_apis(&self, member: &str) -> BTreeSet<String> {
        let reached = self.reach(member, true);
        BACKEND_APIS
            .iter()
            .filter(|api| reached.contains(**api))
            .map(|api| (*api).to_string())
            .collect()
    }

    /// Transitive closure over member edges, optionally following third-party
    /// edges out of the lockfile as well.
    fn reach(&self, member: &str, third_party: bool) -> BTreeSet<String> {
        let mut reached = BTreeSet::new();
        let mut pending: Vec<String> = self.next(member, third_party);
        while let Some(current) = pending.pop() {
            if !reached.insert(current.clone()) {
                continue;
            }
            pending.extend(self.next(&current, third_party));
        }
        reached
    }

    /// One node's outgoing edges.
    fn next(&self, package: &str, third_party: bool) -> Vec<String> {
        if let Some(edges) = self.edges.get(package) {
            return edges.iter().cloned().collect();
        }
        if !third_party {
            return Vec::new();
        }
        self.locked
            .get(package)
            .map(|edges| edges.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// The shortest edge chain from one package to another, rendered for a
    /// reader who has to remove one of its edges.
    ///
    /// A finding that names only the endpoints leaves the reader to rediscover
    /// the intermediate, which is the whole cost of a transitive violation.
    fn path_to(&self, from: &str, to: &str) -> String {
        let mut previous: BTreeMap<String, String> = BTreeMap::new();
        let mut queue: Vec<String> = vec![from.to_string()];
        let mut seen: BTreeSet<String> = [from.to_string()].into_iter().collect();
        let mut at = 0usize;
        while at < queue.len() {
            let current = queue[at].clone();
            at += 1;
            for next in self.next(&current, true) {
                if !seen.insert(next.clone()) {
                    continue;
                }
                previous.insert(next.clone(), current.clone());
                if next == to {
                    let mut chain = vec![to.to_string()];
                    let mut step = to.to_string();
                    while let Some(before) = previous.get(&step) {
                        chain.push(before.clone());
                        step = before.clone();
                    }
                    chain.reverse();
                    return chain.join(" -> ");
                }
                queue.push(next);
            }
        }
        format!("{from} -> ... -> {to}")
    }
}

/// The layering the ownership registry declares.
struct Registry {
    /// Directly declared edges per package.
    declared: BTreeMap<String, BTreeSet<String>>,
    /// Declared layer per package.
    layers: BTreeMap<String, String>,
    /// Neutrality decision per layer name.
    neutrality: BTreeMap<String, bool>,
}

impl Registry {
    /// Read the registry and hold it against the member roster.
    fn read(tree: &Tree, members: &BTreeSet<String>) -> Result<Self, GateError> {
        let table = tree.read_toml("docs/CRATE_OWNERSHIP.toml")?;
        let crates = table
            .get("crate")
            .and_then(toml::Value::as_array)
            .filter(|entries| !entries.is_empty())
            .ok_or_else(|| {
                GateError::new(
                    "docs/CRATE_OWNERSHIP.toml declares no [[crate]] entries",
                    "record each crate's layer and allowed internal edges; a closure read \
                     from an empty registry allows nothing and reports nothing",
                )
            })?;
        let mut declared = BTreeMap::new();
        let mut layers = BTreeMap::new();
        for entry in crates {
            let Some(package) = entry.get("package").and_then(toml::Value::as_str) else {
                return Err(GateError::new(
                    "a docs/CRATE_OWNERSHIP.toml [[crate]] entry declares no package",
                    "name the package the entry describes",
                ));
            };
            let edges: BTreeSet<String> = entry
                .get("dependency")
                .and_then(toml::Value::as_array)
                .map(|list| {
                    list.iter()
                        .filter_map(|dependency| dependency.get("package"))
                        .filter_map(toml::Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            declared.insert(package.to_string(), edges);
            layers.insert(
                package.to_string(),
                entry
                    .get("layer")
                    .and_then(toml::Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            );
        }

        let unregistered: Vec<&String> = members
            .iter()
            .filter(|member| !declared.contains_key(*member))
            .collect();
        if !unregistered.is_empty() {
            let names: Vec<&str> = unregistered.iter().map(|name| name.as_str()).collect();
            return Err(GateError::new(
                format!(
                    "workspace member(s) with no docs/CRATE_OWNERSHIP.toml entry: {}",
                    names.join(", ")
                ),
                "record the crate's layer and allowed internal edges in the registry, then \
                 run `python3 scripts/crate_ownership.py --write`; an unregistered crate has \
                 an empty closure, so every edge it has would report at once",
            ));
        }

        let neutrality: BTreeMap<String, bool> = NEUTRAL_LAYERS
            .iter()
            .map(|(layer, neutral)| ((*layer).to_string(), *neutral))
            .collect();
        let used: BTreeSet<&str> = members
            .iter()
            .map(|member| layers.get(member).map_or("", String::as_str))
            .collect();
        let undecided: Vec<&str> = used
            .iter()
            .filter(|layer| !neutrality.contains_key(**layer))
            .copied()
            .collect();
        if !undecided.is_empty() {
            return Err(GateError::new(
                format!(
                    "layer(s) a member declares with no neutrality decision: {}",
                    undecided.join(", ")
                ),
                "record whether the layer is substrate-neutral in NEUTRAL_LAYERS in \
                 xtask/src/gates/layering.rs; a layer with no decision would be skipped",
            ));
        }
        let stale: Vec<&str> = neutrality
            .keys()
            .map(String::as_str)
            .filter(|layer| !used.contains(*layer))
            .collect();
        if !stale.is_empty() {
            return Err(GateError::new(
                format!(
                    "neutrality decision(s) no member uses: {}",
                    stale.join(", ")
                ),
                "delete the entry from NEUTRAL_LAYERS in xtask/src/gates/layering.rs; a \
                 decision for a layer nobody declares records a rule that stopped covering \
                 anything",
            ));
        }

        Ok(Self {
            declared,
            layers,
            neutrality,
        })
    }

    /// Every package the given one may reach, through its own declared edges and
    /// through theirs.
    fn closure(&self, package: &str) -> BTreeSet<String> {
        let mut reached = BTreeSet::new();
        let mut pending: Vec<String> = self
            .declared
            .get(package)
            .map(|edges| edges.iter().cloned().collect())
            .unwrap_or_default();
        while let Some(current) = pending.pop() {
            if !reached.insert(current.clone()) {
                continue;
            }
            if let Some(edges) = self.declared.get(&current) {
                pending.extend(edges.iter().cloned());
            }
        }
        reached
    }

    /// The layer the registry gives a package.
    fn layer(&self, package: &str) -> &str {
        self.layers.get(package).map_or("", String::as_str)
    }

    /// Whether a package's layer is substrate-neutral.
    fn neutral(&self, package: &str) -> bool {
        self.neutrality
            .get(self.layer(package))
            .copied()
            .unwrap_or(false)
    }
}

/// One member's direct production edges, with its own default features activated.
///
/// An optional dependency is an edge only when a feature in the default closure
/// activates it, which is the edge set a plain build has. A weak activation
/// (`dep?/feature`) is not an activation.
fn direct_edges(manifest: &toml::Table, workspace_deps: &toml::Table) -> BTreeSet<String> {
    let activated = activated_dependencies(manifest);
    let mut edges = BTreeSet::new();
    for (_, host) in dependency_hosts(manifest) {
        for table in PRODUCTION_TABLES {
            for (key, spec) in entries(&host, table) {
                let optional = spec
                    .get("optional")
                    .and_then(toml::Value::as_bool)
                    .unwrap_or(false);
                if optional && !activated.contains(&key) {
                    continue;
                }
                edges.insert(target_package(&key, &spec, workspace_deps));
            }
        }
    }
    edges
}

/// Dependency keys the manifest's default features activate.
fn activated_dependencies(manifest: &toml::Table) -> BTreeSet<String> {
    let features = manifest
        .get("features")
        .and_then(toml::Value::as_table)
        .cloned()
        .unwrap_or_default();
    let mut enabled = BTreeSet::new();
    let mut pending = vec!["default".to_string()];
    let mut activated = BTreeSet::new();
    while let Some(feature) = pending.pop() {
        if !enabled.insert(feature.clone()) {
            continue;
        }
        // A feature named after an optional dependency activates it implicitly.
        activated.insert(feature.clone());
        let Some(list) = features.get(&feature).and_then(toml::Value::as_array) else {
            continue;
        };
        for entry in list.iter().filter_map(toml::Value::as_str) {
            if let Some(dependency) = entry.strip_prefix("dep:") {
                activated.insert(dependency.to_string());
            } else if let Some((dependency, _)) = entry.split_once('/') {
                match dependency.strip_suffix('?') {
                    // `dep?/feature` enables the feature only if something else
                    // already pulled the dependency in, so it is not an edge.
                    Some(_) => {}
                    None => {
                        activated.insert(dependency.to_string());
                        pending.push(dependency.to_string());
                    }
                }
            } else {
                pending.push(entry.to_string());
            }
        }
    }
    activated
}

/// Each locked package's direct edges, which is how a third-party crate's own
/// dependencies become visible without resolving features.
fn locked_edges(lock: &toml::Table) -> BTreeMap<String, BTreeSet<String>> {
    let mut edges: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let Some(packages) = lock.get("package").and_then(toml::Value::as_array) else {
        return edges;
    };
    for package in packages {
        let Some(name) = package.get("name").and_then(toml::Value::as_str) else {
            continue;
        };
        let entry = edges.entry(name.to_string()).or_default();
        let Some(dependencies) = package.get("dependencies").and_then(toml::Value::as_array) else {
            continue;
        };
        for dependency in dependencies.iter().filter_map(toml::Value::as_str) {
            // A lockfile edge is `name`, `name version`, or `name version source`.
            let dependency = dependency.split_whitespace().next().unwrap_or(dependency);
            entry.insert(dependency.to_string());
        }
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(text: &str) -> toml::Table {
        toml::from_str(text).expect("fixture parses")
    }

    #[test]
    fn an_optional_dependency_is_an_edge_only_when_a_default_feature_activates_it() {
        let manifest = table(
            r#"
            [features]
            default = ["fast"]
            fast = ["dep:present"]
            slow = ["dep:absent"]

            [dependencies]
            present = { version = "1", optional = true }
            absent = { version = "1", optional = true }
            always = "1"
            "#,
        );
        let edges = direct_edges(&manifest, &toml::Table::new());
        assert!(edges.contains("present"), "{edges:?}");
        assert!(edges.contains("always"), "{edges:?}");
        assert!(!edges.contains("absent"), "{edges:?}");
    }

    #[test]
    fn a_weak_activation_is_not_an_edge_and_a_bare_feature_name_is() {
        let manifest = table(
            r#"
            [features]
            default = ["weak", "bare"]
            weak = ["maybe?/extra"]
            bare = []

            [dependencies]
            maybe = { version = "1", optional = true }
            bare = { version = "1", optional = true }
            "#,
        );
        let edges = direct_edges(&manifest, &toml::Table::new());
        assert!(!edges.contains("maybe"), "{edges:?}");
        assert!(edges.contains("bare"), "{edges:?}");
    }

    #[test]
    fn a_renamed_workspace_dependency_resolves_to_the_package_it_names() {
        let workspace = table(
            r#"
            renamed = { version = "1", package = "real-name" }
            "#,
        );
        let manifest = table(
            r#"
            [dependencies]
            renamed = { workspace = true }
            "#,
        );
        let edges = direct_edges(&manifest, &workspace);
        assert!(edges.contains("real-name"), "{edges:?}");
    }

    #[test]
    fn development_edges_are_not_production_edges() {
        let manifest = table(
            r#"
            [dev-dependencies]
            harness = "1"

            [build-dependencies]
            generator = "1"
            "#,
        );
        let edges = direct_edges(&manifest, &toml::Table::new());
        assert!(!edges.contains("harness"), "{edges:?}");
        assert!(edges.contains("generator"), "{edges:?}");
    }

    #[test]
    fn a_lockfile_edge_keeps_only_the_package_name() {
        let lock = table(
            r#"
            [[package]]
            name = "wgpu"
            dependencies = ["naga 0.19.0", "raw-window-handle"]

            [[package]]
            name = "naga"
            "#,
        );
        let edges = locked_edges(&lock);
        assert_eq!(
            edges.get("wgpu").expect("wgpu is locked"),
            &["naga".to_string(), "raw-window-handle".to_string()]
                .into_iter()
                .collect::<BTreeSet<String>>()
        );
        assert!(edges.contains_key("naga"));
    }

    #[test]
    fn a_neutral_crate_that_reaches_a_backend_api_only_through_a_third_party_crate_is_found() {
        let graph = Graph {
            members: ["neutral".to_string()].into_iter().collect(),
            directories: [("neutral".to_string(), "neutral".to_string())]
                .into_iter()
                .collect(),
            edges: [("neutral".to_string(), ["middle".to_string()].into())]
                .into_iter()
                .collect(),
            locked: [
                ("middle".to_string(), ["wgpu".to_string()].into()),
                ("wgpu".to_string(), BTreeSet::new()),
            ]
            .into_iter()
            .collect(),
        };
        assert_eq!(
            graph.reachable_backend_apis("neutral"),
            ["wgpu".to_string()].into_iter().collect::<BTreeSet<_>>()
        );
        assert_eq!(graph.path_to("neutral", "wgpu"), "neutral -> middle -> wgpu");
        assert!(graph.reachable_members("neutral").is_empty());
    }

    #[test]
    fn a_declared_closure_follows_the_edges_of_the_crates_it_names() {
        let registry = Registry {
            declared: [
                ("top".to_string(), ["middle".to_string()].into()),
                ("middle".to_string(), ["bottom".to_string()].into()),
                ("bottom".to_string(), BTreeSet::new()),
            ]
            .into_iter()
            .collect(),
            layers: [("top".to_string(), "runtime".to_string())]
                .into_iter()
                .collect(),
            neutrality: [("runtime".to_string(), true)].into_iter().collect(),
        };
        assert_eq!(
            registry.closure("top"),
            ["middle".to_string(), "bottom".to_string()]
                .into_iter()
                .collect::<BTreeSet<_>>()
        );
        assert!(registry.neutral("top"));
        assert!(!registry.neutral("unknown"));
    }
}
