//! The crate graph a consumer compiles enables no host reference oracle.
//!
//! WHY: `cpu-parity` gates the host reference oracles in `vyre-primitives`, and
//! three domain features of `vyre-libs` used to name it. `cargo add vyre-libs`
//! therefore compiled CPU execution into the shipped library, against the rule
//! that no CPU execution lives outside `vyre-reference`. It also masked the
//! domain edges a feature-isolation sweep exists to find, because the oracle
//! modules re-enabled whatever a domain had forgotten to name.
//!
//! Deleting the three edges fixes the incident. This closes the class: the
//! contract is the whole dependency graph a default `vyre-libs` compiles, not
//! the three features that happened to carry the edge. A new domain feature, a
//! new dependency, or a dependency that turns the oracles on for its own
//! reasons all reintroduce the defect through a path nobody would think to
//! name, and each of them turns this red on the commit that adds it.
//!
//! Every member of the graph is derived at run time from the manifests cargo
//! itself reads: the workspace roster from the root manifest, each package's
//! feature table from its own manifest, and the activation rules from the
//! feature entries. Nothing here restates a package list, a feature list, or an
//! edge.
//!
//! What this does NOT catch: an oracle that ships ungated, or one gated behind
//! a feature named something else. Gating is what makes a module optional, and
//! a module with no gate is a different defect with a different measure.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use toml::{Table, Value};
use vyre_test_support::monorepo::vyre_workspace_root;

/// The feature that compiles the host reference oracles.
const ORACLE_FEATURE: &str = "cpu-parity";

/// The package a library consumer names. Its default build is the shipped
/// surface this contract is about.
const SHIPPED_ROOT: &str = "vyre-libs";

#[test]
fn the_default_consumer_graph_enables_no_cpu_oracle() {
    let workspace = Workspace::load();
    let resolved = workspace.resolve(&Request::default_only());

    let offenders: Vec<String> = resolved
        .iter()
        .filter(|(_, features)| features.contains(ORACLE_FEATURE))
        .map(|(package, _)| package.clone())
        .collect();

    assert!(
        offenders.is_empty(),
        "Fix: a default `{SHIPPED_ROOT}` build must enable `{ORACLE_FEATURE}` nowhere; \
         it is on for {offenders:?}. Remove the feature edge that reaches it, or move \
         the code that needs an oracle into vyre-reference."
    );
}

#[test]
fn the_resolver_descends_into_every_dependency_the_shipped_build_pulls() {
    let workspace = Workspace::load();
    let resolved = workspace.resolve(&Request::default_only());

    let expected: BTreeSet<&str> = workspace.manifests[SHIPPED_ROOT]
        .dependencies
        .iter()
        .filter(|dependency| {
            !dependency.optional && workspace.manifests.contains_key(&dependency.package)
        })
        .map(|dependency| dependency.package.as_str())
        .collect();

    assert!(
        expected.len() > 1,
        "Fix: `{SHIPPED_ROOT}` must declare more than one workspace dependency for this \
         contract to mean anything; the manifest parse produced {expected:?}"
    );
    for package in expected {
        assert!(
            resolved.contains_key(package),
            "Fix: the resolver stopped before `{package}`, which `{SHIPPED_ROOT}` depends on \
             unconditionally. A graph that does not descend cannot see an oracle further down."
        );
    }
}

#[test]
fn the_resolver_reports_an_oracle_that_a_feature_edge_turns_on() {
    let workspace = Workspace::load();

    // The mutation is the manifest's own oracle feature, requested explicitly.
    // Its entries name the packages an oracle build reaches, so the packages
    // this must report are read off the same edge rather than listed here.
    let edge = workspace.manifests[SHIPPED_ROOT]
        .features
        .get(ORACLE_FEATURE)
        .unwrap_or_else(|| {
            panic!("Fix: `{SHIPPED_ROOT}` must declare `{ORACLE_FEATURE}` for this contract")
        });
    let mut must_report: BTreeSet<String> = BTreeSet::new();
    must_report.insert(SHIPPED_ROOT.to_string());
    for entry in edge {
        if let Some((alias, feature)) = entry.split_once('/') {
            if feature == ORACLE_FEATURE {
                let alias = alias.trim_end_matches('?');
                must_report.insert(workspace.manifests[SHIPPED_ROOT].package_of(alias));
            }
        }
    }
    assert!(
        must_report.len() > 1,
        "Fix: `{ORACLE_FEATURE}` must cross a package boundary for this contract to prove the \
         resolver follows one; it reads as {edge:?}"
    );

    let resolved = workspace.resolve(&Request::with_feature(ORACLE_FEATURE));
    let reported: BTreeSet<String> = resolved
        .iter()
        .filter(|(_, features)| features.contains(ORACLE_FEATURE))
        .map(|(package, _)| package.clone())
        .collect();

    assert!(
        must_report.is_subset(&reported),
        "Fix: requesting `{ORACLE_FEATURE}` must report every package the edge turns it on for. \
         Expected at least {must_report:?}, got {reported:?}. A resolver that misses one here \
         would also miss it in the default build."
    );
}

/// What a build asks of the root package.
struct Request {
    /// Features named on the command line.
    explicit: BTreeSet<String>,
    /// Whether the root's `default` feature is on.
    default: bool,
}

impl Request {
    /// `cargo add vyre-libs`: default features, nothing else.
    fn default_only() -> Self {
        Self {
            explicit: BTreeSet::new(),
            default: true,
        }
    }

    /// A default build with one extra feature named.
    fn with_feature(feature: &str) -> Self {
        let mut explicit = BTreeSet::new();
        explicit.insert(feature.to_string());
        Self {
            explicit,
            default: true,
        }
    }
}

/// Every workspace package, keyed by package name.
struct Workspace {
    manifests: BTreeMap<String, Manifest>,
}

impl Workspace {
    /// Parse the root manifest's member roster and every member manifest.
    fn load() -> Self {
        let root = vyre_workspace_root();
        let root_manifest = read_table(&root.join("Cargo.toml"));
        let workspace_table = root_manifest
            .get("workspace")
            .and_then(Value::as_table)
            .expect("Fix: the root manifest must declare a [workspace] table");
        let inherited = workspace_table
            .get("dependencies")
            .and_then(Value::as_table)
            .cloned()
            .unwrap_or_default();
        let members = workspace_table
            .get("members")
            .and_then(Value::as_array)
            .expect("Fix: the root manifest must declare workspace.members");

        let mut manifests = BTreeMap::new();
        for member in members {
            let relative = member
                .as_str()
                .expect("Fix: every workspace member must be a path string");
            let path = root.join(relative).join("Cargo.toml");
            if !path.is_file() {
                continue;
            }
            let table = read_table(&path);
            let Some(package) = table
                .get("package")
                .and_then(Value::as_table)
                .and_then(|package| package.get("name"))
                .and_then(Value::as_str)
            else {
                continue;
            };
            manifests.insert(package.to_string(), Manifest::parse(&table, &inherited));
        }

        assert!(
            manifests.contains_key(SHIPPED_ROOT),
            "Fix: the workspace roster must hold `{SHIPPED_ROOT}`; parsed {:?}",
            manifests.keys().collect::<Vec<_>>()
        );
        Self { manifests }
    }

    /// Resolve the package graph a build of [`SHIPPED_ROOT`] compiles, and the
    /// features each package in it ends up with.
    ///
    /// Feature activation is monotone: a package's feature set only grows, and
    /// the package universe is the finite workspace roster, so the fixpoint
    /// terminates. Iterating to a fixpoint rather than walking once is what
    /// makes a `dep?/feature` entry correct: it applies only once its
    /// dependency is known to be in the graph, which an earlier pass may not
    /// have decided yet.
    fn resolve(&self, request: &Request) -> BTreeMap<String, BTreeSet<String>> {
        let mut wanted: BTreeMap<String, Wanted> = BTreeMap::new();
        wanted.insert(
            SHIPPED_ROOT.to_string(),
            Wanted {
                explicit: request.explicit.clone(),
                default: request.default,
            },
        );

        let mut activated: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        loop {
            let mut changed = false;
            for package in wanted.keys().cloned().collect::<Vec<_>>() {
                let Some(manifest) = self.manifests.get(&package) else {
                    continue;
                };
                let (explicit, default) = {
                    let entry = &wanted[&package];
                    (entry.explicit.clone(), entry.default)
                };
                let mut roots = explicit;
                if default && manifest.features.contains_key("default") {
                    roots.insert("default".to_string());
                }
                let (own, external) = manifest.closure(&roots);

                let live = manifest.live_dependencies(&own, &external);
                for (alias, extra) in manifest.propagated(&external, &live) {
                    let dependency = manifest.dependency(&alias);
                    if !self.manifests.contains_key(&dependency.package) {
                        continue;
                    }
                    // A newly reached package is itself a change, even when it
                    // arrives with no features and no default: the loop would
                    // otherwise settle before ever expanding it.
                    changed |= !wanted.contains_key(&dependency.package);
                    let target = wanted.entry(dependency.package.clone()).or_insert(Wanted {
                        explicit: BTreeSet::new(),
                        default: false,
                    });
                    if dependency.default_features && !target.default {
                        target.default = true;
                        changed = true;
                    }
                    for feature in dependency.features.iter().cloned().chain(extra) {
                        changed |= target.explicit.insert(feature);
                    }
                }

                if activated.get(&package) != Some(&own) {
                    changed = true;
                    activated.insert(package, own);
                }
            }
            if !changed {
                break;
            }
        }
        activated
    }
}

/// One package's requested feature state during resolution.
struct Wanted {
    explicit: BTreeSet<String>,
    default: bool,
}

/// The parts of one manifest feature resolution needs.
struct Manifest {
    /// The `[features]` table.
    features: BTreeMap<String, Vec<String>>,
    /// Every normal and build dependency, across plain and target tables.
    dependencies: Vec<Dependency>,
}

impl Manifest {
    fn parse(table: &Table, inherited: &Table) -> Self {
        let features = table
            .get("features")
            .and_then(Value::as_table)
            .map(|features| {
                features
                    .iter()
                    .map(|(name, entries)| {
                        let entries = entries
                            .as_array()
                            .map(|entries| {
                                entries
                                    .iter()
                                    .filter_map(Value::as_str)
                                    .map(str::to_string)
                                    .collect()
                            })
                            .unwrap_or_default();
                        (name.clone(), entries)
                    })
                    .collect()
            })
            .unwrap_or_default();

        let mut dependencies = Vec::new();
        collect_dependencies(table, inherited, &mut dependencies);
        if let Some(targets) = table.get("target").and_then(Value::as_table) {
            for platform in targets.values() {
                if let Some(platform) = platform.as_table() {
                    collect_dependencies(platform, inherited, &mut dependencies);
                }
            }
        }

        Self {
            features,
            dependencies,
        }
    }

    /// Look up a dependency by the name its table uses.
    fn dependency(&self, alias: &str) -> &Dependency {
        self.dependencies
            .iter()
            .find(|dependency| dependency.alias == alias)
            .unwrap_or_else(|| panic!("Fix: no dependency named `{alias}` in this manifest"))
    }

    /// The package a dependency alias resolves to, after any rename.
    fn package_of(&self, alias: &str) -> String {
        self.dependency(alias).package.clone()
    }

    /// Expand `roots` over this package's own feature table.
    ///
    /// Returns the feature names that end up on, and the raw entries that name
    /// another package (`dep:name`, `name/feature`, `name?/feature`), which
    /// only mean something once the dependency they name is resolved.
    fn closure(&self, roots: &BTreeSet<String>) -> (BTreeSet<String>, Vec<String>) {
        let mut own = BTreeSet::new();
        let mut external = Vec::new();
        let mut stack: Vec<String> = roots.iter().cloned().collect();
        while let Some(name) = stack.pop() {
            if name.starts_with("dep:") || name.contains('/') {
                external.push(name);
                continue;
            }
            if !own.insert(name.clone()) {
                continue;
            }
            if let Some(entries) = self.features.get(&name) {
                stack.extend(entries.iter().cloned());
            }
        }
        (own, external)
    }

    /// Dependency aliases the given feature state pulls into the build.
    ///
    /// A non-optional dependency is always in. An optional one joins when a
    /// live feature names it through `dep:name`, through an unconditional
    /// `name/feature`, or through the implicit feature cargo gives an optional
    /// dependency that no `dep:` entry mentions.
    fn live_dependencies(&self, own: &BTreeSet<String>, external: &[String]) -> BTreeSet<String> {
        let mut live: BTreeSet<String> = self
            .dependencies
            .iter()
            .filter(|dependency| !dependency.optional)
            .map(|dependency| dependency.alias.clone())
            .collect();
        for dependency in self.dependencies.iter().filter(|entry| entry.optional) {
            if own.contains(&dependency.alias) {
                live.insert(dependency.alias.clone());
            }
        }
        for entry in external {
            if let Some(alias) = entry.strip_prefix("dep:") {
                live.insert(alias.to_string());
            } else if let Some((alias, _)) = entry.split_once('/') {
                if !alias.ends_with('?') {
                    live.insert(alias.to_string());
                }
            }
        }
        live.retain(|alias| {
            self.dependencies
                .iter()
                .any(|dependency| dependency.alias == *alias)
        });
        live
    }

    /// Per live dependency alias, the features this package's feature state
    /// asks that dependency to turn on.
    fn propagated(
        &self,
        external: &[String],
        live: &BTreeSet<String>,
    ) -> BTreeMap<String, BTreeSet<String>> {
        let mut propagated: BTreeMap<String, BTreeSet<String>> = live
            .iter()
            .map(|alias| (alias.clone(), BTreeSet::new()))
            .collect();
        for entry in external {
            let Some((alias, feature)) = entry.split_once('/') else {
                continue;
            };
            let alias = alias.trim_end_matches('?');
            if let Some(features) = propagated.get_mut(alias) {
                features.insert(feature.to_string());
            }
        }
        propagated
    }
}

/// One dependency edge, with the workspace inheritance already applied.
struct Dependency {
    /// Name the dependency table uses, which is what a feature entry names.
    alias: String,
    /// Package the alias resolves to, after any `package =` rename.
    package: String,
    optional: bool,
    default_features: bool,
    features: Vec<String>,
}

impl Dependency {
    fn parse(alias: &str, spec: &Value, inherited: &Table) -> Self {
        let Some(table) = spec.as_table() else {
            return Self {
                alias: alias.to_string(),
                package: alias.to_string(),
                optional: false,
                default_features: true,
                features: Vec::new(),
            };
        };

        let base = table
            .get("workspace")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            .then(|| inherited.get(alias).and_then(Value::as_table))
            .flatten();

        let features = |source: Option<&Table>| -> Vec<String> {
            source
                .and_then(|source| source.get("features"))
                .and_then(Value::as_array)
                .map(|entries| {
                    entries
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default()
        };
        let mut merged = features(base);
        merged.extend(features(Some(table)));

        let default_features = table
            .get("default-features")
            .and_then(Value::as_bool)
            .or_else(|| {
                base.and_then(|base| base.get("default-features"))
                    .and_then(Value::as_bool)
            })
            .unwrap_or(true);

        let package = table
            .get("package")
            .and_then(Value::as_str)
            .or_else(|| {
                base.and_then(|base| base.get("package"))
                    .and_then(Value::as_str)
            })
            .unwrap_or(alias);

        Self {
            alias: alias.to_string(),
            package: package.to_string(),
            optional: table
                .get("optional")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            default_features,
            features: merged,
        }
    }
}

/// Append every normal and build dependency one table declares.
///
/// Development dependencies are absent on purpose: they are not in the graph a
/// consumer compiles, which is why this crate may enable the oracles for its
/// own tests without shipping them.
fn collect_dependencies(tables: &Table, inherited: &Table, into: &mut Vec<Dependency>) {
    for key in ["dependencies", "build-dependencies"] {
        let Some(entries) = tables.get(key).and_then(Value::as_table) else {
            continue;
        };
        for (alias, spec) in entries {
            into.push(Dependency::parse(alias, spec, inherited));
        }
    }
}

fn read_table(path: &Path) -> Table {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", path.display()));
    text.parse::<Table>()
        .unwrap_or_else(|error| panic!("Fix: cannot parse {}: {error}", path.display()))
}
