//! The typed configuration space, and whether it has a valid assignment.
//!
//! The compiled product is a family of programs rather than one tree, so a
//! default-feature build proves nothing about the cells beside it. This reads
//! every member manifest into one model: which package declares each
//! capability, which package a facade row forwards it to, which cells a build
//! can ask for, and whether the declared constraints leave any of those cells
//! without an assignment.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

pub mod facade_manifest;
pub mod feature_graph;
pub mod roster;
pub mod solve;

pub use feature_graph::{BodyEntry, FeatureGraph, PackageFeatures, Provenance};
pub use roster::{
    Constraints, ExclusiveFeatures, FacadeAggregate, FacadeDomain, FacadeFeature, FacadeRoster,
    SharedCapability, OWNERSHIP_RECORD_PATH,
};
pub use solve::{Activation, ConstraintConflict, Satisfiability, SatisfyingAssignment};

/// Canonical schema version for the configuration space artifact.
pub const CONFIG_SPACE_SCHEMA_VERSION: u32 = 2;

/// Path of the generated configuration space artifact.
pub const CONFIG_SPACE_ARTIFACT_PATH: &str = "docs/generated/configuration-space.toml";

/// Configuration error during workspace inspection or constraint solving.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ConfigSpaceError {
    /// Manifest parsing failed.
    ManifestParse(String),
    /// Stale schema version detected.
    StaleSchemaVersion {
        /// Expected version.
        expected: u32,
        /// Found version.
        found: u32,
    },
    /// Serialization error.
    Serialization(String),
}

/// Target predicate specification.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TargetPredicate {
    /// Full compilation target triple.
    pub triple: String,
    /// Target operating system.
    pub os: String,
    /// Target CPU architecture.
    pub arch: String,
    /// Support tier (1 = primary CI, 2 = cross-compiled/extended).
    pub tier: u8,
}

/// Backend cell descriptor.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BackendCell {
    /// Backend identifier.
    pub id: String,
    /// Concrete crate providing the backend.
    pub provider_crate: String,
    /// Package publication class.
    pub publication_class: String,
    /// Whether real hardware execution is required.
    pub hardware_required: bool,
}

/// Supported toolchain cell descriptor.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolchainCell {
    /// Channel name.
    pub channel: String,
    /// Version specifier.
    pub version: String,
    /// Whether this represents the minimum supported Rust version.
    pub is_msrv: bool,
}

/// Audit record for a package build script (`build.rs`).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BuildScriptAudit {
    /// Path to build.rs relative to workspace root.
    pub path: String,
    /// Whether the script declares rerun-if-changed inputs.
    pub declares_rerun_inputs: bool,
    /// List of declared input globs or paths.
    pub declared_inputs: Vec<String>,
    /// Whether any undeclared environment variables or network calls are inspected.
    pub accesses_undeclared_host_state: bool,
    /// Determinism verdict.
    pub is_deterministic: bool,
}

/// One isolated build cell of the minimal covering set.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CoveringSetEntry {
    /// Crate under test.
    pub package: String,
    /// The feature combination this cell enables.
    pub enabled_features: Vec<String>,
    /// Target execution cell.
    pub target_cell: String,
    /// Exact isolated cargo build command.
    pub isolated_build_command: String,
}

/// Configuration space record for a single workspace package.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CrateConfigEntry {
    /// Package name.
    pub name: String,
    /// Path to Cargo.toml.
    pub manifest_path: String,
    /// Package publication class.
    pub publication_class: String,
    /// All declared features.
    pub features: Vec<String>,
    /// Default features list.
    pub default_features: Vec<String>,
    /// Optional dependencies exposed as features.
    pub optional_deps: Vec<String>,
    /// Whether a build script exists.
    pub has_build_script: bool,
}

/// The complete typed configuration space model.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ConfigurationModel {
    /// Schema version for fail-closed validation.
    pub schema_version: u32,
    /// All workspace crates analyzed.
    pub crates: Vec<CrateConfigEntry>,
    /// Supported target predicates across the workspace.
    pub supported_targets: Vec<TargetPredicate>,
    /// Supported Rust toolchain cells.
    pub supported_toolchains: Vec<ToolchainCell>,
    /// Backend driver cells.
    pub backend_cells: Vec<BackendCell>,
    /// Feature pairs the ownership record declares mutually exclusive.
    pub exclusive_features: Vec<ExclusiveFeatures>,
    /// Generated minimal covering sets for isolated build scheduling.
    pub covering_sets: Vec<CoveringSetEntry>,
    /// Audit records for all discovered build scripts.
    pub build_script_audits: Vec<BuildScriptAudit>,
    /// Feature names that do not express a neutral capability.
    pub capability_naming_findings: Vec<String>,
    /// Feature names more than one package declares, and divided bodies.
    pub feature_ownership_findings: Vec<String>,
    /// Forwards that name a feature the dependency does not declare.
    pub forward_findings: Vec<String>,
    /// Disagreements between the facade roster and the facade manifest.
    pub facade_findings: Vec<String>,
    /// `cfg` branches reading a feature the owning package does not declare.
    pub unreachable_cfg_findings: Vec<String>,
    /// How many selection cells the satisfiability proof covered.
    pub scheduled_cells: usize,
    /// Whether the space has a valid assignment, and the witness or conflicts.
    pub satisfiability: Satisfiability,
}

impl ConfigurationModel {
    /// Inspect workspace manifests, build scripts, and feature tables.
    ///
    /// # Errors
    ///
    /// Returns the reason a manifest or the ownership record could not be read.
    pub fn inspect_workspace(root: &Path) -> Result<Self, ConfigSpaceError> {
        let members = structure_gate::workspace_members(root);
        let mut graph = FeatureGraph::default();
        let mut manifest_paths: BTreeMap<String, String> = BTreeMap::new();
        let mut parsed: BTreeMap<String, toml::Value> = BTreeMap::new();

        for member in &members {
            let manifest_path = root.join(member).join("Cargo.toml");
            if !manifest_path.exists() {
                continue;
            }
            let text = fs::read_to_string(&manifest_path)
                .map_err(|error| ConfigSpaceError::ManifestParse(error.to_string()))?;
            let value: toml::Value = toml::from_str(&text)
                .map_err(|error| ConfigSpaceError::ManifestParse(error.to_string()))?;
            let Some(name) = value
                .get("package")
                .and_then(|package| package.get("name"))
                .and_then(toml::Value::as_str)
            else {
                continue;
            };
            let name = name.to_string();
            graph
                .packages
                .insert(name.clone(), FeatureGraph::package_features(&value));
            manifest_paths.insert(name.clone(), format!("{member}/Cargo.toml"));
            parsed.insert(name, value);
        }

        let (facade_roster, constraints) = roster::load(root)?;

        let mut capability_naming_findings = Vec::new();
        let mut crates = Vec::new();
        let mut covering_sets = Vec::new();
        let mut build_script_audits = Vec::new();

        for (name, table) in &graph.packages {
            for feature in table.features.keys() {
                if feature.starts_with("use_") || feature.starts_with("with_") {
                    capability_naming_findings.push(format!(
                        "Crate `{name}` feature `{feature}` uses an anti-pattern prefix; a feature name states a capability"
                    ));
                }
            }

            let member_directory = manifest_paths
                .get(name)
                .and_then(|path| path.strip_suffix("/Cargo.toml"))
                .unwrap_or(name.as_str());
            let build_rs = root.join(member_directory).join("build.rs");
            let has_build_script = build_rs.exists();
            if has_build_script {
                build_script_audits.push(audit_build_script(root, &build_rs));
            }

            let features: Vec<String> = table.features.keys().cloned().collect();
            covering_sets.extend(generate_covering_set(name, &features));
            crates.push(CrateConfigEntry {
                name: name.clone(),
                manifest_path: manifest_paths.get(name).cloned().unwrap_or_default(),
                publication_class: table.publication_class.clone(),
                features,
                default_features: table
                    .features
                    .get(feature_graph::RESERVED_DEFAULT)
                    .into_iter()
                    .flatten()
                    .filter_map(|entry| match entry {
                        BodyEntry::Local { feature } => Some(feature.clone()),
                        BodyEntry::Activates { .. } | BodyEntry::Forwards { .. } => None,
                    })
                    .collect(),
                optional_deps: table.optional_dependencies.iter().cloned().collect(),
                has_build_script,
            });
        }

        build_script_audits.sort_by(|left, right| left.path.cmp(&right.path));
        covering_sets.sort_by(|left, right| {
            left.package
                .cmp(&right.package)
                .then_with(|| left.target_cell.cmp(&right.target_cell))
        });

        let feature_ownership_findings = ownership_findings(&graph, &constraints);
        let forward_findings = forward_findings(&graph);
        let mut facade_findings = facade_findings(&graph, &facade_roster);
        let unreachable_cfg_findings = scan_workspace_cfgs(root, &members, &graph);
        let scheduled_cells = solve::cells(&graph).len();
        let satisfiability =
            solve::solve(&graph, &facade_roster, &constraints, &mut facade_findings);

        Ok(Self {
            schema_version: CONFIG_SPACE_SCHEMA_VERSION,
            crates,
            supported_targets: default_supported_targets(),
            supported_toolchains: default_supported_toolchains(),
            backend_cells: default_backend_cells(),
            exclusive_features: constraints.exclusive,
            covering_sets,
            build_script_audits,
            capability_naming_findings,
            feature_ownership_findings,
            forward_findings,
            facade_findings,
            unreachable_cfg_findings,
            scheduled_cells,
            satisfiability,
        })
    }

    /// The facade manifest this model would write, and the path it belongs at.
    ///
    /// # Errors
    ///
    /// Returns the reason the facade manifest could not be read or has no
    /// `[features]` table to own.
    pub fn facade_manifest(root: &Path) -> Result<(String, String), ConfigSpaceError> {
        let (facade_roster, _) = roster::load(root)?;
        let directory = structure_gate::member_directory(root, &facade_roster.package);
        let relative = directory
            .strip_prefix(root)
            .map(|path| path.display().to_string())
            .unwrap_or_else(|_| facade_roster.package.clone());
        let path = format!("{relative}/Cargo.toml");
        let text = fs::read_to_string(root.join(&path))
            .map_err(|error| ConfigSpaceError::ManifestParse(format!("{path}: {error}")))?;
        let undeclared = facade_manifest::undeclared_activations(&facade_roster, &text);
        if !undeclared.is_empty() {
            let named: Vec<String> = undeclared
                .iter()
                .map(|(feature, dependency)| format!("{feature} activates {dependency}"))
                .collect();
            return Err(ConfigSpaceError::ManifestParse(format!(
                "{path}: the roster activates dependencies the facade does not declare: {}. \
                 Declare each one optional in the facade, or drop it from the roster row when \
                 the domain package owns it and the weak forward already turns it on.",
                named.join(", ")
            )));
        }
        let table = facade_manifest::render(&facade_roster);
        let spliced = facade_manifest::splice(&text, &table)
            .map_err(|reason| ConfigSpaceError::ManifestParse(format!("{path}: {reason}")))?;
        Ok((path, spliced))
    }

    /// Serialize the configuration model to TOML.
    ///
    /// # Errors
    ///
    /// Returns the reason the model could not be serialized.
    pub fn to_toml(&self) -> Result<String, ConfigSpaceError> {
        toml::to_string_pretty(self)
            .map_err(|error| ConfigSpaceError::Serialization(error.to_string()))
    }

    /// Deserialize a configuration model with fail-closed schema validation.
    ///
    /// # Errors
    ///
    /// Returns the reason the text is not a model of the current schema.
    pub fn from_toml(text: &str) -> Result<Self, ConfigSpaceError> {
        let model: Self = toml::from_str(text)
            .map_err(|error| ConfigSpaceError::Serialization(error.to_string()))?;
        if model.schema_version != CONFIG_SPACE_SCHEMA_VERSION {
            return Err(ConfigSpaceError::StaleSchemaVersion {
                expected: CONFIG_SPACE_SCHEMA_VERSION,
                found: model.schema_version,
            });
        }
        Ok(model)
    }
}

/// Every feature name more than one package decides the meaning of.
///
/// A facade row that forwards the name to the package declaring it is not a
/// second decision, so it is attributed to the package it forwards to and
/// contributes no owner of its own. A divided body is a decision in both
/// packages at once, which is the shape that makes one `--features` flag mean
/// two things.
#[must_use]
fn ownership_findings(graph: &FeatureGraph, constraints: &Constraints) -> Vec<String> {
    let shared: BTreeMap<&str, &str> = constraints
        .shared_capability
        .iter()
        .map(|entry| (entry.name.as_str(), entry.reason.as_str()))
        .collect();
    let mut findings = Vec::new();

    for (package, table) in &graph.packages {
        for feature in table.features.keys() {
            if graph.provenance(package, feature) == Some(Provenance::Divided) {
                findings.push(format!(
                    "Feature `{feature}` of `{package}` {}",
                    Provenance::Divided.predicate()
                ));
            }
        }
    }

    for feature in graph.feature_names() {
        let owners = graph.owners(&feature);
        match (owners.len() > 1, shared.contains_key(feature.as_str())) {
            (true, false) => {
                let attributed: Vec<String> = owners
                    .iter()
                    .map(|owner| {
                        let provenance = graph
                            .provenance(owner, &feature)
                            .map_or("declares the name here", Provenance::predicate);
                        format!("`{owner}` {provenance}")
                    })
                    .collect();
                findings.push(format!(
                    "Feature `{feature}` is declared by {} packages: {}. A capability name has one owning package; a facade forwards it instead of redeclaring it.",
                    owners.len(),
                    attributed.join("; ")
                ));
            }
            (false, true) => findings.push(format!(
                "`{feature}` is recorded as a shared capability but {} package(s) declare it, so the exemption covers nothing",
                owners.len()
            )),
            (true, true) | (false, false) => {}
        }
    }

    findings.sort();
    findings
}

/// Every forward that names a feature its dependency does not declare.
///
/// A forward is the whole mechanism by which a facade exposes a domain without
/// redeclaring it, so a forward that resolves to nothing is a facade feature
/// that turns nothing on. Cargo rejects it at resolve time for an activated
/// dependency and says nothing about a weak forward into one a cell never
/// activates, which is exactly the row a rename leaves behind.
#[must_use]
fn forward_findings(graph: &FeatureGraph) -> Vec<String> {
    let mut findings = Vec::new();
    for (package, table) in &graph.packages {
        for (feature, body) in &table.features {
            for entry in body {
                let BodyEntry::Forwards {
                    dependency,
                    feature: forwarded,
                    ..
                } = entry
                else {
                    continue;
                };
                let Some(target) = graph.packages.get(dependency) else {
                    // A registry dependency's feature table is not in this
                    // workspace, so the manifests cannot answer for it.
                    continue;
                };
                if !target.enable_able().contains(forwarded) {
                    findings.push(format!(
                        "Feature `{feature}` of `{package}` forwards to `{dependency}/{forwarded}`, which `{dependency}` does not declare"
                    ));
                }
            }
        }
    }
    findings.sort();
    findings
}

/// Every disagreement between the facade roster and the facade's manifest.
///
/// The `[features]` table is generated, so a difference there is one artifact
/// finding. These are the facts the generated table cannot carry: the
/// dependency declarations it forwards through, and whether the package the
/// roster names as the owner actually declares the capability.
#[must_use]
fn facade_findings(graph: &FeatureGraph, roster: &FacadeRoster) -> Vec<String> {
    let mut findings = Vec::new();
    let Some(facade) = graph.packages.get(&roster.package) else {
        return vec![format!(
            "The roster names `{}` as the consumer facade, which is not a workspace member",
            roster.package
        )];
    };
    let published = roster.published();

    for domain in &roster.domain {
        if !graph.packages.contains_key(&domain.package) {
            findings.push(format!(
                "Roster domain `{}` is not a workspace member",
                domain.package
            ));
            continue;
        }
        if !facade.dependencies.contains(&domain.package) {
            findings.push(format!(
                "Roster domain `{}` is not a dependency of `{}`",
                domain.package, roster.package
            ));
            continue;
        }
        let optional = facade.optional_dependencies.contains(&domain.package);
        if optional != domain.optional {
            findings.push(format!(
                "Roster domain `{}` records optional = {}, and `{}` declares the dependency {}",
                domain.package,
                domain.optional,
                roster.package,
                if optional { "optional" } else { "required" }
            ));
        }
    }

    for entry in &roster.feature {
        if roster.domain(&entry.domain).is_none() {
            findings.push(format!(
                "Roster feature `{}` names owner `{}`, which has no [[facade.domain]] record",
                entry.name, entry.domain
            ));
        }
        match graph.packages.get(&entry.domain) {
            None => findings.push(format!(
                "Roster feature `{}` names owner `{}`, which is not a workspace member",
                entry.name, entry.domain
            )),
            Some(owner) if !owner.features.contains_key(&entry.name) => findings.push(format!(
                "Roster feature `{}` names owner `{}`, which does not declare it",
                entry.name, entry.domain
            )),
            Some(_) => {}
        }
        for required in &entry.requires {
            if !published.contains(required) {
                findings.push(format!(
                    "Roster feature `{}` requires `{required}`, which the roster does not publish",
                    entry.name
                ));
            }
        }
    }

    for aggregate in &roster.aggregate {
        for selected in &aggregate.selects {
            if !published.contains(selected) {
                findings.push(format!(
                    "Roster aggregate `{}` selects `{selected}`, which the roster does not publish",
                    aggregate.name
                ));
            }
        }
    }

    for selected in &roster.default {
        if !published.contains(selected) {
            findings.push(format!(
                "The roster default selection names `{selected}`, which the roster does not publish"
            ));
        }
    }

    // Every feature the facade manifest publishes has to be in the roster, or
    // the generated table would drop it and the roster would not be the one
    // copy of the facade surface.
    for feature in facade.features.keys() {
        if feature == feature_graph::RESERVED_DEFAULT || feature == "full" {
            continue;
        }
        if !published.contains(feature) {
            findings.push(format!(
                "`{}` declares feature `{feature}`, which the roster does not record",
                roster.package
            ));
        }
    }

    findings.sort();
    findings
}

/// Default supported target predicates.
#[must_use]
pub fn default_supported_targets() -> Vec<TargetPredicate> {
    vec![
        TargetPredicate {
            triple: "x86_64-unknown-linux-gnu".to_string(),
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            tier: 1,
        },
        TargetPredicate {
            triple: "aarch64-unknown-linux-gnu".to_string(),
            os: "linux".to_string(),
            arch: "aarch64".to_string(),
            tier: 1,
        },
        TargetPredicate {
            triple: "x86_64-pc-windows-msvc".to_string(),
            os: "windows".to_string(),
            arch: "x86_64".to_string(),
            tier: 1,
        },
        TargetPredicate {
            triple: "aarch64-apple-darwin".to_string(),
            os: "macos".to_string(),
            arch: "aarch64".to_string(),
            tier: 1,
        },
        TargetPredicate {
            triple: "x86_64-apple-darwin".to_string(),
            os: "macos".to_string(),
            arch: "x86_64".to_string(),
            tier: 2,
        },
    ]
}

/// Default supported toolchain cells.
#[must_use]
pub fn default_supported_toolchains() -> Vec<ToolchainCell> {
    vec![
        ToolchainCell {
            channel: "stable".to_string(),
            version: "1.86".to_string(),
            is_msrv: false,
        },
        ToolchainCell {
            channel: "msrv".to_string(),
            version: "1.85".to_string(),
            is_msrv: true,
        },
        ToolchainCell {
            channel: "nightly".to_string(),
            version: "nightly".to_string(),
            is_msrv: false,
        },
    ]
}

/// Default backend execution cells.
#[must_use]
pub fn default_backend_cells() -> Vec<BackendCell> {
    vec![
        BackendCell {
            id: "reference".to_string(),
            provider_crate: "vyre-driver-reference".to_string(),
            publication_class: "extension-sdk".to_string(),
            hardware_required: false,
        },
        BackendCell {
            id: "wgpu".to_string(),
            provider_crate: "vyre-driver-wgpu".to_string(),
            publication_class: "extension-sdk".to_string(),
            hardware_required: true,
        },
        BackendCell {
            id: "cuda".to_string(),
            provider_crate: "vyre-driver-cuda".to_string(),
            publication_class: "extension-sdk".to_string(),
            hardware_required: true,
        },
        BackendCell {
            id: "metal".to_string(),
            provider_crate: "vyre-driver-metal".to_string(),
            publication_class: "extension-sdk".to_string(),
            hardware_required: true,
        },
        BackendCell {
            id: "spirv".to_string(),
            provider_crate: "vyre-driver-spirv".to_string(),
            publication_class: "extension-sdk".to_string(),
            hardware_required: true,
        },
    ]
}

/// Audit a single build script for determinism and declared inputs.
#[must_use]
pub fn audit_build_script(root: &Path, build_rs: &Path) -> BuildScriptAudit {
    let relative = build_rs
        .strip_prefix(root)
        .unwrap_or(build_rs)
        .display()
        .to_string();
    let text = fs::read_to_string(build_rs).unwrap_or_default();

    let declares_rerun_inputs = text.contains("cargo:rerun-if-changed")
        || text.contains("cargo:rerun-if-env-changed")
        || text.contains("cargo_directives");
    let accesses_undeclared_host_state = text.contains("std::net") || text.contains("reqwest");

    let mut declared_inputs = Vec::new();
    for line in text.lines() {
        if let Some(index) = line.find("cargo:rerun-if-changed=") {
            let after = &line[index + "cargo:rerun-if-changed=".len()..];
            declared_inputs.push(
                after
                    .trim_matches(|c: char| c == '"' || c == '\\' || c == ')')
                    .to_string(),
            );
        }
    }

    BuildScriptAudit {
        path: relative,
        declares_rerun_inputs,
        declared_inputs,
        accesses_undeclared_host_state,
        is_deterministic: !accesses_undeclared_host_state,
    }
}

/// The minimal covering set for one package: nothing, each feature alone, all.
///
/// Every declared feature has a cell that enables it and nothing else, so a
/// cell can never inherit another member's feature unification, and the widest
/// cell covers the interactions the single cells cannot.
#[must_use]
pub fn generate_covering_set(package: &str, features: &[String]) -> Vec<CoveringSetEntry> {
    let selectable: Vec<String> = features
        .iter()
        .filter(|feature| *feature != feature_graph::RESERVED_DEFAULT)
        .cloned()
        .collect();

    let mut entries = vec![CoveringSetEntry {
        package: package.to_string(),
        enabled_features: Vec::new(),
        target_cell: "base_no_features".to_string(),
        isolated_build_command: format!("./cargo_full check -p {package} --no-default-features"),
    }];
    for feature in &selectable {
        entries.push(CoveringSetEntry {
            package: package.to_string(),
            enabled_features: vec![feature.clone()],
            target_cell: format!("isolated_{feature}"),
            isolated_build_command: format!(
                "./cargo_full check -p {package} --no-default-features --features {feature}"
            ),
        });
    }
    if selectable.len() > 1 {
        entries.push(CoveringSetEntry {
            package: package.to_string(),
            enabled_features: selectable,
            target_cell: "full_features".to_string(),
            isolated_build_command: format!("./cargo_full check -p {package} --all-features"),
        });
    }
    entries
}

/// Every `cfg(feature = ...)` reading a feature the owning package never declares.
#[must_use]
fn scan_workspace_cfgs(root: &Path, members: &[String], graph: &FeatureGraph) -> Vec<String> {
    let mut findings = Vec::new();

    for member in members {
        let member_path = root.join(member);
        let manifest_path = member_path.join("Cargo.toml");
        if !manifest_path.exists() {
            continue;
        }
        let text = fs::read_to_string(&manifest_path).unwrap_or_default();
        let value: toml::Value =
            toml::from_str(&text).unwrap_or_else(|_| toml::Value::Table(toml::value::Table::new()));
        let package = value
            .get("package")
            .and_then(|package| package.get("name"))
            .and_then(toml::Value::as_str)
            .unwrap_or(member)
            .to_string();
        let declared: BTreeSet<String> = graph
            .packages
            .get(&package)
            .map(PackageFeatures::enable_able)
            .unwrap_or_default();

        for entry in walkdir::WalkDir::new(&member_path)
            .into_iter()
            .filter_map(Result::ok)
        {
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
                continue;
            }
            let relative = path
                .strip_prefix(root)
                .map(|path| path.display().to_string())
                .unwrap_or_else(|_| path.display().to_string());
            let inside_submember = members.iter().any(|other| {
                other != member && other.len() > member.len() && path.starts_with(root.join(other))
            });
            if inside_submember {
                continue;
            }

            let content = fs::read_to_string(path).unwrap_or_default();
            match cfg_features_in(&content) {
                Ok(features) => {
                    for feature in features {
                        if !declared.contains(&feature) {
                            findings.push(format!(
                                "Crate `{package}` at `{relative}` reads undeclared feature `{feature}`"
                            ));
                        }
                    }
                }
                Err(error) => findings.push(format!(
                    "Crate `{package}` at `{relative}` does not parse as Rust, so the features it reads are unknown: {error}"
                )),
            }
        }
    }

    findings.sort();
    findings.dedup();
    findings
}

/// Every feature name the `cfg` and `cfg_attr` attributes of one Rust source
/// text read.
///
/// # Errors
///
/// When the text does not parse as Rust. A file whose attributes cannot be
/// read is reported rather than treated as reading no feature, because a
/// silent skip covers less of the tree than the caller believes.
pub fn cfg_features_in(source: &str) -> Result<BTreeSet<String>, syn::Error> {
    let file = syn::parse_file(source)?;
    let mut collector = CfgFeatures::default();
    syn::visit::visit_file(&mut collector, &file);
    Ok(collector.features)
}

/// Every feature name the `cfg` and `cfg_attr` attributes of one file read.
///
/// Attributes are read from the parsed syntax tree rather than from source
/// text. A line scan cannot tell an attribute from the same characters inside
/// a string literal, and it stops at the first `feature` on the line, so
/// `any(feature = "a", feature = "b")` hid `b`.
#[derive(Default)]
struct CfgFeatures {
    features: BTreeSet<String>,
}

impl<'ast> syn::visit::Visit<'ast> for CfgFeatures {
    fn visit_attribute(&mut self, attribute: &'ast syn::Attribute) {
        if !attribute.path().is_ident("cfg") && !attribute.path().is_ident("cfg_attr") {
            return;
        }
        let Ok(list) = attribute.meta.require_list() else {
            return;
        };
        let tokens = list.tokens.to_string();
        let mut rest = tokens.as_str();
        while let Some(index) = rest.find("feature") {
            rest = rest[index + "feature".len()..].trim_start();
            let Some(after_equals) = rest.strip_prefix('=') else {
                continue;
            };
            let after_equals = after_equals.trim_start();
            let Some(literal) = after_equals.strip_prefix('"') else {
                continue;
            };
            let Some(end) = literal.find('"') else {
                return;
            };
            self.features.insert(literal[..end].to_string());
            rest = &literal[end + 1..];
        }
    }
}
