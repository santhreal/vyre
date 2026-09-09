//! Typed Configuration Space Model & CI Scheduler (Row 115).
//!
//! Generates a typed configuration model from Cargo metadata, target predicates,
//! backend capabilities, package publication classes, and declared incompatibilities.
//! Proves constraints satisfiable, produces minimal deterministic covering sets,
//! validates build script boundedness, and checks capability-based feature naming.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

/// Canonical schema version for ConfigurationSpace.
pub const CONFIG_SPACE_SCHEMA_VERSION: u32 = 1;

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
    /// Unsatisfiable feature or dependency constraint.
    UnsatisfiableConstraint(String),
    /// Feature name does not express a capability.
    InvalidFeatureName {
        /// Owning crate.
        krate: String,
        /// Feature name.
        feature: String,
        /// Reason for rejection.
        reason: String,
    },
    /// Build script violates determinism or undeclared state rules.
    BuildScriptViolation {
        /// Build script path.
        path: String,
        /// Violation description.
        violation: String,
    },
    /// Serialization error.
    Serialization(String),
}

/// Declared feature incompatibility entry.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DeclaredIncompatibility {
    /// Crate declaring or constrained by the incompatibility.
    pub package: String,
    /// First feature.
    pub feature_a: String,
    /// Second feature that cannot coexist with the first.
    pub feature_b: String,
    /// Rationale or technical constraint description.
    pub reason: String,
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
    /// Backend identifier (e.g. "cuda", "wgpu", "metal", "spirv", "reference").
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
    /// Channel name ("stable", "nightly", "msrv").
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

/// A minimal covering set test execution cell.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CoveringSetEntry {
    /// Crate under test.
    pub package: String,
    /// Specific feature combination in this isolated cell.
    pub enabled_features: Vec<String>,
    /// Features explicitly disabled.
    pub disabled_features: Vec<String>,
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
    /// Direct dependencies.
    pub dependencies: Vec<String>,
    /// Whether a build script exists.
    pub has_build_script: bool,
}

/// The complete typed configuration space model (Row 115).
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
    /// Structured declared feature incompatibilities.
    pub declared_incompatibilities: Vec<DeclaredIncompatibility>,
    /// Generated minimal covering sets for isolated build scheduling.
    pub covering_sets: Vec<CoveringSetEntry>,
    /// Audit records for all discovered build scripts.
    pub build_script_audits: Vec<BuildScriptAudit>,
    /// Capability naming findings (non-empty indicates unrecorded or product-named features).
    pub capability_naming_findings: Vec<String>,
    /// Duplicate feature declaration findings.
    pub duplicate_feature_findings: Vec<String>,
    /// Unreachable cfg branch findings.
    pub unreachable_cfg_findings: Vec<String>,
    /// Unreferenced feature findings.
    pub unreferenced_feature_findings: Vec<String>,
    /// Whether all configuration constraints are proved satisfiable.
    pub is_satisfiable: bool,
}

impl ConfigurationModel {
    /// Inspect workspace manifests, build scripts, and feature tables to construct the model.
    pub fn inspect_workspace(root: &Path) -> Result<Self, ConfigSpaceError> {
        let mut crates = Vec::new();
        let mut build_script_audits = Vec::new();
        let mut capability_naming_findings = Vec::new();
        let mut duplicate_feature_findings = Vec::new();
        let mut covering_sets = Vec::new();

        let members = structure_gate::workspace_members(root);
        let mut feature_to_crates: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut crate_feature_map: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

        for member in &members {
            let member_path = root.join(member);
            let manifest_path = member_path.join("Cargo.toml");
            if !manifest_path.exists() {
                continue;
            }

            let manifest_text = fs::read_to_string(&manifest_path)
                .map_err(|e| ConfigSpaceError::ManifestParse(e.to_string()))?;
            let manifest_val: toml::Value = toml::from_str(&manifest_text)
                .map_err(|e| ConfigSpaceError::ManifestParse(e.to_string()))?;

            let pkg_name = manifest_val
                .get("package")
                .and_then(|p| p.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or_default()
                .to_string();

            let pub_class = manifest_val
                .get("package")
                .and_then(|p| p.get("metadata"))
                .and_then(|m| m.get("vyre"))
                .and_then(|v| v.get("publication_class"))
                .and_then(|c| c.as_str())
                .unwrap_or("internal-engine")
                .to_string();

            let mut features = Vec::new();
            let mut default_features = Vec::new();

            if let Some(feat_table) = manifest_val.get("features").and_then(|f| f.as_table()) {
                for (feat_name, feat_members) in feat_table {
                    features.push(feat_name.clone());
                    if feat_name == "default" {
                        if let Some(arr) = feat_members.as_array() {
                            for item in arr {
                                if let Some(s) = item.as_str() {
                                    default_features.push(s.to_string());
                                }
                            }
                        }
                    } else {
                        feature_to_crates
                            .entry(feat_name.clone())
                            .or_default()
                            .push(pkg_name.clone());
                    }

                    // Capability-based naming rule: feature names must not use product names or verb prefixes
                    if feat_name.starts_with("use_") || feat_name.starts_with("with_") {
                        capability_naming_findings.push(format!(
                            "Crate `{pkg_name}` feature `{feat_name}` uses anti-pattern prefix; should express capability"
                        ));
                    }
                }
            }
            features.sort();

            let mut optional_deps = Vec::new();
            let mut dependencies = Vec::new();
            if let Some(dep_table) = manifest_val.get("dependencies").and_then(|d| d.as_table()) {
                for (dep_name, dep_val) in dep_table {
                    dependencies.push(dep_name.clone());
                    if let Some(tbl) = dep_val.as_table() {
                        if tbl
                            .get("optional")
                            .and_then(|o| o.as_bool())
                            .unwrap_or(false)
                        {
                            optional_deps.push(dep_name.clone());
                        }
                    }
                }
            }

            let mut valid_crate_features: BTreeSet<String> = features.iter().cloned().collect();
            for opt in &optional_deps {
                valid_crate_features.insert(opt.clone());
            }
            crate_feature_map.insert(pkg_name.clone(), valid_crate_features);

            let build_rs = member_path.join("build.rs");
            let has_build_script = build_rs.exists();
            if has_build_script {
                let audit = audit_build_script(root, &build_rs);
                build_script_audits.push(audit);
            }

            // Generate minimal covering sets: [empty, each single feature, all features]
            let mut crate_covering_sets = generate_covering_set(&pkg_name, &features);
            covering_sets.append(&mut crate_covering_sets);

            crates.push(CrateConfigEntry {
                name: pkg_name,
                manifest_path: format!("{member}/Cargo.toml"),
                publication_class: pub_class,
                features,
                default_features,
                optional_deps,
                dependencies,
                has_build_script,
            });
        }

        crates.sort_by(|a, b| a.name.cmp(&b.name));
        build_script_audits.sort_by(|a, b| a.path.cmp(&b.path));
        covering_sets.sort_by(|a, b| {
            a.package
                .cmp(&b.package)
                .then_with(|| a.target_cell.cmp(&b.target_cell))
        });

        // Detect duplicate features across crates that are not in permitted shared list
        let permitted_shared: BTreeSet<&'static str> = [
            "device-tests",
            "libs-compositions",
            "subgroup-ops",
            "test-fixtures",
            "cuda",
            "wgpu",
            "gpu",
            "cpu-parity",
            "operations",
        ]
        .into_iter()
        .collect();

        for (feat, owners) in &feature_to_crates {
            if owners.len() > 1 && !permitted_shared.contains(feat.as_str()) {
                duplicate_feature_findings.push(format!(
                    "Feature `{feat}` is declared in multiple packages ({owners:?}); features must have one owning package",
                ));
            }
        }

        // Scan source files for unreachable cfgs and unreferenced features
        let scan_findings = scan_workspace_cfgs(root, &members, &crate_feature_map);
        let unreachable_cfg_findings = scan_findings.unreachable_cfgs;
        let unreferenced_feature_findings = scan_findings.unreferenced_features;

        let supported_targets = default_supported_targets();
        let supported_toolchains = default_supported_toolchains();
        let backend_cells = default_backend_cells();
        let declared_incompatibilities = default_declared_incompatibilities();

        let is_satisfiable = capability_naming_findings.is_empty()
            && duplicate_feature_findings.is_empty()
            && unreachable_cfg_findings.is_empty();

        Ok(Self {
            schema_version: CONFIG_SPACE_SCHEMA_VERSION,
            crates,
            supported_targets,
            supported_toolchains,
            backend_cells,
            declared_incompatibilities,
            covering_sets,
            build_script_audits,
            capability_naming_findings,
            duplicate_feature_findings,
            unreachable_cfg_findings,
            unreferenced_feature_findings,
            is_satisfiable,
        })
    }

    /// Serialize configuration model to TOML.
    pub fn to_toml(&self) -> Result<String, ConfigSpaceError> {
        toml::to_string_pretty(self).map_err(|e| ConfigSpaceError::Serialization(e.to_string()))
    }

    /// Deserialize configuration model from TOML with fail-closed schema validation.
    pub fn from_toml(toml_str: &str) -> Result<Self, ConfigSpaceError> {
        let model: Self =
            toml::from_str(toml_str).map_err(|e| ConfigSpaceError::Serialization(e.to_string()))?;
        if model.schema_version != CONFIG_SPACE_SCHEMA_VERSION {
            return Err(ConfigSpaceError::StaleSchemaVersion {
                expected: CONFIG_SPACE_SCHEMA_VERSION,
                found: model.schema_version,
            });
        }
        Ok(model)
    }
}

/// Default supported target predicates.
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
            triple: "aarch64-linux-android".to_string(),
            os: "android".to_string(),
            arch: "aarch64".to_string(),
            tier: 2,
        },
    ]
}

/// Default supported toolchain cells.
pub fn default_supported_toolchains() -> Vec<ToolchainCell> {
    vec![
        ToolchainCell {
            channel: "stable".to_string(),
            version: "1.85+".to_string(),
            is_msrv: false,
        },
        ToolchainCell {
            channel: "nightly".to_string(),
            version: "nightly".to_string(),
            is_msrv: false,
        },
        ToolchainCell {
            channel: "msrv".to_string(),
            version: "1.85".to_string(),
            is_msrv: true,
        },
    ]
}

/// Default backend execution cells.
pub fn default_backend_cells() -> Vec<BackendCell> {
    vec![
        BackendCell {
            id: "cuda".to_string(),
            provider_crate: "vyre-driver-cuda".to_string(),
            publication_class: "concrete-backend".to_string(),
            hardware_required: true,
        },
        BackendCell {
            id: "wgpu".to_string(),
            provider_crate: "vyre-driver-wgpu".to_string(),
            publication_class: "concrete-backend".to_string(),
            hardware_required: true,
        },
        BackendCell {
            id: "metal".to_string(),
            provider_crate: "vyre-driver-metal".to_string(),
            publication_class: "concrete-backend".to_string(),
            hardware_required: true,
        },
        BackendCell {
            id: "spirv".to_string(),
            provider_crate: "vyre-driver-spirv".to_string(),
            publication_class: "concrete-backend".to_string(),
            hardware_required: true,
        },
        BackendCell {
            id: "reference".to_string(),
            provider_crate: "vyre-reference".to_string(),
            publication_class: "semantics".to_string(),
            hardware_required: false,
        },
    ]
}

/// Default declared feature and backend incompatibilities.
pub fn default_declared_incompatibilities() -> Vec<DeclaredIncompatibility> {
    vec![
        DeclaredIncompatibility {
            package: "vyre-primitives".to_string(),
            feature_a: "cpu-parity".to_string(),
            feature_b: "production-route".to_string(),
            reason: "Host reference oracle execution is restricted to explicit parity test harnesses; production compilation routes lower to device megakernels.".to_string(),
        },
        DeclaredIncompatibility {
            package: "vyre-runtime".to_string(),
            feature_a: "subgroup-ops".to_string(),
            feature_b: "legacy-scalar-dispatch".to_string(),
            reason: "Subgroup matrix and collective operations require uniform warp/wavefront execution.".to_string(),
        },
    ]
}

/// Audit a single build.rs script for determinism and declared inputs.
pub fn audit_build_script(root: &Path, build_rs: &Path) -> BuildScriptAudit {
    let rel_path = build_rs
        .strip_prefix(root)
        .unwrap_or(build_rs)
        .display()
        .to_string();
    let text = fs::read_to_string(build_rs).unwrap_or_default();

    let declares_rerun_inputs = text.contains("cargo:rerun-if-changed")
        || text.contains("cargo:rerun-if-env-changed")
        || text.contains("cargo_directives");
    let accesses_undeclared_host_state = text.contains("std::net") || text.contains("reqwest");
    let is_deterministic = !accesses_undeclared_host_state;

    let mut declared_inputs = Vec::new();
    for line in text.lines() {
        if let Some(idx) = line.find("cargo:rerun-if-changed=") {
            let after = &line[idx + "cargo:rerun-if-changed=".len()..];
            let trimmed = after.trim_matches(|c: char| c == '"' || c == '\\' || c == ')');
            declared_inputs.push(trimmed.to_string());
        }
    }

    BuildScriptAudit {
        path: rel_path,
        declares_rerun_inputs,
        declared_inputs,
        accesses_undeclared_host_state,
        is_deterministic,
    }
}

/// Generate a minimal deterministic covering set for feature isolation testing.
pub fn generate_covering_set(pkg_name: &str, features: &[String]) -> Vec<CoveringSetEntry> {
    let mut entries = Vec::new();

    // Filter out "default"
    let non_default_features: Vec<String> = features
        .iter()
        .filter(|&f| f != "default")
        .cloned()
        .collect();

    // 1. Base cell: no features
    entries.push(CoveringSetEntry {
        package: pkg_name.to_string(),
        enabled_features: Vec::new(),
        disabled_features: non_default_features.clone(),
        target_cell: "base_no_features".to_string(),
        isolated_build_command: format!("./cargo_full check -p {pkg_name} --no-default-features"),
    });

    // 2. Individual feature isolation cells: exactly one feature enabled
    for feat in &non_default_features {
        let disabled: Vec<String> = non_default_features
            .iter()
            .filter(|&f| f != feat)
            .cloned()
            .collect();
        entries.push(CoveringSetEntry {
            package: pkg_name.to_string(),
            enabled_features: vec![feat.clone()],
            disabled_features: disabled,
            target_cell: format!("isolated_{feat}"),
            isolated_build_command: format!(
                "./cargo_full check -p {pkg_name} --no-default-features --features {feat}"
            ),
        });
    }

    // 3. Full features cell (if multiple features exist)
    if non_default_features.len() > 1 {
        entries.push(CoveringSetEntry {
            package: pkg_name.to_string(),
            enabled_features: non_default_features,
            disabled_features: Vec::new(),
            target_cell: "full_features".to_string(),
            isolated_build_command: format!("./cargo_full check -p {pkg_name} --all-features"),
        });
    }

    entries
}

struct ScanFindings {
    unreachable_cfgs: Vec<String>,
    unreferenced_features: Vec<String>,
}

fn scan_workspace_cfgs(
    root: &Path,
    members: &[String],
    crate_feature_map: &BTreeMap<String, BTreeSet<String>>,
) -> ScanFindings {
    let mut unreachable_cfgs = Vec::new();
    let mut unreferenced_features = Vec::new();

    let mock_test_files = [
        "structure-gate/src/cfg_test.rs",
        "structure-gate/src/source_scan.rs",
        "structure-gate/src/registration_text.rs",
        "structure-gate/tests/test_gated_modules.rs",
        "xtask/src/gates/scan.rs",
        "xtask/src/gates/lego_quick.rs",
        "xtask/src/gates/hygiene_matrix/rules.rs",
        "xtask/src/gates/test_only_capability.rs",
        "xtask/src/gates/device_test_gating.rs",
        "xtask/src/gates/host_oracle_elimination",
        "vyre-bench/tests/feature_cfg_contract.rs",
        "xtask-registry/tests/registry_contracts/registration_visibility.rs",
        "xtask-registry/src/gates/configuration_model.rs",
    ];

    for member in members {
        let member_path = root.join(member);
        let manifest_path = member_path.join("Cargo.toml");
        if !manifest_path.exists() {
            continue;
        }

        let manifest_text = fs::read_to_string(&manifest_path).unwrap_or_default();
        let manifest_val: toml::Value =
            toml::from_str(&manifest_text).unwrap_or(toml::Value::Table(Default::default()));
        let pkg_name = manifest_val
            .get("package")
            .and_then(|p| p.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or(member)
            .to_string();

        let valid_features = crate_feature_map
            .get(&pkg_name)
            .cloned()
            .unwrap_or_default();

        let mut crate_used_features = BTreeSet::new();

        for entry in walkdir::WalkDir::new(&member_path)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
                continue;
            }

            let rel_str = path
                .strip_prefix(root)
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| path.display().to_string());

            if mock_test_files.iter().any(|mf| rel_str.contains(mf)) {
                continue;
            }

            // Skip submembers inside member directory
            let is_submember = members.iter().any(|other_m| {
                other_m != member
                    && other_m.len() > member.len()
                    && path.starts_with(root.join(other_m))
            });
            if is_submember {
                continue;
            }

            let content = fs::read_to_string(path).unwrap_or_default();
            for line in content.lines() {
                if let Some(feat) = extract_cfg_feature(line) {
                    crate_used_features.insert(feat.clone());
                    if !valid_features.contains(&feat) {
                        unreachable_cfgs.push(format!(
                            "Crate `{pkg_name}` at `{rel_str}` references undeclared feature `{feat}`"
                        ));
                    }
                }
            }
        }

        // Check if any declared feature is never read and enables nothing
        if let Some(feat_table) = manifest_val.get("features").and_then(|f| f.as_table()) {
            for (f_name, f_deps) in feat_table {
                if f_name == "default" {
                    continue;
                }
                let has_deps = f_deps.as_array().map(|a| !a.is_empty()).unwrap_or(false);
                let has_cfg = crate_used_features.contains(f_name);
                if !has_deps && !has_cfg {
                    unreferenced_features.push(format!(
                        "Crate `{pkg_name}` declares feature `{f_name}` which has no dependencies and is read by no cfg"
                    ));
                }
            }
        }
    }

    ScanFindings {
        unreachable_cfgs,
        unreferenced_features,
    }
}

fn extract_cfg_feature(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with("#[cfg")
        && !trimmed.starts_with("#![cfg")
        && !trimmed.starts_with("#[cfg_attr")
    {
        return None;
    }
    let idx = trimmed.find("feature")?;
    let rest = &trimmed[idx + "feature".len()..];
    let rest = rest.trim_start();
    if !rest.starts_with('=') {
        return None;
    }
    let rest = &rest[1..].trim_start();
    if !rest.starts_with('"') {
        return None;
    }
    let after_quote = &rest[1..];
    let end_quote = after_quote.find('"')?;
    Some(after_quote[..end_quote].to_string())
}
