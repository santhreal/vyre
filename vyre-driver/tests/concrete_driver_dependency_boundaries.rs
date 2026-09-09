//! Proves that no concrete driver depends on a semantic library, schedule search, application policy, or peer driver.
//!
//! WHY: `vyre-driver` owns the neutral driver contract: capability negotiation,
//! physical-IR consumption, typed resource ABI, module creation, submission, and evidence.
//! Concrete drivers must remain leaf device drivers and cannot import:
//! - Semantic libraries (`layer = "libraries"` or `layer = "primitives"`)
//! - Schedule search (`layer = "pass-engine"`)
//! - Application policy (`layer = "runtime"`, `layer = "packaging"`, or consumers)
//! - Peer concrete drivers (`layer = "concrete-backend"`)
//!
//! This test derives the crate list and layer assignments directly from `docs/CRATE_OWNERSHIP.toml`
//! and validates both `Cargo.toml` production dependencies and production `.rs` source imports.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;
use toml::Value as TomlValue;
use vyre_test_support::monorepo::vyre_workspace_root;

#[derive(Debug, Deserialize)]
struct MetadataOutput {
    packages: Vec<PackageMetadata>,
}

#[derive(Debug, Deserialize)]
struct PackageMetadata {
    name: String,
    dependencies: Vec<DependencyMetadata>,
}

#[derive(Debug, Deserialize)]
struct DependencyMetadata {
    name: String,
    kind: Option<String>,
}

/// Discovered layers and their member crates parsed from `docs/CRATE_OWNERSHIP.toml`.
#[derive(Debug)]
struct OwnershipModel {
    concrete_drivers: BTreeSet<String>,
    semantic_libraries: BTreeSet<String>,
    schedule_search_crates: BTreeSet<String>,
    application_policy_crates: BTreeSet<String>,
}

fn load_ownership_model(root: &Path) -> OwnershipModel {
    let ownership_path = root.join("docs/CRATE_OWNERSHIP.toml");
    let content = fs::read_to_string(&ownership_path)
        .unwrap_or_else(|e| panic!("Fix: failed to read {}: {e}", ownership_path.display()));
    let toml: TomlValue = toml::from_str(&content)
        .unwrap_or_else(|e| panic!("Fix: failed to parse {}: {e}", ownership_path.display()));

    let mut concrete_drivers = BTreeSet::new();
    let mut semantic_libraries = BTreeSet::new();
    let mut schedule_search_crates = BTreeSet::new();
    let mut application_policy_crates = BTreeSet::new();

    // Default built-in classifications matching CRATE_OWNERSHIP doctrine
    semantic_libraries.insert("vyre-libs".to_string());
    semantic_libraries.insert("vyre-primitives".to_string());
    schedule_search_crates.insert("vyre-pass-engine".to_string());
    application_policy_crates.insert("vyre-runtime".to_string());
    application_policy_crates.insert("vyre-aot".to_string());
    application_policy_crates.insert("vyre-safetensors".to_string());
    application_policy_crates.insert("vyre-bench".to_string());

    if let Some(crates) = toml.get("crate").and_then(|c| c.as_array()) {
        for entry in crates {
            let package = entry.get("package").and_then(|p| p.as_str());
            let layer = entry.get("layer").and_then(|l| l.as_str());

            if let (Some(pkg), Some(layer_name)) = (package, layer) {
                match layer_name {
                    "concrete-backend" => {
                        concrete_drivers.insert(pkg.to_string());
                    }
                    "libraries" | "primitives" => {
                        semantic_libraries.insert(pkg.to_string());
                    }
                    "pass-engine" => {
                        schedule_search_crates.insert(pkg.to_string());
                    }
                    "runtime" | "packaging" => {
                        application_policy_crates.insert(pkg.to_string());
                    }
                    _ => {}
                }
            }
        }
    }

    assert!(
        concrete_drivers.len() >= 5,
        "Fix: expected at least 5 concrete drivers in CRATE_OWNERSHIP.toml, found {}: {:?}",
        concrete_drivers.len(),
        concrete_drivers
    );

    OwnershipModel {
        concrete_drivers,
        semantic_libraries,
        schedule_search_crates,
        application_policy_crates,
    }
}

fn load_cargo_metadata() -> MetadataOutput {
    let root = vyre_workspace_root();
    let output = Command::new("cargo")
        .args([
            "metadata",
            "--format-version",
            "1",
            "--manifest-path",
            root.join("Cargo.toml").to_str().unwrap(),
            "--no-deps",
        ])
        .output()
        .expect("Fix: `cargo metadata` must succeed in the workspace");

    assert!(
        output.status.success(),
        "Fix: cargo metadata exited with non-zero code: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    serde_json::from_slice(&output.stdout).expect("Fix: cargo metadata output must parse as JSON")
}

fn validate_concrete_driver_dependencies(
    packages: &[PackageMetadata],
    model: &OwnershipModel,
) -> Result<(), Vec<String>> {
    let mut violations = Vec::new();

    for pkg in packages {
        if !model.concrete_drivers.contains(&pkg.name) {
            continue;
        }

        for dep in &pkg.dependencies {
            // Check only normal/production dependencies (not dev-dependencies or build-dependencies).
            let is_production_dep = dep.kind.is_none() || dep.kind.as_deref() == Some("normal");
            if !is_production_dep {
                continue;
            }

            // 1. Semantic libraries check
            if model.semantic_libraries.contains(&dep.name) {
                violations.push(format!(
                    "Concrete driver `{}` depends on forbidden semantic library `{}` in production dependencies",
                    pkg.name, dep.name
                ));
            }

            // 2. Schedule search check
            if model.schedule_search_crates.contains(&dep.name) {
                violations.push(format!(
                    "Concrete driver `{}` depends on forbidden schedule search crate `{}` in production dependencies",
                    pkg.name, dep.name
                ));
            }

            // 3. Application policy check
            if model.application_policy_crates.contains(&dep.name) {
                violations.push(format!(
                    "Concrete driver `{}` depends on forbidden application policy crate `{}` in production dependencies",
                    pkg.name, dep.name
                ));
            }

            // 4. Peer concrete driver check
            if model.concrete_drivers.contains(&dep.name) && dep.name != pkg.name {
                violations.push(format!(
                    "Concrete driver `{}` depends on peer concrete driver `{}` in production dependencies",
                    pkg.name, dep.name
                ));
            }
        }
    }

    if violations.is_empty() {
        Ok(())
    } else {
        Err(violations)
    }
}

#[test]
fn concrete_drivers_have_no_forbidden_production_dependencies() {
    let root = vyre_workspace_root();
    let model = load_ownership_model(&root);
    let metadata = load_cargo_metadata();

    if let Err(violations) = validate_concrete_driver_dependencies(&metadata.packages, &model) {
        panic!(
            "Fix: concrete drivers violate dependency boundaries:\n{}",
            violations.join("\n")
        );
    }
}

#[test]
fn concrete_driver_source_files_have_no_forbidden_production_imports() {
    let root = vyre_workspace_root();
    let model = load_ownership_model(&root);

    let forbidden_identifiers = [
        ("vyre_libs", "semantic library"),
        ("vyre_pass_engine", "schedule search"),
    ];

    for driver in &model.concrete_drivers {
        let src_dir = root.join(driver).join("src");
        if !src_dir.exists() {
            continue;
        }

        let mut rs_files = Vec::new();
        collect_rs_files(&src_dir, &mut rs_files);

        for file_path in rs_files {
            let content = fs::read_to_string(&file_path)
                .unwrap_or_else(|e| panic!("Fix: cannot read {}: {e}", file_path.display()));

            for (forbidden, category) in &forbidden_identifiers {
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("//") || trimmed.starts_with("/*") {
                        continue;
                    }
                    if (trimmed.starts_with("use ") || trimmed.starts_with("extern crate "))
                        && trimmed.contains(forbidden)
                    {
                        panic!(
                            "Fix: concrete driver `{driver}` file `{}` contains forbidden {category} import: `{line}`",
                            file_path.display()
                        );
                    }
                }
            }
        }
    }
}

fn collect_rs_files(dir: &Path, files: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_rs_files(&path, files);
            } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
                files.push(path);
            }
        }
    }
}

#[test]
fn dependency_validator_catches_adversarial_semantic_library_dependency() {
    let model = OwnershipModel {
        concrete_drivers: BTreeSet::from(["vyre-driver-mock".to_string()]),
        semantic_libraries: BTreeSet::from(["vyre-libs".to_string()]),
        schedule_search_crates: BTreeSet::new(),
        application_policy_crates: BTreeSet::new(),
    };
    let mock_packages = vec![PackageMetadata {
        name: "vyre-driver-mock".to_string(),
        dependencies: vec![DependencyMetadata {
            name: "vyre-libs".to_string(),
            kind: None,
        }],
    }];

    let result = validate_concrete_driver_dependencies(&mock_packages, &model);
    assert!(result.is_err());
    let errs = result.unwrap_err();
    assert!(errs
        .iter()
        .any(|msg| msg.contains("forbidden semantic library `vyre-libs`")));
}

#[test]
fn dependency_validator_catches_adversarial_schedule_search_dependency() {
    let model = OwnershipModel {
        concrete_drivers: BTreeSet::from(["vyre-driver-mock".to_string()]),
        semantic_libraries: BTreeSet::new(),
        schedule_search_crates: BTreeSet::from(["vyre-pass-engine".to_string()]),
        application_policy_crates: BTreeSet::new(),
    };
    let mock_packages = vec![PackageMetadata {
        name: "vyre-driver-mock".to_string(),
        dependencies: vec![DependencyMetadata {
            name: "vyre-pass-engine".to_string(),
            kind: None,
        }],
    }];

    let result = validate_concrete_driver_dependencies(&mock_packages, &model);
    assert!(result.is_err());
    let errs = result.unwrap_err();
    assert!(errs
        .iter()
        .any(|msg| msg.contains("forbidden schedule search crate `vyre-pass-engine`")));
}

#[test]
fn dependency_validator_catches_adversarial_peer_driver_dependency() {
    let model = OwnershipModel {
        concrete_drivers: BTreeSet::from([
            "vyre-driver-cuda".to_string(),
            "vyre-driver-wgpu".to_string(),
        ]),
        semantic_libraries: BTreeSet::new(),
        schedule_search_crates: BTreeSet::new(),
        application_policy_crates: BTreeSet::new(),
    };
    let mock_packages = vec![PackageMetadata {
        name: "vyre-driver-cuda".to_string(),
        dependencies: vec![DependencyMetadata {
            name: "vyre-driver-wgpu".to_string(),
            kind: None,
        }],
    }];

    let result = validate_concrete_driver_dependencies(&mock_packages, &model);
    assert!(result.is_err());
    let errs = result.unwrap_err();
    assert!(errs
        .iter()
        .any(|msg| msg.contains("peer concrete driver `vyre-driver-wgpu`")));
}

#[test]
fn dependency_validator_catches_adversarial_application_policy_dependency() {
    let model = OwnershipModel {
        concrete_drivers: BTreeSet::from(["vyre-driver-spirv".to_string()]),
        semantic_libraries: BTreeSet::new(),
        schedule_search_crates: BTreeSet::new(),
        application_policy_crates: BTreeSet::from(["vyre-runtime".to_string()]),
    };
    let mock_packages = vec![PackageMetadata {
        name: "vyre-driver-spirv".to_string(),
        dependencies: vec![DependencyMetadata {
            name: "vyre-runtime".to_string(),
            kind: None,
        }],
    }];

    let result = validate_concrete_driver_dependencies(&mock_packages, &model);
    assert!(result.is_err());
    let errs = result.unwrap_err();
    assert!(errs
        .iter()
        .any(|msg| msg.contains("forbidden application policy crate `vyre-runtime`")));
}
