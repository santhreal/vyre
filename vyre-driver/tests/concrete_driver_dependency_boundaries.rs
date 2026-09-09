//! Proves that no concrete driver depends on a semantic library, application policy, or peer driver.
//!
//! WHY: `vyre-driver` owns the neutral driver contract: capability negotiation,
//! physical-IR consumption, typed resource ABI, module creation, submission, and evidence.
//! Concrete drivers (vyre-driver-cuda, vyre-driver-wgpu, vyre-driver-metal, vyre-driver-spirv,
//! vyre-driver-reference) must remain leaf device drivers and cannot import:
//! - Semantic libraries (`vyre-libs`, `vyre-primitives`)
//! - Application policy (`vyre-aot`, `vyre-safetensors`, `vyre-bench`, `vyre-runtime`)
//! - Peer concrete drivers (`vyre-driver-*`)
//!
//! This test derives the dependency graph at runtime using `cargo metadata`.

use std::process::Command;

use serde::Deserialize;
use vyre_test_support::monorepo::vyre_workspace_root;

const SHARED_DRIVER: &str = "vyre-driver";
const CONCRETE_DRIVER_PREFIX: &str = "vyre-driver-";

/// Semantic library crates that concrete drivers must not import in production.
const FORBIDDEN_SEMANTIC_LIBRARIES: &[&str] = &["vyre-libs", "vyre-primitives"];

/// Application policy crates that concrete drivers must not import in production.
const FORBIDDEN_APPLICATION_POLICIES: &[&str] =
    &["vyre-aot", "vyre-safetensors", "vyre-bench", "vyre-runtime"];

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

fn validate_concrete_driver_dependencies(packages: &[PackageMetadata]) -> Result<(), Vec<String>> {
    let mut violations = Vec::new();

    for pkg in packages {
        if !pkg.name.starts_with(CONCRETE_DRIVER_PREFIX) {
            continue;
        }

        for dep in &pkg.dependencies {
            // Check only normal/production dependencies (not dev-dependencies or build-dependencies).
            // In cargo metadata, normal dependencies have `kind: null` (or `kind: None`).
            let is_production_dep = dep.kind.is_none() || dep.kind.as_deref() == Some("normal");
            if !is_production_dep {
                continue;
            }

            // 1. Semantic libraries check
            if FORBIDDEN_SEMANTIC_LIBRARIES.contains(&dep.name.as_str()) {
                violations.push(format!(
                    "Concrete driver `{}` depends on forbidden semantic library `{}` in production dependencies",
                    pkg.name, dep.name
                ));
            }

            // 2. Application policy check
            if FORBIDDEN_APPLICATION_POLICIES.contains(&dep.name.as_str()) {
                violations.push(format!(
                    "Concrete driver `{}` depends on forbidden application policy crate `{}` in production dependencies",
                    pkg.name, dep.name
                ));
            }

            // 3. Peer concrete driver check
            if dep.name.starts_with(CONCRETE_DRIVER_PREFIX)
                && dep.name != pkg.name
                && dep.name != SHARED_DRIVER
            {
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
    let metadata = load_cargo_metadata();
    if let Err(violations) = validate_concrete_driver_dependencies(&metadata.packages) {
        panic!(
            "Fix: concrete drivers violate dependency boundaries:\n{}",
            violations.join("\n")
        );
    }
}

#[test]
fn dependency_validator_catches_adversarial_semantic_library_dependency() {
    let mock_packages = vec![PackageMetadata {
        name: "vyre-driver-mock".to_string(),
        dependencies: vec![DependencyMetadata {
            name: "vyre-libs".to_string(),
            kind: None,
        }],
    }];

    let result = validate_concrete_driver_dependencies(&mock_packages);
    assert!(result.is_err());
    let errs = result.unwrap_err();
    assert!(errs
        .iter()
        .any(|msg| msg.contains("forbidden semantic library `vyre-libs`")));
}

#[test]
fn dependency_validator_catches_adversarial_peer_driver_dependency() {
    let mock_packages = vec![PackageMetadata {
        name: "vyre-driver-cuda".to_string(),
        dependencies: vec![DependencyMetadata {
            name: "vyre-driver-wgpu".to_string(),
            kind: None,
        }],
    }];

    let result = validate_concrete_driver_dependencies(&mock_packages);
    assert!(result.is_err());
    let errs = result.unwrap_err();
    assert!(errs
        .iter()
        .any(|msg| msg.contains("peer concrete driver `vyre-driver-wgpu`")));
}

#[test]
fn dependency_validator_catches_adversarial_application_policy_dependency() {
    let mock_packages = vec![PackageMetadata {
        name: "vyre-driver-spirv".to_string(),
        dependencies: vec![DependencyMetadata {
            name: "vyre-runtime".to_string(),
            kind: None,
        }],
    }];

    let result = validate_concrete_driver_dependencies(&mock_packages);
    assert!(result.is_err());
    let errs = result.unwrap_err();
    assert!(errs
        .iter()
        .any(|msg| msg.contains("forbidden application policy crate `vyre-runtime`")));
}
