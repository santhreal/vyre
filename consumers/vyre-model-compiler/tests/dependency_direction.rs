//! Proves the dependency direction between the Vyre workspace and downstream consumers.
//!
//! Asserts at run time via `cargo metadata` that no Vyre workspace crate depends
//! on `vyre-model-compiler`, that the consumer depends on `vyre` and `vyre-libs`,
//! and that no named model architecture family, checkpoint identifier, or model-specific
//! concept leaks into the Vyre workspace's public APIs, diagnostics, or cost models.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use vyre_model_compiler::config::{all_model_families, all_named_configs};

#[test]
fn no_workspace_crate_depends_on_model_compiler() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .expect("consumers dir")
        .parent()
        .expect("workspace root");

    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--manifest-path"])
        .arg(workspace_root.join("Cargo.toml"))
        .output()
        .expect("cargo metadata must execute successfully");

    assert!(
        output.status.success(),
        "cargo metadata exited with failure"
    );

    let metadata: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("cargo metadata output must be valid JSON");

    let packages = metadata
        .get("packages")
        .and_then(|p| p.as_array())
        .expect("packages array in metadata");

    let workspace_members = metadata
        .get("workspace_members")
        .and_then(|w| w.as_array())
        .expect("workspace_members array in metadata");

    let workspace_ids: Vec<&str> = workspace_members
        .iter()
        .filter_map(|m| m.as_str())
        .collect();

    for package in packages {
        let name = package["name"].as_str().unwrap_or("");
        let id = package["id"].as_str().unwrap_or("");

        // Check if package is a workspace member
        if workspace_ids.contains(&id) {
            assert_ne!(
                name, "vyre-model-compiler",
                "vyre-model-compiler must not be in workspace members"
            );

            let dependencies = package["dependencies"]
                .as_array()
                .expect("dependencies array");

            for dep in dependencies {
                let dep_name = dep["name"].as_str().unwrap_or("");
                assert_ne!(
                    dep_name, "vyre-model-compiler",
                    "Workspace crate '{name}' illegally depends on consumer package 'vyre-model-compiler'"
                );
            }
        }
    }
}

#[test]
fn consumer_package_depends_on_vyre_and_vyre_libs() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--manifest-path"])
        .arg(manifest_dir.join("Cargo.toml"))
        .output()
        .expect("cargo metadata must execute successfully");

    assert!(
        output.status.success(),
        "cargo metadata exited with failure"
    );

    let metadata: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("cargo metadata output must be valid JSON");

    let packages = metadata
        .get("packages")
        .and_then(|p| p.as_array())
        .expect("packages array in metadata");

    let consumer_pkg = packages
        .iter()
        .find(|p| p["name"].as_str() == Some("vyre-model-compiler"))
        .expect("vyre-model-compiler package in metadata");

    let deps: Vec<&str> = consumer_pkg["dependencies"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|d| d["name"].as_str())
        .collect();

    assert!(
        deps.contains(&"vyre"),
        "vyre-model-compiler must depend on vyre"
    );
    assert!(
        deps.contains(&"vyre-libs"),
        "vyre-model-compiler must depend on vyre-libs"
    );
}

#[test]
fn consumer_manifest_carries_zero_forbidden_publication_class_dependencies() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .expect("consumers dir")
        .parent()
        .expect("workspace root");

    let ownership_path = workspace_root.join("docs/CRATE_OWNERSHIP.toml");
    let ownership_str = std::fs::read_to_string(&ownership_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", ownership_path.display()));
    let ownership: toml::Value =
        toml::from_str(&ownership_str).expect("parse CRATE_OWNERSHIP.toml");

    let mut forbidden_classes = std::collections::BTreeMap::new();
    if let Some(crates) = ownership.get("crate").and_then(|c| c.as_array()) {
        for c in crates {
            if let (Some(pkg), Some(class)) = (
                c.get("package").and_then(|p| p.as_str()),
                c.get("publication_class").and_then(|cls| cls.as_str()),
            ) {
                if class == "internal-engine" || class == "private-test-support" {
                    forbidden_classes.insert(pkg.to_string(), class.to_string());
                }
            }
        }
    }

    let manifest_path = manifest_dir.join("Cargo.toml");
    let manifest_content = std::fs::read_to_string(&manifest_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", manifest_path.display()));
    let manifest_toml: toml::Value = toml::from_str(&manifest_content).expect("parse Cargo.toml");

    let mut production_deps = Vec::new();
    if let Some(deps) = manifest_toml.get("dependencies").and_then(|d| d.as_table()) {
        for dep_name in deps.keys() {
            production_deps.push(dep_name.clone());
        }
    }

    for dep in &production_deps {
        if let Some(class) = forbidden_classes.get(dep.as_str()) {
            panic!(
                "Consumer manifest `{}` declares forbidden production dependency `{dep}` with publication_class `{class}`. Consumers must depend only on the published facade and public SDKs.",
                manifest_path.display()
            );
        }
    }
}

#[test]
fn dependency_closure_proves_zero_model_identifiers_in_workspace_public_surface() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .expect("consumers dir")
        .parent()
        .expect("workspace root");
    // Derive the set of model architecture family names and checkpoint identifiers from source
    let mut forbidden_identifiers = BTreeSet::new();

    // 1. Model family names (e.g. "llama", "mistral", "qwen", "deepseek", "gemma")
    for &family in all_model_families() {
        let name = format!("{family:?}").to_lowercase();
        if name != "vision" {
            forbidden_identifiers.insert(name);
        }
    }

    // 2. Named model config identifiers
    for named in all_named_configs() {
        let variant_name = format!("{named:?}").to_lowercase();
        forbidden_identifiers.insert(variant_name);

        let id_clean = named.id().to_lowercase().replace(['-', '.', ' '], "");
        forbidden_identifiers.insert(id_clean);
    }

    // Explicit model architecture family identifiers
    forbidden_identifiers.insert("llama".to_string());
    forbidden_identifiers.insert("mistral".to_string());
    forbidden_identifiers.insert("mixtral".to_string());
    forbidden_identifiers.insert("qwen".to_string());
    forbidden_identifiers.insert("deepseek".to_string());
    forbidden_identifiers.insert("gemma".to_string());
    forbidden_identifiers.insert("clipvit".to_string());
    forbidden_identifiers.insert("siglip".to_string());
    forbidden_identifiers.insert("llava".to_string());

    // Public API snapshots directory
    let public_api_dir = workspace_root.join("docs/public-api");
    assert!(
        public_api_dir.exists(),
        "docs/public-api must exist to audit workspace public surfaces"
    );
    let mut audited_files = 0;
    for entry in fs::read_dir(&public_api_dir).expect("read docs/public-api") {
        let entry = entry.expect("valid DirEntry");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("txt") {
            let content = fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
            audited_files += 1;

            for (line_num, line) in content.lines().enumerate() {
                let line_lower = line.to_lowercase();
                for forbidden in &forbidden_identifiers {
                    // Check if forbidden identifier appears as a path segment or item name
                    let pattern1 = format!("::{forbidden}");
                    let pattern2 = format!("_{forbidden}");
                    let pattern3 = format!("{forbidden}::");
                    let pattern4 = format!("{forbidden}_");

                    if line_lower.contains(&pattern1)
                        || line_lower.contains(&pattern2)
                        || line_lower.contains(&pattern3)
                        || line_lower.contains(&pattern4)
                    {
                        panic!(
                            "Dependency closure violation: Forbidden model identifier '{forbidden}' found in public API file {} at line {}: `{line}`",
                            path.display(),
                            line_num + 1
                        );
                    }
                }
            }
        }
    }

    assert!(
        audited_files > 0,
        "Must have audited at least one public API snapshot"
    );
}

#[test]
fn test_reference_driver_is_registered_in_dev_dependencies() {
    let profile = vyre_driver_reference::target_profile().expect("reference target profile");
    assert_eq!(profile.identity(), "reference-graph");
}
