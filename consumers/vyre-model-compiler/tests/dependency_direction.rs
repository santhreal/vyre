//! Proves the dependency direction between the Vyre workspace and downstream consumers.
//!
//! Asserts at run time via `cargo metadata` that no Vyre workspace crate depends
//! on `vyre-model-compiler`, and that the consumer depends on `vyre` and `vyre-libs`.

use std::path::PathBuf;
use std::process::Command;

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

    assert!(output.status.success(), "cargo metadata exited with failure");

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

    assert!(output.status.success(), "cargo metadata exited with failure");

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
    let ownership: toml::Value = toml::from_str(&ownership_str).expect("parse CRATE_OWNERSHIP.toml");

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
    let manifest_toml: toml::Value = toml::from_str(&manifest_content)
        .unwrap_or_else(|e| panic!("failed to parse {}: {e}", manifest_path.display()));

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
fn test_reference_driver_is_registered_in_dev_dependencies() {
    let backend_id = vyre_driver_reference::registered_backend_id();
    assert_eq!(backend_id, Some("reference"));
}
