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
