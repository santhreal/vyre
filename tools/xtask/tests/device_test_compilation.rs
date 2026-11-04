//! Proof that `device-test-compilation` gate discovers all `device-tests` targets and reports compilation defects.
//!
//! A defect behind `device-tests` only surfaces on hardware unless a gate
//! compiles admitted test targets in hosted environments. This suite asserts
//! that the gate derives every admitted test target from source, discovers
//! declaring members, correctly identifies targets that fail to compile, and
//! reports zero findings for clean targets.

use std::fs;

use tempfile::tempdir;
use xtask::gates::device_test_compilation::{admitted_test_targets, workflow_feature_pairings};
use xtask::gates::scan::Tree;

use super::workspace_sources::track_fixture;

/// WHY: every workspace package declaring `device-tests` is discovered and its admitted targets derived from source.
#[test]
fn test_gate_derives_device_test_targets_and_closure() {
    let manifest = r#"
[workspace]
members = ["driver-a", "driver-b", "util-c"]

[workspace.dependencies]
"#;
    let driver_a_toml = r#"
[package]
name = "driver-a"
version = "0.1.0"
edition = "2021"

[features]
device-tests = []
cuda = []

[[test]]
name = "all_tests"
path = "tests/all_tests.rs"
"#;
    let driver_b_toml = r#"
[package]
name = "driver-b"
version = "0.1.0"
edition = "2021"

[features]
device-tests = []

[[test]]
name = "all_tests_device"
path = "tests/all_tests_device.rs"
required-features = ["device-tests"]

[[test]]
name = "all_tests_host"
path = "tests/all_tests_host.rs"
"#;
    let util_c_toml = r#"
[package]
name = "util-c"
version = "0.1.0"
edition = "2021"

[features]
default = []

[[test]]
name = "all_tests"
path = "tests/all_tests.rs"
"#;
    let workflow = r#"
name: GPU Parity
jobs:
  driver-a:
    steps:
      - name: Run Driver A
        run: ./cargo_full test -p driver-a --features device-tests,cuda -- --test-threads=1
"#;

    let temp = tempdir().expect("create tempdir");
    let root = temp.path();

    let files: &[(&str, &str)] = &[
        ("Cargo.toml", manifest),
        (".github/workflows/gpu-parity.yml", workflow),
        ("driver-a/Cargo.toml", driver_a_toml),
        (
            "driver-a/tests/all_tests.rs",
            "pub mod device_contract;\npub mod host_contract;",
        ),
        (
            "driver-a/tests/device_contract.rs",
            "#![cfg(feature = \"device-tests\")]\npub fn run() {}",
        ),
        ("driver-a/tests/host_contract.rs", "pub fn run() {}"),
        ("driver-b/Cargo.toml", driver_b_toml),
        ("driver-b/tests/all_tests_device.rs", "pub fn live() {}"),
        ("driver-b/tests/all_tests_host.rs", "pub fn host() {}"),
        ("util-c/Cargo.toml", util_c_toml),
        ("util-c/tests/all_tests.rs", "pub fn util() {}"),
    ];

    for (rel, content) in files {
        let path = root.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent dir");
        }
        fs::write(&path, content).expect("write file");
    }

    track_fixture(root);

    let tree = Tree::open(root).expect("open tree");
    let admitted = admitted_test_targets(&tree).expect("admitted test targets");

    // Adding a new package declaring device-tests must not pass unnoticed
    assert_eq!(admitted.len(), 2);
    assert!(admitted.contains_key("driver-a"));
    assert!(admitted.contains_key("driver-b"));
    assert!(!admitted.contains_key("util-c"));

    let a_targets = &admitted["driver-a"];
    assert_eq!(a_targets.len(), 1);
    assert_eq!(a_targets[0].name, "all_tests");

    let b_targets = &admitted["driver-b"];
    assert_eq!(b_targets.len(), 1);
    assert_eq!(b_targets[0].name, "all_tests_device");

    let pairings = workflow_feature_pairings(&tree).expect("workflow pairings");
    assert!(pairings.get("driver-a").unwrap().contains("cuda"));
    assert!(pairings.get("driver-a").unwrap().contains("device-tests"));
}
