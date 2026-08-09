//! Executable contracts for the generated canonical deployment loader.

#![allow(dead_code, unreachable_pub)]

mod common;
#[path = "../templates/artifact.rs.tmpl"]
mod generated_loader;

use std::fs;
use std::path::Path;

use serde_json::json;
use vyre_aot::{package_artifact, Target};

fn package(dir: &Path) -> serde_json::Value {
    let envelope = common::compiled_artifact();
    package_artifact(
        dir,
        &envelope,
        Target::Ptx,
        &[1_u8, 2, 3, 5, 8, 13, 21, 34],
        "generated-loader-contract",
        "",
    )
    .expect("canonical package must write");
    serde_json::from_slice(&fs::read(dir.join("manifest.json")).unwrap()).unwrap()
}

fn write_manifest(dir: &Path, manifest: &serde_json::Value) {
    fs::write(
        dir.join("manifest.json"),
        serde_json::to_vec_pretty(manifest).expect("manifest JSON must serialize"),
    )
    .expect("manifest write must succeed");
}

#[test]
fn generated_loader_rejects_schema_mismatch_before_envelope_reads() {
    let dir = tempfile::tempdir().expect("tempdir must be available");
    let mut manifest = package(dir.path());
    manifest["schema"] = json!("vyre-aot-manifest-v2");
    write_manifest(dir.path(), &manifest);
    fs::remove_file(dir.path().join("artifact.vmk.lzma")).unwrap();

    let error = generated_loader::load_bundle(dir.path())
        .expect_err("stale manifest must be rejected")
        .to_string();

    assert!(error.contains("unsupported manifest schema"), "{error}");
    assert!(!error.contains("read bundle file"), "{error}");
}

#[test]
fn generated_loader_rejects_path_escape_before_envelope_reads() {
    let dir = tempfile::tempdir().expect("tempdir must be available");
    let mut manifest = package(dir.path());
    manifest["envelope_file"] = json!("../artifact.vmk.lzma");
    write_manifest(dir.path(), &manifest);

    let error = generated_loader::load_bundle(dir.path())
        .expect_err("escaping path must be rejected")
        .to_string();

    assert!(error.contains("escapes the bundle root"), "{error}");
}

#[test]
fn generated_loader_rejects_envelope_digest_mismatch() {
    let dir = tempfile::tempdir().expect("tempdir must be available");
    let mut manifest = package(dir.path());
    manifest["envelope_sha256_hex"] =
        json!("0000000000000000000000000000000000000000000000000000000000000000");
    write_manifest(dir.path(), &manifest);

    let error = generated_loader::load_bundle(dir.path())
        .expect_err("mismatched envelope digest must be rejected")
        .to_string();

    assert!(error.contains("envelope SHA-256 mismatch"), "{error}");
}

#[test]
fn generated_loader_projects_target_module_and_abi_from_envelope() {
    let dir = tempfile::tempdir().expect("tempdir must be available");
    let expected = common::compiled_artifact();
    package_artifact(
        dir.path(),
        &expected,
        Target::Ptx,
        &[1_u8, 2, 3, 5, 8, 13, 21, 34],
        "generated-loader-contract",
        "",
    )
    .expect("canonical package must write");

    let loaded = generated_loader::load_bundle(dir.path()).expect("valid bundle must load");

    assert_eq!(loaded.kernel_bytes, b"target-payload-fixture");
    assert_eq!(loaded.weight_bytes, [1_u8, 2, 3, 5, 8, 13, 21, 34]);
    assert_eq!(loaded.manifest.entry_point, "main");
    assert_eq!(loaded.manifest.dispatch.workgroup_size, [64, 1, 1]);
    assert_eq!(loaded.manifest.dispatch.grid_size, [1, 1, 1]);
    assert_eq!(loaded.manifest.buffers.len(), 2);
    assert_eq!(loaded.manifest.buffers[0].name, "params");
    assert_eq!(loaded.manifest.buffers[1].name, "out");
    assert_eq!(
        loaded.envelope.neutral().digest(),
        expected.neutral().digest()
    );
}
