//! Executable contracts for the generated canonical deployment loader.

use crate::fixture_target;

// The template is the shipped loader source, compiled here so a contract runs
// against the bytes a deployment receives. Compiling it proves it parses; the
// contracts below read every projected field, so a field whose meaning changes
// fails one of them instead of sitting unread behind a lint allowance.
#[path = "../templates/artifact.rs.tmpl"]
pub mod generated_loader;

use std::fs;
use std::path::Path;

use serde_json::json;
use vyre_aot::package_artifact;

/// Weight payload every case in this file packages.
const WEIGHT_BYTES: &[u8] = &[1_u8, 2, 3, 5, 8, 13, 21, 34];

/// Package name every case in this file packages under.
const ARTIFACT_NAME: &str = "generated-loader-contract";

/// Deployment note every case in this file packages, projected verbatim.
const NOTES: &str = "canonical deployment note";

fn package(dir: &Path) -> serde_json::Value {
    let envelope = fixture_target::compiled_artifact();
    package_artifact(
        dir,
        &envelope,
        fixture_target::fixture_target(),
        WEIGHT_BYTES,
        ARTIFACT_NAME,
        NOTES,
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

/// WHY: persisted manifests from the pre-target-identity schema must fail before
/// any artifact bytes are trusted. This does not validate later schema migrations.
#[test]
fn generated_loader_rejects_stale_v3_schema_before_envelope_reads() {
    let dir = tempfile::tempdir().expect("tempdir must be available");
    let mut manifest = package(dir.path());
    manifest["schema"] = json!("vyre-aot-manifest-v3");
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

/// WHY: the generated loader is what a deployment runs, and every field it
/// projects is a binding decision the launcher acts on. Names alone leave
/// element counts, element sizes, memory classes and access classes unread, so
/// a projection that swapped two of them would still pass. Each record is
/// destructured exhaustively, which turns a field added to the template into a
/// compile error here until this contract states what it must carry.
#[test]
fn generated_loader_projects_every_manifest_field_from_the_packaged_envelope() {
    let dir = tempfile::tempdir().expect("tempdir must be available");
    let expected = fixture_target::compiled_artifact();
    package_artifact(
        dir.path(),
        &expected,
        fixture_target::fixture_target(),
        WEIGHT_BYTES,
        ARTIFACT_NAME,
        NOTES,
    )
    .expect("canonical package must write");

    let loaded = generated_loader::load_bundle(dir.path()).expect("valid bundle must load");

    assert_eq!(loaded.kernel_bytes, b"target-payload-fixture");
    assert_eq!(loaded.weight_bytes, WEIGHT_BYTES);
    assert_eq!(
        loaded.envelope.neutral().digest(),
        expected.neutral().digest()
    );

    let generated_loader::Manifest {
        aot_version,
        artifact_name,
        target,
        target_payload_format,
        entry_point,
        dispatch,
        buffers,
        notes,
    } = loaded.manifest;
    assert_eq!(aot_version, vyre_aot::VERSION);
    assert_eq!(artifact_name, ARTIFACT_NAME);
    assert_eq!(target, fixture_target::fixture_target().as_str());
    assert_eq!(target_payload_format, "fixture-target-format");
    assert_eq!(entry_point, "main");
    assert_eq!(notes, NOTES);

    let generated_loader::DispatchConfig {
        workgroup_size,
        grid_size,
        dynamic_shared_bytes,
    } = dispatch;
    assert_eq!(workgroup_size, [64, 1, 1]);
    assert_eq!(grid_size, [1, 1, 1]);
    assert_eq!(dynamic_shared_bytes, 0);

    assert_eq!(
        projected_buffers(&buffers),
        vec![
            (
                "params".to_string(),
                0,
                256,
                4,
                generated_loader::BufferMemoryKind::Global,
                generated_loader::BufferAccessKind::ReadOnly,
            ),
            (
                "out".to_string(),
                1,
                64,
                4,
                generated_loader::BufferMemoryKind::Global,
                generated_loader::BufferAccessKind::WriteOnly,
            ),
        ],
        "the loader must project the declared binding slot, element count, element size, memory class and access class of every canonical resource"
    );
}

/// Every field of every projected buffer entry, in binding order.
///
/// Destructuring is exhaustive, so a field added to the template's
/// `BufferEntry` stops this file compiling until the contract above states what
/// the loader must project into it.
fn projected_buffers(
    buffers: &[generated_loader::BufferEntry],
) -> Vec<(
    String,
    u32,
    u64,
    u64,
    generated_loader::BufferMemoryKind,
    generated_loader::BufferAccessKind,
)> {
    buffers
        .iter()
        .map(|entry| {
            let generated_loader::BufferEntry {
                name,
                binding,
                element_count,
                element_size_bytes,
                memory_kind,
                access,
            } = entry;
            (
                name.clone(),
                *binding,
                *element_count,
                *element_size_bytes,
                *memory_kind,
                *access,
            )
        })
        .collect()
}
