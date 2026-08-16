//! Runtime cache blob contracts over the public `vyre_aot` surface.

mod common;

use vyre_aot::cache::*;
use vyre_foundation::ir::{BufferDecl, DataType, Node, Program};

fn add_one_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(64),
            BufferDecl::output("out", 1, DataType::U32)
                .with_count(64)
                .with_output_byte_range(0..256),
        ],
        [64, 1, 1],
        vec![Node::return_()],
    )
}

#[test]
fn fingerprint_hex_is_64_lowercase_chars() {
    let bytes: [u8; 32] = [0xAB; 32];
    let hex = fingerprint_hex(&bytes);
    assert_eq!(hex.len(), 64);
    assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    assert!(hex.chars().all(|c| !c.is_ascii_uppercase()));
}

#[test]
fn emit_writes_canonical_envelope_with_blake3_footer() {
    let program = add_one_program();
    let artifact = common::artifact_over(
        &program,
        "fixture-target-format",
        (0..1024).map(|index| (index % 251) as u8).collect(),
    );
    let envelope_bytes = artifact.to_bytes().unwrap();
    let dir = tempfile::tempdir().expect("Fix: tempdir must succeed");
    let path = emit_runtime_cache_blob(&artifact, dir.path())
        .expect("Fix: emit must succeed for a valid canonical artifact");

    let blob = std::fs::read(&path).expect("Fix: blob must be readable");
    assert_eq!(blob.len(), envelope_bytes.len() + 32);
    let payload = &blob[..envelope_bytes.len()];
    let footer = &blob[envelope_bytes.len()..];
    assert_eq!(payload, envelope_bytes);
    assert_eq!(footer, blake3::hash(&envelope_bytes).as_bytes());
}

#[test]
fn emit_filename_matches_runtime_fingerprint() {
    let program = add_one_program();
    let artifact = common::artifact_over(
        &program,
        "fixture-target-format",
        b"\x00\x01\x02\x03".to_vec(),
    );
    let dir = tempfile::tempdir().expect("Fix: tempdir must succeed");
    let path = emit_runtime_cache_blob(&artifact, dir.path()).expect("Fix: emit must succeed");

    let expected_fingerprint = artifact.neutral().digest().0;
    let expected_filename = format!("{}.bin", fingerprint_hex(&expected_fingerprint));
    assert_eq!(
        path.file_name().and_then(|s| s.to_str()),
        Some(expected_filename.as_str())
    );
}
