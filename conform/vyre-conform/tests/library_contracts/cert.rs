//! `cert` contracts over the public `vyre_conform` surface.

use vyre_conform_spec::Certificate;
use vyre::ir::OpId;
use vyre_conform::cert::*;

#[test]
fn round_trips_through_json() {
    let op_id: OpId = "primitive.bitwise.xor".into();
    let cert = Certificate::new(
        op_id.to_string(),
        "backend-a",
        "0.5.0",
        vec!["Commutative".to_string(), "Associative".to_string()],
    );
    let json = cert.to_json().unwrap();
    let back: Certificate = serde_json::from_str(&json).unwrap();
    assert_eq!(cert, back);
}

#[test]
fn issue_populates_fingerprints_and_timestamp() {
    let sig = "01".repeat(64);
    let key = "02".repeat(32);
    let op_id: OpId = "vyre-libs::nn::softmax".into();
    let input = IssueInput {
        op_id: &op_id,
        program_wire_bytes: b"wire-bytes",
        witness_bytes: b"witness-bytes",
        backend_id: "backend-a",
        backend_version: "24.0.5",
        laws_verified: vec!["Commutative".to_string()],
        timestamp: "2026-04-20T00:00:00Z",
        signature_ed25519: &sig,
        pubkey: &key,
    };
    let cert = issue_certificate(input).unwrap();
    assert_eq!(cert.op_id.as_str(), "vyre-libs::nn::softmax");
    assert_eq!(cert.backend_id, "backend-a");
    assert_eq!(cert.program_blake3.len(), 64);
    assert_eq!(cert.witness_set_blake3.len(), 64);
    assert_eq!(cert.timestamp, "2026-04-20T00:00:00Z");
}

#[test]
fn issue_rejects_empty_program_wire() {
    let op_id: OpId = "x".into();
    let input = IssueInput {
        op_id: &op_id,
        program_wire_bytes: b"",
        witness_bytes: b"w",
        backend_id: "b",
        backend_version: "v",
        laws_verified: vec![],
        timestamp: "2026-04-20T00:00:00Z",
        signature_ed25519: "a",
        pubkey: "b",
    };
    assert!(matches!(
        issue_certificate(input),
        Err(CertificateError::EmptyProgramWire)
    ));
}

#[test]
fn verify_structural_catches_tbd_sentinel() {
    let cert = Certificate::new("x", "backend-a", "0.5.0", vec![]);
    let err = verify_structural(&cert).unwrap_err();
    assert!(matches!(err, CertificateError::UnsetField(_)));
}

#[test]
fn verify_structural_accepts_real_cert() {
    let sig = "ab".repeat(32);
    let key = "cd".repeat(16);
    let op_id: OpId = "x".into();
    let input = IssueInput {
        op_id: &op_id,
        program_wire_bytes: b"p",
        witness_bytes: b"w",
        backend_id: "backend-a",
        backend_version: "24.0.5",
        laws_verified: vec![],
        timestamp: "2026-04-20T00:00:00Z",
        signature_ed25519: &sig,
        pubkey: &key,
    };
    let cert = issue_certificate(input).unwrap();
    verify_structural(&cert).expect("Fix: issued cert must pass structural verify; restore this invariant before continuing.");
}
