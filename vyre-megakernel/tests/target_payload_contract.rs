//! Regression contracts for canonical neutral artifacts with attached target payloads.

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Program, ProgramGraph, ShapeDim, TensorContract,
    ValueLifetime,
};
use vyre_megakernel::{
    compile, ArtifactNodeId, ArtifactRoute, ArtifactValueId, CompileOptions, DiagnosticCode,
    MegakernelArtifact, MegakernelArtifactEnvelope, TargetEntryPoint, TargetPayload,
    TargetPayloadFormat, TargetResourceAccess, TargetResourceBinding, TargetResourceMemory,
    ValidatedCompileRequest,
};

fn neutral_artifact(workgroup_size: [u32; 3]) -> MegakernelArtifact {
    let mut graph = ProgramGraph::new();
    graph
        .add_external_value(
            "input",
            TensorContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(8)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
        )
        .expect("fixture resource must be valid");
    graph
        .add_node(
            "entry",
            Program::wrapped(
                vec![BufferDecl::read("input", 0, DataType::U32).with_count(8)],
                workgroup_size,
                Vec::new(),
            ),
            Vec::new(),
            Vec::new(),
        )
        .expect("fixture node must be valid");
    let request = ValidatedCompileRequest::new(
        graph,
        CompileOptions::new(ArtifactRoute::Static, BTreeMap::new(), 1_000_000),
    )
    .expect("fixture request must validate");
    compile(&request).expect("fixture request must compile")
}

fn format(version: u16) -> TargetPayloadFormat {
    TargetPayloadFormat::new("test.target-binary", version).expect("fixture format must be valid")
}

fn entry() -> TargetEntryPoint {
    TargetEntryPoint {
        name: "entry".into(),
        node: ArtifactNodeId(0),
        grid_size: [4, 1, 1],
        dynamic_shared_bytes: 64,
        resource_bindings: vec![TargetResourceBinding {
            resource: ArtifactValueId(0),
            slot: 3,
            memory: TargetResourceMemory::Global,
            access: TargetResourceAccess::ReadOnly,
        }],
    }
}

/// Regression: packaging target bytes must retain the exact neutral artifact, entry IDs, and bytes.
#[test]
fn neutral_envelope_and_target_payload_round_trip_exactly() {
    let neutral = neutral_artifact([8, 1, 1]);
    let neutral_digest = neutral.digest();
    let payload = TargetPayload::new(&neutral, format(7), vec![entry()], vec![0, 3, 7, 255])
        .expect("valid target payload must bind");
    let payload_digest = payload.digest();
    let mut envelope = MegakernelArtifactEnvelope::new(neutral);
    envelope
        .attach_target_payload(payload)
        .expect("matching payload must attach");

    let decoded = MegakernelArtifactEnvelope::from_bytes(
        &envelope.to_bytes().expect("envelope must encode"),
    )
    .expect("canonical envelope must decode");
    let decoded_payload = decoded
        .require_target_payload(&format(7))
        .expect("exact format must be compatible");

    assert_eq!(decoded.neutral().digest(), neutral_digest);
    assert_eq!(decoded_payload.neutral_artifact(), neutral_digest);
    assert_eq!(decoded_payload.digest(), payload_digest);
    assert_eq!(decoded_payload.bytes(), &[0, 3, 7, 255]);
    assert_eq!(decoded_payload.entries()[0].node, ArtifactNodeId(0));
    assert_eq!(
        decoded_payload.entries()[0].resource_bindings[0].resource,
        ArtifactValueId(0)
    );
    assert_eq!(decoded.neutral().geometry()[0].workgroup_size, [8, 1, 1]);
}

/// Regression: target bytes materialized from one neutral artifact must not attach to another digest.
#[test]
fn target_payload_rejects_a_different_neutral_artifact_digest() {
    let first = neutral_artifact([8, 1, 1]);
    let second = neutral_artifact([16, 1, 1]);
    assert_ne!(first.digest(), second.digest());
    let payload = TargetPayload::new(&first, format(1), vec![entry()], vec![9, 8, 7])
        .expect("payload must bind to its source artifact");
    let mut wrong_envelope = MegakernelArtifactEnvelope::new(second);

    let error = wrong_envelope
        .attach_target_payload(payload)
        .expect_err("wrong neutral artifact must fail deterministically");
    assert_eq!(
        error.diagnostic.code,
        DiagnosticCode::TargetPayloadAssociationMismatch
    );
    assert_eq!(error.diagnostic.path, "target_payload.neutral_artifact");
}

/// Regression: consumers must reject both unsupported payload schemas and format versions.
#[test]
fn target_payload_schema_and_format_version_skew_are_rejected() {
    let neutral = neutral_artifact([8, 1, 1]);
    let payload = TargetPayload::new(&neutral, format(2), vec![entry()], vec![1, 2, 3])
        .expect("payload must construct");
    let mut payload_bytes = payload.to_bytes().expect("payload must encode");
    payload_bytes[4..6].copy_from_slice(&2u16.to_le_bytes());
    let schema_error = TargetPayload::from_bytes(&payload_bytes)
        .expect_err("unsupported attachment schema must fail before body admission");
    assert_eq!(
        schema_error.diagnostic.code,
        DiagnosticCode::TargetPayloadVersionSkew
    );
    assert_eq!(schema_error.diagnostic.path, "target_payload.schema_version");

    let mut envelope = MegakernelArtifactEnvelope::new(neutral);
    envelope
        .attach_target_payload(payload)
        .expect("matching neutral association must attach");
    let format_error = envelope
        .require_target_payload(&format(3))
        .expect_err("wrong target format version must fail");
    assert_eq!(
        format_error.diagnostic.code,
        DiagnosticCode::TargetPayloadVersionSkew
    );
    assert_eq!(
        format_error.diagnostic.path,
        "envelope.target_payloads.format.version"
    );
}

/// Regression: mutation of target bytes must fail the target identity before attachment.
#[test]
fn corrupted_target_payload_identity_is_rejected() {
    let neutral = neutral_artifact([8, 1, 1]);
    let payload = TargetPayload::new(&neutral, format(1), vec![entry()], vec![11, 22, 33])
        .expect("payload must construct");
    let mut bytes = payload.to_bytes().expect("payload must encode");
    bytes[10] ^= 1;

    let error = TargetPayload::from_bytes(&bytes)
        .expect_err("payload mutation must fail its domain-separated digest");
    assert_eq!(
        error.diagnostic.code,
        DiagnosticCode::TargetPayloadDigestMismatch
    );
    assert_eq!(error.diagnostic.path, "target_payload.digest");
}
