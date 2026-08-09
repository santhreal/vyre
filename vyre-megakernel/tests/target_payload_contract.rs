//! Regression contracts for canonical neutral artifacts with attached target payloads.

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Program, ProgramGraph, ShapeDim, ValueContract,
    ValueLifetime,
};
use vyre_megakernel::{
    attach_target, compile, Artifact, ArtifactEnvelope, ArtifactNodeId, ArtifactValueId,
    CompileRequest, DiagnosticCode, Digest, ExternalFacts, FusionGroupId, SearchBudget,
    TargetCompileError, TargetCompiler, TargetEntryPoint, TargetModuleBundle, TargetModuleImage,
    TargetPayload, TargetPayloadFormat, TargetResourceAccess, TargetResourceBinding,
    TargetResourceMemory,
};

fn neutral_artifact(workgroup_size: [u32; 3]) -> Artifact {
    let mut graph = ProgramGraph::new();
    graph
        .add_external_value(
            "input",
            ValueContract {
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
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        SearchBudget::new(1, 1, 1, 0, 1_000_000_000),
        1_000_000,
    )
    .validate()
    .expect("fixture request must validate");
    compile(&request).expect("fixture request must compile")
}

fn format(version: u16) -> TargetPayloadFormat {
    TargetPayloadFormat::new("test.target-binary", version).expect("fixture format must be valid")
}
struct FixtureCompiler {
    format: TargetPayloadFormat,
}

impl TargetCompiler for FixtureCompiler {
    fn format(&self) -> &TargetPayloadFormat {
        &self.format
    }

    fn compile(&self, artifact: &Artifact) -> Result<TargetPayload, TargetCompileError> {
        TargetPayload::new(artifact, self.format.clone(), vec![entry()], vec![4, 2])
            .map_err(Into::into)
    }
}

/// WHY: every product must associate pure target output through one authenticated envelope seam.
#[test]
fn target_compiler_attaches_exactly_one_authenticated_payload() {
    let neutral = neutral_artifact([8, 1, 1]);
    let neutral_digest = neutral.digest();
    let compiler = FixtureCompiler { format: format(9) };

    let envelope = attach_target(neutral, &compiler).expect("target attachment must succeed");

    assert_eq!(envelope.neutral().digest(), neutral_digest);
    let payload = envelope
        .require_target_payload(compiler.format())
        .expect("compiler format must be attached");
    assert_eq!(payload.neutral_artifact(), neutral_digest);
    assert_eq!(payload.bytes(), &[4, 2]);
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
    let mut envelope = ArtifactEnvelope::new(neutral);
    envelope
        .attach_target_payload(payload)
        .expect("matching payload must attach");

    let decoded = ArtifactEnvelope::from_bytes(&envelope.to_bytes().expect("envelope must encode"))
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
    let mut wrong_envelope = ArtifactEnvelope::new(second);

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
    payload_bytes[4..6].copy_from_slice(&3u16.to_le_bytes());
    let schema_error = TargetPayload::from_bytes(&payload_bytes)
        .expect_err("unsupported attachment schema must fail before body admission");
    assert_eq!(
        schema_error.diagnostic.code,
        DiagnosticCode::TargetPayloadVersionSkew
    );
    assert_eq!(
        schema_error.diagnostic.path,
        "target_payload.schema_version"
    );

    let mut envelope = ArtifactEnvelope::new(neutral);
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

/// WHY: authenticated payload bytes must not admit duplicate or out-of-order selected modules.
#[test]
fn target_module_bundle_rejects_noncanonical_module_order() {
    let image = |stage, group| TargetModuleImage {
        group: FusionGroupId(group),
        stage,
        entry_point: format!("group_{group}"),
        bytes: vec![group as u8],
    };
    for modules in [
        vec![image(1, 0), image(0, 1)],
        vec![image(0, 1), image(0, 0)],
        vec![image(0, 0), image(0, 0)],
    ] {
        let bytes = serde_json::to_vec(&TargetModuleBundle {
            schema_version: vyre_megakernel::TARGET_MODULE_BUNDLE_SCHEMA_VERSION,
            modules,
        })
        .expect("fixture bundle must encode");
        let error = TargetModuleBundle::from_bytes(&bytes)
            .expect_err("noncanonical stage/group order must fail admission");
        assert!(
            error.to_string().contains("canonical stage/group order"),
            "unexpected admission error: {error}"
        );
    }
}
