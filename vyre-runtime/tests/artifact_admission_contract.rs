//! Regression contracts for canonical runtime artifact admission.

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Program, ProgramGraph, ShapeDim, TensorContract,
    ValueLifetime,
};
use vyre_megakernel::{
    compile, ArtifactNodeId, ArtifactRoute, ArtifactValueId, CompileOptions, DiagnosticCode, Digest,
    MegakernelArtifact, MegakernelArtifactEnvelope, TargetEntryPoint, TargetPayload,
    TargetPayloadFormat, TargetResourceAccess, TargetResourceBinding, TargetResourceMemory,
    ValidatedCompileRequest,
};
use vyre_runtime::artifact_admission::{admit_artifact, ArtifactAdmissionError};

const FRAME_HEADER_BYTES: usize = 10;
const FRAME_DIGEST_BYTES: usize = 32;
const ENVELOPE_DIGEST_DOMAIN: &[u8] = b"vyre-megakernel-envelope-v1\0";
const TARGET_PAYLOAD_DIGEST_DOMAIN: &[u8] = b"vyre-megakernel-target-payload-v1\0";

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

fn format(identity: &str, version: u16) -> TargetPayloadFormat {
    TargetPayloadFormat::new(identity, version).expect("fixture format must be valid")
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

fn payload(
    neutral: &MegakernelArtifact,
    payload_format: TargetPayloadFormat,
    bytes: &[u8],
) -> TargetPayload {
    TargetPayload::new(neutral, payload_format, vec![entry()], bytes.to_vec())
        .expect("fixture payload must be valid")
}

fn envelope_bytes(
    neutral: MegakernelArtifact,
    payloads: impl IntoIterator<Item = TargetPayload>,
) -> Vec<u8> {
    let mut envelope = MegakernelArtifactEnvelope::new(neutral);
    for payload in payloads {
        envelope
            .attach_target_payload(payload)
            .expect("fixture payload must attach");
    }
    envelope.to_bytes().expect("fixture envelope must encode")
}

fn assert_diagnostic(
    error: &ArtifactAdmissionError,
    code: DiagnosticCode,
    path: &str,
    fix: &str,
) {
    let diagnostic = error.diagnostic();
    assert_eq!(diagnostic.code, code);
    assert_eq!(diagnostic.path, path);
    assert_eq!(diagnostic.fix, fix);
}

fn frame_body(frame: &[u8]) -> &[u8] {
    let body_len = u32::from_le_bytes(frame[6..10].try_into().expect("fixed header slice")) as usize;
    &frame[FRAME_HEADER_BYTES..FRAME_HEADER_BYTES + body_len]
}

fn encode_frame(magic: &[u8; 4], version: u16, domain: &[u8], body: &[u8]) -> Vec<u8> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&version.to_le_bytes());
    hasher.update(&(body.len() as u64).to_le_bytes());
    hasher.update(body);
    let digest = hasher.finalize();

    let mut frame = Vec::with_capacity(FRAME_HEADER_BYTES + body.len() + FRAME_DIGEST_BYTES);
    frame.extend_from_slice(magic);
    frame.extend_from_slice(&version.to_le_bytes());
    frame.extend_from_slice(&(body.len() as u32).to_le_bytes());
    frame.extend_from_slice(body);
    frame.extend_from_slice(digest.as_bytes());
    frame
}

fn replace_once(bytes: &[u8], old: &[u8], new: &[u8]) -> Vec<u8> {
    let offset = bytes
        .windows(old.len())
        .position(|window| window == old)
        .expect("fixture fragment must occur");
    let mut replaced = Vec::with_capacity(bytes.len() - old.len() + new.len());
    replaced.extend_from_slice(&bytes[..offset]);
    replaced.extend_from_slice(new);
    replaced.extend_from_slice(&bytes[offset + old.len()..]);
    replaced
}

fn replace_nested_payload(
    envelope: &[u8],
    original_payload: &[u8],
    replacement_payload: &[u8],
) -> Vec<u8> {
    let original_json = serde_json::to_vec(original_payload).expect("bytes must serialize");
    let replacement_json = serde_json::to_vec(replacement_payload).expect("bytes must serialize");
    let body = replace_once(frame_body(envelope), &original_json, &replacement_json);
    encode_frame(b"VME0", 1, ENVELOPE_DIGEST_DOMAIN, &body)
}

fn reassociate_payload(
    payload: &[u8],
    original_neutral: Digest,
    replacement_neutral: Digest,
) -> Vec<u8> {
    let original_json = serde_json::to_vec(&original_neutral).expect("digest must serialize");
    let replacement_json = serde_json::to_vec(&replacement_neutral).expect("digest must serialize");
    let body = replace_once(frame_body(payload), &original_json, &replacement_json);
    encode_frame(b"VTP0", 1, TARGET_PAYLOAD_DIGEST_DOMAIN, &body)
}

/// Regression: admission returns the exact requested bytes and canonical neutral identity.
#[test]
fn admits_exact_payload_and_neutral_digest() {
    let neutral = neutral_artifact([8, 1, 1]);
    let neutral_digest = neutral.digest();
    let required = format("test.target-a", 7);
    let bytes = envelope_bytes(
        neutral,
        [payload(
            &neutral_artifact([8, 1, 1]),
            required.clone(),
            &[0, 3, 7, 255],
        )],
    );

    let admitted = admit_artifact(&bytes, &required).expect("exact payload must be admitted");

    assert_eq!(admitted.neutral().digest(), neutral_digest);
    assert_eq!(admitted.target_payload().format(), &required);
    assert_eq!(admitted.target_payload().neutral_artifact(), neutral_digest);
    assert_eq!(admitted.target_payload().bytes(), &[0, 3, 7, 255]);
}

/// Regression: canonical serialization remains deterministic across attachment order and admission.
#[test]
fn canonical_serialization_round_trip_is_deterministic() {
    let neutral = neutral_artifact([8, 1, 1]);
    let first = payload(&neutral, format("test.target-a", 1), &[1, 2, 3]);
    let second = payload(&neutral, format("test.target-b", 4), &[4, 5, 6]);
    let forward = envelope_bytes(neutral.clone(), [first.clone(), second.clone()]);
    let reverse = envelope_bytes(neutral, [second, first]);
    assert_eq!(forward, reverse);

    let admitted = admit_artifact(&forward, &format("test.target-b", 4))
        .expect("canonical envelope must admit");
    assert_eq!(
        admitted
            .into_envelope()
            .to_bytes()
            .expect("admitted envelope must encode"),
        forward
    );
}

/// Regression: malformed, truncated, and corrupted envelopes retain canonical diagnostics.
#[test]
fn rejects_malformed_truncated_and_corrupted_envelopes() {
    let required = format("test.target-a", 1);
    let malformed = admit_artifact(b"not an envelope", &required)
        .expect_err("short malformed bytes must fail");
    assert_diagnostic(
        &malformed,
        DiagnosticCode::MalformedArtifact,
        "envelope.header",
        "supply one complete canonical frame",
    );

    let neutral = neutral_artifact([8, 1, 1]);
    let mut truncated = envelope_bytes(
        neutral.clone(),
        [payload(&neutral, required.clone(), &[1, 2, 3])],
    );
    truncated.pop();
    let truncated_error = admit_artifact(&truncated, &required)
        .expect_err("truncated canonical bytes must fail");
    assert_diagnostic(
        &truncated_error,
        DiagnosticCode::MalformedArtifact,
        "envelope.body_length",
        "supply exactly one complete canonical frame",
    );

    let mut corrupted = envelope_bytes(
        neutral.clone(),
        [payload(&neutral, required.clone(), &[1, 2, 3])],
    );
    corrupted[FRAME_HEADER_BYTES] ^= 1;
    let corrupted_error = admit_artifact(&corrupted, &required)
        .expect_err("corrupted canonical bytes must fail authentication");
    assert_diagnostic(
        &corrupted_error,
        DiagnosticCode::DigestMismatch,
        "envelope.digest",
        "discard the corrupted bytes and regenerate them",
    );
}

/// Regression: envelope framing schema skew fails before target selection.
#[test]
fn rejects_envelope_framing_version_skew() {
    let neutral = neutral_artifact([8, 1, 1]);
    let required = format("test.target-a", 1);
    let mut bytes = envelope_bytes(
        neutral.clone(),
        [payload(&neutral, required.clone(), &[1, 2, 3])],
    );
    bytes[4..6].copy_from_slice(&2_u16.to_le_bytes());

    let error = admit_artifact(&bytes, &required).expect_err("unknown envelope schema must fail");
    assert_diagnostic(
        &error,
        DiagnosticCode::VersionSkew,
        "envelope.schema_version",
        "recompile or re-materialize with a compatible schema version",
    );
}

/// Regression: an absent payload identity fails visibly without selecting another attachment.
#[test]
fn rejects_missing_payload_identity_without_fallback() {
    let neutral = neutral_artifact([8, 1, 1]);
    let bytes = envelope_bytes(
        neutral.clone(),
        [
            payload(&neutral, format("test.target-a", 1), &[1]),
            payload(&neutral, format("test.target-b", 1), &[2]),
        ],
    );

    let error = admit_artifact(&bytes, &format("test.target-c", 1))
        .expect_err("unattached identity must not fall back");
    assert_diagnostic(
        &error,
        DiagnosticCode::IncompatibleTargetPayload,
        "envelope.target_payloads.format.identity",
        "attach a compatible payload or materialize one from the neutral artifact",
    );
}

/// Regression: matching identity with the wrong version never falls back to another payload.
#[test]
fn rejects_same_identity_wrong_version_without_fallback() {
    let neutral = neutral_artifact([8, 1, 1]);
    let bytes = envelope_bytes(
        neutral.clone(),
        [
            payload(&neutral, format("test.target-a", 1), &[1]),
            payload(&neutral, format("test.target-b", 2), &[2]),
        ],
    );

    let error = admit_artifact(&bytes, &format("test.target-a", 2))
        .expect_err("same-identity version skew must not select another version-two payload");
    assert_diagnostic(
        &error,
        DiagnosticCode::TargetPayloadVersionSkew,
        "envelope.target_payloads.format.version",
        "materialize the neutral artifact with the exact required target format version",
    );
}

/// Regression: nested target-byte corruption reaches canonical payload authentication and fails.
#[test]
fn rejects_corrupted_nested_payload_through_canonical_decode() {
    let neutral = neutral_artifact([8, 1, 1]);
    let required = format("test.target-a", 1);
    let target = payload(&neutral, required.clone(), &[11, 22, 33]);
    let target_bytes = target.to_bytes().expect("fixture payload must encode");
    let envelope = envelope_bytes(neutral, [target]);
    let mut corrupted_target = target_bytes.clone();
    corrupted_target[FRAME_HEADER_BYTES] ^= 1;
    let corrupted_envelope =
        replace_nested_payload(&envelope, &target_bytes, &corrupted_target);

    let error = admit_artifact(&corrupted_envelope, &required)
        .expect_err("nested payload corruption must fail its canonical digest");
    assert_diagnostic(
        &error,
        DiagnosticCode::TargetPayloadDigestMismatch,
        "target_payload.digest",
        "discard the corrupted bytes and regenerate them",
    );
}

/// Regression: a payload naming another neutral digest fails canonical association validation.
#[test]
fn rejects_payload_association_mismatch_through_canonical_decode() {
    let neutral = neutral_artifact([8, 1, 1]);
    let other_neutral = neutral_artifact([16, 1, 1]);
    let required = format("test.target-a", 1);
    let target = payload(&neutral, required.clone(), &[11, 22, 33]);
    let target_bytes = target.to_bytes().expect("fixture payload must encode");
    let envelope = envelope_bytes(neutral.clone(), [target]);
    let reassociated_target =
        reassociate_payload(&target_bytes, neutral.digest(), other_neutral.digest());
    let reassociated_envelope =
        replace_nested_payload(&envelope, &target_bytes, &reassociated_target);

    let error = admit_artifact(&reassociated_envelope, &required)
        .expect_err("wrong neutral association must fail canonical validation");
    assert_diagnostic(
        &error,
        DiagnosticCode::TargetPayloadAssociationMismatch,
        "target_payload.neutral_artifact",
        "discard the payload and materialize bytes from this exact neutral artifact",
    );
}
