//! Regression contracts for canonical runtime artifact admission.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::LazyLock;

use vyre_driver::materialize::{DeviceSpec, MaterializerDevice};
use vyre_driver::BackendRegistration;
use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BindingSet, BoundResource, Completion,
    Device, ResidentOwner, Resource, VyreBackend,
};
use vyre_foundation::diagnostics::{DiagnosticStage, RetryClass};
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, GraphInput, GraphOutput, Program, ProgramGraph, ShapeDim,
    ValueContract, ValueLifetime,
};
use vyre_megakernel::{
    Artifact, ArtifactEnvelope, ArtifactNodeId, ArtifactValueId, CompileObjective, CompileRequest,
    DeviceFacts, Digest, ExternalFacts, ObjectiveMetric, SearchBudget, TargetCompileError,
    TargetCompiler, TargetPayload, TargetPayloadFormat, TargetProfile, TargetResourceAccess,
    TargetResourceBinding, TargetResourceMemory,
};
use vyre_runtime::artifact_admission::{
    admit_artifact, admit_cached_artifact, admit_envelope, ArtifactAdmissionError, ArtifactSession,
    RetainedArtifactSession,
};
use vyre_runtime::persistent_executor::{PersistentExecutor, ResidentQueueState};
use vyre_runtime::pipeline_cache::{
    InMemoryPipelineCache, PipelineCacheStore, PipelineFingerprint,
};
use vyre_runtime::recovery::{classify_backend_error, recover_artifact_session};

#[path = "../../tests/support/artifact_fixtures.rs"]
mod artifact_fixtures;
#[path = "../../tests/support/fixture_instance.rs"]
mod fixture_instance;

use fixture_instance::{completion, FixtureInstance};

use artifact_fixtures::{
    compile_graph, contract, entry_over, entry_point, graph_over, neutral_artifact,
    single_input_graph,
};
use vyre_test_support::pass_programs::{add_program, copy_program};

const FRAME_HEADER_BYTES: usize = 10;
const FRAME_DIGEST_BYTES: usize = 32;
const ENVELOPE_DIGEST_DOMAIN: &[u8] = b"vyre-megakernel-envelope-v2\0";
const TARGET_PAYLOAD_DIGEST_DOMAIN: &[u8] = b"vyre-megakernel-target-payload-v3\0";

fn resident_queue_artifact() -> Artifact {
    let retained = contract(
        DataType::U32,
        4,
        BufferAccess::ReadWrite,
        ValueLifetime::Retained,
    );
    let values: Vec<(&str, ValueContract)> = ["control", "ring_buffer", "debug_log", "io_queue"]
        .into_iter()
        .map(|name| (name, retained.clone()))
        .collect();
    compile_graph(graph_over("queue", [1, 1, 1], &values), 0)
}

fn resident_projection_artifact() -> Artifact {
    let program = Program::wrapped(
        vec![
            BufferDecl::read_write("out_flags", 0, DataType::U32).with_count(16),
            BufferDecl::read("pattern_bitmap", 1, DataType::U32).with_count(8),
            BufferDecl::read("rule_bitmap", 2, DataType::U32).with_count(8),
        ],
        [1, 1, 1],
        Vec::new(),
    );
    let graph = ProgramGraph::from_program("resident-projection", program)
        .expect("resident projection graph must be valid");
    compile_graph(graph, 0)
}

fn resident_projection_payload(neutral: &Artifact) -> TargetPayload {
    let bindings = [
        (
            ArtifactValueId(0),
            TargetResourceMemory::Global,
            TargetResourceAccess::ReadWrite,
        ),
        (
            ArtifactValueId(1),
            TargetResourceMemory::Constant,
            TargetResourceAccess::ReadOnly,
        ),
        (
            ArtifactValueId(2),
            TargetResourceMemory::Constant,
            TargetResourceAccess::ReadOnly,
        ),
    ]
    .into_iter()
    .enumerate()
    .map(|(slot, (resource, memory, access))| TargetResourceBinding {
        resource,
        group: 0,
        slot: slot as u32,
        memory,
        access,
    })
    .collect();
    TargetPayload::new(
        neutral,
        format("test.cache-target", 1),
        profile("test.cache-target", 1),
        vec![entry_over(
            neutral,
            "resident-projection",
            ArtifactNodeId(0),
            bindings,
        )],
        vec![7, 8, 9],
    )
    .expect("resident projection payload must be valid")
}

fn queue_payload(neutral: &Artifact) -> TargetPayload {
    let bindings = neutral
        .resources()
        .iter()
        .enumerate()
        .map(|(slot, resource)| TargetResourceBinding {
            resource: resource.value,
            group: 0,
            slot: slot as u32,
            memory: TargetResourceMemory::Global,
            access: TargetResourceAccess::ReadWrite,
        })
        .collect();
    TargetPayload::new(
        neutral,
        format("test.cache-target", 1),
        profile("test.cache-target", 1),
        vec![entry_over(neutral, "queue", ArtifactNodeId(0), bindings)],
        vec![1, 2, 3],
    )
    .unwrap()
}

fn format(identity: &str, version: u16) -> TargetPayloadFormat {
    TargetPayloadFormat::new(identity, version).expect("fixture format must be valid")
}

fn profile(identity: &str, generation: u64) -> TargetProfile {
    TargetProfile::new(identity, generation, [64, 1, 1], 64, 1_024, 0)
        .expect("fixture profile must be valid")
}

fn payload(neutral: &Artifact, payload_format: TargetPayloadFormat, bytes: &[u8]) -> TargetPayload {
    let generation = u64::from(payload_format.version());
    let profile = profile(payload_format.identity(), generation);
    TargetPayload::new(
        neutral,
        payload_format,
        profile,
        vec![entry_point(neutral)],
        bytes.to_vec(),
    )
    .expect("fixture payload must be valid")
}

fn envelope_bytes(neutral: Artifact, payloads: impl IntoIterator<Item = TargetPayload>) -> Vec<u8> {
    let mut envelope = ArtifactEnvelope::new(neutral);
    for payload in payloads {
        envelope
            .attach_target_payload(payload)
            .expect("fixture payload must attach");
    }
    envelope.to_bytes().expect("fixture envelope must encode")
}

fn assert_diagnostic(error: &ArtifactAdmissionError, code: &str, path: &str, fix: &str) {
    let diagnostic = error.diagnostic();
    assert_eq!(diagnostic.code.as_str(), code);
    assert_eq!(diagnostic.stage, DiagnosticStage::Admit);
    assert_eq!(
        diagnostic
            .location
            .as_ref()
            .and_then(|location| location.path.as_deref()),
        Some(path)
    );
    assert_eq!(diagnostic.suggested_fix.as_deref(), Some(fix));
}

fn frame_body(frame: &[u8]) -> &[u8] {
    let body_len =
        u32::from_le_bytes(frame[6..10].try_into().expect("fixed header slice")) as usize;
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
    encode_frame(b"VME0", 2, ENVELOPE_DIGEST_DOMAIN, &body)
}

fn reassociate_payload(
    payload: &[u8],
    original_neutral: Digest,
    replacement_neutral: Digest,
) -> Vec<u8> {
    let original_json = serde_json::to_vec(&original_neutral).expect("digest must serialize");
    let replacement_json = serde_json::to_vec(&replacement_neutral).expect("digest must serialize");
    let body = replace_once(frame_body(payload), &original_json, &replacement_json);
    encode_frame(b"VTP0", 3, TARGET_PAYLOAD_DIGEST_DOMAIN, &body)
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
    let malformed =
        admit_artifact(b"not an envelope", &required).expect_err("short malformed bytes must fail");
    assert_diagnostic(
        &malformed,
        "MKC014_MALFORMED_ARTIFACT",
        "envelope.header",
        "supply one complete canonical frame",
    );

    let neutral = neutral_artifact([8, 1, 1]);
    let mut truncated = envelope_bytes(
        neutral.clone(),
        [payload(&neutral, required.clone(), &[1, 2, 3])],
    );
    truncated.pop();
    let truncated_error =
        admit_artifact(&truncated, &required).expect_err("truncated canonical bytes must fail");
    assert_diagnostic(
        &truncated_error,
        "MKC014_MALFORMED_ARTIFACT",
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
        "MKC016_DIGEST_MISMATCH",
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
    bytes[4..6].copy_from_slice(&3_u16.to_le_bytes());

    let error = admit_artifact(&bytes, &required).expect_err("unknown envelope schema must fail");
    assert_diagnostic(
        &error,
        "MKC015_VERSION_SKEW",
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
        "MKC021_INCOMPATIBLE_TARGET_PAYLOAD",
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
        "MKC018_TARGET_PAYLOAD_VERSION_SKEW",
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
    let corrupted_envelope = replace_nested_payload(&envelope, &target_bytes, &corrupted_target);

    let error = admit_artifact(&corrupted_envelope, &required)
        .expect_err("nested payload corruption must fail its canonical digest");
    assert_diagnostic(
        &error,
        "MKC019_TARGET_PAYLOAD_DIGEST_MISMATCH",
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
        "MKC020_TARGET_PAYLOAD_ASSOCIATION_MISMATCH",
        "target_payload.neutral_artifact",
        "discard the payload and materialize bytes from this exact neutral artifact",
    );
}

/// Regression: already-decoded envelopes must admit through the same exact-format index path.
#[test]
fn admits_owned_envelope_by_exact_format_index() {
    let neutral = neutral_artifact([8, 1, 1]);
    let required = format("test.target-b", 4);
    let bytes = envelope_bytes(
        neutral.clone(),
        [
            payload(&neutral, format("test.target-a", 1), &[1, 2, 3]),
            payload(&neutral, required.clone(), &[9, 8, 7, 6]),
        ],
    );
    let envelope = ArtifactEnvelope::from_bytes(&bytes).expect("fixture envelope must decode");

    let admitted =
        admit_envelope(envelope, &required).expect("exact owned-envelope format must admit");

    assert_eq!(admitted.neutral().digest(), neutral.digest());
    assert_eq!(admitted.target_payload().format(), &required);
    assert_eq!(admitted.target_payload().bytes(), &[9, 8, 7, 6]);
}

/// Regression: producer-packaged envelopes (AOT shape) must admit without recompilation.
///
/// This locks the compile → envelope bytes → `admit_artifact` seam used by AOT packages
/// and runtime-cache blobs. It intentionally stays on the canonical envelope types so
/// runtime tests do not take a reverse dependency on `vyre-aot`.
#[test]
fn packaged_envelope_admits_through_runtime_without_recompile() {
    let neutral = neutral_artifact([64, 1, 1]);
    let expected_neutral = neutral.digest();
    let node = neutral.nodes()[0].id;
    let resource = neutral.resources()[0].value;
    // Match the fixture package payload format identity and version.
    let required = format("fixture-target-format", 1);
    let attached = TargetPayload::new(
        &neutral,
        required.clone(),
        profile("fixture-target-format", 1),
        vec![entry_over(
            &neutral,
            "main",
            node,
            vec![TargetResourceBinding {
                resource,
                group: 0,
                slot: 0,
                memory: TargetResourceMemory::Global,
                access: TargetResourceAccess::ReadOnly,
            }],
        )],
        b"target-payload-fixture".to_vec(),
    )
    .expect("packaged fixture payload must bind");
    let expected_payload = attached.digest();
    let envelope_bytes = envelope_bytes(neutral, [attached]);

    let admitted = admit_artifact(&envelope_bytes, &required)
        .expect("packaged envelope must admit at the runtime boundary");
    let owned = admit_envelope(
        ArtifactEnvelope::from_bytes(&envelope_bytes).expect("packaged envelope must re-decode"),
        &required,
    )
    .expect("owned packaged envelope must admit identically");

    assert_eq!(admitted.neutral().digest(), expected_neutral);
    assert_eq!(admitted.target_payload().digest(), expected_payload);
    assert_eq!(admitted.target_payload().format(), &required);
    assert_eq!(admitted.target_payload().bytes(), b"target-payload-fixture");
    assert_eq!(owned.neutral().digest(), expected_neutral);
    assert_eq!(owned.target_payload().digest(), expected_payload);
}

/// Regression: DiskCache/AOT payload hits must admit through the envelope seam.
///
/// `PipelineCacheStore` returns verified payload bytes only. Treating those
/// bytes as executable without admission is the ARCH-001/010 bypass.
#[test]
fn cached_envelope_payload_admits_and_miss_is_none() {
    let neutral = neutral_artifact([8, 1, 1]);
    let required = format("test.cache-target", 1);
    let bytes = envelope_bytes(
        neutral.clone(),
        [payload(&neutral, required.clone(), &[4, 5, 6, 7])],
    );
    let fp = PipelineFingerprint([0xAB; 32]);
    let store = InMemoryPipelineCache::default();
    store.put(fp, bytes);

    let admitted = admit_cached_artifact(&store, &fp, &required)
        .expect("cached envelope payload must admit")
        .expect("cache hit must yield Some");
    assert_eq!(admitted.neutral().digest(), neutral.digest());
    assert_eq!(admitted.target_payload().bytes(), &[4, 5, 6, 7]);

    let miss_fp = PipelineFingerprint([0xCD; 32]);
    let miss = admit_cached_artifact(&store, &miss_fp, &required)
        .expect("cache miss must not be an admission error");
    assert!(miss.is_none(), "missing fingerprint must return Ok(None)");

    // Garbage payload is a hit at the blob layer but fails admission.
    let bad_fp = PipelineFingerprint([0xEF; 32]);
    store.put(bad_fp, b"not-an-envelope".to_vec());
    let err = admit_cached_artifact(&store, &bad_fp, &required)
        .expect_err("non-envelope cache payload must fail admission");
    assert!(
        !err.to_string().is_empty(),
        "admission error must carry diagnostics"
    );
}

static MATERIALIZER_CALLS: AtomicU64 = AtomicU64::new(0);
static TEST_SUPPORTED_OPS: LazyLock<HashSet<vyre_foundation::ir::OpId>> =
    LazyLock::new(HashSet::new);

struct TestMaterializer {
    device: MaterializerDevice,
}

impl ArtifactMaterializer for TestMaterializer {
    fn device(&self) -> &dyn Device {
        &self.device
    }

    fn materialize(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<Box<dyn ArtifactInstance>, BackendError> {
        if payload.neutral_artifact() != artifact.digest() {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: test payload association must match the neutral artifact.".to_string(),
            });
        }
        let retained: BTreeSet<ArtifactValueId> = artifact
            .resources()
            .iter()
            .filter(|resource| resource.lifetime == vyre_megakernel::ResourceLifetime::Retained)
            .map(|resource| resource.value)
            .collect();
        Ok(FixtureInstance::submitting(
            artifact,
            payload,
            self.device.identity(),
            move |artifact, bindings| increment_retained(artifact, bindings, &retained),
        ))
    }
}

/// Complete `bindings`, bumping every retained host counter by one.
///
/// The increment is what the concurrency contracts observe: a retained value
/// read back after a submission must carry the value this instance wrote, not
/// the bytes the caller supplied.
fn increment_retained(
    artifact: Digest,
    bindings: BindingSet,
    retained: &BTreeSet<ArtifactValueId>,
) -> Result<Completion, BackendError> {
    if bindings.artifact() != artifact {
        return Err(BackendError::InvalidProgram {
            fix: "Fix: test bindings must name the materialized artifact.".to_string(),
        });
    }
    let readback = bindings
        .resources()
        .iter()
        .filter_map(|(value, resource)| {
            retained.contains(value).then(|| match resource {
                BoundResource::Host(bytes) => {
                    let mut next = bytes.clone();
                    if next.len() == 4 {
                        let counter = u32::from_le_bytes([next[0], next[1], next[2], next[3]]);
                        std::thread::yield_now();
                        next = counter.wrapping_add(1).to_le_bytes().to_vec();
                    }
                    Ok((*value, next))
                }
                BoundResource::Resident(_) => Err(BackendError::UnsupportedFeature {
                    name: "test resident resource".to_string(),
                    backend: "test-artifact".to_string(),
                }),
            })
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    Ok(Completion {
        retained: readback,
        ..completion(artifact)
    })
}

fn test_backend_factory() -> Result<Box<dyn VyreBackend>, BackendError> {
    Err(BackendError::UnsupportedFeature {
        name: "legacy raw Program backend".to_string(),
        backend: "test-artifact".to_string(),
    })
}

fn test_supported_ops() -> &'static HashSet<vyre_foundation::ir::OpId> {
    &TEST_SUPPORTED_OPS
}

fn test_materializer_factory() -> Result<Box<dyn ArtifactMaterializer>, BackendError> {
    MATERIALIZER_CALLS.fetch_add(1, Ordering::AcqRel);
    Ok(Box::new(TestMaterializer {
        device: MaterializerDevice::acquire(DeviceSpec {
            backend: "test-artifact",
            device: "test-device".to_string(),
            format_extension: "test.cache-target",
            format_version: 1,
            profile: profile("test.cache-target", 1),
        })?,
    }))
}

static TEST_REGISTRATION: BackendRegistration = BackendRegistration {
    id: "test-artifact",
    target_id: vyre_foundation::operation::TargetId::expect_valid("test-artifact"),
    payload_format: None,
    reference_oracle: false,
    factory: test_backend_factory,
    supported_ops: test_supported_ops,
    semantic_operations: test_supported_ops,
    target_compiler: None,
    materializer: Some(test_materializer_factory),
};

static RECORDED_VALUE_ID: AtomicU32 = AtomicU32::new(u32::MAX);
static RECORDED_BYTE_LEN: AtomicU64 = AtomicU64::new(0);
static RECORDED_FIRST_BYTE: AtomicU32 = AtomicU32::new(0);

/// Record the first host binding a submission carried, then complete.
///
/// The recorded value id, byte length and first byte are what the
/// representative-input contracts assert: the runtime must bind the exact bytes
/// the caller supplied, so a fixture that only counted submissions could not
/// see a substituted buffer.
fn record_host_binding(artifact: Digest, bindings: BindingSet) -> Result<Completion, BackendError> {
    for (value, resource) in bindings.resources() {
        if let BoundResource::Host(bytes) = resource {
            RECORDED_VALUE_ID.store(value.0, Ordering::Release);
            RECORDED_BYTE_LEN.store(bytes.len() as u64, Ordering::Release);
            RECORDED_FIRST_BYTE.store(
                bytes.first().copied().unwrap_or(0) as u32,
                Ordering::Release,
            );
        }
    }
    Ok(Completion {
        device_ns: Some(42),
        ..completion(artifact)
    })
}

struct RecordingMaterializer {
    device: MaterializerDevice,
}

impl ArtifactMaterializer for RecordingMaterializer {
    fn device(&self) -> &dyn Device {
        &self.device
    }

    fn materialize(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<Box<dyn ArtifactInstance>, BackendError> {
        Ok(FixtureInstance::submitting(
            artifact,
            payload,
            self.device.identity(),
            record_host_binding,
        ))
    }
}

struct RecordingCompiler {
    format: TargetPayloadFormat,
    profile: TargetProfile,
}

impl TargetCompiler for RecordingCompiler {
    fn format(&self) -> &TargetPayloadFormat {
        &self.format
    }

    fn profile(&self) -> &TargetProfile {
        &self.profile
    }

    fn compile(&self, artifact: &Artifact) -> Result<TargetPayload, TargetCompileError> {
        TargetPayload::new(
            artifact,
            self.format.clone(),
            self.profile.clone(),
            vec![entry_point(artifact)],
            vec![4, 2],
        )
        .map_err(Into::into)
    }
}

fn recording_compiler_factory() -> Result<Box<dyn TargetCompiler>, BackendError> {
    Ok(Box::new(RecordingCompiler {
        format: format("test.cache-target", 1),
        profile: profile("test.cache-target", 1),
    }))
}

fn recording_materializer_factory() -> Result<Box<dyn ArtifactMaterializer>, BackendError> {
    Ok(Box::new(RecordingMaterializer {
        device: MaterializerDevice::acquire(DeviceSpec {
            backend: "recording-artifact",
            device: "recording-device".to_string(),
            format_extension: "test.cache-target",
            format_version: 1,
            profile: profile("test.cache-target", 1),
        })?,
    }))
}

static RECORDING_REGISTRATION: BackendRegistration = BackendRegistration {
    id: "recording-artifact",
    target_id: vyre_foundation::operation::TargetId::expect_valid("test-artifact"),
    payload_format: Some("test.cache-target"),
    reference_oracle: false,
    factory: test_backend_factory,
    supported_ops: test_supported_ops,
    semantic_operations: test_supported_ops,
    target_compiler: Some(recording_compiler_factory),
    materializer: Some(recording_materializer_factory),
};

/// WHY: DeviceFinalists binds exact non-zero representative bytes for each host input.
#[test]
fn finalist_measurement_binds_exact_representative_inputs() {
    let graph = single_input_graph([8, 1, 1]);
    let facts = ExternalFacts::new(Digest([1; 32]), BTreeMap::new());
    let non_zero_bytes = vec![0xAB; 32];
    let representative_inputs =
        BTreeMap::from([(vyre_foundation::ir::GraphValueId(0), non_zero_bytes.clone())]);

    let device = DeviceFacts::unknown().with_device_timestamps(true);
    let budget = SearchBudget::new(1, 1, 1, 1, 1_000_000_000);
    let request = CompileRequest::new(
        graph,
        facts,
        device,
        budget,
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
    .with_representative_inputs(representative_inputs)
    .validate()
    .expect("compile request must validate");

    let session = ArtifactSession::compile(&RECORDING_REGISTRATION, &request)
        .expect("measured compilation must succeed with representative inputs");
    assert!(session.artifact().is_ok());
    assert_eq!(RECORDED_VALUE_ID.load(Ordering::Acquire), 0);
    assert_eq!(RECORDED_BYTE_LEN.load(Ordering::Acquire), 32);
    assert_eq!(RECORDED_FIRST_BYTE.load(Ordering::Acquire), 0xAB);
}

/// WHY: DeviceFinalists fails closed when representative inputs are missing for a host-input resource.
#[test]
fn finalist_measurement_fails_closed_on_missing_representative_inputs() {
    let graph = graph_over(
        "entry",
        [8, 1, 1],
        &[
            (
                "input",
                contract(
                    DataType::U32,
                    8,
                    BufferAccess::ReadOnly,
                    ValueLifetime::Invocation,
                ),
            ),
            (
                "constant",
                contract(
                    DataType::U32,
                    8,
                    BufferAccess::ReadOnly,
                    ValueLifetime::Constant,
                ),
            ),
        ],
    );
    let mut facts = ExternalFacts::new(Digest([1; 32]), BTreeMap::new());
    facts
        .constant_identities
        .insert(vyre_foundation::ir::GraphValueId(1), Digest([2; 32]));
    let representative_inputs =
        BTreeMap::from([(vyre_foundation::ir::GraphValueId(0), vec![0xAB; 32])]);
    let device = DeviceFacts::unknown().with_device_timestamps(true);
    let budget = SearchBudget::new(1, 1, 1, 1, 1_000_000_000);
    let request = CompileRequest::new(
        graph,
        facts,
        device,
        budget,
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
    .with_representative_inputs(representative_inputs)
    .validate()
    .expect("compile request must validate");

    let error = ArtifactSession::compile(&RECORDING_REGISTRATION, &request)
        .err()
        .expect("measured compilation must fail closed when representative inputs are missing");
    let error_text = error.to_string();
    assert!(
        error_text.contains("missing representative input for host-input resource")
            && error_text.contains("`constant`"),
        "Fix: missing representative input must fail closed with an explicit error, got: {error_text}"
    );
}

/// WHY: resident projection follows the authenticated target ABI, including
/// read-only constant slots, rather than silently dropping non-global memory.
#[test]
fn resident_bindings_include_global_and_constant_target_resources() {
    let neutral = resident_projection_artifact();
    let payload = resident_projection_payload(&neutral);
    let session =
        ArtifactSession::from_bytes(&TEST_REGISTRATION, &envelope_bytes(neutral, [payload]))
            .expect("authenticated resident projection must materialize");
    let owner = ResidentOwner::new().expect("resident owner identity must be available");
    let resources = [0, 1, 2].map(|id| Resource::Resident(owner.handle(id)));

    let bindings = session
        .resident_bindings(&resources)
        .expect("global and constant target resources must all bind");
    assert_eq!(bindings.resources().len(), 3);
    for id in 0..3 {
        assert_eq!(
            bindings.resources().get(&ArtifactValueId(id)),
            Some(&BoundResource::Resident(resources[id as usize].clone()))
        );
    }

    let error = session
        .resident_bindings(&resources[..2])
        .expect_err("missing a constant target resource must fail closed");
    assert!(
        error
            .to_string()
            .contains("target entry requires 3 resident resource(s), but the caller supplied 2"),
        "count failure must report the authenticated target ABI: {error}"
    );
}

/// WHY: resident binding maps by authenticated canonical value identity rather than
/// vector index, so target slots emitted in non-canonical order bind their exact resource.
#[test]
fn resident_bindings_preserve_canonical_value_identity_with_reordered_target_slots() {
    let neutral = resident_projection_artifact();
    let reordered_bindings = vec![
        TargetResourceBinding {
            resource: ArtifactValueId(2),
            group: 0,
            slot: 0,
            memory: TargetResourceMemory::Constant,
            access: TargetResourceAccess::ReadOnly,
        },
        TargetResourceBinding {
            resource: ArtifactValueId(0),
            group: 0,
            slot: 1,
            memory: TargetResourceMemory::Global,
            access: TargetResourceAccess::ReadWrite,
        },
        TargetResourceBinding {
            resource: ArtifactValueId(1),
            group: 0,
            slot: 2,
            memory: TargetResourceMemory::Constant,
            access: TargetResourceAccess::ReadOnly,
        },
    ];
    let payload = TargetPayload::new(
        &neutral,
        format("test.cache-target", 1),
        profile("test.cache-target", 1),
        vec![entry_over(
            &neutral,
            "resident-projection",
            ArtifactNodeId(0),
            reordered_bindings,
        )],
        vec![7, 8, 9],
    )
    .expect("reordered resident projection payload must be valid");

    let session =
        ArtifactSession::from_bytes(&TEST_REGISTRATION, &envelope_bytes(neutral, [payload]))
            .expect("authenticated resident projection must materialize");
    let owner = ResidentOwner::new().expect("resident owner identity must be available");
    let resources = [100, 200, 300].map(|id| Resource::Resident(owner.handle(id)));

    let bindings = session
        .resident_bindings(&resources)
        .expect("resident resources must bind by canonical value identity");
    assert_eq!(
        bindings.resources().get(&ArtifactValueId(0)),
        Some(&BoundResource::Resident(resources[0].clone()))
    );
    assert_eq!(
        bindings.resources().get(&ArtifactValueId(1)),
        Some(&BoundResource::Resident(resources[1].clone()))
    );
    assert_eq!(
        bindings.resources().get(&ArtifactValueId(2)),
        Some(&BoundResource::Resident(resources[2].clone()))
    );
}

/// WHY: program_resident_bindings binds non-shared buffers in program declaration
/// order regardless of whether the order matches artifact ABI slot order.
#[test]
fn program_resident_bindings_maps_by_buffer_name_when_declarations_are_reordered() {
    let neutral = resident_projection_artifact();
    let payload = resident_projection_payload(&neutral);
    let session =
        ArtifactSession::from_bytes(&TEST_REGISTRATION, &envelope_bytes(neutral, [payload]))
            .expect("authenticated resident projection must materialize");
    let owner = ResidentOwner::new().expect("resident owner identity must be available");
    let res_rule = Resource::Resident(owner.handle(50));
    let res_out = Resource::Resident(owner.handle(60));
    let res_pattern = Resource::Resident(owner.handle(70));

    let reordered_program = Program::wrapped(
        vec![
            BufferDecl::read("rule_bitmap", 0, DataType::U32).with_count(8),
            BufferDecl::read_write("out_flags", 1, DataType::U32).with_count(16),
            BufferDecl::read("pattern_bitmap", 2, DataType::U32).with_count(8),
        ],
        [1, 1, 1],
        Vec::new(),
    );

    let bindings = session
        .program_resident_bindings(
            &reordered_program,
            &[res_rule.clone(), res_out.clone(), res_pattern.clone()],
        )
        .expect("program resident bindings must resolve by buffer name");

    // rule_bitmap is value 2, out_flags is value 0, pattern_bitmap is value 1
    assert_eq!(
        bindings.resources().get(&ArtifactValueId(2)),
        Some(&BoundResource::Resident(res_rule))
    );
    assert_eq!(
        bindings.resources().get(&ArtifactValueId(0)),
        Some(&BoundResource::Resident(res_out))
    );
    assert_eq!(
        bindings.resources().get(&ArtifactValueId(1)),
        Some(&BoundResource::Resident(res_pattern))
    );
}

/// WHY: resident bindings by name and value fail closed on unknown resources and conflicting duplicates.
#[test]
fn resident_bindings_fail_closed_on_duplicate_mismatch_and_unknown_names() {
    let neutral = resident_projection_artifact();
    let payload = resident_projection_payload(&neutral);
    let session =
        ArtifactSession::from_bytes(&TEST_REGISTRATION, &envelope_bytes(neutral, [payload]))
            .expect("authenticated resident projection must materialize");
    let owner = ResidentOwner::new().expect("resident owner identity must be available");
    let res_a = Resource::Resident(owner.handle(1));
    let res_b = Resource::Resident(owner.handle(2));
    let res_c = Resource::Resident(owner.handle(3));

    let by_name_ok = session.resident_bindings_by_name([
        ("out_flags", &res_a),
        ("pattern_bitmap", &res_b),
        ("rule_bitmap", &res_c),
    ]);
    assert!(by_name_ok.is_ok());

    let unknown_name = session.resident_bindings_by_name([
        ("unknown_buffer", &res_a),
        ("pattern_bitmap", &res_b),
        ("rule_bitmap", &res_c),
    ]);
    let err = unknown_name.expect_err("unknown buffer name must fail closed");
    assert!(err.to_string().contains("unknown_buffer"));

    let by_val_ok = session.resident_bindings_by_value([
        (ArtifactValueId(0), res_a.clone()),
        (ArtifactValueId(1), res_b.clone()),
        (ArtifactValueId(2), res_c.clone()),
    ]);
    assert!(by_val_ok.is_ok());

    let unknown_val = session.resident_bindings_by_value([
        (ArtifactValueId(99), res_a.clone()),
        (ArtifactValueId(1), res_b.clone()),
        (ArtifactValueId(2), res_c.clone()),
    ]);
    let err = unknown_val.expect_err("unknown value must fail closed");
    assert!(err.to_string().contains("99"));

    let duplicate_conflict = session.resident_bindings_by_value([
        (ArtifactValueId(0), res_a),
        (ArtifactValueId(0), res_b),
        (ArtifactValueId(1), res_c.clone()),
        (ArtifactValueId(2), res_c),
    ]);
    let err = duplicate_conflict.expect_err("conflicting duplicate value must fail closed");
    assert!(err.to_string().contains("conflicting resident resources"));
}

/// WHY: bootstrap and recovery must authenticate and rematerialize without a compiler facet.
#[test]
fn artifact_session_bootstrap_and_recovery_use_only_materialization() {
    let neutral = neutral_artifact([8, 1, 1]);
    let digest = neutral.digest();
    let required = format("test.cache-target", 1);
    let bytes = envelope_bytes(
        neutral.clone(),
        [payload(&neutral, required, &[4, 5, 6, 7])],
    );
    let calls_before = MATERIALIZER_CALLS.load(Ordering::Acquire);
    let session = ArtifactSession::from_bytes(&TEST_REGISTRATION, &bytes)
        .expect("authenticated artifact must materialize");
    assert_eq!(session.artifact().unwrap(), digest);
    let first_device = session.device().unwrap();
    let completion = session
        .submit_and_wait(session.bindings().unwrap())
        .expect("materialized instance must submit");
    assert_eq!(completion.artifact, digest);
    let second_device = session
        .rematerialize()
        .expect("recovery must reacquire and rematerialize");
    assert!(second_device.generation > first_device.generation);
    assert_eq!(session.artifact().unwrap(), digest);
    assert!(MATERIALIZER_CALLS.load(Ordering::Acquire) >= calls_before + 2);
}

/// WHY: recovery must branch on the stable error variant and preserve other failures.
#[test]
fn artifact_recovery_requires_structured_device_loss() {
    let neutral = neutral_artifact([8, 1, 1]);
    let bytes = envelope_bytes(
        neutral.clone(),
        [payload(
            &neutral,
            format("test.cache-target", 1),
            &[4, 5, 6, 7],
        )],
    );
    let session = ArtifactSession::from_bytes(&TEST_REGISTRATION, &bytes).unwrap();
    let first_device = session.device().unwrap();
    let failure = BackendError::DeviceLost {
        backend: first_device.backend.to_string(),
        device: first_device.device.clone(),
        generation: first_device.generation,
        message: "fault injection".to_string(),
    };
    assert_eq!(classify_backend_error(&failure), RetryClass::NewDevice);
    let recovered = recover_artifact_session(&session, failure).unwrap();
    assert!(recovered.generation > first_device.generation);
    assert_eq!(session.artifact().unwrap(), neutral.digest());

    let permanent = BackendError::InvalidProgram {
        fix: "Fix: reject malformed test bindings.".to_string(),
    };
    assert_eq!(classify_backend_error(&permanent), RetryClass::Never);
    let transient = BackendError::DeviceOutOfMemory {
        requested: 4096,
        available: 1024,
    };
    assert_eq!(classify_backend_error(&transient), RetryClass::SameDevice);
    let poisoned = BackendError::PoisonedLock {
        lock_error: "fixture poison".to_string(),
    };
    assert_eq!(classify_backend_error(&poisoned), RetryClass::SameDevice);
    let unclassified = BackendError::DispatchFailed {
        code: None,
        message: "fixture dispatch failure".to_string(),
    };
    assert_eq!(classify_backend_error(&unclassified), RetryClass::Never);
    let error = recover_artifact_session(&session, permanent).unwrap_err();
    assert!(matches!(
        error,
        vyre_runtime::artifact_admission::ArtifactSessionError::Backend(
            BackendError::InvalidProgram { .. }
        )
    ));
}

/// WHY: retained and ephemeral policies must preserve one neutral artifact identity.
#[test]
fn retained_and_ephemeral_sessions_share_artifact_identity() {
    let neutral = neutral_artifact([8, 1, 1]);
    let digest = neutral.digest();
    let bytes = envelope_bytes(
        neutral.clone(),
        [payload(
            &neutral,
            format("test.cache-target", 1),
            &[9, 8, 7],
        )],
    );
    let ephemeral = ArtifactSession::from_bytes(&TEST_REGISTRATION, &bytes).unwrap();
    let retained_session = ArtifactSession::from_bytes(&TEST_REGISTRATION, &bytes).unwrap();
    let retained = RetainedArtifactSession::new(retained_session, BTreeMap::new()).unwrap();
    assert_eq!(ephemeral.artifact().unwrap(), digest);
    assert_eq!(retained.artifact().unwrap(), digest);
    let completion = retained
        .submit_and_wait(retained_session_bindings(digest))
        .unwrap();
    assert_eq!(completion.artifact, digest);
}

/// WHY: concurrent submit_and_wait calls on one RetainedArtifactSession must serialize atomically.
#[test]
fn concurrent_retained_submissions_are_serialized_and_atomic() {
    use std::sync::Arc;
    use std::thread;

    let retained_contract = contract(
        DataType::U32,
        1,
        BufferAccess::ReadWrite,
        ValueLifetime::Retained,
    );
    let values = vec![("counter", retained_contract)];
    let neutral = compile_graph(graph_over("counter_kernel", [1, 1, 1], &values), 0);
    let digest = neutral.digest();
    let bytes = envelope_bytes(neutral.clone(), [queue_payload(&neutral)]);

    let retained_val = neutral.resources()[0].value;
    let mut initial_map = BTreeMap::new();
    initial_map.insert(retained_val, 0u32.to_le_bytes().to_vec());

    let session = ArtifactSession::from_bytes(&TEST_REGISTRATION, &bytes).unwrap();
    let retained = Arc::new(RetainedArtifactSession::new(session, initial_map).unwrap());

    const THREADS: usize = 8;
    const ITERS_PER_THREAD: usize = 20;
    const TOTAL_SUBMISSIONS: u32 = (THREADS * ITERS_PER_THREAD) as u32;

    let mut handles = Vec::new();
    for _ in 0..THREADS {
        let retained_clone = Arc::clone(&retained);
        handles.push(thread::spawn(move || {
            for _ in 0..ITERS_PER_THREAD {
                let completion = retained_clone
                    .submit_and_wait(BindingSet::new(digest))
                    .unwrap();
                assert_eq!(completion.artifact, digest);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let final_completion = retained.submit_and_wait(BindingSet::new(digest)).unwrap();
    let final_bytes = &final_completion.retained[&retained_val];
    let final_count = u32::from_le_bytes([
        final_bytes[0],
        final_bytes[1],
        final_bytes[2],
        final_bytes[3],
    ]);
    assert_eq!(
        final_count,
        TOTAL_SUBMISSIONS + 1,
        "concurrent retained submissions must serialize without lost updates"
    );
}

/// WHY: the shipped persistent route must authenticate, retain, submit, and recover one artifact.
#[test]
fn persistent_executor_uses_retained_artifact_lifecycle() {
    let neutral = resident_queue_artifact();
    let digest = neutral.digest();
    let bytes = envelope_bytes(neutral.clone(), [queue_payload(&neutral)]);
    let initial = ResidentQueueState {
        control: vec![1; 16],
        ring: vec![2; 16],
        debug_log: vec![3; 16],
        io_queue: vec![4; 16],
    };
    let executor =
        PersistentExecutor::from_bytes(&TEST_REGISTRATION, &bytes, initial.clone()).unwrap();
    assert_eq!(executor.artifact().unwrap(), digest);
    let completion = executor.submit_and_wait(initial.clone()).unwrap();
    assert_eq!(completion.state, initial);

    let first_device = executor.device().unwrap();
    let recovered = executor
        .recover(BackendError::DeviceLost {
            backend: first_device.backend.to_string(),
            device: first_device.device.clone(),
            generation: first_device.generation,
            message: "fault injection".to_string(),
        })
        .unwrap();
    assert!(recovered.generation > first_device.generation);
    assert_eq!(executor.artifact().unwrap(), digest);
}

fn retained_session_bindings(artifact: Digest) -> BindingSet {
    BindingSet::new(artifact)
}

fn fixed_contract(access: BufferAccess, lifetime: ValueLifetime) -> ValueContract {
    ValueContract {
        dtype: DataType::U32,
        shape: vec![ShapeDim::Known(4)],
        access,
        lifetime,
    }
}

/// Three nodes over three caller-supplied values, producing an invocation
/// intermediate, a retained successor, and one graph output.
///
/// `input`, `constant` and `retained` are graph externals: nothing in the graph
/// produces them, so their contents at launch can only come from the caller.
/// `intermediate`, `retained.next` and `result` are each produced by a node.
fn three_stage_graph() -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let input = graph
        .add_external_value(
            "input",
            fixed_contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
        )
        .expect("graph input must be valid");
    let constant = graph
        .add_external_value(
            "constant",
            fixed_contract(BufferAccess::ReadOnly, ValueLifetime::Constant),
        )
        .expect("graph constant must be valid");
    let retained = graph
        .add_external_value(
            "retained",
            fixed_contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
        )
        .expect("graph retained value must be valid");
    let (_, produced) = graph
        .add_node(
            "zeta",
            add_program("input", "constant", "intermediate"),
            vec![
                GraphInput {
                    buffer: "input".into(),
                    value: input,
                    contract: fixed_contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
                },
                GraphInput {
                    buffer: "constant".into(),
                    value: constant,
                    contract: fixed_contract(BufferAccess::ReadOnly, ValueLifetime::Constant),
                },
            ],
            vec![GraphOutput {
                buffer: "intermediate".into(),
                name: "intermediate".into(),
                contract: fixed_contract(BufferAccess::ReadWrite, ValueLifetime::Invocation),
                retained_successor_of: None,
            }],
        )
        .expect("producer node must be valid");
    let (_, succeeded) = graph
        .add_node(
            "alpha",
            copy_program("intermediate", "retained"),
            vec![
                GraphInput {
                    buffer: "intermediate".into(),
                    value: produced[0],
                    contract: fixed_contract(BufferAccess::ReadWrite, ValueLifetime::Invocation),
                },
                GraphInput {
                    buffer: "retained".into(),
                    value: retained,
                    contract: fixed_contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
                },
            ],
            vec![GraphOutput {
                buffer: "retained".into(),
                name: "retained.next".into(),
                contract: fixed_contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
                retained_successor_of: Some(retained),
            }],
        )
        .expect("retained-succession node must be valid");
    graph
        .add_node(
            "omega",
            copy_program("retained.next", "result"),
            vec![GraphInput {
                buffer: "retained.next".into(),
                value: succeeded[0],
                contract: fixed_contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
            }],
            vec![GraphOutput {
                buffer: "result".into(),
                name: "result".into(),
                contract: fixed_contract(BufferAccess::ReadWrite, ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .expect("consumer node must be valid");
    graph
}

fn payload_over_every_node(neutral: &Artifact) -> TargetPayload {
    let bindings = neutral
        .resources()
        .iter()
        .enumerate()
        .map(|(slot, resource)| TargetResourceBinding {
            resource: resource.value,
            group: 0,
            slot: u32::try_from(slot).expect("fixture slot must fit u32"),
            memory: TargetResourceMemory::Global,
            access: TargetResourceAccess::ReadWrite,
        })
        .collect::<Vec<_>>();
    let entries = neutral
        .nodes()
        .iter()
        .map(|node| entry_over(neutral, "main", node.id, bindings.clone()))
        .collect();
    TargetPayload::new(
        neutral,
        format("test.cache-target", 1),
        profile("test.cache-target", 1),
        entries,
        vec![5, 6, 7],
    )
    .expect("fixture payload must be valid")
}

/// WHY: closes the class "the host input set is derived from what entries touch
/// instead of from what the graph produces". The set asked the caller for one
/// buffer per value any entry read, so a program the compiler split across
/// fusion groups demanded a buffer for every inter-group intermediate: five
/// buffers here where the graph has three externals. A caller that supplied the
/// three it authored got a count refusal naming a number it could not derive
/// from its own graph.
///
/// The expectation is the graph's own external values, not a recomputation of
/// the runtime's rule, so a rule that starts counting produced values again goes
/// red here whatever route it takes to them.
///
/// What it does not catch: whether the bytes reach the right device buffer. That
/// is the materializer's contract, and the conformance certificate covers it by
/// comparing results against the reference.
#[test]
fn host_inputs_are_the_graph_externals_not_every_value_an_entry_reads() {
    let neutral = compile_graph(three_stage_graph(), 3);
    let payload = payload_over_every_node(&neutral);
    let session =
        ArtifactSession::from_bytes(&TEST_REGISTRATION, &envelope_bytes(neutral, [payload]))
            .expect("authenticated three-stage artifact must materialize");

    let externals = ["input", "constant", "retained"].map(|name| {
        session
            .resource(name)
            .unwrap_or_else(|error| panic!("Fix: `{name}` must be a canonical ABI value: {error}"))
    });
    let buffers = [&[0_u8; 16][..], &[0; 16][..], &[0; 16][..]];

    let bindings = session
        .host_bindings(&buffers)
        .expect("Fix: the caller supplies one buffer per graph external, so three must bind.");
    assert_eq!(
        bindings.resources().keys().copied().collect::<Vec<_>>(),
        {
            let mut expected = externals.to_vec();
            expected.sort_unstable();
            expected
        },
        "Fix: host inputs must be exactly the values no entry produces; an inter-group intermediate is device state."
    );

    let produced = ["intermediate", "retained.next", "result"].map(|name| {
        session
            .resource(name)
            .unwrap_or_else(|error| panic!("Fix: `{name}` must be a canonical ABI value: {error}"))
    });
    for value in produced {
        assert!(
            !bindings.resources().contains_key(&value),
            "Fix: value {} is produced by an entry, so the caller must not be asked for its initial bytes.",
            value.0
        );
    }

    let error = session
        .host_bindings(&[
            &[0_u8; 16][..],
            &[0; 16][..],
            &[0; 16][..],
            &[0; 16][..],
            &[0; 16][..],
        ])
        .expect_err("Fix: one buffer per value an entry reads must be refused, not accepted.")
        .to_string();
    assert!(
        error.contains("requires 3 host input buffer(s), but the caller supplied 5"),
        "Fix: the refusal must name the arity the graph implies so a caller can act on it; got: {error}"
    );
}
