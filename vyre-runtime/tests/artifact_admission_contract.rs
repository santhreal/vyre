//! Regression contracts for canonical runtime artifact admission.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::LazyLock;

use vyre_driver::backend::BackendRegistration;
use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BindingSet, BoundResource, Completion,
    Device, DeviceIdentity, Submission, VyreBackend,
};
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Program, ProgramGraph, ShapeDim, ValueContract,
    ValueLifetime,
};
use vyre_megakernel::{
    compile, Artifact, ArtifactEnvelope, ArtifactNodeId, ArtifactValueId, CompileRequest,
    DiagnosticCode, Digest, ExternalFacts, SearchBudget, TargetEntryPoint, TargetPayload,
    TargetPayloadFormat, TargetResourceAccess, TargetResourceBinding, TargetResourceMemory,
};
use vyre_runtime::{
    admit_artifact, admit_cached_artifact, admit_envelope, classify_backend_error,
    recover_artifact_session, ArtifactAdmissionError, ArtifactSession, InMemoryPipelineCache,
    PersistentExecutor, PipelineCacheStore, PipelineFingerprint, RecoveryClass, ResidentQueueState,
    RetainedArtifactSession,
};

const FRAME_HEADER_BYTES: usize = 10;
const FRAME_DIGEST_BYTES: usize = 32;
const ENVELOPE_DIGEST_DOMAIN: &[u8] = b"vyre-megakernel-envelope-v2\0";
const TARGET_PAYLOAD_DIGEST_DOMAIN: &[u8] = b"vyre-megakernel-target-payload-v2\0";

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

fn resident_queue_artifact() -> Artifact {
    let mut graph = ProgramGraph::new();
    let buffers = ["control", "ring_buffer", "debug_log", "io_queue"];
    for name in buffers {
        graph
            .add_external_value(
                name,
                ValueContract {
                    dtype: DataType::U32,
                    shape: vec![ShapeDim::Known(4)],
                    access: BufferAccess::ReadWrite,
                    lifetime: ValueLifetime::Retained,
                },
            )
            .unwrap();
    }
    graph
        .add_node(
            "queue",
            Program::wrapped(
                buffers
                    .into_iter()
                    .enumerate()
                    .map(|(slot, name)| {
                        BufferDecl::read_write(name, slot as u32, DataType::U32).with_count(4)
                    })
                    .collect(),
                [1, 1, 1],
                Vec::new(),
            ),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        SearchBudget::new(1, 1, 1, 0, 1_000_000_000),
        1_000_000,
    )
    .validate()
    .unwrap();
    compile(&request).unwrap()
}

fn queue_payload(neutral: &Artifact) -> TargetPayload {
    let bindings = neutral
        .resources()
        .iter()
        .enumerate()
        .map(|(slot, resource)| TargetResourceBinding {
            resource: resource.value,
            slot: slot as u32,
            memory: TargetResourceMemory::Global,
            access: TargetResourceAccess::ReadWrite,
        })
        .collect();
    TargetPayload::new(
        neutral,
        format("test.cache-target", 1),
        vec![TargetEntryPoint {
            name: "queue".to_string(),
            node: ArtifactNodeId(0),
            grid_size: [1, 1, 1],
            dynamic_shared_bytes: 0,
            resource_bindings: bindings,
        }],
        vec![1, 2, 3],
    )
    .unwrap()
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

fn payload(neutral: &Artifact, payload_format: TargetPayloadFormat, bytes: &[u8]) -> TargetPayload {
    TargetPayload::new(neutral, payload_format, vec![entry()], bytes.to_vec())
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

fn assert_diagnostic(error: &ArtifactAdmissionError, code: DiagnosticCode, path: &str, fix: &str) {
    let diagnostic = error.diagnostic();
    assert_eq!(diagnostic.code, code);
    assert_eq!(diagnostic.path, path);
    assert_eq!(diagnostic.fix, fix);
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
    encode_frame(b"VTP0", 2, TARGET_PAYLOAD_DIGEST_DOMAIN, &body)
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
    let truncated_error =
        admit_artifact(&truncated, &required).expect_err("truncated canonical bytes must fail");
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
    bytes[4..6].copy_from_slice(&3_u16.to_le_bytes());

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
    let corrupted_envelope = replace_nested_payload(&envelope, &target_bytes, &corrupted_target);

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
    // Match the AOT package payload format identity contract (`Target::Ptx.aot_target_id()` + version 1).
    let required = format("secondary_text", 1);
    let attached = TargetPayload::new(
        &neutral,
        required.clone(),
        vec![TargetEntryPoint {
            name: "main".into(),
            node,
            grid_size: [1, 1, 1],
            dynamic_shared_bytes: 0,
            resource_bindings: vec![TargetResourceBinding {
                resource,
                slot: 0,
                memory: TargetResourceMemory::Global,
                access: TargetResourceAccess::ReadOnly,
            }],
        }],
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

static MATERIALIZER_GENERATION: AtomicU64 = AtomicU64::new(0);
static MATERIALIZER_CALLS: AtomicU64 = AtomicU64::new(0);
static TEST_SUPPORTED_OPS: LazyLock<HashSet<vyre_foundation::ir::OpId>> =
    LazyLock::new(HashSet::new);

struct TestDevice {
    identity: DeviceIdentity,
    format: TargetPayloadFormat,
}

impl Device for TestDevice {
    fn identity(&self) -> &DeviceIdentity {
        &self.identity
    }

    fn target_format(&self) -> &TargetPayloadFormat {
        &self.format
    }

    fn is_healthy(&self) -> bool {
        true
    }
}

struct TestMaterializer {
    device: TestDevice,
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
        Ok(Box::new(TestInstance {
            artifact: artifact.digest(),
            payload: payload.digest(),
            device: self.device.identity.clone(),
            retained: artifact
                .resources()
                .iter()
                .filter(|resource| resource.lifetime == vyre_megakernel::ResourceLifetime::Retained)
                .map(|resource| resource.value)
                .collect(),
        }))
    }
}

struct TestInstance {
    artifact: Digest,
    payload: Digest,
    device: DeviceIdentity,
    retained: BTreeSet<ArtifactValueId>,
}

impl ArtifactInstance for TestInstance {
    fn artifact(&self) -> Digest {
        self.artifact
    }

    fn payload(&self) -> Digest {
        self.payload
    }

    fn device(&self) -> &DeviceIdentity {
        &self.device
    }

    fn submit(&self, bindings: BindingSet) -> Result<Box<dyn Submission>, BackendError> {
        if bindings.artifact() != self.artifact {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: test bindings must name the materialized artifact.".to_string(),
            });
        }
        let retained = bindings
            .resources()
            .iter()
            .filter_map(|(value, resource)| {
                self.retained.contains(value).then(|| match resource {
                    BoundResource::Host(bytes) => Ok((*value, bytes.clone())),
                    BoundResource::Resident(_) => Err(BackendError::UnsupportedFeature {
                        name: "test resident resource".to_string(),
                        backend: "test-artifact".to_string(),
                    }),
                })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        Ok(Box::new(TestSubmission(Some(Completion {
            artifact: self.artifact,
            outputs: BTreeMap::new(),
            retained,
            device_ns: None,
        }))))
    }
}

struct TestSubmission(Option<Completion>);

impl Submission for TestSubmission {
    fn is_ready(&self) -> bool {
        true
    }

    fn wait(mut self: Box<Self>) -> Result<Completion, BackendError> {
        self.0.take().ok_or_else(|| BackendError::InvalidProgram {
            fix: "Fix: consume the test submission only once.".to_string(),
        })
    }
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
    let generation = MATERIALIZER_GENERATION.fetch_add(1, Ordering::AcqRel) + 1;
    Ok(Box::new(TestMaterializer {
        device: TestDevice {
            identity: DeviceIdentity {
                backend: "test-artifact",
                device: "test-device".to_string(),
                generation,
            },
            format: format("test.cache-target", 1),
        },
    }))
}

static TEST_REGISTRATION: BackendRegistration = BackendRegistration {
    id: "test-artifact",
    factory: test_backend_factory,
    supported_ops: test_supported_ops,
    target_compiler: None,
    materializer: Some(test_materializer_factory),
};

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
    assert_eq!(classify_backend_error(&failure), RecoveryClass::DeviceLoss);
    let recovered = recover_artifact_session(&session, failure).unwrap();
    assert!(recovered.generation > first_device.generation);
    assert_eq!(session.artifact().unwrap(), neutral.digest());

    let permanent = BackendError::InvalidProgram {
        fix: "Fix: reject malformed test bindings.".to_string(),
    };
    let error = recover_artifact_session(&session, permanent).unwrap_err();
    assert!(matches!(
        error,
        vyre_runtime::ArtifactSessionError::Backend(BackendError::InvalidProgram { .. })
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
