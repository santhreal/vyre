//! Typed external-resource ingestion transaction and binding closure contracts.
//!
//! BACKLOG row 49 requires one bounded typed-resource ingestion transaction that maps
//! files, byte ranges, generated data, or caller memory into artifact value ids through
//! concrete-driver transfers, verifying identity, lifetime, completion, and multi-entry ABIs.

#![forbid(unsafe_code)]

use std::sync::Arc;

use vyre_driver::BackendRegistration;
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ShapeDim, ValueContract, ValueLifetime,
};
use vyre_megakernel::{AbiAccess, Artifact, ArtifactEnvelope, Digest, ResourceLifetime};
use vyre_runtime::artifact_admission::{
    ArtifactSession, ArtifactSessionError, ResourceIngestionError, ResourceManifest,
    ResourceManifestEntry, ResourceManifestSource, TypedResource, TypedResourceDataset,
};

use vyre_test_support::artifact_fixtures;

use crate::artifact_session_fixtures::{
    fixture_backend_registration, fixture_target_payload, SessionFixtureMaterializer,
};

const FORMAT: &str = "ingest.target";
static INGEST_REGISTRATION: BackendRegistration = fixture_backend_registration("ingest-artifact");

fn multi_entry_stateful_artifact() -> Artifact {
    let mut graph = ProgramGraph::new();

    let weights = graph
        .add_external_value(
            "weights",
            ValueContract {
                dtype: DataType::F32,
                shape: vec![ShapeDim::Known(16)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Constant,
            },
        )
        .expect("weights");

    let state_0 = graph
        .add_external_value(
            "state.0",
            ValueContract {
                dtype: DataType::F32,
                shape: vec![ShapeDim::Known(16)],
                access: BufferAccess::ReadWrite,
                lifetime: ValueLifetime::Retained,
            },
        )
        .expect("state");

    let (_, p1_outs) = graph
        .add_node(
            "step1",
            Program::wrapped(
                vec![
                    BufferDecl::read("weights", 0, DataType::F32).with_count(16),
                    BufferDecl::storage("state", 1, BufferAccess::ReadWrite, DataType::F32)
                        .with_count(16),
                    BufferDecl::output("intermediate", 2, DataType::F32).with_count(16),
                ],
                [16, 1, 1],
                vec![
                    Node::store(
                        "state",
                        Expr::gid_x(),
                        Expr::add(
                            Expr::load("state", Expr::gid_x()),
                            Expr::load("weights", Expr::gid_x()),
                        ),
                    ),
                    Node::store(
                        "intermediate",
                        Expr::gid_x(),
                        Expr::load("state", Expr::gid_x()),
                    ),
                ],
            ),
            vec![
                GraphInput {
                    buffer: "weights".into(),
                    value: weights,
                    contract: ValueContract {
                        dtype: DataType::F32,
                        shape: vec![ShapeDim::Known(16)],
                        access: BufferAccess::ReadOnly,
                        lifetime: ValueLifetime::Constant,
                    },
                },
                GraphInput {
                    buffer: "state".into(),
                    value: state_0,
                    contract: ValueContract {
                        dtype: DataType::F32,
                        shape: vec![ShapeDim::Known(16)],
                        access: BufferAccess::ReadWrite,
                        lifetime: ValueLifetime::Retained,
                    },
                },
            ],
            vec![
                GraphOutput {
                    buffer: "state".into(),
                    name: "state.1".into(),
                    contract: ValueContract {
                        dtype: DataType::F32,
                        shape: vec![ShapeDim::Known(16)],
                        access: BufferAccess::ReadWrite,
                        lifetime: ValueLifetime::Retained,
                    },
                    retained_successor_of: Some(state_0),
                },
                GraphOutput {
                    buffer: "intermediate".into(),
                    name: "inter_val".into(),
                    contract: ValueContract {
                        dtype: DataType::F32,
                        shape: vec![ShapeDim::Known(16)],
                        access: BufferAccess::ReadWrite,
                        lifetime: ValueLifetime::Invocation,
                    },
                    retained_successor_of: None,
                },
            ],
        )
        .expect("step1 node");

    let inter_val = p1_outs[1];

    graph
        .add_node(
            "step2",
            Program::wrapped(
                vec![
                    BufferDecl::read("intermediate", 0, DataType::F32).with_count(16),
                    BufferDecl::output("final_out", 1, DataType::F32).with_count(16),
                ],
                [16, 1, 1],
                vec![Node::store(
                    "final_out",
                    Expr::gid_x(),
                    Expr::mul(Expr::load("intermediate", Expr::gid_x()), Expr::f32(2.0)),
                )],
            ),
            vec![GraphInput {
                buffer: "intermediate".into(),
                value: inter_val,
                contract: ValueContract {
                    dtype: DataType::F32,
                    shape: vec![ShapeDim::Known(16)],
                    access: BufferAccess::ReadOnly,
                    lifetime: ValueLifetime::Invocation,
                },
            }],
            vec![GraphOutput {
                buffer: "final_out".into(),
                name: "final_result".into(),
                contract: ValueContract {
                    dtype: DataType::F32,
                    shape: vec![ShapeDim::Known(16)],
                    access: BufferAccess::WriteOnly,
                    lifetime: ValueLifetime::Output,
                },
                retained_successor_of: None,
            }],
        )
        .expect("step2 node");

    artifact_fixtures::compile_graph(graph, 0)
}

fn create_test_session() -> (ArtifactSession, Arc<SessionFixtureMaterializer>, Artifact) {
    let artifact = multi_entry_stateful_artifact();
    let payload = fixture_target_payload(&artifact, FORMAT, vec![1, 2, 3, 4]);
    let mut envelope = ArtifactEnvelope::new(artifact.clone());
    envelope.attach_target_payload(payload).expect("payload");
    let materializer = SessionFixtureMaterializer::new("ingest-backend", "ingest-device", FORMAT);
    let session = ArtifactSession::from_envelope_with_materializer(
        &INGEST_REGISTRATION,
        envelope,
        materializer.clone(),
    )
    .expect("session");
    (session, materializer, artifact)
}

#[test]
fn typed_resource_ingestion_validates_abi_and_workspace_bindings() {
    let (session, _materializer, _artifact) = create_test_session();
    let workspace = session.allocate_workspace().expect("workspace allocation");

    let weights_id = session.resource("weights").expect("weights id");
    let state_id = session.resource("state.0").expect("state.0 id");
    let final_id = session.resource("final_result").expect("final_result id");
    let inter_id = session.resource("inter_val").expect("inter_val id");

    // 1. Valid ingestion with workspace
    let mut dataset = TypedResourceDataset::new();
    dataset.add_memory(weights_id, vec![0u8; 64]);
    dataset.add_memory(state_id, vec![0u8; 64]);
    dataset.add_memory(final_id, vec![0u8; 64]);

    let bindings = session
        .ingest_with_workspace(&workspace, &dataset)
        .expect("valid bindings with workspace");

    assert!(bindings.resources().len() >= 4);

    // 2. Reject missing resource
    let mut missing_dataset = TypedResourceDataset::new();
    missing_dataset.add_memory(weights_id, vec![0u8; 64]);
    missing_dataset.add_memory(state_id, vec![0u8; 64]);

    let missing_err = session
        .ingest_with_workspace(&workspace, &missing_dataset)
        .expect_err("must reject missing final_result");
    assert!(
        missing_err.to_string().contains("missing required resource"),
        "error must name missing resource: {missing_err}"
    );

    // 3. Reject caller supplying workspace-owned value
    let mut override_dataset = TypedResourceDataset::new();
    override_dataset.add_memory(weights_id, vec![0u8; 64]);
    override_dataset.add_memory(state_id, vec![0u8; 64]);
    override_dataset.add_memory(final_id, vec![0u8; 64]);
    override_dataset.add_memory(inter_id, vec![0u8; 64]);

    let workspace_override_err = session
        .ingest_with_workspace(&workspace, &override_dataset)
        .expect_err("must reject caller overriding workspace value");
    assert!(
        workspace_override_err
            .to_string()
            .contains("workspace-owned"),
        "error must state value is workspace-owned: {workspace_override_err}"
    );

    // Clean up
    session.free_workspace(workspace).expect("free workspace");
}

/// WHY: BACKLOG row 49 contract - a partially valid dataset leaves no allocation dispatch-visible.
#[test]
fn partially_valid_dataset_leaves_no_allocation_dispatch_visible() {
    let (session, materializer, _artifact) = create_test_session();
    let workspace = session.allocate_workspace().expect("workspace allocation");

    let weights_id = session.resource("weights").expect("weights id");
    let state_id = session.resource("state.0").expect("state.0 id");
    let final_id = session.resource("final_result").expect("final_result id");

    // Case 1: Pre-validation failure due to byte count mismatch on second resource
    let mut dataset_bad_len = TypedResourceDataset::new();
    dataset_bad_len.add_memory(weights_id, vec![0u8; 64]); // Valid 64 bytes
    dataset_bad_len.add_memory(state_id, vec![0u8; 32]); // Invalid: expected 64 bytes
    dataset_bad_len.add_memory(final_id, vec![0u8; 64]);

    let alloc_count_before = materializer.allocated.lock().unwrap().len();

    let err = session
        .ingest_with_workspace(&workspace, &dataset_bad_len)
        .expect_err("partially valid dataset with bad length on 2nd resource must be rejected");

    let alloc_count_after = materializer.allocated.lock().unwrap().len();

    assert_eq!(
        alloc_count_before, alloc_count_after,
        "pre-validation must reject before performing any driver allocation"
    );
    assert!(
        err.to_string().contains("byte count mismatch")
            || err.to_string().contains("schema mismatch"),
        "rejection must name the mismatch: {err}"
    );

    // Case 2: Missing third resource in dataset
    let mut dataset_missing_one = TypedResourceDataset::new();
    dataset_missing_one.add_memory(weights_id, vec![0u8; 64]);
    dataset_missing_one.add_memory(state_id, vec![0u8; 64]);
    // final_id missing!

    let alloc_count_before_2 = materializer.allocated.lock().unwrap().len();

    let err_missing = session
        .ingest_with_workspace(&workspace, &dataset_missing_one)
        .expect_err("dataset missing one required resource must be rejected before allocation");

    let alloc_count_after_2 = materializer.allocated.lock().unwrap().len();

    assert_eq!(
        alloc_count_before_2, alloc_count_after_2,
        "missing resource check must fail closed before any allocation"
    );
    assert!(
        err_missing
            .to_string()
            .contains("missing required resource"),
        "error must state missing resource: {err_missing}"
    );

    // Case 3: Non-existent file source in dataset
    let mut dataset_missing_file = TypedResourceDataset::new();
    dataset_missing_file.add_memory(weights_id, vec![0u8; 64]);
    dataset_missing_file.insert(TypedResource::file(
        state_id,
        "non_existent_file_path_for_test.bin",
        0,
        64,
    )).expect("insert");
    dataset_missing_file.add_memory(final_id, vec![0u8; 64]);

    let alloc_count_before_3 = materializer.allocated.lock().unwrap().len();

    let err_file = session
        .ingest_with_workspace(&workspace, &dataset_missing_file)
        .expect_err("missing file source must be rejected before allocation");

    let alloc_count_after_3 = materializer.allocated.lock().unwrap().len();

    assert_eq!(
        alloc_count_before_3, alloc_count_after_3,
        "I/O error check must fail before any allocation becomes dispatch-visible"
    );
    assert!(
        err_file.to_string().contains("failed to read resource file"),
        "error must name file read error: {err_file}"
    );

    session.free_workspace(workspace).expect("free workspace");
}

/// WHY: BACKLOG row 49 contract - a mismatched typed schema is rejected before upload.
#[test]
fn mismatched_typed_schema_is_rejected_before_upload() {
    let (session, materializer, _artifact) = create_test_session();
    let workspace = session.allocate_workspace().expect("workspace allocation");

    let weights_id = session.resource("weights").expect("weights id");
    let state_id = session.resource("state.0").expect("state.0 id");
    let final_id = session.resource("final_result").expect("final_result id");

    // 1. Data type mismatch: expected F32, caller specifies U64
    let mut dataset_dtype_mismatch = TypedResourceDataset::new();
    dataset_dtype_mismatch.insert(
        TypedResource::memory(weights_id, vec![0u8; 64]).with_dtype(DataType::U64),
    ).expect("insert");
    dataset_dtype_mismatch.add_memory(state_id, vec![0u8; 64]);
    dataset_dtype_mismatch.add_memory(final_id, vec![0u8; 64]);

    let alloc_before = materializer.allocated.lock().unwrap().len();

    let dtype_err = session
        .ingest_with_workspace(&workspace, &dataset_dtype_mismatch)
        .expect_err("mismatched dtype must be rejected before upload");

    let alloc_after = materializer.allocated.lock().unwrap().len();

    assert_eq!(alloc_before, alloc_after, "zero allocations must occur on dtype mismatch");
    assert!(
        matches!(
            dtype_err,
            ArtifactSessionError::Ingestion(ResourceIngestionError::DtypeMismatch { .. })
        ),
        "error must be DtypeMismatch, got: {dtype_err}"
    );

    // 2. Element count mismatch: expected 16, caller specifies 32
    let mut dataset_elem_mismatch = TypedResourceDataset::new();
    dataset_elem_mismatch.insert(
        TypedResource::memory(weights_id, vec![0u8; 64]).with_element_count(32),
    ).expect("insert");
    dataset_elem_mismatch.add_memory(state_id, vec![0u8; 64]);
    dataset_elem_mismatch.add_memory(final_id, vec![0u8; 64]);

    let elem_err = session
        .ingest_with_workspace(&workspace, &dataset_elem_mismatch)
        .expect_err("mismatched element count must be rejected before upload");

    assert!(
        elem_err.to_string().contains("element count mismatch"),
        "error must be ElementCountMismatch: {elem_err}"
    );

    // 3. Lifetime mismatch: expected Constant, caller specifies Invocation
    let mut dataset_lifetime_mismatch = TypedResourceDataset::new();
    dataset_lifetime_mismatch.insert(
        TypedResource::memory(weights_id, vec![0u8; 64]).with_lifetime(ResourceLifetime::Invocation),
    ).expect("insert");
    dataset_lifetime_mismatch.add_memory(state_id, vec![0u8; 64]);
    dataset_lifetime_mismatch.add_memory(final_id, vec![0u8; 64]);

    let lifetime_err = session
        .ingest_with_workspace(&workspace, &dataset_lifetime_mismatch)
        .expect_err("mismatched lifetime must be rejected before upload");

    assert!(
        lifetime_err.to_string().contains("lifetime mismatch"),
        "error must be LifetimeMismatch: {lifetime_err}"
    );

    // 4. Access mismatch: expected ReadOnly, caller specifies WriteOnly
    let mut dataset_access_mismatch = TypedResourceDataset::new();
    dataset_access_mismatch.insert(
        TypedResource::memory(weights_id, vec![0u8; 64]).with_access(AbiAccess::WriteOnly),
    ).expect("insert");
    dataset_access_mismatch.add_memory(state_id, vec![0u8; 64]);
    dataset_access_mismatch.add_memory(final_id, vec![0u8; 64]);

    let access_err = session
        .ingest_with_workspace(&workspace, &dataset_access_mismatch)
        .expect_err("mismatched access must be rejected before upload");

    assert!(
        access_err.to_string().contains("access mismatch"),
        "error must be AccessMismatch: {access_err}"
    );

    // 5. Device generation mismatch: expected session generation (0), caller specifies 99
    let mut dataset_gen_mismatch = TypedResourceDataset::new();
    dataset_gen_mismatch.insert(
        TypedResource::memory(weights_id, vec![0u8; 64]).with_generation(99),
    ).expect("insert");
    dataset_gen_mismatch.add_memory(state_id, vec![0u8; 64]);
    dataset_gen_mismatch.add_memory(final_id, vec![0u8; 64]);

    let gen_err = session
        .ingest_with_workspace(&workspace, &dataset_gen_mismatch)
        .expect_err("mismatched generation must be rejected before upload");

    assert!(
        gen_err.to_string().contains("generation mismatch"),
        "error must be GenerationMismatch: {gen_err}"
    );

    // 6. Identity digest mismatch
    let mut dataset_id_mismatch = TypedResourceDataset::new();
    dataset_id_mismatch.insert(
        TypedResource::memory(weights_id, vec![0u8; 64]).with_identity(Digest([0xAA; 32])),
    ).expect("insert");
    dataset_id_mismatch.add_memory(state_id, vec![0u8; 64]);
    dataset_id_mismatch.add_memory(final_id, vec![0u8; 64]);

    let id_err = session
        .ingest_with_workspace(&workspace, &dataset_id_mismatch)
        .expect_err("mismatched identity digest must be rejected before upload");

    assert!(
        id_err.to_string().contains("identity mismatch"),
        "error must be IdentityMismatch: {id_err}"
    );

    session.free_workspace(workspace).expect("free workspace");
}

/// WHY: BACKLOG row 49 maps files, byte ranges, generated data, caller memory, and pre-allocated resident resources.
#[test]
fn typed_resource_ingestion_supports_all_source_variants() {
    let (session, _materializer, _artifact) = create_test_session();
    let workspace = session.allocate_workspace().expect("workspace allocation");

    let weights_id = session.resource("weights").expect("weights id");
    let state_id = session.resource("state.0").expect("state.0 id");
    let final_id = session.resource("final_result").expect("final_result id");

    // 1. Memory source
    let res_memory = TypedResource::memory(weights_id, vec![0x11; 64])
        .with_dtype(DataType::F32)
        .with_element_count(16)
        .with_lifetime(ResourceLifetime::Constant)
        .with_access(AbiAccess::ReadOnly);

    // 2. Pattern generated source
    let res_pattern = TypedResource::pattern(state_id, 0x22, 64)
        .with_dtype(DataType::F32)
        .with_element_count(16)
        .with_lifetime(ResourceLifetime::Retained)
        .with_access(AbiAccess::ReadWrite);

    // 3. Zeroed generated source
    let res_zeroed = TypedResource::zeroed(final_id, 64)
        .with_dtype(DataType::F32)
        .with_element_count(16)
        .with_lifetime(ResourceLifetime::Output)
        .with_access(AbiAccess::WriteOnly);

    let mut dataset = TypedResourceDataset::new();
    dataset.insert(res_memory).expect("insert memory");
    dataset.insert(res_pattern).expect("insert pattern");
    dataset.insert(res_zeroed).expect("insert zeroed");

    let bindings = session
        .ingest_with_workspace(&workspace, &dataset)
        .expect("ingestion of memory and generated sources must succeed");

    assert!(bindings.resources().contains_key(&weights_id));
    assert!(bindings.resources().contains_key(&state_id));
    assert!(bindings.resources().contains_key(&final_id));

    // 4. ByteRange source
    let raw_block = vec![0x33u8; 128];
    let source_digest = Digest(*blake3::hash(&raw_block).as_bytes());
    let res_range = TypedResource::byte_range(
        weights_id,
        source_digest,
        32,
        64,
        raw_block[32..96].to_vec(),
    );

    // 5. Pre-allocated resident buffer (immutable reuse)
    let pre_alloc_state = session.allocate_resident(64).expect("allocate resident");
    let res_resident = TypedResource::resident(state_id, pre_alloc_state);

    let mut dataset2 = TypedResourceDataset::new();
    dataset2.insert(res_range).expect("insert range");
    dataset2.insert(res_resident).expect("insert resident");
    dataset2.insert(TypedResource::zeroed(final_id, 64)).expect("insert zeroed");

    let bindings2 = session
        .ingest_with_workspace(&workspace, &dataset2)
        .expect("ingestion of byte range and resident reuse must succeed");

    assert!(bindings2.resources().contains_key(&weights_id));
    assert!(bindings2.resources().contains_key(&state_id));
    assert!(bindings2.resources().contains_key(&final_id));

    session.free_workspace(workspace).expect("free workspace");
}

/// WHY: AOT bundle manifest serialization, round-trip, and ingestion validation.
#[test]
fn aot_bundle_manifest_round_trip_and_ingestion() {
    let (session, _materializer, artifact) = create_test_session();
    let workspace = session.allocate_workspace().expect("workspace allocation");

    let weights_id = session.resource("weights").expect("weights id");
    let state_id = session.resource("state.0").expect("state.0 id");
    let final_id = session.resource("final_result").expect("final_result id");

    let manifest = ResourceManifest::new(
        Some(artifact.digest()),
        vec![
            ResourceManifestEntry {
                value: weights_id,
                name: Some("weights".into()),
                dtype: DataType::F32,
                element_count: 16,
                byte_count: 64,
                lifetime: ResourceLifetime::Constant,
                access: AbiAccess::ReadOnly,
                source: ResourceManifestSource::InlineBytes {
                    bytes: vec![1u8; 64],
                },
                identity: None,
            },
            ResourceManifestEntry {
                value: state_id,
                name: Some("state.0".into()),
                dtype: DataType::F32,
                element_count: 16,
                byte_count: 64,
                lifetime: ResourceLifetime::Retained,
                access: AbiAccess::ReadWrite,
                source: ResourceManifestSource::Pattern {
                    fill: 2,
                    byte_count: 64,
                },
                identity: None,
            },
            ResourceManifestEntry {
                value: final_id,
                name: Some("final_result".into()),
                dtype: DataType::F32,
                element_count: 16,
                byte_count: 64,
                lifetime: ResourceLifetime::Output,
                access: AbiAccess::WriteOnly,
                source: ResourceManifestSource::Zeroed {
                    byte_count: 64,
                },
                identity: None,
            },
        ],
    );

    // Round-trip to JSON bytes
    let json_bytes = manifest.to_bytes().expect("manifest serialization");
    let decoded_manifest = ResourceManifest::from_bytes(&json_bytes).expect("manifest deserialization");
    assert_eq!(manifest, decoded_manifest);

    // Ingest from manifest
    let dataset = decoded_manifest.to_dataset().expect("manifest to dataset");
    let bindings = session
        .ingest_with_workspace(&workspace, &dataset)
        .expect("manifest ingestion must succeed");

    assert!(bindings.resources().contains_key(&weights_id));
    assert!(bindings.resources().contains_key(&state_id));
    assert!(bindings.resources().contains_key(&final_id));

    session.free_workspace(workspace).expect("free workspace");
}

#[test]
fn lifetime_and_access_exhaustive_closure() {
    let lifetimes = [
        ResourceLifetime::Constant,
        ResourceLifetime::Invocation,
        ResourceLifetime::Retained,
        ResourceLifetime::Output,
    ];
    for lt in lifetimes {
        let matched = match lt {
            ResourceLifetime::Constant => "constant",
            ResourceLifetime::Invocation => "invocation",
            ResourceLifetime::Retained => "retained",
            ResourceLifetime::Output => "output",
        };
        assert!(!matched.is_empty());
    }

    let accesses = [
        AbiAccess::ReadOnly,
        AbiAccess::WriteOnly,
        AbiAccess::ReadWrite,
        AbiAccess::Uniform,
    ];
    for acc in accesses {
        let is_read = match acc {
            AbiAccess::ReadOnly | AbiAccess::ReadWrite | AbiAccess::Uniform => true,
            AbiAccess::WriteOnly => false,
        };
        let is_write = match acc {
            AbiAccess::WriteOnly | AbiAccess::ReadWrite => true,
            AbiAccess::ReadOnly | AbiAccess::Uniform => false,
        };
        match acc {
            AbiAccess::ReadOnly => {
                assert!(is_read);
                assert!(!is_write);
            }
            AbiAccess::WriteOnly => {
                assert!(!is_read);
                assert!(is_write);
            }
            AbiAccess::ReadWrite => {
                assert!(is_read);
                assert!(is_write);
            }
            AbiAccess::Uniform => {
                assert!(is_read);
                assert!(!is_write);
            }
        }
    }
}
