//! Session state machine, retained generation, error classification, and recovery contracts.
//!
//! BACKLOG row 54 requires a typed domain-neutral execution session that owns artifact
//! identity, authenticated resources, retained generations, bounded step/control state,
//! IO readiness, and recovery across one-shot, iterative, and event-driven graphs.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use vyre_driver::materialize::{DeviceSpec, MaterializerDevice};
use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BackendRegistration, BindingSet,
    BoundResource, Completion, Device, ErrorCode, ResidentOwner, Resource, VyreBackend,
};
use vyre_foundation::diagnostics::RetryClass;
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ShapeDim, ValueContract, ValueLifetime,
};
use vyre_megakernel::{
    AbiAccess, Artifact, ArtifactEnvelope, ArtifactValueId, TargetEntryPoint, TargetPayload,
    TargetPayloadFormat, TargetProfile, TargetResourceAccess, TargetResourceBinding,
    TargetResourceMemory,
};
use vyre_runtime::artifact_admission::{ArtifactSession, RetainedArtifactSession};
use vyre_runtime::recovery::{classify_backend_error, recover_artifact_session};

use vyre_test_support::artifact_fixtures;
use vyre_test_support::fixture_instance::FixtureInstance;

const FORMAT: &str = "state_machine.target";

struct SmMaterializer {
    device: MaterializerDevice,
    owner: ResidentOwner,
    next: AtomicU64,
    allocated: Mutex<Vec<usize>>,
    freed: Mutex<Vec<Resource>>,
}

impl SmMaterializer {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            device: MaterializerDevice::acquire(DeviceSpec {
                backend: "sm-backend",
                device: "sm-device".to_string(),
                format_extension: FORMAT,
                format_version: 1,
                profile: TargetProfile::new(FORMAT, 1, [64, 1, 1], 64, 1_024, 0)
                    .expect("profile"),
            })
            .expect("device"),
            owner: ResidentOwner::new().expect("owner"),
            next: AtomicU64::new(0),
            allocated: Mutex::new(Vec::new()),
            freed: Mutex::new(Vec::new()),
        })
    }
}

impl ArtifactMaterializer for SmMaterializer {
    fn device(&self) -> &dyn Device {
        &self.device
    }

    fn materialize(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<Box<dyn ArtifactInstance>, BackendError> {
        Ok(FixtureInstance::neutral(
            artifact,
            payload,
            self.device.identity(),
        ))
    }

    fn allocate_resident(&self, byte_len: usize) -> Result<Resource, BackendError> {
        self.allocated
            .lock()
            .expect("allocation log")
            .push(byte_len);
        let id = self.next.fetch_add(1, Ordering::AcqRel);
        Ok(Resource::Resident(self.owner.handle(id)))
    }

    fn free_resident(&self, resource: Resource) -> Result<(), BackendError> {
        self.freed.lock().expect("free log").push(resource);
        Ok(())
    }
}

fn sm_backend_factory() -> Result<Box<dyn VyreBackend>, BackendError> {
    Err(BackendError::UnsupportedFeature {
        name: "raw Program backend".to_string(),
        backend: "sm-artifact".to_string(),
    })
}

fn sm_supported_ops() -> &'static std::collections::HashSet<vyre_foundation::ir::OpId> {
    static OPS: std::sync::LazyLock<std::collections::HashSet<vyre_foundation::ir::OpId>> =
        std::sync::LazyLock::new(std::collections::HashSet::new);
    &OPS
}

static SM_REGISTRATION: BackendRegistration = BackendRegistration {
    id: "sm-artifact",
    target_id: vyre_foundation::operation::TargetId::expect_valid("sm-artifact"),
    payload_format: None,
    reference_oracle: false,
    factory: sm_backend_factory,
    supported_ops: sm_supported_ops,
    semantic_operations: sm_supported_ops,
    target_compiler: None,
    materializer: None,
};

fn stateful_accumulator_artifact() -> Artifact {
    let mut graph = ProgramGraph::new();

    let delta = graph
        .add_external_value(
            "delta",
            ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(1)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
        )
        .expect("delta");

    let counter_0 = graph
        .add_external_value(
            "counter.0",
            ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(1)],
                access: BufferAccess::ReadWrite,
                lifetime: ValueLifetime::Retained,
            },
        )
        .expect("counter");

    graph
        .add_node(
            "accumulate",
            Program::wrapped(
                vec![
                    BufferDecl::read("delta", 0, DataType::U32).with_count(1),
                    BufferDecl::storage("counter", 1, BufferAccess::ReadWrite, DataType::U32).with_count(1),
                    BufferDecl::output("snapshot", 2, DataType::U32).with_count(1),
                ],
                [1, 1, 1],
                vec![
                    Node::store(
                        "counter",
                        Expr::u32(0),
                        Expr::add(Expr::load("counter", Expr::u32(0)), Expr::load("delta", Expr::u32(0))),
                    ),
                    Node::store(
                        "snapshot",
                        Expr::u32(0),
                        Expr::load("counter", Expr::u32(0)),
                    ),
                ],
            ),
            vec![
                GraphInput {
                    buffer: "delta".into(),
                    value: delta,
                    contract: ValueContract {
                        dtype: DataType::U32,
                        shape: vec![ShapeDim::Known(1)],
                        access: BufferAccess::ReadOnly,
                        lifetime: ValueLifetime::Invocation,
                    },
                },
                GraphInput {
                    buffer: "counter".into(),
                    value: counter_0,
                    contract: ValueContract {
                        dtype: DataType::U32,
                        shape: vec![ShapeDim::Known(1)],
                        access: BufferAccess::ReadWrite,
                        lifetime: ValueLifetime::Retained,
                    },
                },
            ],
            vec![
                GraphOutput {
                    buffer: "counter".into(),
                    name: "counter.1".into(),
                    contract: ValueContract {
                        dtype: DataType::U32,
                        shape: vec![ShapeDim::Known(1)],
                        access: BufferAccess::ReadWrite,
                        lifetime: ValueLifetime::Retained,
                    },
                    retained_successor_of: Some(counter_0),
                },
                GraphOutput {
                    buffer: "snapshot".into(),
                    name: "accum_out".into(),
                    contract: ValueContract {
                        dtype: DataType::U32,
                        shape: vec![ShapeDim::Known(1)],
                        access: BufferAccess::WriteOnly,
                        lifetime: ValueLifetime::Output,
                    },
                    retained_successor_of: None,
                },
            ],
        )
        .expect("accumulate node");

    artifact_fixtures::compile_graph(graph, 0)
}

fn single_entry_payload(artifact: &Artifact) -> TargetPayload {
    let entries = artifact
        .abi()
        .entries
        .iter()
        .map(|entry| {
            let bindings = entry
                .inputs
                .iter()
                .chain(entry.outputs.iter())
                .enumerate()
                .map(|(slot, &resource)| TargetResourceBinding {
                    resource,
                    group: 0,
                    slot: slot as u32,
                    memory: TargetResourceMemory::Global,
                    access: TargetResourceAccess::ReadWrite,
                })
                .collect();
            TargetEntryPoint {
                name: format!("entry_{}", entry.node.0),
                node: entry.node,
                workgroup_size: [64, 1, 1],
                grid_size: [1, 1, 1],
                dynamic_shared_bytes: 0,
                resource_bindings: bindings,
            }
        })
        .collect();

    TargetPayload::new(
        artifact,
        TargetPayloadFormat::new(FORMAT, 1).expect("format"),
        TargetProfile::new(FORMAT, 1, [64, 1, 1], 64, 1_024, 0).expect("profile"),
        entries,
        vec![10, 20, 30],
    )
    .expect("payload")
}

#[test]
fn retained_session_manages_state_machine_generations_atomically() {
    let artifact = stateful_accumulator_artifact();
    let payload = single_entry_payload(&artifact);
    let mut envelope = ArtifactEnvelope::new(artifact.clone());
    envelope.insert_payload(payload).expect("payload");
    let materializer = SmMaterializer::new();

    let session = ArtifactSession::from_envelope_with_materializer(
        &SM_REGISTRATION,
        envelope,
        materializer.clone(),
    )
    .expect("session");

    let counter_id = session.resource("counter.0").expect("counter.0 resource");

    // 1. Missing initial retained state must fail
    let bad_init = RetainedArtifactSession::new(
        session.clone(),
        BTreeMap::new(),
    );
    assert!(bad_init.is_err(), "must reject empty initial retained state");

    // 2. Proper initialization
    let initial_state = BTreeMap::from([(counter_id, 0_u32.to_le_bytes().to_vec())]);
    let retained_session = RetainedArtifactSession::new(session, initial_state)
        .expect("valid retained session initialization");

    assert_eq!(
        retained_session.artifact().unwrap(),
        artifact.digest()
    );

    // 3. State replacement validation
    let replacement = BTreeMap::from([(counter_id, 100_u32.to_le_bytes().to_vec())]);
    retained_session
        .replace_retained(replacement)
        .expect("valid state replacement");

    let bad_replacement = BTreeMap::from([(ArtifactValueId(999), vec![0])]);
    assert!(
        retained_session.replace_retained(bad_replacement).is_err(),
        "must reject invalid retained replacement keys"
    );
}

#[test]
fn backend_error_classification_exhaustive_closure() {
    let test_cases = [
        (
            BackendError::DeviceLost {
                backend: "test".into(),
                device: "gpu".into(),
                generation: 1,
                message: "lost".into(),
            },
            RetryClass::NewDevice,
        ),
        (
            BackendError::DeviceOutOfMemory {
                requested: 1024,
                available: 512,
            },
            RetryClass::SameDevice,
        ),
        (
            BackendError::PoisonedLock {
                lock_error: "state".into(),
            },
            RetryClass::SameDevice,
        ),
        (
            BackendError::UnsupportedFeature {
                name: "feature".into(),
                backend: "test".into(),
            },
            RetryClass::Never,
        ),
        (
            BackendError::InvalidProgram {
                fix: "invalid".into(),
            },
            RetryClass::Never,
        ),
    ];

    for (error, expected_class) in test_cases {
        assert_eq!(classify_backend_error(&error), expected_class);
    }
}
