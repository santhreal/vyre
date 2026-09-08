//! Typed external-resource ingestion transaction and binding closure contracts.
//!
//! BACKLOG row 49 requires one bounded typed-resource ingestion transaction that maps
//! files, byte ranges, generated data, or caller memory into artifact value ids through
//! concrete-driver transfers, verifying identity, lifetime, completion, and multi-entry ABIs.

#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use vyre_driver::materialize::{DeviceSpec, MaterializerDevice};
use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BackendRegistration, BoundResource,
    Device, ResidentOwner, Resource, VyreBackend,
};
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ShapeDim, ValueContract, ValueLifetime,
};
use vyre_megakernel::{
    AbiAccess, Artifact, ArtifactEnvelope, ArtifactValueId, ResourceLifetime, TargetEntryPoint,
    TargetPayload, TargetPayloadFormat, TargetProfile, TargetResourceAccess, TargetResourceBinding,
    TargetResourceMemory,
};
use vyre_runtime::artifact_admission::ArtifactSession;

use vyre_test_support::artifact_fixtures;
use vyre_test_support::fixture_instance::FixtureInstance;

const FORMAT: &str = "ingest.target";

struct IngestMaterializer {
    device: MaterializerDevice,
    owner: ResidentOwner,
    next: AtomicU64,
    allocated: Mutex<Vec<usize>>,
    freed: Mutex<Vec<Resource>>,
}

impl IngestMaterializer {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            device: MaterializerDevice::acquire(DeviceSpec {
                backend: "ingest-backend",
                device: "ingest-device".to_string(),
                format_extension: FORMAT,
                format_version: 1,
                profile: TargetProfile::new(FORMAT, 1, [64, 1, 1], 64, 1_024, 0)
                    .expect("profile"),
            })
            .expect("device"),
            owner: ResidentOwner::new("ingest-backend", "ingest-device"),
            next: AtomicU64::new(0),
            allocated: Mutex::new(Vec::new()),
            freed: Mutex::new(Vec::new()),
        })
    }
}

impl ArtifactMaterializer for IngestMaterializer {
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

fn ingest_backend_factory() -> Result<Box<dyn VyreBackend>, BackendError> {
    Err(BackendError::UnsupportedFeature {
        name: "raw Program backend".to_string(),
        backend: "ingest-artifact".to_string(),
    })
}

fn ingest_supported_ops() -> &'static std::collections::HashSet<vyre_foundation::ir::OpId> {
    static OPS: std::sync::LazyLock<std::collections::HashSet<vyre_foundation::ir::OpId>> =
        std::sync::LazyLock::new(std::collections::HashSet::new);
    &OPS
}

static INGEST_REGISTRATION: BackendRegistration = BackendRegistration {
    id: "ingest-artifact",
    target_id: vyre_foundation::operation::TargetId::expect_valid("ingest-artifact"),
    payload_format: None,
    reference_oracle: false,
    factory: ingest_backend_factory,
    supported_ops: ingest_supported_ops,
    semantic_operations: ingest_supported_ops,
    target_compiler: None,
    materializer: None,
};

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
                    BufferDecl::storage("state", 1, BufferAccess::ReadWrite, DataType::F32).with_count(16),
                    BufferDecl::output("intermediate", 2, DataType::F32).with_count(16),
                ],
                [16, 1, 1],
                vec![
                    Node::store(
                        "state",
                        Expr::gid_x(),
                        Expr::add(Expr::load("state", Expr::gid_x()), Expr::load("weights", Expr::gid_x())),
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

fn multi_entry_payload(artifact: &Artifact) -> TargetPayload {
    let entries = artifact
        .abi()
        .entries
        .iter()
        .map(|entry| {
            let recorded = artifact
                .geometry()
                .iter()
                .find(|record| record.node == entry.node)
                .expect("geometry");
            let bindings = entry
                .resources
                .iter()
                .enumerate()
                .map(|(slot, resource)| TargetResourceBinding {
                    resource: resource.value,
                    group: 0,
                    slot: slot as u32,
                    memory: TargetResourceMemory::Global,
                    access: match resource.access {
                        AbiAccess::ReadOnly => TargetResourceAccess::ReadOnly,
                        AbiAccess::WriteOnly => TargetResourceAccess::WriteOnly,
                        AbiAccess::ReadWrite => TargetResourceAccess::ReadWrite,
                        AbiAccess::WorkgroupLocal => TargetResourceAccess::WorkgroupLocal,
                    },
                })
                .collect();
            TargetEntryPoint {
                name: entry.name.clone(),
                node: entry.node,
                geometry: recorded.workgroup,
                resource_bindings: bindings,
            }
        })
        .collect();

    TargetPayload::new(
        artifact,
        TargetPayloadFormat::new(FORMAT, 1).expect("format"),
        TargetProfile::new(FORMAT, 1, [64, 1, 1], 64, 1_024, 0).expect("profile"),
        entries,
        vec![1, 2, 3, 4],
    )
    .expect("payload")
}

#[test]
fn typed_resource_ingestion_validates_abi_and_workspace_bindings() {
    let artifact = multi_entry_stateful_artifact();
    let payload = multi_entry_payload(&artifact);
    let envelope = ArtifactEnvelope::new(artifact.clone(), payload).expect("envelope");
    let materializer = IngestMaterializer::new();

    let session = ArtifactSession::from_envelope_with_materializer(
        &INGEST_REGISTRATION,
        envelope,
        materializer.clone(),
    )
    .expect("session");

    let workspace = session.allocate_workspace().expect("workspace allocation");

    // Allocate caller resources
    let weights_res = session.allocate_resident(64).expect("weights res");
    let state_res = session.allocate_resident(64).expect("state res");
    let final_out_res = session.allocate_resident(64).expect("final out res");

    // 1. Valid binding with workspace
    let bindings = session
        .resident_bindings_with_workspace(
            &workspace,
            [
                ("weights", &weights_res),
                ("state.0", &state_res),
                ("final_result", &final_out_res),
            ],
        )
        .expect("valid bindings with workspace");

    assert!(bindings.resources().len() >= 4);

    // 2. Reject missing resource
    let missing_err = session
        .resident_bindings_with_workspace(
            &workspace,
            [("weights", &weights_res), ("state.0", &state_res)],
        )
        .expect_err("must reject missing final_result");
    assert!(missing_err.to_string().contains("requires resident resource"));

    // 3. Reject caller supplying workspace-owned value
    let workspace_override_err = session
        .resident_bindings_with_workspace(
            &workspace,
            [
                ("weights", &weights_res),
                ("state.0", &state_res),
                ("final_result", &final_out_res),
                ("inter_val", &weights_res),
            ],
        )
        .expect_err("must reject caller overriding workspace value");
    assert!(workspace_override_err.to_string().contains("workspace-owned"));

    // Clean up
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
        match lt {
            ResourceLifetime::Constant => assert_eq!(lt, ResourceLifetime::Constant),
            ResourceLifetime::Invocation => assert_eq!(lt, ResourceLifetime::Invocation),
            ResourceLifetime::Retained => assert_eq!(lt, ResourceLifetime::Retained),
            ResourceLifetime::Output => assert_eq!(lt, ResourceLifetime::Output),
        }
    }

    let accesses = [
        AbiAccess::ReadOnly,
        AbiAccess::WriteOnly,
        AbiAccess::ReadWrite,
        AbiAccess::WorkgroupLocal,
    ];
    for acc in accesses {
        match acc {
            AbiAccess::ReadOnly => assert_eq!(acc, AbiAccess::ReadOnly),
            AbiAccess::WriteOnly => assert_eq!(acc, AbiAccess::WriteOnly),
            AbiAccess::ReadWrite => assert_eq!(acc, AbiAccess::ReadWrite),
            AbiAccess::WorkgroupLocal => assert_eq!(acc, AbiAccess::WorkgroupLocal),
        }
    }
}
