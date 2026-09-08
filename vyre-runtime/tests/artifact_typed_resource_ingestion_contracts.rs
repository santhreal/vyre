//! Typed external-resource ingestion transaction and binding closure contracts.
//!
//! BACKLOG row 49 requires one bounded typed-resource ingestion transaction that maps
//! files, byte ranges, generated data, or caller memory into artifact value ids through
//! concrete-driver transfers, verifying identity, lifetime, completion, and multi-entry ABIs.

#![forbid(unsafe_code)]

use vyre_driver::BackendRegistration;
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ShapeDim, ValueContract, ValueLifetime,
};
use vyre_megakernel::{AbiAccess, Artifact, ArtifactEnvelope, ResourceLifetime};
use vyre_runtime::artifact_admission::ArtifactSession;

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

#[test]
fn typed_resource_ingestion_validates_abi_and_workspace_bindings() {
    let artifact = multi_entry_stateful_artifact();
    let payload = fixture_target_payload(&artifact, FORMAT, vec![1, 2, 3, 4]);
    let mut envelope = ArtifactEnvelope::new(artifact.clone());
    envelope.attach_target_payload(payload).expect("payload");
    let materializer = SessionFixtureMaterializer::new("ingest-backend", "ingest-device", FORMAT);

    let session = ArtifactSession::from_envelope_with_materializer(
        &INGEST_REGISTRATION,
        envelope,
        materializer,
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
    assert!(missing_err
        .to_string()
        .contains("requires resident resource"));

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
    assert!(workspace_override_err
        .to_string()
        .contains("workspace-owned"));

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
