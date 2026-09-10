//! WGPU connected-graph production route contracts.
//!
//! Representative connected graphs from unrelated domains must execute
//! through `CompileRequest -> ArtifactEnvelope -> TargetPayload -> ArtifactInstance -> BindingSet -> Completion`
//! and match independent semantics under declared tolerances.

#![cfg(all(test, feature = "device-tests"))]

use std::collections::BTreeMap;

use vyre_driver::BoundResource;
use vyre_driver_wgpu::{registered_backend_id, WGPU_BACKEND_ID};
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ValueContract, ValueLifetime,
};
use vyre_megakernel::{
    attach_target, compile, CompileObjective, CompileRequest, Digest, ExternalFacts,
    ObjectiveMetric, SearchBudget,
};
use vyre_runtime::artifact_admission::ArtifactSession;

fn contract(access: BufferAccess, lifetime: ValueLifetime, count: u64) -> ValueContract {
    ValueContract::dense_1d(DataType::U32, count, access, lifetime)
}

fn budget() -> SearchBudget {
    SearchBudget::new(64, 1_000_000, 4, 0, 10_000_000)
}

fn facts() -> ExternalFacts {
    ExternalFacts::new(Digest([0x56; 32]), BTreeMap::new())
}

#[test]
fn wgpu_executes_pure_dataflow_connected_graph() {
    let _ = registered_backend_id();
    let registration =
        vyre_driver::backend_registration(WGPU_BACKEND_ID).expect("registered WGPU backend");
    let device = registration
        .acquire()
        .expect("WGPU device acquisition must succeed");
    assert!(!device.device_lost(), "WGPU device must not be lost");
    let device_facts = device.device_profile().compile_facts();

    // Pure Dataflow: 3-stage connected pipeline
    // Node 0: Y = 3 * X + 5
    // Node 1: S = sum(Y)
    // Node 2: Z = Y + S
    let count = 4_u64;
    let mut graph = ProgramGraph::new();

    let in_x = graph
        .add_external_value(
            "in_x",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
        )
        .unwrap();

    let node0_prog = Program::wrapped(
        vec![
            BufferDecl::read("x", 0, DataType::U32).with_count(count as u32),
            BufferDecl::written("y", 1, BufferAccess::WriteOnly, DataType::U32)
                .with_count(count as u32),
        ],
        [count as u32, 1, 1],
        vec![Node::store(
            "y",
            Expr::gid_x(),
            Expr::add(
                Expr::mul(Expr::load("x", Expr::gid_x()), Expr::u32(3)),
                Expr::u32(5),
            ),
        )],
    );

    let (_, val_y) = graph
        .add_node(
            "scale_node",
            node0_prog,
            vec![GraphInput {
                buffer: "x".into(),
                value: in_x,
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
            }],
            vec![GraphOutput {
                buffer: "y".into(),
                name: "y".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Invocation, count),
                retained_successor_of: None,
            }],
        )
        .unwrap();

    let node1_prog = Program::wrapped(
        vec![
            BufferDecl::read("y_in", 0, DataType::U32).with_count(count as u32),
            BufferDecl::written("sum_out", 1, BufferAccess::WriteOnly, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "sum_out",
            Expr::u32(0),
            Expr::add(
                Expr::add(
                    Expr::load("y_in", Expr::u32(0)),
                    Expr::load("y_in", Expr::u32(1)),
                ),
                Expr::add(
                    Expr::load("y_in", Expr::u32(2)),
                    Expr::load("y_in", Expr::u32(3)),
                ),
            ),
        )],
    );

    let (_, val_s) = graph
        .add_node(
            "sum_node",
            node1_prog,
            vec![GraphInput {
                buffer: "y_in".into(),
                value: val_y[0],
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
            }],
            vec![GraphOutput {
                buffer: "sum_out".into(),
                name: "sum_out".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Invocation, 1),
                retained_successor_of: None,
            }],
        )
        .unwrap();

    let node2_prog = Program::wrapped(
        vec![
            BufferDecl::read("y_norm_in", 0, DataType::U32).with_count(count as u32),
            BufferDecl::read("s_in", 1, DataType::U32).with_count(1),
            BufferDecl::output("z_out", 2, DataType::U32).with_count(count as u32),
        ],
        [count as u32, 1, 1],
        vec![Node::store(
            "z_out",
            Expr::gid_x(),
            Expr::add(
                Expr::load("y_norm_in", Expr::gid_x()),
                Expr::load("s_in", Expr::u32(0)),
            ),
        )],
    );

    let _ = graph
        .add_node(
            "norm_node",
            node2_prog,
            vec![
                GraphInput {
                    buffer: "y_norm_in".into(),
                    value: val_y[0],
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
                },
                GraphInput {
                    buffer: "s_in".into(),
                    value: val_s[0],
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, 1),
                },
            ],
            vec![GraphOutput {
                buffer: "z_out".into(),
                name: "z_out".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output, count),
                retained_successor_of: None,
            }],
        )
        .unwrap();

    let request = CompileRequest::new(
        graph,
        facts(),
        device_facts,
        budget(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
    )
    .validate()
    .expect("compile request must validate");

    let compiler = registration.target_compiler().expect("target compiler");
    let artifact = compile(&request).expect("compile must succeed");
    assert_eq!(artifact.nodes().len(), 3);
    assert_ne!(artifact.digest(), Digest([0; 32]));

    let envelope = attach_target(artifact, compiler.as_ref()).expect("attach target");
    let payloads = envelope.target_payloads();
    assert_eq!(payloads.len(), 1, "attach_target must attach one payload");
    assert_ne!(payloads[0].digest(), Digest([0; 32]));

    let session = ArtifactSession::from_envelope(registration, envelope).expect("materialization");
    let mut bindings = session.bindings().expect("binding set");

    let input_bytes = [1_u32, 2, 3, 4]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let output_bytes = vec![0u8; 16];

    if let Ok(in_val) = session.resource("in_x") {
        bindings.insert(in_val, BoundResource::Host(input_bytes));
    }
    if let Ok(out_val) = session.resource("z_out") {
        bindings.insert(out_val, BoundResource::Host(output_bytes));
    }

    let completion = session
        .submit_and_wait(bindings)
        .expect("execution must complete");

    assert_ne!(completion.artifact, Digest([0; 32]));
    assert_eq!(completion.artifact, session.artifact().unwrap());
    assert_ne!(session.payload().unwrap(), Digest([0; 32]));

    let expected_z = [58_u32, 61, 64, 67]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();

    let out_res = session.resource("z_out").expect("z_out resource");
    let actual_z = completion
        .outputs
        .get(&out_res)
        .expect("z_out output bytes");
    assert_eq!(
        actual_z, &expected_z,
        "output must match independent oracle"
    );
}
