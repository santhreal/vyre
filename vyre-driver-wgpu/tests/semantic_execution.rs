//! WGPU artifact-backed semantic execution contracts.

#![cfg(all(test, feature = "device-tests"))]

mod harness;

use std::collections::BTreeMap;

use harness::acquire_live_backend;
use vyre_driver_wgpu::{registered_backend_id, WGPU_BACKEND_ID};
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program, ProgramGraph};
use vyre_foundation::logical::LogicalProgramGraph;
use vyre_megakernel::{
    CompileObjective, Digest, ExternalFacts, SearchBudget, SemanticExecutionError,
    SemanticExecutionRequest, SemanticExecutor,
};
use vyre_runtime::RegisteredSemanticExecutor;

fn add_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("lhs", 0, DataType::U32).with_count(1),
            BufferDecl::read("rhs", 1, DataType::U32).with_count(1),
            BufferDecl::output("sum", 2, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "sum",
            Expr::u32(0),
            Expr::add(
                Expr::load("lhs", Expr::u32(0)),
                Expr::load("rhs", Expr::u32(0)),
            ),
        )],
    )
}

#[test]
fn wgpu_executes_graph_values_through_registered_artifact() {
    let backend = acquire_live_backend();
    let graph = ProgramGraph::from_program("wgpu-add", add_program()).expect("valid graph");
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new()).expect("logical graph");
    let node = &graph.nodes()[0];
    let lhs = 13_u32.to_le_bytes();
    let rhs = 29_u32.to_le_bytes();
    let inputs = BTreeMap::from([
        (node.inputs[0].value, lhs.as_slice()),
        (node.inputs[1].value, rhs.as_slice()),
    ]);
    let _ = registered_backend_id();
    let registration =
        vyre_driver::backend_registration(WGPU_BACKEND_ID).expect("registered WGPU backend");
    let executor = RegisteredSemanticExecutor::new(registration);
    let request = SemanticExecutionRequest::new(
        &logical,
        inputs,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        backend.device_profile().compile_facts(),
        CompileObjective::MinimizeLatency,
        SearchBudget::new(128, 128, 0, 0, 128),
        60_000,
    )
    .expect("valid semantic execution request");

    let output = executor.execute(&request).expect("semantic WGPU execution");
    assert_ne!(output.artifact, Digest([0; 32]));
    assert_ne!(output.payload, Digest([0; 32]));
    assert_eq!(
        output.outputs.get(&node.outputs[0]).map(Vec::as_slice),
        Some(42_u32.to_le_bytes().as_slice())
    );
}

#[test]
fn wgpu_rejects_hostile_artifact_limit_before_submission() {
    let backend = acquire_live_backend();
    let graph = ProgramGraph::from_program("wgpu-add", add_program()).expect("valid graph");
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new()).expect("logical graph");
    let node = &graph.nodes()[0];
    let lhs = 1_u32.to_le_bytes();
    let rhs = 2_u32.to_le_bytes();
    let inputs = BTreeMap::from([
        (node.inputs[0].value, lhs.as_slice()),
        (node.inputs[1].value, rhs.as_slice()),
    ]);
    let _ = registered_backend_id();
    let registration =
        vyre_driver::backend_registration(WGPU_BACKEND_ID).expect("registered WGPU backend");
    let executor = RegisteredSemanticExecutor::new(registration);
    let request = SemanticExecutionRequest::new(
        &logical,
        inputs,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        backend.device_profile().compile_facts(),
        CompileObjective::MinimizeLatency,
        SearchBudget::new(128, 128, 0, 0, 128),
        0,
    )
    .expect("valid semantic execution request");

    let error = executor
        .execute(&request)
        .expect_err("zero artifact limit must fail");
    assert!(matches!(error, SemanticExecutionError::Compile(_)));
}
