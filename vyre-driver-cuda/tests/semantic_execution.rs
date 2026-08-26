//! CUDA artifact-backed semantic execution contracts.

#![cfg(all(test, feature = "device-tests"))]

use std::collections::BTreeMap;

use vyre_driver_cuda::{registered_backend_id, CUDA_BACKEND_ID};
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
fn cuda_executes_graph_values_through_registered_artifact() {
    let _ = registered_backend_id();
    let registration =
        vyre_driver::backend_registration(CUDA_BACKEND_ID).expect("registered CUDA backend");
    let device = registration.acquire().expect("live CUDA backend");
    let graph = ProgramGraph::from_program("cuda-add", add_program()).expect("valid graph");
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new()).expect("logical graph");
    let node = &graph.nodes()[0];
    let lhs = 17_u32.to_le_bytes();
    let rhs = 25_u32.to_le_bytes();
    let inputs = BTreeMap::from([
        (node.inputs[0].value, lhs.as_slice()),
        (node.inputs[1].value, rhs.as_slice()),
    ]);
    let executor = RegisteredSemanticExecutor::new(registration);
    let request = SemanticExecutionRequest::new(
        &logical,
        inputs,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        device.device_profile().compile_facts(),
        CompileObjective::MinimizeLatency,
        SearchBudget::new(128, 128, 0, 0, 128),
        60_000,
    )
    .expect("valid semantic execution request");

    let output = executor.execute(&request).expect("semantic CUDA execution");
    assert_ne!(output.artifact, Digest([0; 32]));
    assert_ne!(output.payload, Digest([0; 32]));
    assert_eq!(
        output.outputs.get(&node.outputs[0]).map(Vec::as_slice),
        Some(42_u32.to_le_bytes().as_slice())
    );
}

#[test]
fn cuda_rejects_hostile_artifact_limit_before_submission() {
    let _ = registered_backend_id();
    let registration =
        vyre_driver::backend_registration(CUDA_BACKEND_ID).expect("registered CUDA backend");
    let device = registration.acquire().expect("live CUDA backend");
    let graph = ProgramGraph::from_program("cuda-add", add_program()).expect("valid graph");
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new()).expect("logical graph");
    let node = &graph.nodes()[0];
    let lhs = 1_u32.to_le_bytes();
    let rhs = 2_u32.to_le_bytes();
    let inputs = BTreeMap::from([
        (node.inputs[0].value, lhs.as_slice()),
        (node.inputs[1].value, rhs.as_slice()),
    ]);
    let executor = RegisteredSemanticExecutor::new(registration);
    let request = SemanticExecutionRequest::new(
        &logical,
        inputs,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        device.device_profile().compile_facts(),
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
