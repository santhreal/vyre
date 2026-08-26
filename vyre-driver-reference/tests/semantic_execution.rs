//! Reference semantic execution contracts over graph-value identities and hostile policy.

use std::collections::BTreeMap;

use vyre_driver_reference::ReferenceSemanticExecutor;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program, ProgramGraph};
use vyre_foundation::logical::LogicalProgramGraph;
use vyre_megakernel::{
    CompileObjective, DeviceFacts, Digest, ExternalFacts, SearchBudget, SemanticExecutionError,
    SemanticExecutionRequest, SemanticExecutor,
};

const BUDGET: SearchBudget = SearchBudget::new(8, 64, 0, 0, 1_000);

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

fn request_for<'a>(
    logical: &'a LogicalProgramGraph<'a>,
    inputs: BTreeMap<vyre_foundation::ir::GraphValueId, &'a [u8]>,
    budget: SearchBudget,
    max_artifact_bytes: u64,
) -> Result<SemanticExecutionRequest<'a>, SemanticExecutionError> {
    SemanticExecutionRequest::new(
        logical,
        inputs,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        CompileObjective::MinimizeLatency,
        budget,
        max_artifact_bytes,
    )
}

#[test]
fn reference_executes_graph_values_with_registered_artifact_identities() {
    let graph = ProgramGraph::from_program("add", add_program()).expect("valid graph");
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new()).expect("logical graph");
    let node = &graph.nodes()[0];
    let lhs = 7_u32.to_le_bytes();
    let rhs = 11_u32.to_le_bytes();
    let inputs = BTreeMap::from([
        (node.inputs[0].value, lhs.as_slice()),
        (node.inputs[1].value, rhs.as_slice()),
    ]);

    let request = request_for(&logical, inputs, BUDGET, 1_000_000)
        .expect("complete graph bindings form a valid semantic request");
    let output = ReferenceSemanticExecutor
        .execute(&request)
        .expect("reference semantic execution");

    assert_ne!(output.artifact, Digest([0; 32]));
    assert_ne!(output.payload, Digest([0; 32]));
    assert_ne!(output.artifact, output.payload);
    assert_eq!(output.outputs.len(), 1);
    assert_eq!(
        output.outputs.get(&node.outputs[0]).map(Vec::as_slice),
        Some(18_u32.to_le_bytes().as_slice())
    );
}

#[test]
fn reference_rejects_hostile_policy_and_graph_value_inputs() {
    let graph = ProgramGraph::from_program("add", add_program()).expect("valid graph");
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new()).expect("logical graph");
    let node = &graph.nodes()[0];
    let lhs = 1_u32.to_le_bytes();
    let rhs = 2_u32.to_le_bytes();

    let missing = BTreeMap::from([(node.inputs[0].value, lhs.as_slice())]);
    let error = request_for(&logical, missing, BUDGET, 1_000_000)
        .expect_err("missing graph value must fail before execution");
    assert!(matches!(error, SemanticExecutionError::InvalidRequest(_)));

    let extra = BTreeMap::from([
        (node.inputs[0].value, lhs.as_slice()),
        (node.inputs[1].value, rhs.as_slice()),
        (vyre_foundation::ir::GraphValueId(999), lhs.as_slice()),
    ]);
    let error = request_for(&logical, extra, BUDGET, 1_000_000)
        .expect_err("undeclared graph value must fail before execution");
    assert!(matches!(error, SemanticExecutionError::InvalidRequest(_)));
    assert!(error.to_string().contains("undeclared input graph value"));

    let complete = BTreeMap::from([
        (node.inputs[0].value, lhs.as_slice()),
        (node.inputs[1].value, rhs.as_slice()),
    ]);
    let measurement_budget = SearchBudget::new(8, 64, 0, 1, 1_000);
    let request = request_for(&logical, complete, measurement_budget, 1_000_000)
        .expect("complete graph bindings form a valid semantic request");
    let error = ReferenceSemanticExecutor
        .execute(&request)
        .expect_err("reference parity cannot fabricate device measurements");
    assert!(matches!(error, SemanticExecutionError::InvalidRequest(_)));

    let complete = BTreeMap::from([
        (node.inputs[0].value, lhs.as_slice()),
        (node.inputs[1].value, rhs.as_slice()),
    ]);
    let request = request_for(&logical, complete, BUDGET, 0)
        .expect("complete graph bindings form a valid semantic request");
    let error = ReferenceSemanticExecutor
        .execute(&request)
        .expect_err("zero artifact byte limit must fail");
    assert!(matches!(error, SemanticExecutionError::Compile(_)));
}
