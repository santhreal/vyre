//! Reference semantic execution contracts over graph-value identities and hostile policy.

use std::collections::BTreeMap;

use vyre_driver_reference::ReferenceSemanticExecutor;
use vyre_foundation::logical::LogicalProgramGraph;
use vyre_megakernel::{
    DeviceFacts, Digest, SearchBudget, SemanticExecutionError, SemanticExecutor,
};
use vyre_test_support::semantic_requests::{
    add_bindings, add_graph, assert_executes_retained_accumulate, request,
};

/// The reference target ranks with the open cost model and measures nothing.
const BUDGET: SearchBudget = SearchBudget::new(8, 64, 0, 0, 1_000);

#[test]
fn reference_executes_graph_values_with_registered_artifact_identities() {
    let graph = add_graph("add");
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new()).expect("logical graph");
    let lhs = 7_u32.to_le_bytes();
    let rhs = 11_u32.to_le_bytes();
    let inputs = add_bindings(&graph, lhs.as_slice(), rhs.as_slice());

    let request = request(&logical, inputs, DeviceFacts::unknown(), BUDGET, 1_000_000)
        .expect("complete graph bindings form a valid semantic request");
    let output = ReferenceSemanticExecutor
        .execute(&request)
        .expect("reference semantic execution");

    assert_ne!(output.artifact, Digest([0; 32]));
    assert_ne!(output.payload, Digest([0; 32]));
    assert_ne!(output.artifact, output.payload);
    assert_eq!(output.outputs.len(), 1);
    assert_eq!(
        output
            .outputs
            .get(&graph.nodes()[0].outputs[0])
            .map(Vec::as_slice),
        Some(18_u32.to_le_bytes().as_slice())
    );
}

#[test]
fn reference_rejects_hostile_policy_and_graph_value_inputs() {
    let graph = add_graph("add");
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new()).expect("logical graph");
    let node = &graph.nodes()[0];
    let lhs = 1_u32.to_le_bytes();
    let rhs = 2_u32.to_le_bytes();
    let facts = DeviceFacts::unknown();

    let missing = BTreeMap::from([(node.inputs[0].value, lhs.as_slice())]);
    let error = request(&logical, missing, facts, BUDGET, 1_000_000)
        .expect_err("missing graph value must fail before execution");
    assert!(matches!(error, SemanticExecutionError::InvalidRequest(_)));

    let mut extra = add_bindings(&graph, lhs.as_slice(), rhs.as_slice());
    extra.insert(vyre_foundation::ir::GraphValueId(999), lhs.as_slice());
    let error = request(&logical, extra, facts, BUDGET, 1_000_000)
        .expect_err("undeclared graph value must fail before execution");
    assert!(matches!(error, SemanticExecutionError::InvalidRequest(_)));
    assert!(error.to_string().contains("undeclared input graph value"));

    let complete = add_bindings(&graph, lhs.as_slice(), rhs.as_slice());
    let measurement_budget = SearchBudget::new(8, 64, 0, 1, 1_000);
    let request_with_measurement =
        request(&logical, complete, facts, measurement_budget, 1_000_000)
            .expect("complete graph bindings form a valid semantic request");
    let error = ReferenceSemanticExecutor
        .execute(&request_with_measurement)
        .expect_err("reference parity cannot fabricate device measurements");
    assert!(matches!(error, SemanticExecutionError::InvalidRequest(_)));

    let complete = add_bindings(&graph, lhs.as_slice(), rhs.as_slice());
    let request_without_budget = request(&logical, complete, facts, BUDGET, 0)
        .expect("complete graph bindings form a valid semantic request");
    let error = ReferenceSemanticExecutor
        .execute(&request_without_budget)
        .expect_err("zero artifact byte ceiling must fail");
    assert!(matches!(error, SemanticExecutionError::Compile(_)));
}

#[test]
fn reference_returns_retained_state_the_graph_carried() {
    assert_executes_retained_accumulate(
        &ReferenceSemanticExecutor,
        DeviceFacts::unknown(),
        "accumulate",
    );
}
