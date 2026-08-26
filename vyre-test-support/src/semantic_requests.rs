//! Shared graph, bindings, and request builders for the semantic execution seam.
//!
//! Every backend proves the same contract over the same two-input add graph:
//! complete graph-value bindings admit a request, and a zero artifact byte
//! ceiling is refused before submission. Only the executor and the target facts
//! differ, so the graph, the bindings, and the request are stated once here.

use std::collections::BTreeMap;

use vyre_foundation::ir::{BufferDecl, DataType, Expr, GraphValueId, Node, Program, ProgramGraph};
use vyre_foundation::logical::LogicalProgramGraph;
use vyre_megakernel::{
    CompileObjective, CompileRequest, DeviceFacts, Digest, ExternalFacts, SearchBudget,
    SemanticExecutionError, SemanticExecutionOutput, SemanticExecutionPolicy,
    SemanticExecutionRequest, SemanticExecutor, ValidatedCompileRequest,
};

/// A policy targeting an unknown device under the latency objective.
///
/// Nothing about the target is granted, so a suite proving the seam rather than
/// a device states the budget and the artifact ceiling and nothing else. The
/// external facts digest varies per suite because it identifies the caller's
/// fact set, not the target.
#[must_use]
pub fn unknown_policy(
    external_digest: Digest,
    budget: SearchBudget,
    max_artifact_bytes: u64,
) -> SemanticExecutionPolicy {
    SemanticExecutionPolicy::new(
        ExternalFacts::new(external_digest, BTreeMap::new()),
        DeviceFacts::unknown(),
        CompileObjective::MinimizeLatency,
        budget,
        max_artifact_bytes,
    )
}

/// The budget a device-backed contract runs under: search allowed, measurement not.
pub const DEVICE_BUDGET: SearchBudget = SearchBudget::new(128, 128, 0, 0, 128);

/// The first summand every shared contract binds.
pub const LHS: u32 = 13;
/// The second summand every shared contract binds.
pub const RHS: u32 = 29;

/// Two `u32` inputs summed into one `u32` output.
pub fn add_program() -> Program {
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

/// The add program as a graph named `name`.
pub fn add_graph(name: &str) -> ProgramGraph {
    ProgramGraph::from_program(name, add_program()).expect("valid graph")
}

/// Bindings for both graph inputs of the add graph's only node.
pub fn add_bindings<'a>(
    graph: &ProgramGraph,
    lhs: &'a [u8],
    rhs: &'a [u8],
) -> BTreeMap<GraphValueId, &'a [u8]> {
    let node = &graph.nodes()[0];
    BTreeMap::from([(node.inputs[0].value, lhs), (node.inputs[1].value, rhs)])
}

/// A request over `logical` with target facts, budget, and artifact ceiling stated.
pub fn request<'a>(
    logical: &'a LogicalProgramGraph<'a>,
    inputs: BTreeMap<GraphValueId, &'a [u8]>,
    target_facts: DeviceFacts,
    budget: SearchBudget,
    max_artifact_bytes: u64,
) -> Result<SemanticExecutionRequest<'a>, SemanticExecutionError> {
    SemanticExecutionRequest::new(
        logical,
        inputs,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        target_facts,
        CompileObjective::MinimizeLatency,
        budget,
        max_artifact_bytes,
    )
}

/// A validated request over `graph` for an unknown device with no external values.
///
/// Nothing about the target is granted and the external fact set is empty, so a
/// suite proving a compiler or a target module states the graph, the budget, and
/// the artifact ceiling and nothing else.
///
/// # Panics
/// Panics when the request is rejected, which for a nonzero ceiling means the
/// graph itself is invalid and the fixture is what is wrong.
#[must_use]
pub fn validated_unknown_request(
    graph: ProgramGraph,
    budget: SearchBudget,
    max_artifact_bytes: u64,
) -> ValidatedCompileRequest {
    CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        budget,
        max_artifact_bytes,
    )
    .validate()
    .expect("a valid graph under a nonzero artifact ceiling is admitted")
}

/// An admitted output carrying `outputs` under distinct artifact and payload identities.
///
/// An executor double proves what the seam does with a result, not what a
/// device computed, so the two identities only have to differ from each other
/// and from zero.
#[must_use]
pub fn admitted_output(outputs: BTreeMap<GraphValueId, Vec<u8>>) -> SemanticExecutionOutput {
    SemanticExecutionOutput {
        artifact: Digest([1; 32]),
        payload: Digest([2; 32]),
        outputs,
    }
}

/// Asserts `executor` returns distinct admitted identities and the summed output.
pub fn assert_executes_add(
    executor: &impl SemanticExecutor,
    target_facts: DeviceFacts,
    name: &str,
) {
    let graph = add_graph(name);
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new()).expect("logical graph");
    let lhs = LHS.to_le_bytes();
    let rhs = RHS.to_le_bytes();
    let inputs = add_bindings(&graph, lhs.as_slice(), rhs.as_slice());

    let request = request(&logical, inputs, target_facts, DEVICE_BUDGET, 60_000)
        .expect("complete graph bindings form a valid semantic request");
    let output = executor.execute(&request).expect("semantic execution");

    assert_ne!(output.artifact, Digest([0; 32]));
    assert_ne!(output.payload, Digest([0; 32]));
    assert_eq!(
        output
            .outputs
            .get(&graph.nodes()[0].outputs[0])
            .map(Vec::as_slice),
        Some((LHS + RHS).to_le_bytes().as_slice())
    );
}

/// Asserts `executor` refuses a zero artifact byte ceiling before submission.
pub fn assert_refuses_zero_artifact_limit(
    executor: &impl SemanticExecutor,
    target_facts: DeviceFacts,
    name: &str,
) {
    let graph = add_graph(name);
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new()).expect("logical graph");
    let lhs = LHS.to_le_bytes();
    let rhs = RHS.to_le_bytes();
    let inputs = add_bindings(&graph, lhs.as_slice(), rhs.as_slice());

    let request = request(&logical, inputs, target_facts, DEVICE_BUDGET, 0)
        .expect("complete graph bindings form a valid semantic request");
    let error = executor
        .execute(&request)
        .expect_err("zero artifact byte ceiling must fail");

    assert!(matches!(error, SemanticExecutionError::Compile(_)));
}
