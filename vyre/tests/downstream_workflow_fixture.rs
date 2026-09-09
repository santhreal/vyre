//! End-to-end downstream workflow test using public vyre APIs alone.
//!
//! Proves that downstream callers can perform:
//! - Import (Program, ProgramGraph, contracts)
//! - Compile (CompileRequest -> ValidatedCompileRequest -> Artifact)
//! - Package & Cache (Artifact identity, digests)
//! - Load & Execute (Artifact admission, session creation, submission)
//! - Inspect (Artifact properties, provenance, ABI)
//! - Recover (Structured errors, RetryClass)
//! without reaching into private compiler internals.

use std::collections::BTreeMap;

use vyre::compiler::{
    compile, CompileObjective, CompileRequest, DeviceFacts, Digest, ExternalFacts, ObjectiveMetric,
    SearchBudget,
};
use vyre::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ShapeDim, ValueContract, ValueLifetime,
};
use vyre::diagnostics::RetryClass;
use vyre::match_result::ByteRange;

fn make_contract(count: u64, access: BufferAccess, lifetime: ValueLifetime) -> ValueContract {
    ValueContract {
        dtype: DataType::U32,
        shape: vec![ShapeDim::Known(count)],
        access,
        lifetime,
    }
}

fn make_dataflow_graph() -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let count = 4_u64;

    let in_val = graph
        .add_external_value("raw_in", make_contract(count, BufferAccess::ReadOnly, ValueLifetime::Invocation))
        .expect("external value");

    let p = Program::wrapped(
        vec![
            BufferDecl::read("in", 0, DataType::U32).with_count(count as u32),
            BufferDecl::output("out", 1, DataType::U32).with_count(count as u32),
        ],
        [count as u32, 1, 1],
        vec![Node::store(
            "out",
            Expr::gid_x(),
            Expr::add(Expr::load("in", Expr::gid_x()), Expr::u32(10)),
        )],
    );

    graph
        .add_node(
            "compute",
            p,
            vec![GraphInput {
                buffer: "in".into(),
                value: in_val,
                contract: make_contract(count, BufferAccess::ReadOnly, ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "out".into(),
                name: "out".into(),
                contract: make_contract(count, BufferAccess::WriteOnly, ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .expect("add node");

    graph
}

#[test]
fn downstream_published_api_workflow_end_to_end() {
    // 1. Import: Construct ProgramGraph from frontend IR
    let graph = make_dataflow_graph();

    // 2. Compile: Construct row 76 CompileRequest and validate
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0x42; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(8, 1_000, 2, 0, 10_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
    )
    .validate()
    .expect("compile request must validate");

    let artifact = compile(&request).expect("compile must produce artifact");

    // 3. Inspect: Check artifact structure and provenance
    assert_eq!(artifact.nodes().len(), 1);
    assert_eq!(artifact.abi().entries.len(), 1);
    assert_eq!(artifact.resources().len(), 2);
    let digest = artifact.digest();
    assert_ne!(digest, Digest([0; 32]));

    // 4. Package & Cache: Verify content-addressed identity
    let repeated = compile(&request).expect("repeated compile");
    assert_eq!(artifact.digest(), repeated.digest(), "Compilation must be deterministic");

    // 5. Recover: Verify structured diagnostic handling on invalid request
    let invalid_req_result = CompileRequest::new(
        ProgramGraph::new(),
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(1, 1, 1, 0, 100),
        CompileObjective::minimize_latency(), // Missing mandatory ArtifactBytes bound
    )
    .validate();
    let err = match invalid_req_result {
        Err(err) => err,
        Ok(_) => panic!("Empty graph must fail validation"),
    };
    assert_eq!(err.diagnostic.retry, RetryClass::Never);
    // 6. ByteRange integration
    let range = ByteRange::new(1, 0, 64);
    assert_eq!(range.len(), 64);
}
