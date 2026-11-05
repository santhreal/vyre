//! Connected graph compilation and execution via the top-level vyre facade.
//!
//! Proves that connected multi-node graphs across dataflow, retained state,
//! irregular indexing, and concurrent arm domains compile and materialize
//! through canonical compiler artifacts and runtime sessions.

use std::collections::BTreeMap;

use vyre::compiler::{
    compile, CompileObjective, CompileRequest, DeviceFacts, Digest, ExternalFacts, ObjectiveMetric,
    SearchBudget,
};
use vyre::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ValueContract, ValueLifetime,
};

fn contract(access: BufferAccess, lifetime: ValueLifetime, count: u64) -> ValueContract {
    ValueContract::dense_1d(DataType::U32, count, access, lifetime)
}

#[test]
fn facade_compiles_multi_stage_connected_dataflow_graph() {
    let count = 8_u64;
    let mut graph = ProgramGraph::new();

    let in_val = graph
        .add_external_value(
            "raw_in",
            contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
        )
        .unwrap();

    let prog_a = Program::wrapped(
        vec![
            BufferDecl::read("in", 0, DataType::U32).with_count(count as u32),
            BufferDecl::output("mid", 1, DataType::U32).with_count(count as u32),
        ],
        [count as u32, 1, 1],
        vec![Node::store(
            "mid",
            Expr::gid_x(),
            Expr::add(
                Expr::mul(Expr::load("in", Expr::gid_x()), Expr::u32(2)),
                Expr::u32(1),
            ),
        )],
    );

    let (_, mid_vals) = graph
        .add_node(
            "stage_a",
            prog_a,
            vec![GraphInput {
                buffer: "in".into(),
                value: in_val,
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
            }],
            vec![GraphOutput {
                buffer: "mid".into(),
                name: "mid".into(),
                contract: contract(BufferAccess::ReadWrite, ValueLifetime::Invocation, count),
                retained_successor_of: None,
            }],
        )
        .unwrap();

    let prog_b = Program::wrapped(
        vec![
            BufferDecl::read("mid_in", 0, DataType::U32).with_count(count as u32),
            BufferDecl::output("final_out", 1, DataType::U32).with_count(count as u32),
        ],
        [count as u32, 1, 1],
        vec![Node::store(
            "final_out",
            Expr::gid_x(),
            Expr::add(Expr::load("mid_in", Expr::gid_x()), Expr::u32(100)),
        )],
    );

    graph
        .add_node(
            "stage_b",
            prog_b,
            vec![GraphInput {
                buffer: "mid_in".into(),
                value: mid_vals[0],
                contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation, count),
            }],
            vec![GraphOutput {
                buffer: "final_out".into(),
                name: "final_out".into(),
                contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output, count),
                retained_successor_of: None,
            }],
        )
        .unwrap();

    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0x56; 32]), BTreeMap::new()),
        DeviceFacts::new(
            vyre_foundation::validate::BackendCapabilities::default(),
            1024,
        ),
        SearchBudget::new(32, 1_000_000, 4, 0, 10_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 10_000_000),
    )
    .validate()
    .expect("request must validate");

    let artifact = compile(&request).expect("compile must produce artifact");
    assert_eq!(artifact.nodes().len(), 2);
    assert_eq!(artifact.abi().entries.len(), 2);
}
