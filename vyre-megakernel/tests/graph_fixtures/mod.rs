//! The graph shapes the megakernel suites compile.
//!
//! Three suites ask the planner about the same two-node graph: a producer and a
//! consumer joined by one invocation-scoped value. Each had written its own copy
//! of the builder, identical but for the program each node runs. One owner means
//! a suite that changes the shape changes it for every suite that reads it, which
//! is the point of asking the same planner the same question.

// Every test binary compiles this module on its own, so a fixture a given suite
// does not ask for is unused in that binary.
#![allow(dead_code)]

use vyre_foundation::ir::{
    BufferAccess, DataType, GraphInput, GraphOutput, Program, ProgramGraph, ShapeDim,
    ValueContract, ValueLifetime,
};
use vyre_test_support::pass_programs::copy_program;

/// A `u32` value of symbolic length that lives for one invocation.
pub(crate) fn invocation_contract() -> ValueContract {
    ValueContract {
        dtype: DataType::U32,
        shape: vec![ShapeDim::Symbol("items".into())],
        access: BufferAccess::ReadWrite,
        lifetime: ValueLifetime::Invocation,
    }
}

/// A producer and a consumer joined by one invocation-scoped value.
///
/// `producer` reads `input` and writes `intermediate`; `consumer` reads
/// `intermediate` and writes `output`. The caller supplies both programs, which
/// is the only thing the suites disagree about.
pub(crate) fn producer_consumer_pair(producer: Program, consumer: Program) -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let input = graph
        .add_external_value("input", invocation_contract())
        .unwrap();
    let (_, intermediate) = graph
        .add_node(
            "producer",
            producer,
            vec![GraphInput {
                buffer: "input".into(),
                value: input,
                contract: invocation_contract(),
            }],
            vec![GraphOutput {
                buffer: "intermediate".into(),
                name: "intermediate".into(),
                contract: invocation_contract(),
                retained_successor_of: None,
            }],
        )
        .unwrap();
    graph
        .add_node(
            "consumer",
            consumer,
            vec![GraphInput {
                buffer: "intermediate".into(),
                value: intermediate[0],
                contract: invocation_contract(),
            }],
            vec![GraphOutput {
                buffer: "output".into(),
                name: "output".into(),
                contract: ValueContract {
                    lifetime: ValueLifetime::Output,
                    ..invocation_contract()
                },
                retained_successor_of: None,
            }],
        )
        .unwrap();
    graph
}
/// Two independent arms joined into one graph.
pub(crate) fn two_arm_graph(arm_a: Program, arm_b: Program) -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let in_a = graph
        .add_external_value("in_a", invocation_contract())
        .unwrap();
    let in_b = graph
        .add_external_value("in_b", invocation_contract())
        .unwrap();
    graph
        .add_node(
            "arm_a",
            arm_a,
            vec![GraphInput {
                buffer: "in_a".into(),
                value: in_a,
                contract: invocation_contract(),
            }],
            vec![GraphOutput {
                buffer: "out_a".into(),
                name: "out_a".into(),
                contract: ValueContract {
                    lifetime: ValueLifetime::Output,
                    ..invocation_contract()
                },
                retained_successor_of: None,
            }],
        )
        .unwrap();
    graph
        .add_node(
            "arm_b",
            arm_b,
            vec![GraphInput {
                buffer: "in_b".into(),
                value: in_b,
                contract: invocation_contract(),
            }],
            vec![GraphOutput {
                buffer: "out_b".into(),
                name: "out_b".into(),
                contract: ValueContract {
                    lifetime: ValueLifetime::Output,
                    ..invocation_contract()
                },
                retained_successor_of: None,
            }],
        )
        .unwrap();
    graph
}

/// Independent two-arm graph: two parallel copy operations reading separate inputs and writing separate outputs.
pub(crate) fn independent_two_arm_graph() -> ProgramGraph {
    two_arm_graph(copy_program("in_a", "out_a"), copy_program("in_b", "out_b"))
}

/// RAW conflicting two-arm graph: arm B reads the intermediate output of arm A within the same graph.
pub(crate) fn raw_conflict_two_arm_graph() -> ProgramGraph {
    producer_consumer_pair(
        copy_program("input", "intermediate"),
        copy_program("intermediate", "output"),
    )
}

/// Asymmetric join graph: Node 0 feeds Node 1 and Node 2.
pub(crate) fn asymmetric_join_graph() -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let in0 = graph
        .add_external_value("in0", invocation_contract())
        .unwrap();
    let (_, out0) = graph
        .add_node(
            "n0",
            copy_program("in0", "out0"),
            vec![GraphInput {
                buffer: "in0".into(),
                value: in0,
                contract: invocation_contract(),
            }],
            vec![GraphOutput {
                buffer: "out0".into(),
                name: "out0".into(),
                contract: invocation_contract(),
                retained_successor_of: None,
            }],
        )
        .unwrap();
    graph
        .add_node(
            "n1",
            copy_program("in1", "out1"),
            vec![GraphInput {
                buffer: "in1".into(),
                value: out0[0],
                contract: invocation_contract(),
            }],
            vec![GraphOutput {
                buffer: "out1".into(),
                name: "out1".into(),
                contract: invocation_contract(),
                retained_successor_of: None,
            }],
        )
        .unwrap();
    graph
        .add_node(
            "n2",
            copy_program("in2", "out2"),
            vec![GraphInput {
                buffer: "in2".into(),
                value: out0[0],
                contract: invocation_contract(),
            }],
            vec![GraphOutput {
                buffer: "out2".into(),
                name: "out2".into(),
                contract: ValueContract {
                    lifetime: ValueLifetime::Output,
                    ..invocation_contract()
                },
                retained_successor_of: None,
            }],
        )
        .unwrap();
    graph
}
