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
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ShapeDim, ValueContract, ValueLifetime,
};

/// A `u32` value of symbolic length that lives for one invocation.
pub fn invocation_contract() -> ValueContract {
    ValueContract {
        dtype: DataType::U32,
        shape: vec![ShapeDim::Symbol("items".into())],
        access: BufferAccess::ReadWrite,
        lifetime: ValueLifetime::Invocation,
    }
}

/// A program that copies one element from `input` to `output`.
pub fn copy_program(input: &str, output: &str) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadWrite, DataType::U32),
            BufferDecl::storage(output, 1, BufferAccess::ReadWrite, DataType::U32),
        ],
        [32, 1, 1],
        vec![Node::store(
            output,
            Expr::u32(0),
            Expr::load(input, Expr::u32(0)),
        )],
    )
}

/// A producer and a consumer joined by one invocation-scoped value.
///
/// `producer` reads `input` and writes `intermediate`; `consumer` reads
/// `intermediate` and writes `output`. The caller supplies both programs, which
/// is the only thing the suites disagree about.
pub fn producer_consumer_pair(producer: Program, consumer: Program) -> ProgramGraph {
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
