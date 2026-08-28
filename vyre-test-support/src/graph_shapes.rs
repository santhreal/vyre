//! Graph shapes whose logical facts decide how a region may be distributed.
//!
//! A placement reads three facts off a region: whether a shard may read a point
//! another shard writes, whether the region updates a location it computes, and
//! whether one region consumes what the region before it produced. Each fact has
//! one graph shape that states it, and both the foundation logical tests and the
//! workspace artifact fixtures need the same three shapes. They are built here
//! so a change to one shape reaches every case that reads its facts.

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ValueLifetime,
};

use crate::graph_values::u32_vector;

/// One node that reads the caller-supplied `u32` value of `count` elements it
/// also writes.
///
/// A region reading a value it writes may read a point another shard holds, so
/// its axes address a spatial domain rather than independent elements.
#[must_use]
pub fn in_place_input_graph(count: u32, workgroup: [u32; 3]) -> ProgramGraph {
    let state = u32_vector(count, BufferAccess::ReadWrite, ValueLifetime::Invocation);
    let mut graph = ProgramGraph::new();
    let value = graph
        .add_external_value("state", state.clone())
        .expect("fixture external value must be valid");
    graph
        .add_node(
            "relax",
            Program::wrapped(
                vec![BufferDecl::read_write("state", 0, DataType::U32).with_count(count)],
                workgroup,
                vec![Node::store(
                    "state",
                    Expr::gid_x(),
                    Expr::add(Expr::load("state", Expr::gid_x()), Expr::u32(1)),
                )],
            ),
            vec![GraphInput {
                buffer: "state".into(),
                value,
                contract: state,
            }],
            Vec::new(),
        )
        .expect("fixture in-place node must be valid");
    graph
}

/// One node that atomically updates a caller-visible `u32` value of `count`
/// elements.
///
/// The value is written for the invocation rather than retained across
/// submissions, so the region is atomic without being ordered: a shard holds
/// contributions for points another shard owns.
#[must_use]
pub fn atomic_output_graph(count: u32, workgroup: [u32; 3]) -> ProgramGraph {
    let counters = u32_vector(count, BufferAccess::ReadWrite, ValueLifetime::Invocation);
    let mut graph = ProgramGraph::new();
    graph
        .add_node(
            "scatter",
            Program::wrapped(
                vec![BufferDecl::read_write("out", 0, DataType::U32).with_count(count)],
                workgroup,
                vec![Node::let_bind(
                    "prior",
                    Expr::atomic_add("out", Expr::gid_x(), Expr::u32(1)),
                )],
            ),
            Vec::new(),
            vec![GraphOutput {
                buffer: "out".into(),
                name: "out".into(),
                contract: counters,
                retained_successor_of: None,
            }],
        )
        .expect("fixture atomic node must be valid");
    graph
}

/// Two nodes over `count` elements, the second consuming what the first wrote.
///
/// At one element no axis has a bound to cut, so the only placement that uses
/// more than one device is the one that runs consecutive regions on consecutive
/// devices.
#[must_use]
pub fn chained_graph(count: u32, workgroup: [u32; 3]) -> ProgramGraph {
    let carried = u32_vector(count, BufferAccess::ReadWrite, ValueLifetime::Invocation);
    let mut graph = ProgramGraph::new();
    let (_, produced) = graph
        .add_node(
            "producer",
            Program::wrapped(
                vec![BufferDecl::read_write("mid", 0, DataType::U32).with_count(count)],
                workgroup,
                vec![Node::store("mid", Expr::gid_x(), Expr::u32(1))],
            ),
            Vec::new(),
            vec![GraphOutput {
                buffer: "mid".into(),
                name: "mid".into(),
                contract: carried.clone(),
                retained_successor_of: None,
            }],
        )
        .expect("fixture producer node must be valid");
    graph
        .add_node(
            "consumer",
            Program::wrapped(
                vec![
                    BufferDecl::read_write("mid", 0, DataType::U32).with_count(count),
                    BufferDecl::output("result", 1, DataType::U32).with_count(count),
                ],
                workgroup,
                vec![Node::store(
                    "result",
                    Expr::gid_x(),
                    Expr::load("mid", Expr::gid_x()),
                )],
            ),
            vec![GraphInput {
                buffer: "mid".into(),
                value: produced[0],
                contract: carried,
            }],
            vec![GraphOutput {
                buffer: "result".into(),
                name: "result".into(),
                contract: u32_vector(count, BufferAccess::WriteOnly, ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .expect("fixture consumer node must be valid");
    graph
}
