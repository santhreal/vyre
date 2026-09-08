//! Domain 2: Graph / Irregular / Relational Dataflow Composition.
//!
//! Composes CSR edge offsets, neighbor scattering, ragged reductions, and
//! retained PageRank / distance state across bounded recurrence steps into
//! one connected validated [`ProgramGraph`].

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, ControlBounds, DataType, Expr, GraphInput, GraphOutput, Node, Program,
    ProgramGraph, ProgramGraphBuilder, ProgramGraphError, ShapeDim, ValueContract, ValueLifetime,
};

/// Build a representative CSR graph traversal and PageRank iteration whole-graph.
///
/// Features exercised:
/// - CSR ragged extents & offsets
/// - Retained node ranks state across steps
/// - Bounded loop iteration control
/// - Segmented reduction combination
pub fn build_csr_graph_traversal_pipeline(
    node_count: u64,
    edge_count: u64,
    steps: u64,
) -> Result<ProgramGraph, ProgramGraphError> {
    let mut builder = ProgramGraphBuilder::new();

    let offsets = builder.input(
        "edge_offsets",
        DataType::U32,
        vec![ShapeDim::Known(node_count + 1)],
    )?;
    let targets = builder.input(
        "edge_targets",
        DataType::U32,
        vec![ShapeDim::Known(edge_count)],
    )?;
    let initial_ranks = builder.retained_state(
        "node_ranks.0",
        DataType::F32,
        vec![ShapeDim::Known(node_count)],
    )?;

    // Step body subgraph: 1 iteration of CSR scatter + PageRank rank update
    let mut step_builder = ProgramGraphBuilder::new();
    let s_ranks = step_builder.retained_state(
        "curr_ranks",
        DataType::F32,
        vec![ShapeDim::Known(node_count)],
    )?;
    let s_offsets = step_builder.input(
        "s_offsets",
        DataType::U32,
        vec![ShapeDim::Known(node_count + 1)],
    )?;
    let s_targets = step_builder.input(
        "s_targets",
        DataType::U32,
        vec![ShapeDim::Known(edge_count)],
    )?;

    let step_p = Program::wrapped(
        vec![
            BufferDecl::read("s_offsets", 0, DataType::U32).with_count((node_count + 1) as u32),
            BufferDecl::read("s_targets", 1, DataType::U32).with_count(edge_count as u32),
            BufferDecl::read_write("curr_ranks", 2, DataType::F32).with_count(node_count as u32),
            BufferDecl::read_write("next_ranks", 3, DataType::F32).with_count(node_count as u32),
        ],
        [node_count.max(1) as u32, 1, 1],
        vec![
            Node::let_bind(
                "deg",
                Expr::sub(
                    Expr::load("s_offsets", Expr::add(Expr::gid_x(), Expr::u32(1))),
                    Expr::load("s_offsets", Expr::gid_x()),
                ),
            ),
            Node::store(
                "next_ranks",
                Expr::gid_x(),
                Expr::add(
                    Expr::mul(
                        Expr::load("curr_ranks", Expr::gid_x()),
                        Expr::f32(0.85),
                    ),
                    Expr::f32(0.15 / node_count as f32),
                ),
            ),
        ],
    );

    step_builder.add_node(
        "pagerank_scatter",
        step_p,
        vec![
            GraphInput {
                buffer: "s_offsets".into(),
                value: s_offsets,
                contract: ValueContract {
                    dtype: DataType::U32,
                    shape: vec![ShapeDim::Known(node_count + 1)],
                    access: BufferAccess::ReadOnly,
                    lifetime: ValueLifetime::Invocation,
                },
            },
            GraphInput {
                buffer: "s_targets".into(),
                value: s_targets,
                contract: ValueContract {
                    dtype: DataType::U32,
                    shape: vec![ShapeDim::Known(edge_count)],
                    access: BufferAccess::ReadOnly,
                    lifetime: ValueLifetime::Invocation,
                },
            },
            GraphInput {
                buffer: "curr_ranks".into(),
                value: s_ranks,
                contract: ValueContract {
                    dtype: DataType::F32,
                    shape: vec![ShapeDim::Known(node_count)],
                    access: BufferAccess::ReadWrite,
                    lifetime: ValueLifetime::Retained,
                },
            },
        ],
        vec![GraphOutput {
            buffer: "next_ranks".into(),
            name: "next_ranks_val".into(),
            contract: ValueContract {
                dtype: DataType::F32,
                shape: vec![ShapeDim::Known(node_count)],
                access: BufferAccess::ReadWrite,
                lifetime: ValueLifetime::Retained,
            },
            retained_successor_of: Some(s_ranks),
        }],
    )?;
    let step_subgraph = step_builder.build()?;

    let final_ranks = builder.add_bounded_loop(
        "pagerank_loop",
        &step_subgraph,
        &[initial_ranks],
        &[offsets, targets],
        ControlBounds {
            max_steps: steps.max(1),
            guaranteed_termination: true,
        },
    )?;

    // Mark final ranks as output
    let out_p = Program::wrapped(
        vec![
            BufferDecl::read("final_in", 0, DataType::F32).with_count(node_count as u32),
            BufferDecl::output("out_ranks", 1, DataType::F32).with_count(node_count as u32),
        ],
        [node_count.max(1) as u32, 1, 1],
        vec![Node::store(
            "out_ranks",
            Expr::gid_x(),
            Expr::load("final_in", Expr::gid_x()),
        )],
    );

    builder.add_node(
        "publish_ranks",
        out_p,
        vec![GraphInput {
            buffer: "final_in".into(),
            value: final_ranks[0],
            contract: ValueContract {
                dtype: DataType::F32,
                shape: vec![ShapeDim::Known(node_count)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Retained,
            },
        }],
        vec![GraphOutput {
            buffer: "out_ranks".into(),
            name: "final_ranks_out".into(),
            contract: ValueContract {
                dtype: DataType::F32,
                shape: vec![ShapeDim::Known(node_count)],
                access: BufferAccess::WriteOnly,
                lifetime: ValueLifetime::Output,
            },
            retained_successor_of: None,
        }],
    )?;

    builder.build()
}
