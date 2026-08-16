//! The emitted program: one invocation checks every motif edge.

use vyre_foundation::composition::{trap_program, wrap_anonymous_region};
use vyre_foundation::ir::{BufferAccess, DataType, Expr, Node, Program};

use crate::graph::program_graph::{
    word_buffer, ProgramGraphShape, NAME_EDGE_KIND_MASK, NAME_EDGE_OFFSETS, NAME_EDGE_TARGETS,
};

use super::pattern::MotifEdge;
use super::{MOTIF_HITS_BUFFER, MOTIF_WITNESS_OUT_BUFFER, MOTIF_WORKGROUP_SIZE, OP_ID};

/// Build a Program: one invocation checks every motif edge, records
/// participating endpoint bits only for matched edges, and publishes
/// the participant union if the whole motif matched.
///
/// Invalid motif sizes lower to an explicit trap program. Prior code
/// silently truncated `edges.len() as u32`; this path keeps the failure
/// executable without crashing the host process.
#[must_use]
pub fn motif(shape: ProgramGraphShape, edges: &[MotifEdge], witness_out: &str) -> Program {
    let Ok(edge_count) = u32::try_from(edges.len()) else {
        return trap_program(
            OP_ID,
            Some((witness_out, DataType::U32)),
            "Fix: motif edges.len() exceeds u32::MAX; split the motif or redesign the caller."
                .to_string(),
        );
    };
    let mut buffers = shape.read_only_buffers();
    let per_node = shape.node_count.max(1);
    buffers.push(word_buffer(
        "motif_hits",
        MOTIF_HITS_BUFFER,
        BufferAccess::ReadWrite,
        per_node,
    ));
    buffers.push(word_buffer(
        witness_out,
        MOTIF_WITNESS_OUT_BUFFER,
        BufferAccess::ReadWrite,
        per_node,
    ));

    let clear_outputs = vec![
        Node::store("motif_hits", Expr::var("node"), Expr::u32(0)),
        Node::store(witness_out, Expr::var("node"), Expr::u32(0)),
    ];
    // Motif edges are compile-time operands of this generated program, not
    // runtime graph data. Lowering them as constants removes three input
    // buffers and prevents loop-carried scratch state from making a partial
    // motif look like a complete match.
    let Some(scan_capacity) = edges.len().checked_mul(5) else {
        return trap_program(
            OP_ID,
            Some((witness_out, DataType::U32)),
            "Fix: motif scan node count overflows usize; split the motif before lowering."
                .to_string(),
        );
    };
    let Some(mark_capacity) = edges.len().checked_mul(2) else {
        return trap_program(
            OP_ID,
            Some((witness_out, DataType::U32)),
            "Fix: motif witness mark count overflows usize; split the motif before lowering."
                .to_string(),
        );
    };
    let mut scan_edges: Vec<Node> = Vec::new();
    if let Err(error) = scan_edges.try_reserve(scan_capacity) {
        return trap_program(
            OP_ID,
            Some((witness_out, DataType::U32)),
            format!("Fix: motif lowering could not reserve {scan_capacity} scan nodes: {error}"),
        );
    }
    let mut mark_hits: Vec<Node> = Vec::new();
    if let Err(error) = mark_hits.try_reserve(mark_capacity) {
        return trap_program(
            OP_ID,
            Some((witness_out, DataType::U32)),
            format!("Fix: motif lowering could not reserve {mark_capacity} mark nodes: {error}"),
        );
    }
    for (idx, edge) in edges.iter().enumerate() {
        let edge_found = format!("edge_found_{idx}");
        let edge_start = format!("edge_start_{idx}");
        let edge_end = format!("edge_end_{idx}");
        let edge_index = format!("e_{idx}");
        let actual_dst = format!("actual_dst_{idx}");
        let actual_kind = format!("actual_kind_{idx}");
        scan_edges.push(Node::let_bind(&edge_found, Expr::u32(0)));
        if edge.from < shape.node_count {
            scan_edges.push(Node::let_bind(
                &edge_start,
                Expr::load(NAME_EDGE_OFFSETS, Expr::u32(edge.from)),
            ));
            scan_edges.push(Node::let_bind(
                &edge_end,
                Expr::load(NAME_EDGE_OFFSETS, Expr::u32(edge.from.saturating_add(1))),
            ));
            scan_edges.push(Node::loop_for(
                &edge_index,
                Expr::var(&edge_start),
                Expr::var(&edge_end),
                vec![
                    Node::let_bind(
                        &actual_dst,
                        Expr::load(NAME_EDGE_TARGETS, Expr::var(&edge_index)),
                    ),
                    Node::let_bind(
                        &actual_kind,
                        Expr::load(NAME_EDGE_KIND_MASK, Expr::var(&edge_index)),
                    ),
                    Node::if_then(
                        Expr::and(
                            Expr::eq(Expr::var(&actual_dst), Expr::u32(edge.to)),
                            Expr::ne(
                                Expr::bitand(Expr::var(&actual_kind), Expr::u32(edge.kind_mask)),
                                Expr::u32(0),
                            ),
                        ),
                        vec![Node::assign(&edge_found, Expr::u32(1))],
                    ),
                ],
            ));
        }
        scan_edges.push(Node::if_then(
            Expr::ne(Expr::var(&edge_found), Expr::u32(0)),
            vec![Node::assign(
                "matched_edges",
                Expr::add(Expr::var("matched_edges"), Expr::u32(1)),
            )],
        ));
        if edge.from < shape.node_count {
            mark_hits.push(Node::store(
                "motif_hits",
                Expr::u32(edge.from),
                Expr::u32(1),
            ));
        }
        if edge.to < shape.node_count {
            mark_hits.push(Node::store("motif_hits", Expr::u32(edge.to), Expr::u32(1)));
        }
    }
    let materialize = vec![Node::store(
        witness_out,
        Expr::var("node"),
        Expr::load("motif_hits", Expr::var("node")),
    )];
    let mut publish_full_match = mark_hits;
    publish_full_match.push(Node::loop_for(
        "node",
        Expr::u32(0),
        Expr::u32(shape.node_count),
        materialize,
    ));

    // PHASE7_GRAPH C2: motif is fundamentally serial  -  one thread loops
    // over every motif edge in order and accumulates `matched_edges`.
    // Using a [256,1,1] workgroup with a `gid_x() == 0` gate burns 255
    // idle lanes per workgroup. Dispatch a single 1-lane workgroup
    // instead so the wasted parallelism is gone, and drop the redundant
    // gate.
    Program::wrapped(
        buffers,
        MOTIF_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(
            OP_ID,
            vec![
                Node::loop_for(
                    "node",
                    Expr::u32(0),
                    Expr::u32(shape.node_count),
                    clear_outputs,
                ),
                Node::let_bind("matched_edges", Expr::u32(0)),
                Node::Block(scan_edges),
                Node::if_then(
                    Expr::eq(Expr::var("matched_edges"), Expr::u32(edge_count)),
                    publish_full_match,
                ),
            ],
        )],
    )
}
