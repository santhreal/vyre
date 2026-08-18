//! Shared CSR frontier-step `Program` builder.
//! Forward and reverse traversals use the same ProgramGraph ABI,
//! frontier buffers, edge-kind mask filtering, and packed-NodeSet
//! output writes. The only semantic difference is whether the input
//! frontier is tested at `src` before walking outgoing edges or at
//! `dst` while scanning a source row.
//!

use crate::graph::program_graph::{
    ProgramGraphShape, BINDING_PRIMITIVE_START, NAME_EDGE_KIND_MASK, NAME_EDGE_OFFSETS,
};
use vyre_foundation::ir::{Expr, Node, Program};

/// Canonical binding index for the input frontier bitset.
pub const BINDING_FRONTIER_IN: u32 = BINDING_PRIMITIVE_START;
/// Canonical binding index for the output frontier bitset.
pub const BINDING_FRONTIER_OUT: u32 = BINDING_PRIMITIVE_START + 1;
/// Binding index for the excluded-source mask of the excluding forward step.
pub const BINDING_EXCLUDED_SOURCES: u32 = BINDING_PRIMITIVE_START + 1;
/// Binding index for the output frontier of the excluding forward step.
///
/// The excluded-source mask takes the slot [`BINDING_FRONTIER_OUT`] holds in
/// every other frontier step, so the output frontier sits one slot further out.
pub const BINDING_EXCLUDING_FRONTIER_OUT: u32 = BINDING_PRIMITIVE_START + 2;
pub(crate) const CSR_FRONTIER_STEP_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

/// Dispatch grid for one source-lane CSR frontier step.
#[must_use]
pub const fn csr_frontier_step_dispatch_grid(node_count: u32) -> [u32; 3] {
    vyre_primitives::lane_grid(node_count, CSR_FRONTIER_STEP_WORKGROUP_SIZE[0])
}

/// Direction for a one-step CSR frontier traversal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CsrFrontierStepKind {
    /// If `src` is active, emit each allowed `dst`.
    Forward,
    /// If any allowed `dst` is active, emit `src`.
    Backward,
}

/// Build a one-step CSR frontier traversal under a caller-owned op id.
#[must_use]
pub(crate) fn csr_frontier_step_program(
    op_id: &'static str,
    kind: CsrFrontierStepKind,
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
    allow_mask: u32,
) -> Program {
    match kind {
        CsrFrontierStepKind::Forward => crate::builder::csr::CsrTraversalComposer::forward(
            op_id,
            shape.node_count,
            shape.edge_count,
            allow_mask,
        )
        .with_workgroup_size(CSR_FRONTIER_STEP_WORKGROUP_SIZE)
        .build_forward_step(frontier_in, frontier_out),
        CsrFrontierStepKind::Backward => crate::builder::csr::CsrTraversalComposer::backward(
            op_id,
            shape.node_count,
            shape.edge_count,
            allow_mask,
        )
        .with_workgroup_size(CSR_FRONTIER_STEP_WORKGROUP_SIZE)
        .build_backward_step(frontier_in, frontier_out),
    }
}
/// Build a forward CSR step that excludes active source nodes selected by
/// `excluded_sources`.
#[must_use]
pub(crate) fn csr_forward_step_excluding_program(
    op_id: &'static str,
    shape: ProgramGraphShape,
    frontier_in: &str,
    excluded_sources: &str,
    frontier_out: &str,
    allow_mask: u32,
) -> Program {
    crate::builder::csr::CsrTraversalComposer::forward(
        op_id,
        shape.node_count,
        shape.edge_count,
        allow_mask,
    )
    .with_workgroup_size(CSR_FRONTIER_STEP_WORKGROUP_SIZE)
    .build_forward_step_excluding(frontier_in, excluded_sources, frontier_out)
}

pub(crate) fn edge_scan_body(
    allow_mask: u32,
    before_kind_body: Vec<Node>,
    on_allowed_body: Vec<Node>,
) -> Vec<Node> {
    let mut loop_body = before_kind_body;
    loop_body.push(Node::let_bind(
        "kind_mask",
        Expr::load(NAME_EDGE_KIND_MASK, Expr::var("e")),
    ));
    loop_body.push(Node::if_then(
        Expr::ne(
            Expr::bitand(Expr::var("kind_mask"), Expr::u32(allow_mask)),
            Expr::u32(0),
        ),
        on_allowed_body,
    ));
    edge_bounds_and_loop(loop_body)
}

fn edge_bounds_and_loop(loop_body: Vec<Node>) -> Vec<Node> {
    vec![
        Node::let_bind(
            "edge_start",
            Expr::load(NAME_EDGE_OFFSETS, Expr::var("src")),
        ),
        Node::let_bind(
            "edge_end",
            Expr::load(NAME_EDGE_OFFSETS, Expr::add(Expr::var("src"), Expr::u32(1))),
        ),
        Node::loop_for(
            "e",
            Expr::var("edge_start"),
            Expr::var("edge_end"),
            loop_body,
        ),
    ]
}

#[path = "csr_frontier_step_queue.rs"]
mod csr_frontier_step_queue;
pub(crate) use csr_frontier_step_queue::*;
#[cfg(test)]
pub(crate) fn csr_frontier_step_cpu_ref(
    kind: CsrFrontierStepKind,
    graph: crate::graph::csr_closure_inputs::CsrGraphView<'_>,
    frontier: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    assert!(
        graph.edge_offsets.len() == (graph.node_count as usize) + 1,
        "complete CSR offset table"
    );
    match kind {
        CsrFrontierStepKind::Forward => {
            vyre_reference::composition_witness::csr_forward_traverse_witness(
                graph.node_count,
                graph.edge_offsets,
                graph.edge_targets,
                graph.edge_kind_mask,
                frontier,
                allow_mask,
            )
        }
        CsrFrontierStepKind::Backward => {
            vyre_reference::composition_witness::csr_backward_traverse_witness(
                graph.node_count,
                graph.edge_offsets,
                graph.edge_targets,
                graph.edge_kind_mask,
                frontier,
                allow_mask,
            )
        }
    }
}

#[cfg(test)]
pub(crate) fn csr_frontier_step_cpu_ref_into(
    kind: CsrFrontierStepKind,
    graph: crate::graph::csr_closure_inputs::CsrGraphView<'_>,
    frontier: &[u32],
    allow_mask: u32,
    out: &mut Vec<u32>,
) {
    match kind {
        CsrFrontierStepKind::Forward => {
            vyre_reference::composition_witness::csr_forward_traverse_witness_into(
                graph.node_count,
                graph.edge_offsets,
                graph.edge_targets,
                graph.edge_kind_mask,
                frontier,
                allow_mask,
                out,
            )
        }
        CsrFrontierStepKind::Backward => {
            vyre_reference::composition_witness::csr_backward_traverse_witness_into(
                graph.node_count,
                graph.edge_offsets,
                graph.edge_targets,
                graph.edge_kind_mask,
                frontier,
                allow_mask,
                out,
            )
        }
    }
}

#[cfg(test)]
pub(crate) fn csr_forward_traverse_cpu_ref(
    node_count: u32,
    row_offsets: &[u32],
    col_indices: &[u32],
    edge_kind_mask: &[u32],
    frontier: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    csr_frontier_step_cpu_ref(
        CsrFrontierStepKind::Forward,
        crate::graph::csr_closure_inputs::CsrGraphView::new(
            node_count,
            row_offsets,
            col_indices,
            edge_kind_mask,
        ),
        frontier,
        allow_mask,
    )
}

#[cfg(test)]
pub(crate) fn csr_forward_traverse_cpu_ref_into(
    node_count: u32,
    row_offsets: &[u32],
    col_indices: &[u32],
    edge_kind_mask: &[u32],
    frontier: &[u32],
    allow_mask: u32,
    out: &mut Vec<u32>,
) {
    csr_frontier_step_cpu_ref_into(
        CsrFrontierStepKind::Forward,
        crate::graph::csr_closure_inputs::CsrGraphView::new(
            node_count,
            row_offsets,
            col_indices,
            edge_kind_mask,
        ),
        frontier,
        allow_mask,
        out,
    );
}

#[cfg(test)]
pub(crate) fn csr_backward_traverse_cpu_ref(
    node_count: u32,
    row_offsets: &[u32],
    col_indices: &[u32],
    edge_kind_mask: &[u32],
    frontier: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    csr_frontier_step_cpu_ref(
        CsrFrontierStepKind::Backward,
        crate::graph::csr_closure_inputs::CsrGraphView::new(
            node_count,
            row_offsets,
            col_indices,
            edge_kind_mask,
        ),
        frontier,
        allow_mask,
    )
}

#[cfg(test)]
pub(crate) fn csr_backward_traverse_cpu_ref_into(
    node_count: u32,
    row_offsets: &[u32],
    col_indices: &[u32],
    edge_kind_mask: &[u32],
    frontier: &[u32],
    allow_mask: u32,
    out: &mut Vec<u32>,
) {
    csr_frontier_step_cpu_ref_into(
        CsrFrontierStepKind::Backward,
        crate::graph::csr_closure_inputs::CsrGraphView::new(
            node_count,
            row_offsets,
            col_indices,
            edge_kind_mask,
        ),
        frontier,
        allow_mask,
        out,
    );
}

#[cfg(test)]
mod tests {
    use super::{csr_frontier_step_dispatch_grid, CSR_FRONTIER_STEP_WORKGROUP_SIZE};
    use vyre_reference::composition_witness::{
        csr_backward_traverse_witness, csr_forward_traverse_witness,
    };

    #[test]
    fn generated_csr_frontier_step_uses_block_sized_workgroup() {
        let program = crate::graph::csr_forward_traverse::csr_forward_traverse(
            crate::graph::program_graph::ProgramGraphShape::new(1024, 1536),
            "frontier_in",
            "frontier_out",
            u32::MAX,
        );

        assert_eq!(program.workgroup_size(), CSR_FRONTIER_STEP_WORKGROUP_SIZE);
        assert!(
            program.workgroup_size()[0] > 1,
            "Fix: CSR frontier traversal must not launch one workgroup per source node."
        );
    }

    #[test]
    fn dispatch_grid_packs_source_lanes_into_workgroups() {
        assert_eq!(csr_frontier_step_dispatch_grid(0), [1, 1, 1]);
        assert_eq!(csr_frontier_step_dispatch_grid(1), [1, 1, 1]);
        assert_eq!(csr_frontier_step_dispatch_grid(256), [1, 1, 1]);
        assert_eq!(csr_frontier_step_dispatch_grid(257), [2, 1, 1]);
        assert_eq!(csr_frontier_step_dispatch_grid(513), [3, 1, 1]);
    }

    #[test]
    fn generated_csr_frontier_steps_match_scalar_reference() {
        let mut state = 0xC5A1_F00D_u32;
        for case in 0..2048_u32 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let node_count = (state % 97) + 1;
            let mut offsets = Vec::with_capacity(node_count as usize + 1);
            let mut targets = Vec::new();
            let mut masks = Vec::new();
            offsets.push(0);
            for src in 0..node_count {
                state = state.rotate_left(5) ^ src.wrapping_mul(0x9E37_79B9);
                let degree = state % 5;
                for edge in 0..degree {
                    state = state.rotate_left(7) ^ edge.wrapping_mul(0x85EB_CA6B);
                    let target = match edge % 5 {
                        0 => state % node_count,
                        1 => node_count,
                        2 => u32::MAX,
                        _ => state % (node_count + 3),
                    };
                    targets.push(target);
                    masks.push(1_u32 << (state & 7));
                }
                offsets.push(targets.len() as u32);
            }
            let words = crate::bitset::bitset_words(node_count) as usize;
            let mut frontier = vec![0_u32; words];
            for node in 0..node_count {
                state = state.rotate_left(3) ^ node.wrapping_mul(0x27D4_EB2D);
                if (state & 3) != 0 {
                    frontier[(node / 32) as usize] |= 1_u32 << (node % 32);
                }
            }
            let allow_mask = if case % 11 == 0 {
                0
            } else {
                (1_u32 << (case & 7)) | (1_u32 << ((case + 3) & 7))
            };

            assert_eq!(
                crate::graph::csr_forward_traverse::cpu_ref(
                    node_count, &offsets, &targets, &masks, &frontier, allow_mask,
                ),
                csr_forward_traverse_witness(
                    node_count, &offsets, &targets, &masks, &frontier, allow_mask,
                ),
                "forward case {case}"
            );
            assert_eq!(
                crate::graph::csr_backward_traverse::cpu_ref(
                    node_count, &offsets, &targets, &masks, &frontier, allow_mask,
                ),
                csr_backward_traverse_witness(
                    node_count, &offsets, &targets, &masks, &frontier, allow_mask,
                ),
                "backward case {case}"
            );
        }
    }
}
