//! Reverse CSR frontier expansion over an in-place accumulator bitset.

use vyre_foundation::composition::wrap_anonymous_region;

use vyre_foundation::ir::{Expr, Node, Program};

#[cfg(any(test, feature = "cpu-parity"))]
use crate::bitset::bitset_words;
#[cfg(any(test, feature = "cpu-parity"))]
use crate::graph::csr_closure_inputs::CsrClosureInputs;
use crate::graph::frontier_bits::{set_bit, when_bit_set, BitAccess};
use vyre_primitives::lane_grid;
use crate::graph::program_graph::{
    push_frontier_changed_buffers, ProgramGraphShape, NAME_EDGE_KIND_MASK, NAME_EDGE_OFFSETS,
    NAME_EDGE_TARGETS,
};

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::graph::csr_backward_or_changed";
/// Source-lane workgroup for reverse in-place CSR expansion.
pub const CSR_BACKWARD_OR_CHANGED_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

/// Dispatch grid for a node-parallel reverse in-place CSR expansion pass.
#[must_use]
pub const fn csr_backward_or_changed_parallel_grid(node_count: u32) -> [u32; 3] {
    lane_grid(node_count, CSR_BACKWARD_OR_CHANGED_WORKGROUP_SIZE[0])
}

/// Parallel in-place reverse expansion program for resident fixed-point drivers.
#[must_use]
pub fn csr_backward_or_changed_parallel(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
) -> Program {
    let src = Expr::InvocationId { axis: 0 };
    let body = vec![
        Node::let_bind("edge_start", Expr::load(NAME_EDGE_OFFSETS, src.clone())),
        Node::let_bind(
            "edge_end",
            Expr::load(NAME_EDGE_OFFSETS, Expr::add(src.clone(), Expr::u32(1))),
        ),
        Node::let_bind("hit", Expr::u32(0)),
        Node::loop_for(
            "e",
            Expr::var("edge_start"),
            Expr::var("edge_end"),
            vec![Node::if_then(
                Expr::eq(Expr::var("hit"), Expr::u32(0)),
                vec![
                    Node::let_bind("kind_mask", Expr::load(NAME_EDGE_KIND_MASK, Expr::var("e"))),
                    Node::if_then(
                        Expr::ne(
                            Expr::bitand(Expr::var("kind_mask"), Expr::u32(edge_kind_mask)),
                            Expr::u32(0),
                        ),
                        vec![
                            Node::let_bind("dst", Expr::load(NAME_EDGE_TARGETS, Expr::var("e"))),
                            Node::if_then(
                                Expr::lt(Expr::var("dst"), Expr::u32(shape.node_count)),
                                when_bit_set(
                                    frontier_out,
                                    &Expr::var("dst"),
                                    None,
                                    "dst_word",
                                    "dst_bit",
                                    |word| word,
                                    vec![Node::assign("hit", Expr::u32(1))],
                                ),
                            ),
                        ],
                    ),
                ],
            )],
        ),
        Node::if_then(
            Expr::eq(Expr::var("hit"), Expr::u32(1)),
            set_bit(
                frontier_out,
                &src,
                BitAccess {
                    word: "src_word_idx",
                    mask: "src_bit",
                    value: "old",
                },
                |word| word,
                vec![Node::let_bind(
                    "_changed",
                    Expr::atomic_or(changed, Expr::u32(0), Expr::u32(1)),
                )],
            ),
        ),
    ];
    let mut buffers = shape.read_only_buffers();
    push_frontier_changed_buffers(&mut buffers, frontier_out, changed, shape.node_count);
    Program::wrapped(
        buffers,
        CSR_BACKWARD_OR_CHANGED_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(
            OP_ID,
            vec![Node::if_then(
                Expr::lt(src.clone(), Expr::u32(shape.node_count)),
                body,
            )],
        )],
    )
}

/// CPU reference for one reverse-or-changed expansion pass (snapshot semantics): a
/// source node is added to the frontier when any of its out-neighbors reached by an edge
/// whose kind passes `allow_mask` is present in the INPUT frontier. Returns the
/// updated frontier (input bits are monotonically retained) and `1` iff a new bit was set.
///
/// This reads the pre-pass frontier for the neighbor test, so it is the deterministic
/// single-pass answer. The GPU program [`csr_backward_or_changed_parallel`] reads the
/// live in-place accumulator, so for a multi-hop backward chain a single GPU pass can set
/// MORE bits than one snapshot pass, but both converge to the identical fixed point (the
/// op's contract; proven by the generated backward oracle matrix + fixpoint idempotence).
///
/// The per-edge kind array is `edge_kind_mask` and the scalar edge filter is
/// `allow_mask`, matching every sibling CSR reference. This module used to swap
/// those two names, so `edge_kind_mask` meant the array next door and the scalar
/// here, and a caller who read a sibling first had the two roles inverted.
#[cfg(any(test, feature = "cpu-parity"))]
#[must_use]
pub fn cpu_ref(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier: &[u32],
    allow_mask: u32,
) -> (Vec<u32>, u32) {
    let words = bitset_words(node_count).max(1) as usize;
    let mut out = frontier.to_vec();
    out.resize(words, 0);
    let is_set = |bits: &[u32], node: u32| -> bool {
        bits.get((node >> 5) as usize)
            .is_some_and(|word| word & (1u32 << (node & 31)) != 0)
    };
    let mut changed = 0u32;
    for src in 0..node_count {
        let start = edge_offsets[src as usize] as usize;
        let end = edge_offsets[src as usize + 1] as usize;
        // Match the program's early-out: stop at the first present, kind-passing out-neighbor.
        let mut hit = false;
        for edge in start..end {
            if edge_kind_mask[edge] & allow_mask != 0 {
                let dst = edge_targets[edge];
                if dst < node_count && is_set(frontier, dst) {
                    hit = true;
                    break;
                }
            }
        }
        if hit && !is_set(&out, src) {
            out[(src >> 5) as usize] |= 1u32 << (src & 31);
            changed = 1;
        }
    }
    (out, changed)
}

/// Iterate [`cpu_ref`] to a fixed point (at most `inputs.max_iters` passes): the full set
/// of nodes that can reach an initial-frontier node along kind-passing edges. Returns the
/// converged frontier and `1` iff any pass set a new bit.
#[cfg(any(test, feature = "cpu-parity"))]
#[must_use]
pub fn cpu_ref_closure(inputs: CsrClosureInputs<'_>, frontier: &[u32]) -> (Vec<u32>, u32) {
    let graph = inputs.graph;
    let words = bitset_words(graph.node_count).max(1) as usize;
    let mut out = frontier.to_vec();
    out.resize(words, 0);
    let mut any_changed = 0u32;
    for _ in 0..inputs.max_iters {
        let (next, changed) = cpu_ref(
            graph.node_count,
            graph.edge_offsets,
            graph.edge_targets,
            graph.edge_kind_mask,
            &out,
            inputs.allow_mask,
        );
        out = next;
        if changed == 0 {
            break;
        }
        any_changed = 1;
    }
    (out, any_changed)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || csr_backward_or_changed_parallel(ProgramGraphShape::new(4, 4), "frontier", "changed", 1),
        Some(|| {
            // Graph: 0→1, 0→2, 1→3, 2→3. Reverse-or-changed adds a source node when any
            // of its out-neighbors is already in the frontier. Seed = {1, 2}: only node 0
            // (out-edges to 1 and 2) sees a set out-neighbor, so it is the sole addition.
            //
            // Deliberately a ONE-HOP fixture. Node 0's read of its out-neighbor bits (1, 2)
            // is stable because NO lane writes bits 1 or 2 this pass (the only write is node
            // 0's own bit 0, which nobody reads), so the single pass is race-clean and
            // order-independent, as the lane-reversal race net and the grid-overfire
            // output-invariance net require. A MULTI-HOP backward chain (e.g. seed {3},
            // where node 1 is added then node 0 sees 1's just-set bit) is legitimately
            // order-dependent in a single node-parallel pass; that only the CONVERGED
            // fixed-point is order-independent is the op's real contract, covered by the
            // dedicated iterate-to-fixpoint oracle test, not assertable as a single-pass
            // fixture here.
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[0, 0, 0, 0]),    // pg_nodes
                to_bytes(&[0, 2, 3, 4, 4]), // pg_edge_offsets
                to_bytes(&[1, 2, 3, 3]),    // pg_edge_targets
                to_bytes(&[1, 1, 1, 1]),    // pg_edge_kind_mask
                to_bytes(&[0, 0, 0, 0]),    // pg_node_tags
                to_bytes(&[0b0110]),        // frontier seed = {1, 2}
                to_bytes(&[0]),             // changed
            ]]
        }),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            // Node 0 added (its out-neighbors 1, 2 are set); frontier = {0, 1, 2}, changed = 1.
            vec![vec![to_bytes(&[0b0111]), to_bytes(&[1])]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_frontier_and_changed_bindings() {
        let program = csr_backward_or_changed_parallel(
            ProgramGraphShape::new(4, 3),
            "frontier",
            "changed",
            u32::MAX,
        );
        let names = program
            .buffers()
            .iter()
            .map(|buffer| buffer.name())
            .collect::<Vec<_>>();

        assert!(names.contains(&"frontier"));
        assert!(names.contains(&"changed"));
        assert_eq!(
            program.workgroup_size(),
            CSR_BACKWARD_OR_CHANGED_WORKGROUP_SIZE
        );
    }

    #[test]
    fn parallel_grid_packs_source_lanes_into_blocks() {
        assert_eq!(csr_backward_or_changed_parallel_grid(0), [1, 1, 1]);
        assert_eq!(csr_backward_or_changed_parallel_grid(1), [1, 1, 1]);
        assert_eq!(csr_backward_or_changed_parallel_grid(256), [1, 1, 1]);
        assert_eq!(csr_backward_or_changed_parallel_grid(257), [2, 1, 1]);
        assert_eq!(csr_backward_or_changed_parallel_grid(513), [3, 1, 1]);
    }
}
