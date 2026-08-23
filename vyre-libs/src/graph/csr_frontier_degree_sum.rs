//! `csr_frontier_degree_sum`  -  total outgoing-edge count of an active
//! BFS frontier on a `super::program_graph::ProgramGraph`.
//!
//! `csr_forward_traverse` launches one thread per *source node*, which
//! catastrophically load-imbalances on power-law graphs (one vertex with
//! 1M neighbors monopolises one thread while everyone else does zero
//! work). Load-balanced expansion launches one thread per *active edge*
//! instead; the host needs this count to launch the exact grid.
//!
//! This primitive computes that count. Given:
//!   - `frontier_in`  -  a packed bitset over `node_count`, one bit per
//!     active source node.
//!   - `pg_edge_offsets`  -  the canonical CSR row pointers from
//!     `ProgramGraph`.
//!
//! It emits a single u32 scalar:
//!   - `degree_sum_out[0] = ∑_{v ∈ frontier_in} (edge_offsets[v+1] − edge_offsets[v])`
//!
//! The host reads this scalar between dispatches and uses it to size
//! the next load-balanced expansion kernel. The CPU reference at the
//! bottom of this file documents the contract; the parity harness runs
//! both implementations on the same input.

use vyre_foundation::composition::wrap_anonymous_region;

use vyre_foundation::ir::{BufferAccess, Expr, Node, Program};

use crate::graph::frontier_bits::active_source_lane;
use crate::graph::program_graph::{
    frontier_buffer, word_buffer, ProgramGraphShape, BINDING_PRIMITIVE_START,
};

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::graph::csr_frontier_degree_sum";

/// Canonical binding index for the input frontier bitset.
pub const BINDING_FRONTIER_IN: u32 = BINDING_PRIMITIVE_START;
/// Canonical binding index for the output degree-sum scalar.
pub const BINDING_DEGREE_SUM_OUT: u32 = BINDING_PRIMITIVE_START + 1;
/// One lane per source node; 256 keeps occupancy high without bloating register pressure.
pub const CSR_FRONTIER_DEGREE_SUM_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

/// Dispatch grid that covers every source node exactly once.
#[cfg(test)]
#[must_use]
pub const fn csr_frontier_degree_sum_dispatch_grid(node_count: u32) -> [u32; 3] {
    vyre_primitives::lane_grid(node_count, CSR_FRONTIER_DEGREE_SUM_WORKGROUP_SIZE[0])
}

/// Build the IR `Program` that computes `degree_sum_out[0]` =
/// total outgoing-edge count over the active frontier.
///
/// One thread per node. Each thread:
///   1. Loads its bit from `frontier_in`. If clear, exits.
///   2. Computes its degree as `edge_offsets[gid+1] - edge_offsets[gid]`.
///   3. Atomically adds the degree into `degree_sum_out[0]`.
#[must_use]
pub fn csr_frontier_degree_sum(shape: ProgramGraphShape) -> Program {
    let frontier_in = "frontier_in";
    let degree_sum_out = "degree_sum_out";

    let body = vec![active_source_lane(
        shape.node_count,
        frontier_in,
        None,
        Expr::InvocationId { axis: 0 },
        {
            let [off_lo, off_hi, deg] =
                crate::builder::csr::CsrTraversalComposer::new(OP_ID, OP_ID, shape.node_count)
                    .emit_row_degree(Expr::var("src"), "off_lo", "off_hi", "degree");
            vec![
                off_lo,
                off_hi,
                deg,
                Node::let_bind(
                    "_old",
                    Expr::atomic_add(degree_sum_out, Expr::u32(0), Expr::var("degree")),
                ),
            ]
        },
    )];

    let mut buffers = shape.read_only_buffers();
    buffers.push(frontier_buffer(
        frontier_in,
        BINDING_FRONTIER_IN,
        BufferAccess::ReadOnly,
        shape.node_count,
    ));
    buffers.push(word_buffer(
        degree_sum_out,
        BINDING_DEGREE_SUM_OUT,
        BufferAccess::ReadWrite,
        1,
    ));

    let entry = vec![wrap_anonymous_region(OP_ID, body)];
    Program::wrapped(buffers, CSR_FRONTIER_DEGREE_SUM_WORKGROUP_SIZE, entry)
}

const EXPECTED_CSR_FRONTIER_DEGREE_SUM_OUTPUT_BYTES: [u8; 4] = [3, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || csr_frontier_degree_sum(ProgramGraphShape::new(4, 4)),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![crate::graph::program_graph::sample_program_graph_inputs(&[
                to_bytes(&[0b0011]),              // frontier_in = {0, 1}
                to_bytes(&[0]),                   // degree_sum_out
            ])]
        }),
        Some(|| {
            vec![vec![EXPECTED_CSR_FRONTIER_DEGREE_SUM_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_laws(&["bounded"])
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_reference::composition_witness::csr_frontier_degree_sum_witness;

    fn try_csr_frontier_degree_sum_cpu(
        frontier: &[u32],
        edge_offsets: &[u32],
        node_count: u32,
    ) -> Result<u32, String> {
        if edge_offsets.len() != (node_count as usize) + 1 {
            return Err("Fix: edge_offsets.len() must equal node_count + 1.".to_owned());
        }
        if edge_offsets[0] != 0 {
            return Err("Fix: edge offsets must start at zero.".to_owned());
        }
        if edge_offsets.windows(2).any(|w| w[0] > w[1]) {
            return Err("Fix: non-monotonic CSR offsets detected.".to_owned());
        }
        let expected_words = crate::bitset::bitset_words(node_count) as usize;
        if frontier.len() < expected_words {
            return Err("Fix: frontier bitset shorter than expected words.".to_owned());
        }
        Ok(csr_frontier_degree_sum_witness(
            frontier,
            edge_offsets,
            node_count,
        ))
    }

    fn csr_frontier_degree_sum_cpu(frontier: &[u32], edge_offsets: &[u32], node_count: u32) -> u32 {
        try_csr_frontier_degree_sum_cpu(frontier, edge_offsets, node_count).unwrap_or(u32::MAX)
    }

    #[test]
    fn cpu_ref_empty_frontier_emits_zero() {
        let frontier = vec![0u32];
        let edge_offsets = vec![0u32, 3, 7, 9, 9, 12]; // 5 nodes, 12 edges
        assert_eq!(csr_frontier_degree_sum_cpu(&frontier, &edge_offsets, 5), 0);
    }

    #[test]
    fn cpu_ref_single_node_frontier_returns_its_degree() {
        // Node 0 in frontier. Its degree = edge_offsets[1] - edge_offsets[0] = 3.
        let frontier = vec![1u32];
        let edge_offsets = vec![0u32, 3, 7, 9, 9, 12];
        assert_eq!(csr_frontier_degree_sum_cpu(&frontier, &edge_offsets, 5), 3);
    }

    #[test]
    fn cpu_ref_full_frontier_sums_all_degrees() {
        // All 5 nodes in frontier.
        let frontier = vec![0b11111u32];
        let edge_offsets = vec![0u32, 3, 7, 9, 9, 12];
        // Degrees: 3, 4, 2, 0, 3 → sum 12.
        assert_eq!(csr_frontier_degree_sum_cpu(&frontier, &edge_offsets, 5), 12);
    }

    #[test]
    fn cpu_ref_handles_isolated_nodes_in_frontier() {
        // Node 3 has 0 outgoing edges. Frontier = just {3}. Sum should be 0.
        let frontier = vec![0b1000u32];
        let edge_offsets = vec![0u32, 3, 7, 9, 9, 12];
        assert_eq!(csr_frontier_degree_sum_cpu(&frontier, &edge_offsets, 5), 0);
    }

    #[test]
    fn cpu_ref_partial_frontier_sums_subset() {
        // Frontier = {0, 2}. Degrees 3 + 2 = 5.
        let frontier = vec![0b00101u32];
        let edge_offsets = vec![0u32, 3, 7, 9, 9, 12];
        assert_eq!(csr_frontier_degree_sum_cpu(&frontier, &edge_offsets, 5), 5);
    }

    #[test]
    fn cpu_ref_multi_word_frontier() {
        // 64 nodes, two-word bitset. Set bits at 0, 31, 32, 63. Degrees 1 each.
        let frontier = vec![
            0b1u32 | (1u32 << 31), // word 0: bits 0 and 31
            0b1u32 | (1u32 << 31), // word 1: bits 0 and 31 (= absolute 32 and 63)
        ];
        let edge_offsets = (0..=64u32).collect::<Vec<_>>();
        // Each node has degree 1 (offsets are [0, 1, 2, ..., 64]).
        // Frontier has 4 active nodes; sum = 4.
        assert_eq!(csr_frontier_degree_sum_cpu(&frontier, &edge_offsets, 64), 4);
    }

    #[test]
    fn build_program_returns_well_formed_program() {
        let shape = ProgramGraphShape::new(64, 128);
        let program = csr_frontier_degree_sum(shape);
        assert!(
            program.buffers().len() >= 6,
            "expects pg buffers + frontier_in + degree_sum_out"
        );
        assert_eq!(
            program.workgroup_size(),
            CSR_FRONTIER_DEGREE_SUM_WORKGROUP_SIZE
        );
    }

    #[test]
    fn dispatch_grid_covers_all_source_nodes() {
        assert_eq!(csr_frontier_degree_sum_dispatch_grid(0), [1, 1, 1]);
        assert_eq!(csr_frontier_degree_sum_dispatch_grid(1), [1, 1, 1]);
        assert_eq!(csr_frontier_degree_sum_dispatch_grid(256), [1, 1, 1]);
        assert_eq!(csr_frontier_degree_sum_dispatch_grid(257), [2, 1, 1]);
        assert_eq!(csr_frontier_degree_sum_dispatch_grid(1029), [5, 1, 1]);
    }

    #[test]
    fn checked_cpu_ref_reports_non_monotonic_offsets() {
        let frontier = [0b11u32];
        let edge_offsets = [0, 9, 3];
        let error = try_csr_frontier_degree_sum_cpu(&frontier, &edge_offsets, 2)
            .expect_err("checked degree sum oracle must reject malformed CSR offsets");

        assert!(
            error.contains("non-monotonic CSR offsets"),
            "error should describe the malformed CSR offsets: {error}"
        );
    }

    #[test]
    fn legacy_cpu_ref_pins_degree_sum_on_malformed_offsets() {
        let frontier = [0b11u32];
        let edge_offsets = [0, 9, 3];

        assert_eq!(
            csr_frontier_degree_sum_cpu(&frontier, &edge_offsets, 2),
            u32::MAX
        );
    }

    #[test]
    fn generated_degree_sum_cpu_matches_scalar_reference() {
        let mut state = 0xD36D_5A17_u32;
        for case in 0..4096u32 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let node_count = state % 4097 + 1;
            let mut edge_offsets = Vec::with_capacity(node_count as usize + 1);
            let mut expected = 0u32;
            let words = crate::bitset::bitset_words(node_count) as usize;
            let mut frontier = vec![0u32; words];

            edge_offsets.push(0);
            for src in 0..node_count {
                state = state.rotate_left(7) ^ src.wrapping_mul(0x9E37_79B9);
                let degree = state % 13;
                let active = (state.rotate_right((src & 15) + 1) & 3) != 0;
                if active {
                    frontier[(src / 32) as usize] |= 1u32 << (src % 32);
                    expected += degree;
                }
                edge_offsets.push(edge_offsets.last().copied().unwrap_or(0) + degree);
            }

            assert_eq!(
                try_csr_frontier_degree_sum_cpu(&frontier, &edge_offsets, node_count),
                Ok(expected),
                "generated degree-sum case {case}"
            );
        }
    }

    #[test]
    fn op_id_is_canonical_and_stable() {
        // Operation ids appear in serialized semantic metadata and benchmark
        // attribution; changing one is wire-format-visible.
        assert_eq!(OP_ID, "vyre-libs::graph::csr_frontier_degree_sum");
    }
}
