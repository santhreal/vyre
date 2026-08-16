//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
#![forbid(unsafe_code)]
#![cfg(all(feature = "graph", feature = "cpu-parity"))]
mod graph_sweep_support;
use graph_sweep_support::bitset_words;
#[path = "../../tests/support/csr_sweep/mod.rs"]
mod csr_sweep;

use vyre_libs::graph::csr_backward_traverse;

/// Independent oracle for ONE BACKWARD (reverse) CSR traversal step: a node `src` enters the output
/// frontier iff it has at least one allowed out-edge whose TARGET is currently in the frontier (i.e).
/// the output is the set of PREDECESSORS of the frontier along allowed edges. This is the kernel's
/// documented contract (see the `csr_backward_traverse` inventory example: `frontier_in = {3}` →
/// `frontier_out = {1, 2}`, "both point at 3"). The previous version of this oracle computed the
/// OPPOSITE relation (successors of the frontier: for each frontier `dst`, it marked `dst`'s edge
/// targets), so it disagreed with the (correct) kernel on every graph with asymmetric edges, a false
/// red, fixed here to the true backward/predecessor semantics.
fn oracle_csr_backward_step(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    let words = bitset_words(node_count);
    let mut out = vec![0u32; words];
    let in_frontier = |node: u32| -> bool {
        let word = (node / 32) as usize;
        word < frontier_in.len() && (frontier_in[word] & (1u32 << (node % 32))) != 0
    };
    for src in 0..node_count {
        let edge_start = edge_offsets[src as usize] as usize;
        let edge_end = edge_offsets[src as usize + 1] as usize;
        for e in edge_start..edge_end {
            if (edge_kind_mask[e] & allow_mask) == 0 {
                continue;
            }
            let dst = edge_targets[e];
            if dst < node_count && in_frontier(dst) {
                out[(src / 32) as usize] |= 1u32 << (src % 32);
                break;
            }
        }
    }
    out
}

const CASES: usize = 16384;

#[test]
fn sweep_graph_csr_backward_traverse_volume_oracle_matrix() {
    for (case, node_count, offsets, targets, masks, frontier, allow_mask) in csr_sweep::tuples(
        "single_source_all_kinds",
        CASES as u64,
        0xBAC1A4D1,
        0x9E37_79B9_7F4A_7C15,
    ) {
        let expected = oracle_csr_backward_step(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask,
        );
        let actual = csr_backward_traverse::cpu_ref(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask,
        );
        assert_eq!(
            actual, expected,
            "Fix: csr_backward_traverse volume case {case} node_count={node_count}"
        );
    }
}
