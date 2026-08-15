//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
#![forbid(unsafe_code)]
#![cfg(all(feature = "graph", feature = "cpu-parity"))]
use vyre_primitives::graph::csr_closure_inputs::{CsrClosureInputs, CsrGraphView};
mod graph_sweep_support;
use graph_sweep_support::bitset_words;
#[path = "../../tests/support/csr_sweep/mod.rs"]
mod csr_sweep;

use vyre_primitives::graph::persistent_bfs;

fn oracle_persistent(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
    max_iters: u32,
) -> (Vec<u32>, u32) {
    let words = bitset_words(node_count);
    let mut accum = frontier_in.to_vec();
    accum.resize(words, 0);
    let mut changed = 0u32;
    for _ in 0..max_iters {
        let step = csr_sweep::oracle_forward_step(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            &accum,
            allow_mask,
        );
        let mut step_changed = false;
        for wi in 0..words {
            let before = accum[wi];
            accum[wi] |= step[wi];
            if accum[wi] != before {
                step_changed = true;
            }
        }
        if step_changed {
            changed = 1;
        } else {
            break;
        }
    }
    (accum, changed)
}

const CASES: usize = 16384;

#[test]
fn sweep_graph_persistent_bfs_volume_oracle_matrix() {
    for (case, node_count, offsets, targets, masks, frontier, allow_mask) in csr_sweep::tuples(
        "single_source_all_kinds",
        CASES as u64,
        0xBF5FEFE57,
        0x9E37_79B9_7F4A_7C15,
    ) {
        let max_iters = 1 + (case % 8) as u32;
        let expected = oracle_persistent(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask, max_iters,
        );
        let actual = persistent_bfs::cpu_ref(
            CsrClosureInputs {
                graph: CsrGraphView {
                    node_count: node_count,
                    edge_offsets: &offsets,
                    edge_targets: &targets,
                    edge_kind_mask: &masks,
                },
                allow_mask: allow_mask,
                max_iters: max_iters,
            },
            &frontier,
        );
        assert_eq!(actual, expected, "Fix: persistent_bfs volume case {case}");
    }
}
