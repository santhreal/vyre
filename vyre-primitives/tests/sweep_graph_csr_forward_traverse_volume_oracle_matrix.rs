//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
#![forbid(unsafe_code)]
#![cfg(all(feature = "graph", feature = "cpu-parity"))]
mod graph_sweep_support;
use graph_sweep_support::{bitset_words, generated_csr_frontier};

use vyre_primitives::graph::csr_forward_traverse;

fn oracle_csr_forward_step(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    let words = bitset_words(node_count);
    let mut out = vec![0u32; words];
    for src in 0..node_count {
        let word_idx = (src / 32) as usize;
        let bit_mask = 1u32 << (src % 32);
        if word_idx >= frontier_in.len() || (frontier_in[word_idx] & bit_mask) == 0 {
            continue;
        }
        let edge_start = edge_offsets[src as usize] as usize;
        let edge_end = edge_offsets[src as usize + 1] as usize;
        for e in edge_start..edge_end {
            if (edge_kind_mask[e] & allow_mask) == 0 {
                continue;
            }
            let dst = edge_targets[e];
            if dst < node_count {
                let dst_word = (dst / 32) as usize;
                let dst_bit = 1u32 << (dst % 32);
                out[dst_word] |= dst_bit;
            }
        }
    }
    out
}

const CASES: usize = 16384;

#[test]
fn sweep_graph_csr_forward_traverse_volume_oracle_matrix() {
    for case in 0..CASES {
        let seed = case as u64 ^ 0xF074D4D01;
        let (node_count, offsets, targets, masks, frontier, allow_mask) =
            generated_csr_frontier(seed);
        let expected = oracle_csr_forward_step(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask,
        );
        let actual = csr_forward_traverse::cpu_ref(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask,
        );
        assert_eq!(
            actual, expected,
            "Fix: csr_forward_traverse volume case {case} node_count={node_count}"
        );
    }
}
