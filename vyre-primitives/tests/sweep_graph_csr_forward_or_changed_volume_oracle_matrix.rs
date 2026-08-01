//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
#![forbid(unsafe_code)]
#![cfg(all(feature = "graph", feature = "cpu-parity"))]
mod graph_sweep_support;
use graph_sweep_support::{bitset_words, generated_csr_frontier};

use vyre_primitives::graph::csr_forward_or_changed;

/// Independent model of one `csr_forward_or_changed` pass.
///
/// The pass is IN-PLACE and OR-accumulating, which is what the name says: the
/// output starts as a copy of the input frontier and each frontier source ORs
/// its allowed neighbours into it. Because the scan walks sources in index
/// order over the same buffer it writes, a bit set by a lower-indexed source is
/// visible to a higher-indexed one, so a single pass can advance more than one
/// hop. That is deliberate: the frontier is monotone, extra bits only bring the
/// fixpoint closer, and every path in the family (serial, grid-sync, batched)
/// relies on it.
///
/// This oracle used to start from an all-zero output and read only
/// `frontier_in`, modelling a strict one-hop out-of-place expansion. That
/// contract belongs to no implementation here: it dropped the input frontier
/// from the result and forbade in-pass propagation, so the matrix failed on
/// case 0 with `left: [132, 605487104, 34816]` against `right: [0, 537919488,
/// 34816]`. It stays hand-written here rather than calling the production
/// helper, so it is still an independent check of the same contract.
fn oracle_csr_forward_step(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
) -> (Vec<u32>, u32) {
    let words = bitset_words(node_count);
    let mut out = frontier_in.to_vec();
    out.resize(words, 0);
    let mut changed = 0u32;
    for src in 0..node_count {
        let word_idx = (src / 32) as usize;
        let bit_mask = 1u32 << (src % 32);
        if (out[word_idx] & bit_mask) == 0 {
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
                if out[dst_word] & dst_bit == 0 {
                    out[dst_word] |= dst_bit;
                    changed = 1;
                }
            }
        }
    }
    (out, changed)
}

const CASES: usize = 16384;

#[test]
fn sweep_graph_csr_forward_or_changed_volume_oracle_matrix() {
    for case in 0..CASES {
        let seed = case as u64 ^ 0x07C8A465;
        let (node_count, offsets, targets, masks, frontier, allow_mask) =
            generated_csr_frontier(seed);
        let (step, oracle_changed) = oracle_csr_forward_step(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask,
        );
        let (actual_step, changed) = csr_forward_or_changed::cpu_ref(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask,
        );
        assert_eq!(
            actual_step, step,
            "Fix: forward_or_changed step case {case}"
        );
        // The flag reports whether the pass SET a bit, not whether the buffer
        // ended up different: those agree here only because the frontier is
        // monotone, and the flag is what drives fixpoint termination.
        assert_eq!(
            changed, oracle_changed,
            "Fix: forward_or_changed flag case {case}"
        );
    }
}
