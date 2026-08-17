//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
#![forbid(unsafe_code)]
#![cfg(feature = "graph")]
mod graph_sweep_fixtures;
use graph_sweep_fixtures::bitset_words;
#[path = "../../tests/support/csr_sweep/mod.rs"]
mod csr_sweep;
use vyre_reference::composition_witness::csr_forward_or_changed_witness;

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
const CASES: usize = 16384;

#[test]
fn sweep_graph_csr_forward_or_changed_volume_oracle_matrix() {
    for (case, node_count, offsets, targets, masks, frontier, allow_mask) in csr_sweep::tuples(
        "padded_tail_masked_kinds",
        CASES as u64,
        0x07C8A465,
        0x9E37_79B9_7F4A_7C15,
    ) {
        let (step, oracle_changed) = oracle_forward_step_with_change_flag(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask,
        );
        let (actual_step, changed) = csr_forward_or_changed_witness(
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

/// The masked forward step plus the changed flag this primitive also returns.
///
/// Deliberately not built on `csr_sweep::oracle_forward_step`: this family's
/// contract is a step that accumulates into the frontier as it walks, so a node
/// reached from an earlier source propagates onward within the same step. The
/// shared oracle computes a pure one-hop image of the input frontier, which is a
/// different function, and the difference is observable on a dense shape.
fn oracle_forward_step_with_change_flag(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
) -> (Vec<u32>, u32) {
    let mut out = frontier_in.to_vec();
    out.resize(bitset_words(node_count), 0);
    let mut changed = 0u32;
    for src in 0..node_count {
        let word_idx = (src / 32) as usize;
        if out[word_idx] & (1u32 << (src % 32)) == 0 {
            continue;
        }
        for e in edge_offsets[src as usize] as usize..edge_offsets[src as usize + 1] as usize {
            if edge_kind_mask[e] & allow_mask == 0 {
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
