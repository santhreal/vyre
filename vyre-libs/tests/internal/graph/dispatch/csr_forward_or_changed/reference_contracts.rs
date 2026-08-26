use super::*;
use crate::graph::csr_closure_inputs::{CsrClosureInputs, CsrGraphView};
use vyre_reference::composition_witness::{
    csr_forward_or_changed_closure_witness,
    csr_forward_or_changed_witness as reference_forward_step_with_change_flag,
};

fn reference_forward_closure_via_change_flag(
    inputs: CsrClosureInputs<'_>,
    seed: &[u32],
) -> Vec<u32> {
    let graph = inputs.graph;
    csr_forward_or_changed_closure_witness(
        graph.node_count,
        graph.edge_offsets,
        graph.edge_targets,
        graph.edge_kind_mask,
        seed,
        inputs.allow_mask,
        inputs.max_iters,
    )
}

#[test]
fn step_flips_change_flag_when_new_bits_added() {
    let (off, tgt, msk) = linear_graph();
    let (out, changed) =
        reference_forward_step_with_change_flag(4, &off, &tgt, &msk, &[0b0001], 0xFFFF_FFFF);
    // Seed {0} expands to {0, 1}. New bit added -> flag = 1.
    assert!(out[0] & 0b0010 != 0, "1 must be in expanded frontier");
    assert_eq!(changed, 1, "change flag must flip on new bit");
}

#[test]
fn step_clears_change_flag_at_fixpoint() {
    let (off, tgt, msk) = linear_graph();
    // Saturated frontier: every node already set.
    let (_out, changed) =
        reference_forward_step_with_change_flag(4, &off, &tgt, &msk, &[0b1111], 0xFFFF_FFFF);
    assert_eq!(changed, 0, "no new bits -> flag stays 0");
}

/// forward_closure_via_change_flag terminates at fixpoint and
/// returns the full forward closure. On a chain 0->1->2->3
/// from {0} final = {0,1,2,3}.
#[test]
fn closure_reaches_full_chain_via_change_flag() {
    let (off, tgt, msk) = linear_graph();
    let out = reference_forward_closure_via_change_flag(
        linear_inputs_all(&off, &tgt, &msk, 10),
        &[0b0001],
    );
    assert_eq!(out, vec![0b1111]);
}

/// Adversarial: empty seed must yield empty closure with flag 0
/// on the first iteration (no work).
#[test]
fn empty_seed_yields_empty_closure_no_change() {
    let (off, tgt, msk) = linear_graph();
    let (out, changed) =
        reference_forward_step_with_change_flag(4, &off, &tgt, &msk, &[0u32], 0xFFFF_FFFF);
    assert_eq!(out, vec![0u32]);
    assert_eq!(changed, 0);
}

/// Adversarial: closure must terminate before max_iters even on
/// a graph with a self-loop (the change flag is the only
/// termination signal we trust).
#[test]
fn closure_terminates_with_self_loop_under_max_iters() {
    let (off, tgt, msk) = self_loop_isolated_node_graph();
    let out = reference_forward_closure_via_change_flag(
        CsrClosureInputs::allow_all(CsrGraphView::new(2, &off, &tgt, &msk), 50),
        &[0b01],
    );
    // Self-loop never adds new bits -> terminates immediately.
    assert_eq!(out, vec![0b01]);
}

/// Adversarial: allow_mask filtering. Edges of the wrong kind
/// must not propagate; the change flag must register no change.
#[test]
fn allow_mask_filters_step() {
    let (off, tgt, msk) = single_edge_kind_bit1_graph();
    let (out, changed) = reference_forward_step_with_change_flag(
        2,
        &off,
        &tgt,
        &msk,
        &[0b01],
        0b0001, // demand kind 0
    );
    // No matching edges -> frontier unchanged from seed, no change.
    assert_eq!(out[0] & 0b10, 0);
    assert_eq!(changed, 0);
}
