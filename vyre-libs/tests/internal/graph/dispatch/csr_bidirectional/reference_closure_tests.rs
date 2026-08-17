use super::{
    reference_bidirectional_closure, reference_bidirectional_closure_into,
    reference_bidirectional_step,
};
use crate::graph::csr_closure_inputs::graphs;
use crate::graph::csr_closure_inputs::{CsrClosureInputs, CsrGraphView};

/// Adversarial: closure on disjoint components must not bridge
/// across components. Seed in component A must not flag B.
#[test]
fn closure_does_not_bridge_disjoint_components() {
    // Two-component CSR: 0 -> 1, 2 -> 3 (disjoint).
    let off = vec![0, 1, 1, 2, 2];
    let tgt = vec![1, 3];
    let msk = vec![1, 1];
    let out = reference_bidirectional_closure(
        CsrClosureInputs::allow_all(
            CsrGraphView {
                node_count: 4,
                edge_offsets: &off,
                edge_targets: &tgt,
                edge_kind_mask: &msk,
            },
            5,
        ),
        &[0b0001],
    );
    // Reaches {0, 1} only.
    assert_eq!(out, vec![0b0011]);
}

/// Idempotence: running the step on a saturated bitset returns
/// the same bitset.
#[test]
fn closure_is_idempotent_at_fixpoint() {
    let g = graphs::CHAIN_4;
    let saturated = vec![0b1111];
    let out = reference_bidirectional_step(
        g.node_count,
        g.edge_offsets,
        g.edge_targets,
        g.edge_kind_mask,
        &saturated,
        u32::MAX,
    );
    // Bidirectional step from saturated set keeps everything set.
    assert_eq!(out, saturated);
}

/// Caller-owned scratch reuse: pointer and capacity preserved through closure iterations.
#[test]
fn closure_into_reuses_caller_scratch_buffers() {
    let g = graphs::CHAIN_4;
    let mut current = Vec::with_capacity(16);
    let mut next = Vec::with_capacity(16);
    let cur_ptr = current.as_ptr();
    let nxt_ptr = next.as_ptr();

    reference_bidirectional_closure_into(
        CsrClosureInputs::allow_all(g, 10),
        &[0b0001],
        &mut current,
        &mut next,
    );

    assert_eq!(current, vec![0b1111]);
    assert_eq!(current.as_ptr(), cur_ptr, "current buffer capacity reused");
    assert_eq!(
        next.as_ptr(),
        nxt_ptr,
        "next scratch buffer capacity reused"
    );
}
