use super::*;
use vyre_primitives::graph::csr_closure_inputs::{CsrClosureInputs, CsrGraphView};

/// Adversarial: closure on disjoint components must not bridge
/// across components. Seed in component A must not flag B.
#[test]
fn closure_does_not_bridge_disjoint_components() {
    // Two-component CSR: 0 -> 1, 2 -> 3 (disjoint).
    let off = vec![0, 1, 1, 2, 2];
    let tgt = vec![1, 3];
    let msk = vec![1, 1];
    let out = reference_bidirectional_closure(
        CsrClosureInputs {
            graph: CsrGraphView {
                node_count: 4,
                edge_offsets: &off,
                edge_targets: &tgt,
                edge_kind_mask: &msk,
            },
            allow_mask: 0xFFFF_FFFF,
            max_iters: 5,
        },
        &[0b0001],
    );
    // Reaches {0, 1} only.
    assert_eq!(out, vec![0b0011]);
}

/// Idempotence: running the step on a saturated bitset returns
/// the same bitset.
#[test]
fn closure_is_idempotent_at_fixpoint() {
    let (off, tgt, msk) = linear_graph();
    let saturated = vec![0b1111];
    let out = reference_bidirectional_step(4, &off, &tgt, &msk, &saturated, 0xFFFF_FFFF);
    // Bidirectional step from saturated set keeps everything set.
    assert_eq!(out, saturated);
}
