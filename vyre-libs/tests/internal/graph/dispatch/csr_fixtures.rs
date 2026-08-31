use crate::graph::csr_closure_inputs::{CsrClosureInputs, CsrGraphView};

pub(crate) fn linear_graph() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    // 0 -> 1 -> 2 -> 3
    (vec![0, 1, 2, 3, 3], vec![1, 2, 3], vec![1, 1, 1])
}

pub(crate) fn linear_inputs<'a>(
    off: &'a [u32],
    tgt: &'a [u32],
    msk: &'a [u32],
    allow_mask: u32,
    max_iters: u32,
) -> CsrClosureInputs<'a> {
    CsrClosureInputs {
        graph: CsrGraphView {
            node_count: 4,
            edge_offsets: off,
            edge_targets: tgt,
            edge_kind_mask: msk,
        },
        allow_mask,
        max_iters,
    }
}

pub(crate) fn linear_inputs_all<'a>(
    off: &'a [u32],
    tgt: &'a [u32],
    msk: &'a [u32],
    max_iters: u32,
) -> CsrClosureInputs<'a> {
    linear_inputs(off, tgt, msk, 0xFFFF_FFFF, max_iters)
}

pub(crate) struct SameShapeGraphChanges {
    pub(crate) edge_offsets: Vec<u32>,
    pub(crate) first_targets: Vec<u32>,
    pub(crate) second_targets: Vec<u32>,
    pub(crate) edge_kind_mask: Vec<u32>,
}

impl SameShapeGraphChanges {
    pub(crate) fn new() -> Self {
        Self {
            edge_offsets: vec![0, 1, 2, 3, 3],
            first_targets: vec![1, 2, 3],
            second_targets: vec![2, 3, 0],
            edge_kind_mask: vec![1, 1, 1],
        }
    }

    pub(crate) fn assert_refreshed_targets(&self, recorded: &[Vec<u32>]) {
        assert_eq!(
            recorded,
            &[
                self.first_targets.as_slice(),
                self.second_targets.as_slice()
            ]
        );
    }
}

/// 2-node graph with single edge 0 -> 1 of kind bit 1 (0b0010).
pub(crate) fn single_edge_kind_bit1_graph() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    (vec![0, 1, 1], vec![1], vec![0b0010])
}

/// 2-node graph with self-loop 0 -> 0 of kind 1 and isolated node 1.
pub(crate) fn self_loop_isolated_node_graph() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    (vec![0, 1, 1], vec![0], vec![1])
}

/// 4-node two-component disjoint graph: 0 -> 1, 2 -> 3.
pub(crate) fn two_component_disjoint_graph() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    (vec![0, 1, 1, 2, 2], vec![1, 3], vec![1, 1])
}
