use super::*;
use crate::dispatch_buffers::u32_slice_to_le_bytes;
use crate::graph::dominator_frontier::{
    cpu_ref as reference_dominator_frontier, dominance_frontier, try_dominance_frontier,
};
use std::sync::Mutex;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

mod dispatcher_doubles;

use crate::test_parity_oracles::StaticOutputs;
use dispatcher_doubles::{DominatorInputShapeDispatcher, RecordingDominatorDispatcher};

const DOMINATOR_CONTRACT: &str = "dominance frontier dispatch";

#[test]
fn checked_reference_surfaces_bad_seed_width() {
    let err = try_dominance_frontier(2, &[0, 0, 0], &[], &[0, 0, 0], &[], &[])
        .expect_err("short dominance-frontier seed must fail through substrate wrapper");

    assert!(
        err.contains("seed"),
        "Fix: dominance-frontier checked wrapper must preserve primitive seed diagnostics, got: {err}"
    );
}

/// Linear chain 0 -> 1 -> 2 -> 3. Dominance closure: each node
/// dominates itself and every successor. Predecessors: each
/// non-zero node has the previous as its sole pred.
/// Seed = {0}. Expected frontier: empty (every dominator
/// strictly dominates the merge candidate).
#[test]
fn frontier_of_linear_chain_is_empty() {
    // dom CSR: row 0 = {0,1,2,3}; row 1 = {1,2,3}; row 2 = {2,3}; row 3 = {3}
    let dom_offsets = vec![0, 4, 7, 9, 10];
    let dom_targets = vec![0, 1, 2, 3, 1, 2, 3, 2, 3, 3];
    // pred CSR: row 0 = {}; row 1 = {0}; row 2 = {1}; row 3 = {2}
    let pred_offsets = vec![0, 0, 1, 2, 3];
    let pred_targets = vec![0, 1, 2];
    let seed = vec![0b0001];
    let frontier = dominance_frontier(
        4,
        &dom_offsets,
        &dom_targets,
        &pred_offsets,
        &pred_targets,
        &seed,
    );
    assert_eq!(frontier, vec![0u32]);
    assert_eq!(frontier_size(&frontier), 0);
}

/// Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3.
/// Dominators: 0 dominates {0,1,2,3}; 1,2 dominate themselves;
/// 3 dominates itself.
/// Seed = {1}: 1 dominates a predecessor of 3 (itself), but does
/// not strictly dominate 3 (0 does, not 1). So frontier = {3}.
#[test]
fn frontier_of_diamond_seed_is_merge_node() {
    // dom CSR: 0 -> {0,1,2,3}; 1 -> {1}; 2 -> {2}; 3 -> {3}
    let dom_offsets = vec![0, 4, 5, 6, 7];
    let dom_targets = vec![0, 1, 2, 3, 1, 2, 3];
    // pred CSR: 0 -> {}; 1 -> {0}; 2 -> {0}; 3 -> {1, 2}
    let pred_offsets = vec![0, 0, 1, 2, 4];
    let pred_targets = vec![0, 0, 1, 2];
    let seed = vec![0b0010]; // {1}
    let frontier = dominance_frontier(
        4,
        &dom_offsets,
        &dom_targets,
        &pred_offsets,
        &pred_targets,
        &seed,
    );
    // Expect node 3 in the frontier.
    assert_eq!(frontier, vec![0b1000]);
    assert_eq!(frontier_size(&frontier), 1);
}

/// Closure-bar: substrate consumer must produce the same bitset
/// as a direct primitive call. If the wiring drifts, this
/// fails before any downstream consumer sees stale frontiers.
#[test]
fn matches_primitive_directly() {
    let dom_offsets = vec![0, 4, 5, 6, 7];
    let dom_targets = vec![0, 1, 2, 3, 1, 2, 3];
    let pred_offsets = vec![0, 0, 1, 2, 4];
    let pred_targets = vec![0, 0, 1, 2];
    let seed = vec![0b0011];
    let via_substrate = dominance_frontier(
        4,
        &dom_offsets,
        &dom_targets,
        &pred_offsets,
        &pred_targets,
        &seed,
    );
    let via_primitive = reference_dominator_frontier(
        4,
        &dom_offsets,
        &dom_targets,
        &pred_offsets,
        &pred_targets,
        &seed,
    );
    assert_eq!(via_substrate, via_primitive);
}

/// Adversarial: empty seed must yield an empty frontier. A naive
/// implementation that ignores the seed bit and walks every
/// Region would mark the entire bitset.
#[test]
fn empty_seed_yields_empty_frontier() {
    let dom_offsets = vec![0, 4, 5, 6, 7];
    let dom_targets = vec![0, 1, 2, 3, 1, 2, 3];
    let pred_offsets = vec![0, 0, 1, 2, 4];
    let pred_targets = vec![0, 0, 1, 2];
    let seed = vec![0u32];
    let frontier = dominance_frontier(
        4,
        &dom_offsets,
        &dom_targets,
        &pred_offsets,
        &pred_targets,
        &seed,
    );
    assert_eq!(frontier, vec![0u32]);
    assert_eq!(frontier_size(&frontier), 0);
}

/// Adversarial: seed that strictly dominates the entire graph
/// must NOT include any node in its frontier (a node n is in
/// the frontier of seed s only if s does NOT strictly dominate
/// n).
#[test]
fn seed_dominating_everything_has_empty_frontier() {
    // 0 dominates {0,1,2,3}. Seed = {0} -> frontier should be {}.
    let dom_offsets = vec![0, 4, 5, 6, 7];
    let dom_targets = vec![0, 1, 2, 3, 1, 2, 3];
    let pred_offsets = vec![0, 0, 1, 2, 4];
    let pred_targets = vec![0, 0, 1, 2];
    let seed = vec![0b0001];
    let frontier = dominance_frontier(
        4,
        &dom_offsets,
        &dom_targets,
        &pred_offsets,
        &pred_targets,
        &seed,
    );
    assert_eq!(frontier, vec![0u32]);
}

/// frontier_size must return the popcount of the bitset.
#[test]
fn frontier_size_counts_set_bits() {
    assert_eq!(frontier_size(&[0u32]), 0);
    assert_eq!(frontier_size(&[0b1011u32]), 3);
    assert_eq!(frontier_size(&[0xFFFFFFFFu32, 0b1u32]), 33);
}

#[test]
fn via_decodes_exact_frontier_into_reused_buffer() {
    let dispatcher = StaticOutputs::new(DOMINATOR_CONTRACT, vec![u32_slice_to_le_bytes(&[0b1000])])
        .expecting_grid([1, 1, 1])
        .expecting_inputs(&[6]);
    let dom_offsets = vec![0, 4, 5, 6, 7];
    let dom_targets = vec![0, 1, 2, 3, 1, 2, 3];
    let pred_offsets = vec![0, 0, 1, 2, 4];
    let pred_targets = vec![0, 0, 1, 2];
    let mut out = Vec::with_capacity(4);
    let ptr = out.as_ptr();
    dominance_frontier_via_into(
        &dispatcher,
        4,
        &dom_offsets,
        &dom_targets,
        &pred_offsets,
        &pred_targets,
        &[0b0010],
        &mut out,
    )
    .expect("Fix: dispatch succeeds");
    assert_eq!(out, vec![0b1000]);
    assert_eq!(out.as_ptr(), ptr);
}

#[test]
fn via_with_scratch_reuses_dispatch_storage() {
    let dispatcher = StaticOutputs::new(DOMINATOR_CONTRACT, vec![u32_slice_to_le_bytes(&[0b1000])])
        .expecting_grid([1, 1, 1])
        .expecting_inputs(&[6]);
    let dom_offsets = vec![0, 4, 5, 6, 7];
    let dom_targets = vec![0, 1, 2, 3, 1, 2, 3];
    let pred_offsets = vec![0, 0, 1, 2, 4];
    let pred_targets = vec![0, 0, 1, 2];
    let mut scratch = DominanceFrontierGpuScratch::default();
    let mut out = Vec::with_capacity(1);

    dominance_frontier_via_with_scratch_into(
        &dispatcher,
        4,
        &dom_offsets,
        &dom_targets,
        &pred_offsets,
        &pred_targets,
        &[0b0010],
        &mut scratch,
        &mut out,
    )
    .expect("Fix: dispatch succeeds");
    assert_eq!(out, vec![0b1000]);
    let input_capacities = scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>();
    let out_capacity = out.capacity();
    assert_eq!(scratch.program_builds(), 1);

    dominance_frontier_via_with_scratch_into(
        &dispatcher,
        4,
        &dom_offsets,
        &dom_targets,
        &pred_offsets,
        &pred_targets,
        &[0b0011],
        &mut scratch,
        &mut out,
    )
    .expect("Fix: dispatch succeeds");
    assert_eq!(
        scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>(),
        input_capacities
    );
    assert_eq!(out.capacity(), out_capacity);
    assert_eq!(out, vec![0b1000]);
    assert_eq!(scratch.program_builds(), 1);

    let shorter_dom_offsets = vec![0, 3, 4, 5, 6];
    let shorter_dom_targets = vec![0, 1, 2, 1, 2, 3];
    dominance_frontier_via_with_scratch_into(
        &dispatcher,
        4,
        &shorter_dom_offsets,
        &shorter_dom_targets,
        &pred_offsets,
        &pred_targets,
        &[0b0011],
        &mut scratch,
        &mut out,
    )
    .expect("Fix: changed dominance layout should dispatch");
    assert_eq!(scratch.program_builds(), 2);
}

#[test]
fn via_refreshes_static_graph_inputs_for_same_shape_content_change() {
    let dispatcher = RecordingDominatorDispatcher {
        calls: Mutex::new(Vec::new()),
        output: u32_slice_to_le_bytes(&[0b1000]),
    };
    let dom_offsets = vec![0, 4, 5, 6, 7];
    let dom_targets = vec![0, 1, 2, 3, 1, 2, 3];
    let changed_dom_targets = vec![0, 1, 2, 2, 1, 2, 3];
    let pred_offsets = vec![0, 0, 1, 2, 4];
    let pred_targets = vec![0, 0, 1, 2];
    let mut scratch = DominanceFrontierGpuScratch::default();
    let mut out = Vec::new();

    dominance_frontier_via_with_scratch_into(
        &dispatcher,
        4,
        &dom_offsets,
        &dom_targets,
        &pred_offsets,
        &pred_targets,
        &[0b0010],
        &mut scratch,
        &mut out,
    )
    .expect("Fix: first dominance frontier dispatch should succeed");
    dominance_frontier_via_with_scratch_into(
        &dispatcher,
        4,
        &dom_offsets,
        &changed_dom_targets,
        &pred_offsets,
        &pred_targets,
        &[0b0010],
        &mut scratch,
        &mut out,
    )
    .expect("Fix: same-shape dominance frontier content change should refresh inputs");

    let calls = dispatcher
        .calls
        .lock()
        .expect("Fix: recording dispatcher calls lock should not be poisoned");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0][1], u32_slice_to_le_bytes(&dom_targets));
    assert_eq!(calls[1][1], u32_slice_to_le_bytes(&changed_dom_targets));
    assert_eq!(scratch.program_builds(), 1);
}

#[test]
fn via_reuses_static_graph_inputs_and_refreshes_dynamic_seed() {
    let dispatcher = RecordingDominatorDispatcher {
        calls: Mutex::new(Vec::new()),
        output: u32_slice_to_le_bytes(&[0b1000]),
    };
    let dom_offsets = vec![0, 4, 5, 6, 7];
    let dom_targets = vec![0, 1, 2, 3, 1, 2, 3];
    let pred_offsets = vec![0, 0, 1, 2, 4];
    let pred_targets = vec![0, 0, 1, 2];
    let mut scratch = DominanceFrontierGpuScratch::default();
    let mut out = Vec::new();

    dominance_frontier_via_with_scratch_into(
        &dispatcher,
        4,
        &dom_offsets,
        &dom_targets,
        &pred_offsets,
        &pred_targets,
        &[0b0010],
        &mut scratch,
        &mut out,
    )
    .expect("Fix: first dominance frontier dispatch should succeed");
    let static_capacities = scratch
        .inputs
        .iter()
        .take(4)
        .map(Vec::capacity)
        .collect::<Vec<_>>();
    dominance_frontier_via_with_scratch_into(
        &dispatcher,
        4,
        &dom_offsets,
        &dom_targets,
        &pred_offsets,
        &pred_targets,
        &[0b0011],
        &mut scratch,
        &mut out,
    )
    .expect("Fix: same graph with changed seed should refresh dynamic input only");

    let calls = dispatcher
        .calls
        .lock()
        .expect("Fix: recording dispatcher calls lock should not be poisoned");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0][0], calls[1][0]);
    assert_eq!(calls[0][1], calls[1][1]);
    assert_eq!(calls[0][2], calls[1][2]);
    assert_eq!(calls[0][3], calls[1][3]);
    assert_eq!(calls[0][4], u32_slice_to_le_bytes(&[0b0010]));
    assert_eq!(calls[1][4], u32_slice_to_le_bytes(&[0b0011]));
    assert_eq!(
        scratch
            .inputs
            .iter()
            .take(4)
            .map(Vec::capacity)
            .collect::<Vec<_>>(),
        static_capacities
    );
    assert_eq!(scratch.program_builds(), 1);
}

#[test]
fn via_zero_edge_graph_uses_primitive_padding_plan() {
    let mut out = Vec::new();
    dominance_frontier_via_into(
        &DominatorInputShapeDispatcher,
        1,
        &[0, 0],
        &[],
        &[0, 0],
        &[],
        &[1],
        &mut out,
    )
    .expect("Fix: zero-edge dominance frontier dispatch should use padded target buffers");

    assert_eq!(out, vec![0]);
}

#[test]
fn via_rejects_extra_outputs() {
    let dispatcher = StaticOutputs::new(
        DOMINATOR_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0b1000]),
            u32_slice_to_le_bytes(&[0]),
        ],
    )
    .expecting_grid([1, 1, 1])
    .expecting_inputs(&[6]);
    let dom_offsets = vec![0, 4, 5, 6, 7];
    let dom_targets = vec![0, 1, 2, 3, 1, 2, 3];
    let pred_offsets = vec![0, 0, 1, 2, 4];
    let pred_targets = vec![0, 0, 1, 2];
    let err = dominance_frontier_via(
        &dispatcher,
        4,
        &dom_offsets,
        &dom_targets,
        &pred_offsets,
        &pred_targets,
        &[0b0010],
    )
    .expect_err("extra outputs must be rejected");
    assert!(
        matches!(err, DispatchError::BackendError(_)),
        "unexpected error: {err:?}"
    );
}

#[test]
fn via_rejects_trailing_frontier_bytes() {
    let dispatcher = StaticOutputs::new(DOMINATOR_CONTRACT, vec![vec![0, 0, 0, 0, 1]])
        .expecting_grid([1, 1, 1])
        .expecting_inputs(&[6]);
    let dom_offsets = vec![0, 4, 5, 6, 7];
    let dom_targets = vec![0, 1, 2, 3, 1, 2, 3];
    let pred_offsets = vec![0, 0, 1, 2, 4];
    let pred_targets = vec![0, 0, 1, 2];
    let err = dominance_frontier_via(
        &dispatcher,
        4,
        &dom_offsets,
        &dom_targets,
        &pred_offsets,
        &pred_targets,
        &[0b0010],
    )
    .expect_err("trailing frontier bytes must be rejected");
    assert!(
        matches!(err, DispatchError::BackendError(_)),
        "unexpected error: {err:?}"
    );
}
