use super::super::*;
use crate::dispatch_buffers::u32_slice_to_le_bytes;
use crate::test_parity_oracles::{NeverDispatches, StaticOutputs};
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};
use vyre_primitives::graph::csr_closure_inputs::{graphs, CsrClosureInputs, CsrGraphView};

const BFS_CONTRACT: &str = "persistent BFS expand dispatch";

/// Expands one persistent-BFS step over [`graphs::CHAIN_4`] with every edge kind allowed,
/// returning the changed and converged flags. The contracts below vary the dispatcher, the seed
/// and the iteration budget; the graph itself is incidental to them.
fn linear_expand_into(
    dispatcher: &dyn ProgramDispatcher,
    seed: &[u32],
    max_iters: u32,
    frontier: &mut Vec<u32>,
) -> Result<(u32, u32), DispatchError> {
    bfs_expand_via_into(
        dispatcher,
        CsrClosureInputs::allow_all(graphs::CHAIN_4, max_iters),
        seed,
        frontier,
    )
}

/// [`linear_expand_into`] through caller-owned scratch.
fn linear_expand_with_scratch(
    dispatcher: &dyn ProgramDispatcher,
    seed: &[u32],
    max_iters: u32,
    scratch: &mut PersistentBfsGpuScratch,
    frontier: &mut Vec<u32>,
) -> Result<(u32, u32), DispatchError> {
    bfs_expand_via_with_scratch_into(
        dispatcher,
        CsrClosureInputs::allow_all(graphs::CHAIN_4, max_iters),
        seed,
        scratch,
        frontier,
    )
}

/// [`linear_expand_into`] returning owned frontier storage.
fn linear_expand(
    dispatcher: &dyn ProgramDispatcher,
    seed: &[u32],
    max_iters: u32,
) -> Result<(Vec<u32>, u32, u32), DispatchError> {
    bfs_expand_via(
        dispatcher,
        CsrClosureInputs::allow_all(graphs::CHAIN_4, max_iters),
        seed,
    )
}

#[test]
fn via_into_decodes_exact_outputs_into_reused_frontier() {
    let dispatcher = StaticOutputs::new(
        BFS_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0b1111]),
            u32_slice_to_le_bytes(&[1]),
            u32_slice_to_le_bytes(&[1]),
        ],
    )
    .expecting_grid([1, 1, 1])
    .expecting_inputs(&[9]);
    let mut frontier = Vec::with_capacity(4);
    let ptr = frontier.as_ptr();
    let (changed, converged) = linear_expand_into(&dispatcher, &[0b0001], 4, &mut frontier)
        .expect("Fix: dispatch succeeds");
    assert_eq!(frontier, vec![0b1111]);
    assert_eq!(changed, 1);
    assert_eq!(converged, 1);
    assert_eq!(frontier.as_ptr(), ptr);
}

#[test]
fn via_into_rejects_non_boolean_changed_flag_readback() {
    let dispatcher = StaticOutputs::new(
        BFS_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0b1111]),
            u32_slice_to_le_bytes(&[7]),
            u32_slice_to_le_bytes(&[1]),
        ],
    )
    .expecting_grid([1, 1, 1])
    .expecting_inputs(&[9]);
    let mut frontier = vec![0xDEAD_BEEF];
    let capacity = frontier.capacity();

    let err = linear_expand_into(&dispatcher, &[0b0001], 4, &mut frontier)
        .expect_err("Fix: persistent BFS wrapper must reject malformed changed-flag readback");

    assert!(
        matches!(err, DispatchError::BackendError(_)),
        "unexpected error variant: {err:?}"
    );
    assert_eq!(
        frontier,
        vec![0b1111],
        "frontier readback remains available for diagnostics even when the scalar flag is malformed"
    );
    assert_eq!(frontier.capacity(), capacity);
}

#[test]
fn via_into_rejects_non_boolean_converged_flag_readback() {
    let dispatcher = StaticOutputs::new(
        BFS_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0b1111]),
            u32_slice_to_le_bytes(&[1]),
            u32_slice_to_le_bytes(&[7]),
        ],
    )
    .expecting_grid([1, 1, 1])
    .expecting_inputs(&[9]);
    let mut frontier = Vec::new();

    let err = linear_expand_into(&dispatcher, &[0b0001], 4, &mut frontier).expect_err(
        "Fix: persistent BFS wrapper must reject a non-boolean converged-flag readback",
    );

    assert!(
        matches!(err, DispatchError::BackendError(_)),
        "unexpected error variant: {err:?}"
    );
    assert!(
        err.to_string().contains("converged"),
        "the diagnostic must name the converged signal, got: {err}"
    );
}

#[test]
fn via_large_graph_allocates_changed_active_scratch_without_extra_outputs() {
    let node_count = 513u32;
    let words = ((node_count + 31) / 32) as usize;
    let dispatcher = StaticOutputs::new(
        BFS_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&vec![0u32; words]),
            u32_slice_to_le_bytes(&[0, 0, 0]),
            u32_slice_to_le_bytes(&[1]),
        ],
    )
    .expecting_grid([3, 1, 1])
    .expecting_inputs(&[9])
    .expecting_input_bytes(7, 12);
    let edge_offsets = vec![0u32; node_count as usize + 1];
    let frontier_in = vec![0u32; words];
    let mut frontier = Vec::new();

    let (changed, converged) = bfs_expand_via_into(
        &dispatcher,
        CsrClosureInputs::allow_all(
            CsrGraphView {
                node_count: node_count,
                edge_offsets: &edge_offsets,
                edge_targets: &[],
                edge_kind_mask: &[],
            },
            64,
        ),
        &frontier_in,
        &mut frontier,
    )
    .expect("Fix: large persistent BFS dispatch should allocate internal active scratch.");

    assert_eq!(changed, 0);
    assert_eq!(converged, 1);
    assert_eq!(frontier, vec![0u32; words]);
}

#[test]
fn via_with_scratch_reuses_dispatch_storage() {
    let dispatcher = StaticOutputs::new(
        BFS_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0b1111]),
            u32_slice_to_le_bytes(&[1]),
            u32_slice_to_le_bytes(&[1]),
        ],
    )
    .expecting_grid([1, 1, 1])
    .expecting_inputs(&[9]);
    let mut scratch = PersistentBfsGpuScratch::default();
    let mut frontier = Vec::with_capacity(1);

    let (changed, converged) =
        linear_expand_with_scratch(&dispatcher, &[0b0001], 4, &mut scratch, &mut frontier)
            .expect("Fix: dispatch succeeds");
    assert_eq!(changed, 1);
    assert_eq!(converged, 1);
    assert_eq!(frontier, vec![0b1111]);
    let input_capacities = scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>();
    let frontier_capacity = frontier.capacity();

    let (changed, converged) =
        linear_expand_with_scratch(&dispatcher, &[0b0011], 4, &mut scratch, &mut frontier)
            .expect("Fix: dispatch succeeds");
    assert_eq!(changed, 1);
    assert_eq!(converged, 1);
    assert_eq!(
        scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>(),
        input_capacities
    );
    assert_eq!(frontier.capacity(), frontier_capacity);
}

#[test]
fn via_refreshes_static_graph_inputs_for_same_shape_content_change() {
    let dispatcher = StaticOutputs::new(
        BFS_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0b1111]),
            u32_slice_to_le_bytes(&[1]),
            u32_slice_to_le_bytes(&[1]),
        ],
    )
    .expecting_grid([1, 1, 1])
    .expecting_inputs(&[9])
    .recording_input(2);
    let edge_offsets = vec![0, 1, 2, 3, 3];
    let first_targets = vec![1, 2, 3];
    let second_targets = vec![2, 3, 0];
    let edge_kind_mask = vec![1, 1, 1];
    let mut scratch = PersistentBfsGpuScratch::default();
    let mut frontier = Vec::new();

    for (edge_targets, why) in [
        (
            &first_targets,
            "Fix: first same-shape persistent BFS dispatch should succeed",
        ),
        (
            &second_targets,
            "Fix: second same-shape persistent BFS dispatch should refresh graph inputs",
        ),
    ] {
        bfs_expand_via_with_scratch_into(
            &dispatcher,
            CsrClosureInputs::allow_all(
                CsrGraphView {
                    node_count: 4,
                    edge_offsets: &edge_offsets,
                    edge_targets: edge_targets,
                    edge_kind_mask: &edge_kind_mask,
                },
                4,
            ),
            &[0b0001],
            &mut scratch,
            &mut frontier,
        )
        .expect(why);
    }

    assert_eq!(
        dispatcher.recorded().as_slice(),
        &[first_targets, second_targets]
    );
    let snapshot = scratch.plan_cache.snapshot();
    assert_eq!(snapshot.entries, 1);
    assert_eq!(snapshot.misses, 1);
    assert_eq!(snapshot.hits, 1);
}

#[test]
fn via_zero_iters_validates_and_returns_seed_without_dispatch_or_cache() {
    let mut scratch = PersistentBfsGpuScratch::default();
    let mut frontier = Vec::with_capacity(8);
    let ptr = frontier.as_ptr();
    let (changed, converged) = linear_expand_with_scratch(
        &NeverDispatches("zero-iteration persistent BFS must not dispatch"),
        &[0b0011],
        0,
        &mut scratch,
        &mut frontier,
    )
    .expect("Fix: zero-iteration persistent BFS should validate and return seed frontier");

    assert_eq!(changed, 0);
    assert_eq!(
        converged, 0,
        "a zero-iteration expansion runs no confirming step, so it is not proven converged"
    );
    assert_eq!(frontier, vec![0b0011]);
    assert_eq!(frontier.as_ptr(), ptr);
    assert!(scratch.inputs.is_empty());
    assert_eq!(scratch.static_input_key, None);
}

#[test]
fn via_rejects_extra_outputs() {
    let dispatcher = StaticOutputs::new(
        BFS_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0b1111]),
            u32_slice_to_le_bytes(&[1]),
            u32_slice_to_le_bytes(&[1]),
            u32_slice_to_le_bytes(&[99]),
        ],
    )
    .expecting_grid([1, 1, 1])
    .expecting_inputs(&[9]);
    let err = linear_expand(&dispatcher, &[0b0001], 4).expect_err("extra outputs must be rejected");
    assert!(
        matches!(err, DispatchError::BackendError(_)),
        "unexpected error: {err:?}"
    );
}

#[test]
fn via_rejects_trailing_changed_bytes() {
    let dispatcher = StaticOutputs::new(
        BFS_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0b1111]),
            vec![1, 0, 0, 0, 2],
            u32_slice_to_le_bytes(&[0]),
        ],
    )
    .expecting_grid([1, 1, 1])
    .expecting_inputs(&[9]);
    let err = linear_expand(&dispatcher, &[0b0001], 4)
        .expect_err("trailing changed bytes must be rejected");
    assert!(
        matches!(err, DispatchError::BackendError(_)),
        "unexpected error: {err:?}"
    );
}

#[test]
fn via_rejects_mismatched_edge_arrays() {
    let dispatcher = StaticOutputs::new(
        BFS_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0b1111]),
            u32_slice_to_le_bytes(&[1]),
        ],
    )
    .expecting_grid([1, 1, 1])
    .expecting_inputs(&[9]);
    let err = bfs_expand_via(
        &dispatcher,
        CsrClosureInputs::allow_all(
            CsrGraphView {
                node_count: 2,
                edge_offsets: &[0, 1, 1],
                edge_targets: &[1],
                edge_kind_mask: &[],
            },
            1,
        ),
        &[0b01],
    )
    .expect_err("mismatched edge arrays must be rejected");
    assert!(matches!(err, DispatchError::BadInputs(_)));
}
