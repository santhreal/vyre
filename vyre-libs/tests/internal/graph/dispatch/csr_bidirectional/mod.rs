use super::*;
use crate::dispatch_buffers::u32_slice_to_le_bytes;
fn reference_bidirectional_step(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    vyre_reference::composition_witness::csr_bidirectional_step_witness(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_in,
        allow_mask,
    )
}

use reference_bidirectional_step as reference_csr_bidir;

fn reference_bidirectional_closure_into(
    inputs: CsrClosureInputs<'_>,
    seed: &[u32],
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
) {
    let graph = inputs.graph;
    vyre_reference::composition_witness::csr_bidirectional_closure_witness_into(
        graph.node_count,
        graph.edge_offsets,
        graph.edge_targets,
        graph.edge_kind_mask,
        seed,
        inputs.allow_mask,
        inputs.max_iters,
        current,
        next,
    );
}

fn reference_bidirectional_closure(inputs: CsrClosureInputs<'_>, seed: &[u32]) -> Vec<u32> {
    let graph = inputs.graph;
    vyre_reference::composition_witness::csr_bidirectional_closure_witness(
        graph.node_count,
        graph.edge_offsets,
        graph.edge_targets,
        graph.edge_kind_mask,
        seed,
        inputs.allow_mask,
        inputs.max_iters,
    )
}

use crate::graph::csr_closure_inputs::{CsrClosureInputs, CsrGraphView};
use crate::test_parity_oracles::{NeverDispatches, StaticOutputs};
use reference_bidirectional_closure as reference_csr_bidir_closure;
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

#[path = "../csr_fixtures.rs"]
mod csr_fixtures;
use csr_fixtures::*;

mod reference_closure_tests;

/// 4 nodes dispatched at CSR_FRONTIER_STEP_WORKGROUP_SIZE (256 threads/group):
/// ceil(4/256) == 1 workgroup, NOT 4. The grid was corrected from a 256x
/// over-dispatch in plan_csr_bidirectional_step (vyre-primitives
/// csr-bidir-grid-miscompile), so every contract below asserts 1 workgroup.
const BIDIR_CONTRACT: &str = "bidirectional step dispatch";

/// Runs one bidirectional step over [`linear_graph`] into caller-owned storage. The contracts below
/// vary the dispatcher and the seed; the graph itself is incidental to them.
fn linear_step_into(
    dispatcher: &dyn ProgramDispatcher,
    seed: &[u32],
    out: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let (off, tgt, msk) = linear_graph();
    bidirectional_step_via_into(dispatcher, 4, &off, &tgt, &msk, seed, 0xFFFF_FFFF, out)
}

/// [`linear_step_into`] returning owned storage.
fn linear_step(
    dispatcher: &dyn ProgramDispatcher,
    seed: &[u32],
) -> Result<Vec<u32>, DispatchError> {
    let (off, tgt, msk) = linear_graph();
    bidirectional_step_via(dispatcher, 4, &off, &tgt, &msk, seed, 0xFFFF_FFFF)
}

/// [`linear_step_into`] through caller-owned scratch, with the allow mask exposed because it
/// participates in the cached program key.
fn linear_step_with_scratch(
    dispatcher: &dyn ProgramDispatcher,
    seed: &[u32],
    allow_mask: u32,
    scratch: &mut BidirectionalGpuScratch,
    out: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let (off, tgt, msk) = linear_graph();
    bidirectional_step_via_with_scratch_into(
        dispatcher, 4, &off, &tgt, &msk, seed, allow_mask, scratch, out,
    )
}

/// Iterated [`linear_step_with_scratch`] over caller-owned frontier buffers.
fn linear_closure_with_scratch(
    dispatcher: &dyn ProgramDispatcher,
    seed: &[u32],
    max_iters: u32,
    scratch: &mut BidirectionalGpuScratch,
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let (off, tgt, msk) = linear_graph();
    let inputs = linear_inputs_all(&off, &tgt, &msk, max_iters);
    bidirectional_closure_via_with_scratch_into(dispatcher, inputs, seed, scratch, current, next)
}

#[test]
fn step_includes_forward_and_backward_neighbors() {
    let (off, tgt, msk) = linear_graph();
    // Seed = {1}. Forward = {2}, backward = {0}. Union ⊇ {0, 2}.
    let out = reference_bidirectional_step(4, &off, &tgt, &msk, &[0b0010], 0xFFFF_FFFF);
    assert!(out[0] & 0b0001 != 0, "0 should be in backward step from 1");
    assert!(out[0] & 0b0100 != 0, "2 should be in forward step from 1");
}

#[test]
fn empty_seed_yields_empty_step() {
    let (off, tgt, msk) = linear_graph();
    let out = reference_bidirectional_step(4, &off, &tgt, &msk, &[0u32], 0xFFFF_FFFF);
    assert_eq!(out, vec![0u32]);
}

/// Closure-bar: substrate call equals direct primitive call.
#[test]
fn matches_primitive_directly() {
    let (off, tgt, msk) = linear_graph();
    let seed = vec![0b0010];
    let via_substrate = reference_bidirectional_step(4, &off, &tgt, &msk, &seed, 0xFFFF_FFFF);
    let via_primitive = reference_csr_bidir(4, &off, &tgt, &msk, &seed, 0xFFFF_FFFF);
    assert_eq!(via_substrate, via_primitive);
}

/// Adversarial: kind-mask filter must reject edges whose kinds
/// don't intersect `allow_mask`. The bidirectional step is a
/// pure successor/predecessor union; with no matching edges,
/// no neighbors are flagged (the primitive does not retain
/// the seed in its output).
#[test]
fn allow_mask_filters_out_wrong_edge_kinds() {
    let (off, tgt, msk) = single_edge_kind_bit1_graph();
    let out = reference_bidirectional_step(2, &off, &tgt, &msk, &[0b01], 0b0001);
    let direct = reference_csr_bidir(2, &off, &tgt, &msk, &[0b01], 0b0001);
    // Substrate output must match primitive directly.
    assert_eq!(out, direct);
    // And bit 1 (would-be neighbor via a kind-0 edge that doesn't
    // exist) must NOT be set in the result.
    assert_eq!(out[0] & 0b10, 0);
}

/// bidirectional_closure on a linear chain {0 -> 1 -> 2 -> 3} with
/// seed {0} must reach every node within 3 iterations.
#[test]
fn closure_reaches_full_chain() {
    let (off, tgt, msk) = linear_graph();
    let out = reference_bidirectional_closure(linear_inputs_all(&off, &tgt, &msk, 5), &[0b0001]);
    assert_eq!(out, vec![0b1111]);
}

#[test]
fn closure_into_matches_owned_closure() {
    let (off, tgt, msk) = linear_graph();
    let inputs = linear_inputs_all(&off, &tgt, &msk, 5);
    let owned = reference_bidirectional_closure(inputs, &[0b0001]);
    let mut current = Vec::new();
    let mut next = Vec::new();
    let inputs_into = linear_inputs_all(&off, &tgt, &msk, 5);
    reference_bidirectional_closure_into(inputs_into, &[0b0001], &mut current, &mut next);
    assert_eq!(current, owned);
}

#[test]
fn closure_matches_primitive_directly() {
    let (off, tgt, msk) = linear_graph();
    let seed = [0b0001];
    let inputs = linear_inputs_all(&off, &tgt, &msk, 5);
    let via_substrate = reference_bidirectional_closure(inputs, &seed);
    let via_primitive = reference_csr_bidir_closure(inputs, &seed);
    assert_eq!(via_substrate, via_primitive);
}

#[test]
fn via_step_decodes_exact_output_into_reused_buffer() {
    let dispatcher = StaticOutputs::new(BIDIR_CONTRACT, vec![u32_slice_to_le_bytes(&[0b1010])])
        .expecting_grid([1, 1, 1])
        .expecting_inputs(&[7]);
    let mut out = Vec::with_capacity(4);
    let ptr = out.as_ptr();
    linear_step_into(&dispatcher, &[0b0010], &mut out).expect("Fix: dispatch succeeds");
    assert_eq!(out, vec![0b1010]);
    assert_eq!(out.as_ptr(), ptr);
}

#[test]
fn via_step_with_scratch_reuses_dispatch_storage() {
    let dispatcher = StaticOutputs::new(BIDIR_CONTRACT, vec![u32_slice_to_le_bytes(&[0b1010])])
        .expecting_grid([1, 1, 1])
        .expecting_inputs(&[7]);
    let mut scratch = BidirectionalGpuScratch::default();
    let mut out = Vec::with_capacity(1);

    linear_step_with_scratch(&dispatcher, &[0b0010], 0xFFFF_FFFF, &mut scratch, &mut out)
        .expect("Fix: dispatch succeeds");
    assert_eq!(out, vec![0b1010]);
    let input_capacities = scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>();
    let out_capacity = out.capacity();
    assert_eq!(scratch.program_builds(), 1);

    linear_step_with_scratch(&dispatcher, &[0b0100], 0xFFFF_FFFF, &mut scratch, &mut out)
        .expect("Fix: dispatch succeeds");
    assert_eq!(
        scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>(),
        input_capacities
    );
    assert_eq!(out.capacity(), out_capacity);
    assert_eq!(out, vec![0b1010]);
    assert_eq!(scratch.program_builds(), 1);

    linear_step_with_scratch(&dispatcher, &[0b0100], 0x0000_0001, &mut scratch, &mut out)
        .expect("Fix: changed allow_mask should dispatch");
    assert_eq!(scratch.program_builds(), 2);
}

#[test]
fn via_step_refreshes_static_inputs_when_same_shape_graph_content_changes() {
    let dispatcher = StaticOutputs::new(BIDIR_CONTRACT, vec![u32_slice_to_le_bytes(&[0b1010])])
        .recording_input(2);
    let fixture = SameShapeGraphChanges::new();
    let mut scratch = BidirectionalGpuScratch::default();
    let mut out = Vec::new();

    for (edge_targets, why) in [
        (
            &fixture.first_targets,
            "Fix: first same-shape bidirectional dispatch should succeed",
        ),
        (
            &fixture.second_targets,
            "Fix: second same-shape bidirectional dispatch should refresh static CSR inputs",
        ),
    ] {
        bidirectional_step_via_with_scratch_into(
            &dispatcher,
            4,
            &fixture.edge_offsets,
            edge_targets,
            &fixture.edge_kind_mask,
            &[0b0010],
            0xFFFF_FFFF,
            &mut scratch,
            &mut out,
        )
        .expect(why);
    }

    fixture.assert_refreshed_targets(&dispatcher.recorded());
    assert_eq!(
        scratch.program_builds(),
        1,
        "Fix: same-shape CSR content changes must refresh static inputs without rebuilding the program."
    );
}

#[test]
fn via_step_uses_bridge_zero_inputs_for_graph_scratch() {
    struct InspectingDispatcher;

    impl ProgramDispatcher for InspectingDispatcher {
        fn dispatch(
            &self,
            _program: &Program,
            inputs: &[Vec<u8>],
            grid_override: Option<[u32; 3]>,
        ) -> Result<Vec<Vec<u8>>, DispatchError> {
            // 4 nodes / 256 threads-per-group => ceil(4/256) == 1 workgroup (not 4):
            // the corrected grid from vyre-primitives csr-bidir-grid-miscompile.
            assert_eq!(grid_override, Some([1, 1, 1]));
            assert_eq!(inputs.len(), 7);
            assert_eq!(inputs[0], u32_slice_to_le_bytes(&[0, 0, 0, 0]));
            assert_eq!(inputs[4], u32_slice_to_le_bytes(&[0, 0, 0, 0]));
            assert_eq!(inputs[6], u32_slice_to_le_bytes(&[0]));
            Ok(vec![u32_slice_to_le_bytes(&[0b1010])])
        }
    }

    let out = linear_step(&InspectingDispatcher, &[0b0010]).expect("Fix: dispatch succeeds");

    assert_eq!(out, vec![0b1010]);
}

#[test]
fn via_step_rejects_extra_outputs() {
    let dispatcher = StaticOutputs::new(
        BIDIR_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0b1010]),
            u32_slice_to_le_bytes(&[0]),
        ],
    )
    .expecting_grid([1, 1, 1])
    .expecting_inputs(&[7]);
    let err = linear_step(&dispatcher, &[0b0010]).expect_err("extra outputs must be rejected");
    assert!(
        matches!(err, DispatchError::BackendError(_)),
        "unexpected error: {err:?}"
    );
}

#[test]
fn via_step_rejects_trailing_output_bytes() {
    let dispatcher = StaticOutputs::new(BIDIR_CONTRACT, vec![vec![0, 0, 0, 0, 1]])
        .expecting_grid([1, 1, 1])
        .expecting_inputs(&[7]);
    let err =
        linear_step(&dispatcher, &[0b0010]).expect_err("trailing output bytes must be rejected");
    assert!(
        matches!(err, DispatchError::BackendError(_)),
        "unexpected error: {err:?}"
    );
}

#[test]
fn via_step_rejects_mismatched_edge_arrays() {
    let dispatcher = StaticOutputs::new(BIDIR_CONTRACT, vec![u32_slice_to_le_bytes(&[0b1010])])
        .expecting_grid([1, 1, 1])
        .expecting_inputs(&[7]);
    let err = bidirectional_step_via(&dispatcher, 2, &[0, 1, 1], &[1], &[], &[0b01], 0xFFFF_FFFF)
        .expect_err("mismatched edge arrays must be rejected");
    assert!(matches!(err, DispatchError::BadInputs(_)));
}

#[test]
fn via_step_empty_graph_is_validated_by_primitive_and_does_not_dispatch() {
    let mut out = vec![u32::MAX];
    bidirectional_step_via_into(
        &NeverDispatches("empty bidirectional graph must not dispatch"),
        0,
        &[0],
        &[],
        &[],
        &[],
        u32::MAX,
        &mut out,
    )
    .expect("Fix: canonical empty graph is valid");
    assert!(out.is_empty());
}

#[test]
fn closure_rejects_bad_seed_without_clobbering_reusable_buffers() {
    let mut scratch = BidirectionalGpuScratch::default();
    let mut current = vec![0xCAFE_BABE];
    let mut next = vec![0xDEAD_BEEF];

    let err = linear_closure_with_scratch(
        &NeverDispatches("malformed closure seed must be rejected before dispatch"),
        &[],
        5,
        &mut scratch,
        &mut current,
        &mut next,
    )
    .expect_err("bad seed width must be rejected before mutating reusable buffers");

    assert!(matches!(err, DispatchError::BadInputs(_)));
    assert_eq!(current, vec![0xCAFE_BABE]);
    assert_eq!(next, vec![0xDEAD_BEEF]);
}

#[test]
fn closure_zero_iters_validates_and_returns_seed_without_program_or_dispatch() {
    let mut scratch = BidirectionalGpuScratch::default();
    let mut current = Vec::with_capacity(8);
    let mut next = vec![0xDEAD_BEEF];

    linear_closure_with_scratch(
        &NeverDispatches("zero-iteration bidirectional closure must not dispatch"),
        &[0b0010],
        0,
        &mut scratch,
        &mut current,
        &mut next,
    )
    .expect("Fix: zero-iteration closure should still validate inputs");

    assert_eq!(current, vec![0b0010]);
    assert!(next.is_empty());
    assert_eq!(scratch.program_builds(), 0);
    assert!(scratch.inputs.is_empty());
}

#[test]
fn closure_empty_graph_validates_and_returns_empty_without_program_or_dispatch() {
    let mut scratch = BidirectionalGpuScratch::default();
    let mut current = vec![0xCAFE_BABE];
    let mut next = vec![0xDEAD_BEEF];

    bidirectional_closure_via_with_scratch_into(
        &NeverDispatches("empty bidirectional closure must not dispatch"),
        CsrClosureInputs::allow_all(
            CsrGraphView {
                node_count: 0,
                edge_offsets: &[0],
                edge_targets: &[],
                edge_kind_mask: &[],
            },
            4,
        ),
        &[],
        &mut scratch,
        &mut current,
        &mut next,
    )
    .expect("Fix: canonical empty closure should validate and short-circuit");

    assert!(current.is_empty());
    assert!(next.is_empty());
    assert_eq!(scratch.program_builds(), 0);
    assert!(scratch.inputs.is_empty());
}
