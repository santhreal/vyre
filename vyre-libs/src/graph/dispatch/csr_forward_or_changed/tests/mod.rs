use super::*;
use crate::dispatch_buffers::u32_slice_to_le_bytes;
use crate::test_parity_oracles::{NeverDispatches, StaticOutputs};
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};
use vyre_primitives::graph::csr_closure_inputs::{CsrClosureInputs, CsrGraphView};

mod reference_contracts;

/// Seven inputs is the legacy shape; eight is the changed-history shape. Both
/// are live, so a dispatcher that accepts only one of them would reject a
/// correct plan.
const CSR_CHANGED_CONTRACT: &str = "csr_forward_or_changed dispatch";

fn linear_graph() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    // 0 -> 1 -> 2 -> 3
    (vec![0, 1, 2, 3, 3], vec![1, 2, 3], vec![1, 1, 1])
}

/// Runs the changed-flag closure over [`linear_graph`] seeded at node 0 with every edge kind
/// allowed. The contracts below vary the dispatcher and the iteration budget; the graph itself is
/// incidental to them.
fn linear_closure(
    dispatcher: &dyn ProgramDispatcher,
    max_iters: u32,
) -> Result<Vec<u32>, DispatchError> {
    let (off, tgt, msk) = linear_graph();
    forward_closure_via_change_flag_gpu(
        dispatcher,
        CsrClosureInputs {
            graph: CsrGraphView {
                node_count: 4,
                edge_offsets: &off,
                edge_targets: &tgt,
                edge_kind_mask: &msk,
            },
            allow_mask: 0xFFFF_FFFF,
            max_iters,
        },
        &[0b0001],
    )
}

/// [`linear_closure`] decoding into caller-owned frontier storage.
fn linear_closure_into(
    dispatcher: &dyn ProgramDispatcher,
    max_iters: u32,
    frontier: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let (off, tgt, msk) = linear_graph();
    forward_closure_via_change_flag_gpu_into(
        dispatcher,
        CsrClosureInputs {
            graph: CsrGraphView {
                node_count: 4,
                edge_offsets: &off,
                edge_targets: &tgt,
                edge_kind_mask: &msk,
            },
            allow_mask: 0xFFFF_FFFF,
            max_iters,
        },
        &[0b0001],
        frontier,
    )
}

/// [`linear_closure`] through caller-owned scratch, with the seed and allow mask exposed because
/// they participate in the cached program key.
fn linear_closure_with_scratch(
    dispatcher: &dyn ProgramDispatcher,
    seed: &[u32],
    allow_mask: u32,
    max_iters: u32,
    scratch: &mut ForwardChangedGpuScratch,
    frontier: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let (off, tgt, msk) = linear_graph();
    forward_closure_via_change_flag_gpu_with_scratch_into(
        dispatcher,
        CsrClosureInputs {
            graph: CsrGraphView {
                node_count: 4,
                edge_offsets: &off,
                edge_targets: &tgt,
                edge_kind_mask: &msk,
            },
            allow_mask,
            max_iters,
        },
        seed,
        scratch,
        frontier,
    )
}

#[test]
fn gpu_into_decodes_exact_outputs_into_reused_frontier() {
    let dispatcher = StaticOutputs::new(
        CSR_CHANGED_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0b1111]),
            u32_slice_to_le_bytes(&[0, 0, 0, 0]),
        ],
    )
    .expecting_inputs(&[7, 8]);
    let mut frontier = Vec::with_capacity(4);
    let ptr = frontier.as_ptr();
    linear_closure_into(&dispatcher, 4, &mut frontier).expect("Fix: dispatch succeeds");
    assert_eq!(frontier, vec![0b1111]);
    assert_eq!(frontier.as_ptr(), ptr);
}

#[test]
fn gpu_rejects_extra_outputs() {
    let dispatcher = StaticOutputs::new(
        CSR_CHANGED_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0b1111]),
            u32_slice_to_le_bytes(&[0, 0, 0, 0]),
            u32_slice_to_le_bytes(&[99]),
        ],
    )
    .expecting_inputs(&[7, 8]);
    let err = linear_closure(&dispatcher, 4).expect_err("extra outputs must be rejected");
    assert!(
        matches!(err, DispatchError::BackendError(_)),
        "unexpected error: {err:?}"
    );
}

#[test]
fn gpu_rejects_trailing_changed_bytes() {
    let dispatcher = StaticOutputs::new(
        CSR_CHANGED_CONTRACT,
        vec![u32_slice_to_le_bytes(&[0b1111]), vec![0, 0, 0, 0, 1]],
    )
    .expecting_inputs(&[7, 8]);
    let err = linear_closure(&dispatcher, 4).expect_err("trailing changed bytes must be rejected");
    assert!(
        matches!(err, DispatchError::BackendError(_)),
        "unexpected error: {err:?}"
    );
}

#[test]
fn gpu_rejects_non_boolean_changed_flag() {
    let dispatcher = StaticOutputs::new(
        CSR_CHANGED_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0b1111]),
            u32_slice_to_le_bytes(&[2]),
        ],
    )
    .expecting_inputs(&[7, 8]);
    let err =
        linear_closure(&dispatcher, 1).expect_err("non-boolean changed flag must be rejected");
    assert!(
        matches!(err, DispatchError::BackendError(_)),
        "unexpected error: {err:?}"
    );
}

#[test]
fn gpu_rejects_bad_seed_width_without_clobbering_frontier() {
    let mut scratch = ForwardChangedGpuScratch::default();
    let mut frontier = vec![0xCAFE_BABEu32];
    let capacity = frontier.capacity();

    let err = linear_closure_with_scratch(
        &NeverDispatches("bad seed width must be rejected before dispatch"),
        &[],
        0xFFFF_FFFF,
        5,
        &mut scratch,
        &mut frontier,
    )
    .expect_err("bad seed width must be rejected before mutating reusable frontier storage");

    assert!(matches!(err, DispatchError::BadInputs(_)));
    assert_eq!(frontier, vec![0xCAFE_BABEu32]);
    assert_eq!(frontier.capacity(), capacity);
    assert!(scratch.inputs.is_empty());
    assert_eq!(scratch.program_builds(), 0);
}

#[test]
fn gpu_reuses_dispatch_input_buffers() {
    let dispatcher = StaticOutputs::new(
        CSR_CHANGED_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0b1111]),
            u32_slice_to_le_bytes(&[0, 0, 0, 0]),
        ],
    )
    .expecting_inputs(&[7, 8]);
    let mut scratch =
        ForwardChangedGpuScratch::with_input_capacities(&[32, 32, 32, 32, 32, 32, 32, 8], 1);
    let mut frontier = Vec::with_capacity(4);
    let input_caps = scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>();
    let frontier_ptr = frontier.as_ptr();
    linear_closure_with_scratch(
        &dispatcher,
        &[0b0001],
        0xFFFF_FFFF,
        4,
        &mut scratch,
        &mut frontier,
    )
    .unwrap();
    assert_eq!(
        scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>(),
        input_caps
    );
    assert_eq!(frontier.as_ptr(), frontier_ptr);
    assert_eq!(frontier, vec![0b1111]);
}

#[test]
fn gpu_refreshes_static_inputs_when_same_shape_graph_content_changes() {
    let dispatcher = StaticOutputs::new(
        CSR_CHANGED_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0b0001]),
            u32_slice_to_le_bytes(&[0]),
        ],
    )
    .recording_input(2);
    let edge_offsets = vec![0, 1, 2, 3, 3];
    let first_targets = vec![1, 2, 3];
    let second_targets = vec![2, 3, 0];
    let edge_kind_mask = vec![1, 1, 1];
    let mut scratch = ForwardChangedGpuScratch::default();
    let mut frontier = Vec::new();

    for (edge_targets, why) in [
        (
            &first_targets,
            "Fix: first same-shape dispatch should succeed",
        ),
        (
            &second_targets,
            "Fix: second same-shape dispatch should refresh static CSR inputs",
        ),
    ] {
        forward_closure_via_change_flag_gpu_with_scratch_into(
            &dispatcher,
            CsrClosureInputs::allow_all(
                CsrGraphView {
                    node_count: 4,
                    edge_offsets: &edge_offsets,
                    edge_targets,
                    edge_kind_mask: &edge_kind_mask,
                },
                1,
            ),
            &[0b0001],
            &mut scratch,
            &mut frontier,
        )
        .expect(why);
    }

    let recorded_targets = dispatcher.recorded();
    assert_eq!(
        recorded_targets.as_slice(),
        &[first_targets, second_targets]
    );
    assert_eq!(
        scratch.program_builds(),
        1,
        "Fix: same-shape graph content changes should refresh staged static inputs without rebuilding the primitive program."
    );
}

#[test]
fn gpu_reuses_cached_program_by_primitive_key() {
    let history_dispatcher = StaticOutputs::new(
        CSR_CHANGED_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0b1111]),
            u32_slice_to_le_bytes(&[0, 0, 0, 0]),
        ],
    )
    .expecting_inputs(&[7, 8]);
    let legacy_dispatcher = StaticOutputs::new(
        CSR_CHANGED_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0b1111]),
            u32_slice_to_le_bytes(&[0]),
        ],
    )
    .expecting_inputs(&[7, 8]);
    let mut scratch = ForwardChangedGpuScratch::default();
    let mut frontier = Vec::new();

    // Only the seed width, the allow mask and the changed-history policy participate in the
    // primitive key, so the cumulative build count after each step is the contract.
    let steps: [(&dyn ProgramDispatcher, &[u32], u32, u32, usize, &str); 4] = [
        (
            &history_dispatcher,
            &[0b0001],
            0xFFFF_FFFF,
            4,
            1,
            "Fix: first changed-history dispatch should build one program",
        ),
        (
            &history_dispatcher,
            &[0b0011],
            0xFFFF_FFFF,
            4,
            1,
            "Fix: identical primitive key should reuse the cached program",
        ),
        (
            &history_dispatcher,
            &[0b0001],
            0b0001,
            4,
            2,
            "Fix: changed allow mask should rebuild the primitive program",
        ),
        (
            &legacy_dispatcher,
            &[0b0001],
            0b0001,
            65,
            3,
            "Fix: switching changed-history policy should rebuild the program",
        ),
    ];

    for (dispatcher, seed, allow_mask, max_iters, builds, why) in steps {
        linear_closure_with_scratch(
            dispatcher,
            seed,
            allow_mask,
            max_iters,
            &mut scratch,
            &mut frontier,
        )
        .expect(why);
        assert_eq!(scratch.program_builds(), builds, "{why}");
    }
}

#[test]

fn gpu_rejects_mismatched_edge_arrays() {
    let dispatcher = StaticOutputs::new(
        CSR_CHANGED_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0b1111]),
            u32_slice_to_le_bytes(&[0, 0, 0, 0]),
        ],
    )
    .expecting_inputs(&[7, 8]);
    let err = forward_closure_via_change_flag_gpu(
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

#[test]
fn generated_gpu_seed_copy_bounds_to_primitive_frontier_words() {
    for node_count in 1u32..=512 {
        let frontier_words = node_count.div_ceil(32) as usize;
        let edge_offsets = vec![0; node_count as usize + 1];
        for extra_words in 0..8usize {
            let seed_len = frontier_words + extra_words;
            let seed = (0..seed_len)
                .map(|idx| 0xA5A5_0000u32 ^ idx as u32 ^ node_count)
                .collect::<Vec<_>>();
            let dispatcher = StaticOutputs::new(
                CSR_CHANGED_CONTRACT,
                vec![
                    u32_slice_to_le_bytes(&vec![0; frontier_words]),
                    u32_slice_to_le_bytes(&[0]),
                ],
            )
            .recording_input(5);
            let mut frontier = Vec::new();

            let result = forward_closure_via_change_flag_gpu_into(
                &dispatcher,
                CsrClosureInputs::allow_all(
                    CsrGraphView {
                        node_count,
                        edge_offsets: &edge_offsets,
                        edge_targets: &[],
                        edge_kind_mask: &[],
                    },
                    1,
                ),
                &seed,
                &mut frontier,
            );

            if extra_words == 0 {
                result.expect("Fix: exact-width empty-edge generated CSR closure should dispatch");
                let observed = dispatcher.recorded();
                assert_eq!(
                    observed.len(),
                    1,
                    "node_count={node_count} extra_words={extra_words}"
                );
                assert_eq!(
                    observed[0],
                    seed[..frontier_words],
                    "node_count={node_count} extra_words={extra_words}"
                );
            } else {
                let err = result.expect_err(
                    "Fix: oversized generated seed must be rejected instead of silently truncated",
                );
                assert!(
                    matches!(err, DispatchError::BadInputs(_)),
                    "node_count={node_count} extra_words={extra_words} err={err:?}"
                );
                let observed = dispatcher.recorded();
                assert!(
                    observed.is_empty(),
                    "node_count={node_count} extra_words={extra_words}"
                );
            }
        }
    }
}
