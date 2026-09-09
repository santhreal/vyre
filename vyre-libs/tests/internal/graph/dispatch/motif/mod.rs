use super::*;
use crate::dispatch_buffers::u32_slice_to_le_bytes;
use crate::graph::dispatch::motif::{motif_matches_via, motif_participation_count_via};
use crate::graph::motif::{plan_motif_launch, validate_motif_inputs, MotifEdge};
use vyre_test_support::test_parity_oracles::{policy, SequentialOutputs, StaticOutputs};
use vyre_megakernel::SemanticExecutionError;

const MOTIF_CONTRACT: &str = "motif match dispatch";

fn chain_graph() -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<MotifEdge>) {
    (
        vec![0, 1, 2, 2],
        vec![1, 2],
        vec![1, 1],
        vec![
            MotifEdge {
                from: 0,
                kind_mask: 1,
                to: 1,
            },
            MotifEdge {
                from: 1,
                kind_mask: 1,
                to: 2,
            },
        ],
    )
}

#[test]
fn a_motif_endpoint_outside_the_graph_is_rejected_with_its_index() {
    let bad_motif = [MotifEdge {
        from: 0,
        kind_mask: 1,
        to: 3,
    }];

    let err = validate_motif_inputs(3, &[0, 1, 1, 1], &[1], &[1], &bad_motif)
        .expect_err("a motif endpoint outside the graph must be rejected");

    assert!(
        err.contains("motif_edges[0].to=3 is outside node_count 3"),
        "Fix: motif validation must name the offending edge index and endpoint, got: {err}"
    );
}

#[test]
fn launch_plan_matches_primitive_dispatch_plan() {
    let (offsets, targets, masks, motif) = chain_graph();
    let launch = plan_motif_launch(3, &offsets, &targets, &masks, &motif, "witness")
        .expect("Fix: motif launch planning must accept the canonical chain graph");
    let dispatch =
        crate::graph::motif::plan_motif_dispatch(3, &offsets, &targets, &masks, &motif, "witness")
            .expect("Fix: motif dispatch planning must accept the canonical chain graph");

    assert_eq!(launch.layout(), dispatch.layout());
    assert_eq!(launch.output_words(), dispatch.output_words());
    assert_eq!(launch.edge_storage_words(), dispatch.edge_storage_words());
    let launch_program = launch.program();
    let dispatch_program = dispatch.program();
    assert_eq!(launch_program.entry_op_id, dispatch_program.entry_op_id);
    assert_eq!(launch_program.buffers.len(), dispatch_program.buffers.len());
}

#[test]
fn via_decodes_exact_output_into_reused_buffer() {
    let dispatcher = StaticOutputs::new(
        MOTIF_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[1, 1, 1]),
            u32_slice_to_le_bytes(&[1, 1, 1]),
        ],
    )
    .expecting_inputs(&[5]);
    let (offsets, targets, masks, motif) = chain_graph();
    let mut witness = Vec::with_capacity(4);
    let ptr = witness.as_ptr();
    match_motif_via_into(
        &dispatcher,
        &policy(),
        3,
        &offsets,
        &targets,
        &masks,
        &motif,
        &mut witness,
    )
    .expect("Fix: motif dispatch succeeds");

    assert_eq!(witness, vec![1, 1, 1]);
    assert_eq!(witness.as_ptr(), ptr);
}

#[test]
fn via_refreshes_static_graph_inputs_for_same_shape_content_change() {
    let dispatcher = StaticOutputs::new(
        MOTIF_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[1, 1, 1]),
            u32_slice_to_le_bytes(&[1, 1, 1]),
        ],
    )
    .expecting_inputs(&[5])
    .recording_input(2);
    let (offsets, targets, masks, motif) = chain_graph();
    let changed_targets = vec![2, 2];
    let mut scratch = MotifGpuScratch::default();
    let mut witness = Vec::new();

    match_motif_via_with_scratch_into(
        &dispatcher,
        &policy(),
        3,
        &offsets,
        &targets,
        &masks,
        &motif,
        &mut scratch,
        &mut witness,
    )
    .expect("Fix: first motif same-shape dispatch should succeed");
    match_motif_via_with_scratch_into(
        &dispatcher,
        &policy(),
        3,
        &offsets,
        &changed_targets,
        &masks,
        &motif,
        &mut scratch,
        &mut witness,
    )
    .expect("Fix: second motif same-shape dispatch should refresh graph inputs");

    let recorded = dispatcher.recorded();
    assert_eq!(recorded.as_slice(), &[targets, changed_targets]);
    assert_eq!(
        scratch.program_builds(),
        1,
        "Fix: same-shape motif graph changes should refresh static inputs without rebuilding the generated Program."
    );
}

#[test]
fn via_with_scratch_reuses_dispatch_storage() {
    let dispatcher = StaticOutputs::new(
        MOTIF_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[1, 1, 1]),
            u32_slice_to_le_bytes(&[1, 1, 1]),
        ],
    )
    .expecting_inputs(&[5]);
    let (offsets, targets, masks, motif) = chain_graph();
    let mut scratch = MotifGpuScratch::default();
    let mut witness = Vec::with_capacity(3);

    match_motif_via_with_scratch_into(
        &dispatcher,
        &policy(),
        3,
        &offsets,
        &targets,
        &masks,
        &motif,
        &mut scratch,
        &mut witness,
    )
    .expect("Fix: motif dispatch succeeds");
    let input_capacities = scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>();
    let hit_capacity = scratch.motif_hits.capacity();
    let witness_capacity = witness.capacity();
    assert_eq!(scratch.program_builds(), 1);

    match_motif_via_with_scratch_into(
        &dispatcher,
        &policy(),
        3,
        &offsets,
        &targets,
        &masks,
        &motif,
        &mut scratch,
        &mut witness,
    )
    .expect("Fix: motif dispatch succeeds");

    assert_eq!(
        scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>(),
        input_capacities
    );
    assert_eq!(scratch.motif_hits.capacity(), hit_capacity);
    assert_eq!(witness.capacity(), witness_capacity);
    assert_eq!(scratch.program_builds(), 1);

    let same_shape_different_targets = [2, 2];
    match_motif_via_with_scratch_into(
        &dispatcher,
        &policy(),
        3,
        &offsets,
        &same_shape_different_targets,
        &masks,
        &motif,
        &mut scratch,
        &mut witness,
    )
    .expect("Fix: same-shape motif dispatch succeeds");
    assert_eq!(scratch.program_builds(), 1);

    let mut different_motif = motif.clone();
    different_motif[1].to = 0;
    match_motif_via_with_scratch_into(
        &dispatcher,
        &policy(),
        3,
        &offsets,
        &targets,
        &masks,
        &different_motif,
        &mut scratch,
        &mut witness,
    )
    .expect("Fix: changed motif dispatch succeeds");
    assert_eq!(scratch.program_builds(), 2);
}

#[test]
fn via_rejects_extra_outputs() {
    let dispatcher = StaticOutputs::new(
        MOTIF_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0]),
            u32_slice_to_le_bytes(&[0]),
            u32_slice_to_le_bytes(&[0]),
        ],
    )
    .expecting_inputs(&[5]);
    let err = match_motif_via(&dispatcher, &policy(), 1, &[0, 0], &[], &[], &[])
        .expect_err("extra outputs must be rejected");
    assert!(matches!(err, SemanticExecutionError::Backend(_)));
}

#[test]
fn via_rejects_non_boolean_witness() {
    let dispatcher = StaticOutputs::new(
        MOTIF_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0, 0, 0]),
            u32_slice_to_le_bytes(&[1, 2, 0]),
        ],
    )
    .expecting_inputs(&[5]);
    let (offsets, targets, masks, motif) = chain_graph();
    let err = match_motif_via(
        &dispatcher,
        &policy(),
        3,
        &offsets,
        &targets,
        &masks,
        &motif,
    )
    .expect_err("non-boolean witness output must be rejected");

    assert!(matches!(err, SemanticExecutionError::Backend(_)));
}

#[test]
fn via_rejects_malformed_csr_before_dispatch() {
    let dispatcher = StaticOutputs::new(MOTIF_CONTRACT, Vec::new()).expecting_inputs(&[5]);
    let err = match_motif_via(&dispatcher, &policy(), 2, &[0, 1, 1], &[1], &[], &[])
        .expect_err("mismatched edge arrays must be rejected");
    assert!(matches!(err, SemanticExecutionError::InvalidRequest(_)));
}

#[test]
fn motif_matches_via_dispatches_match_and_reduction() {
    let dispatcher = SequentialOutputs::new(
        "motif_matches_via test",
        vec![
            vec![
                u32_slice_to_le_bytes(&[1, 1, 1]),
                u32_slice_to_le_bytes(&[1, 1, 1]),
            ],
            vec![u32_slice_to_le_bytes(&[1])],
        ],
    );
    let (offsets, targets, masks, motif) = chain_graph();
    let matches = motif_matches_via(
        &dispatcher,
        &policy(),
        3,
        &offsets,
        &targets,
        &masks,
        &motif,
    )
    .expect("motif_matches_via must succeed");
    assert!(matches);
}

#[test]
fn motif_participation_count_via_dispatches_match_and_reduction() {
    let dispatcher = SequentialOutputs::new(
        "motif_participation_count_via test",
        vec![
            vec![
                u32_slice_to_le_bytes(&[1, 1, 1]),
                u32_slice_to_le_bytes(&[1, 1, 1]),
            ],
            vec![u32_slice_to_le_bytes(&[3])],
        ],
    );
    let (offsets, targets, masks, motif) = chain_graph();
    let count = motif_participation_count_via(
        &dispatcher,
        &policy(),
        3,
        &offsets,
        &targets,
        &masks,
        &motif,
    )
    .expect("motif_participation_count_via must succeed");
    assert_eq!(count, 3);
}
