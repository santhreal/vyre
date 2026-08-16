use super::*;
use crate::dispatch_buffers::u32_slice_to_le_bytes;
use crate::test_parity_oracles::StaticOutputs;
use vyre_foundation::program_dispatch::DispatchError;
use crate::graph::motif::{
    cpu_ref as reference_motif, cpu_ref_matches as reference_motif_matches,
    cpu_ref_participation_count as reference_motif_participation_count, plan_motif_launch,
    MotifEdge,
};

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
fn matches_primitive_directly() {
    let (offsets, targets, masks, motif) = chain_graph();
    let via_substrate = match_motif(3, &offsets, &targets, &masks, &motif);
    let via_primitive = reference_motif(3, &offsets, &targets, &masks, &motif);
    assert_eq!(via_substrate, via_primitive);
}

#[test]
fn checked_reference_wrappers_surface_bad_motif_endpoints() {
    let bad_motif = [MotifEdge {
        from: 0,
        kind_mask: 1,
        to: 3,
    }];

    let err = try_motif_participation_count(3, &[0, 1, 1, 1], &[1], &[1], &bad_motif)
        .expect_err("bad motif endpoint must fail through substrate wrapper");

    assert!(
        err.contains("motif_edges[0].to=3 is outside node_count 3"),
        "Fix: substrate motif wrapper must preserve primitive endpoint diagnostics, got: {err}"
    );
    assert!(try_match_motif(3, &[0, 1, 1, 1], &[1], &[1], &bad_motif).is_err());
    assert!(try_motif_matches(3, &[0, 1, 1, 1], &[1], &[1], &bad_motif).is_err());
}

#[test]
fn launch_plan_matches_primitive_dispatch_plan() {
    let (offsets, targets, masks, motif) = chain_graph();
    let launch = plan_motif_launch(3, &offsets, &targets, &masks, &motif, "witness")
        .expect("Fix: motif launch planning must accept the canonical chain graph");
    let dispatch = crate::graph::motif::plan_motif_dispatch(
        3, &offsets, &targets, &masks, &motif, "witness",
    )
    .expect("Fix: motif dispatch planning must accept the canonical chain graph");

    assert_eq!(launch.layout(), dispatch.layout());
    assert_eq!(launch.output_words(), dispatch.output_words());
    assert_eq!(launch.edge_storage_words(), dispatch.edge_storage_words());
    assert_eq!(launch.dispatch_grid(), dispatch.dispatch_grid());
    let launch_program = launch.program();
    let dispatch_program = dispatch.program();
    assert_eq!(launch_program.entry_op_id, dispatch_program.entry_op_id);
    assert_eq!(launch_program.buffers.len(), dispatch_program.buffers.len());
}

#[test]
fn predicate_and_count_match_primitive_output() {
    let (offsets, targets, masks, motif) = chain_graph();
    assert_eq!(
        motif_matches(3, &offsets, &targets, &masks, &motif),
        reference_motif_matches(&offsets, &targets, &masks, &motif)
            && reference_motif_participation_count(3, &offsets, &targets, &masks, &motif) != 0
    );
    assert_eq!(
        motif_participation_count(3, &offsets, &targets, &masks, &motif),
        reference_motif_participation_count(3, &offsets, &targets, &masks, &motif)
    );
}

#[test]
fn missing_edge_clears_match_and_participation() {
    let motif = [MotifEdge {
        from: 1,
        kind_mask: 1,
        to: 2,
    }];
    assert!(!motif_matches(3, &[0, 1, 1, 1], &[1], &[1], &motif));
    assert_eq!(
        motif_participation_count(3, &[0, 1, 1, 1], &[1], &[1], &motif),
        0
    );
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
    .expecting_grid([1, 1, 1])
    .expecting_inputs(&[7]);
    let (offsets, targets, masks, motif) = chain_graph();
    let mut witness = Vec::with_capacity(4);
    let ptr = witness.as_ptr();
    match_motif_via_into(
        &dispatcher,
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
    .expecting_grid([1, 1, 1])
    .expecting_inputs(&[7])
    .recording_input(2);
    let (offsets, targets, masks, motif) = chain_graph();
    let changed_targets = vec![2, 2];
    let mut scratch = MotifGpuScratch::default();
    let mut witness = Vec::new();

    match_motif_via_with_scratch_into(
        &dispatcher,
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
    .expecting_grid([1, 1, 1])
    .expecting_inputs(&[7]);
    let (offsets, targets, masks, motif) = chain_graph();
    let mut scratch = MotifGpuScratch::default();
    let mut witness = Vec::with_capacity(3);

    match_motif_via_with_scratch_into(
        &dispatcher,
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
    .expecting_grid([1, 1, 1])
    .expecting_inputs(&[7]);
    let err = match_motif_via(&dispatcher, 1, &[0, 0], &[], &[], &[])
        .expect_err("extra outputs must be rejected");
    assert!(matches!(err, DispatchError::BackendError(_)));
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
    .expecting_grid([1, 1, 1])
    .expecting_inputs(&[7]);
    let (offsets, targets, masks, motif) = chain_graph();
    let err = match_motif_via(&dispatcher, 3, &offsets, &targets, &masks, &motif)
        .expect_err("non-boolean witness output must be rejected");

    assert!(matches!(err, DispatchError::BackendError(_)));
}

#[test]
fn via_rejects_malformed_csr_before_dispatch() {
    let dispatcher = StaticOutputs::new(MOTIF_CONTRACT, Vec::new())
        .expecting_grid([1, 1, 1])
        .expecting_inputs(&[7]);
    let err = match_motif_via(&dispatcher, 2, &[0, 1, 1], &[1], &[], &[])
        .expect_err("mismatched edge arrays must be rejected");
    assert!(matches!(err, DispatchError::BadInputs(_)));
}
