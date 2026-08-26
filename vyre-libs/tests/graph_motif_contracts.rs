//! Contracts for the motif primitive.

use vyre_libs::graph::motif::{
    plan_motif_dispatch, plan_motif_launch, validate_csr_inputs, validate_motif_inputs, MotifEdge,
    MotifLayout, MotifProgramCacheKey, MOTIF_HITS_BUFFER, MOTIF_WITNESS_OUT_BUFFER,
    MOTIF_WORKGROUP_SIZE, TWO_EDGE_PATH_MOTIF,
};
use vyre_reference::composition_witness::{motif_witness, reduce_count_non_zero_witness};

fn reference_witness(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
) -> Vec<u32> {
    let edges = motif_edges
        .iter()
        .map(|edge| (edge.from, edge.kind_mask, edge.to))
        .collect::<Vec<_>>();
    motif_witness(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        &edges,
    )
}

#[test]
fn validation_rejects_bad_motif_endpoints() {
    let to_outside = [MotifEdge {
        from: 0,
        kind_mask: 1,
        to: 3,
    }];
    let err = validate_motif_inputs(3, &[0, 1, 1, 1], &[1], &[1], &to_outside)
        .expect_err("motif endpoint beyond node_count must fail validation");
    assert!(
        err.contains("motif_edges[0].to=3 is outside node_count 3"),
        "endpoint errors must identify the bad destination, got: {err}"
    );

    let from_outside = [MotifEdge {
        from: 4,
        kind_mask: 1,
        to: 0,
    }];
    let err = validate_motif_inputs(3, &[0, 0, 0, 0], &[], &[], &from_outside)
        .expect_err("motif source beyond node_count must fail validation");
    assert!(
        err.contains("motif_edges[0].from=4 is outside node_count 3"),
        "endpoint errors must identify the bad source, got: {err}"
    );
}

#[test]
fn generated_participant_count_matches_witness_count() {
    for node_count in 2u32..=7 {
        let edge_offsets = (0..=node_count).collect::<Vec<_>>();
        let edge_targets = (0..node_count)
            .map(|node| (node + 1) % node_count)
            .collect::<Vec<_>>();
        let edge_kind_mask = vec![1; node_count as usize];
        for motif_len in 0usize..64 {
            let motif_edges = (0..motif_len)
                .map(|index| {
                    let from = index as u32 % node_count;
                    MotifEdge {
                        from,
                        kind_mask: 1,
                        to: (from + 1) % node_count,
                    }
                })
                .collect::<Vec<_>>();
            let witness = reference_witness(
                node_count,
                &edge_offsets,
                &edge_targets,
                &edge_kind_mask,
                &motif_edges,
            );
            assert_eq!(witness.len(), node_count as usize);
            assert_eq!(
                reduce_count_non_zero_witness(&witness),
                witness.iter().filter(|&&value| value != 0).count() as u32
            );
        }
    }
}

#[test]
fn complete_path_motif_marks_every_participant() {
    let witness = reference_witness(3, &[0, 1, 2, 2], &[1, 2], &[1, 1], &TWO_EDGE_PATH_MOTIF);
    assert_eq!(witness, vec![1, 1, 1]);
}

#[test]
fn missing_motif_edge_clears_all_participants() {
    let witness = reference_witness(3, &[0, 1, 1, 1], &[1], &[1], &TWO_EDGE_PATH_MOTIF);
    assert_eq!(witness, vec![0, 0, 0]);
}

#[test]
fn repeated_reference_calls_do_not_retain_previous_state() {
    let complete = reference_witness(3, &[0, 1, 2, 2], &[1, 2], &[1, 1], &TWO_EDGE_PATH_MOTIF);
    let missing = reference_witness(3, &[0, 1, 1, 1], &[1], &[1], &TWO_EDGE_PATH_MOTIF);
    let complete_again =
        reference_witness(3, &[0, 1, 2, 2], &[1, 2], &[1, 1], &TWO_EDGE_PATH_MOTIF);
    assert_eq!(complete, vec![1, 1, 1]);
    assert_eq!(missing, vec![0, 0, 0]);
    assert_eq!(complete_again, complete);
}

#[test]
fn empty_motif_matches_without_participants() {
    let witness = reference_witness(3, &[0, 1, 2, 2], &[1, 2], &[1, 1], &[]);
    assert_eq!(witness, vec![0, 0, 0]);
    assert_eq!(reduce_count_non_zero_witness(&witness), 0);
}

#[test]
fn validate_csr_inputs_accepts_empty_and_canonical_graphs() {
    assert_eq!(
        validate_motif_inputs(0, &[0], &[], &[], &[]).unwrap(),
        MotifLayout {
            node_count: 0,
            output_words: 0,
            edge_count: 0,
            edge_storage_words: 1,
            motif_edge_count: 0,
        }
    );
    assert_eq!(
        validate_motif_inputs(
            3,
            &[0, 1, 2, 2],
            &[1, 2],
            &[1, 1],
            &[MotifEdge {
                from: 0,
                kind_mask: 1,
                to: 1,
            }],
        )
        .unwrap(),
        MotifLayout {
            node_count: 3,
            output_words: 3,
            edge_count: 2,
            edge_storage_words: 2,
            motif_edge_count: 1,
        }
    );
}

#[test]
fn dispatch_plan_owns_shape_buffers_and_readback_words() {
    let motif_edges = [MotifEdge {
        from: 0,
        kind_mask: 1,
        to: 1,
    }];
    let launch = plan_motif_launch(
        3,
        &[0, 1, 2, 2],
        &[1, 2],
        &[1, 1],
        &motif_edges,
        "witness_out",
    )
    .expect("Fix: canonical motif launch plan should validate without materializing a Program");
    assert_eq!(launch.layout().node_count, 3);
    assert_eq!(launch.output_words(), 3);
    assert_eq!(launch.edge_storage_words(), 2);
    assert_eq!(
        launch.cache_key(),
        &MotifProgramCacheKey {
            node_count: 3,
            edge_count: 2,
            motif_edges: motif_edges.to_vec(),
            witness_out: "witness_out".to_string(),
        }
    );

    let plan = plan_motif_dispatch(
        3,
        &[0, 1, 2, 2],
        &[1, 2],
        &[1, 1],
        &motif_edges,
        "witness_out",
    )
    .expect("Fix: canonical motif dispatch plan should validate");

    assert_eq!(plan.layout().node_count, 3);
    assert_eq!(plan.layout().edge_count, 2);
    assert_eq!(plan.layout().motif_edge_count, 1);
    assert_eq!(plan.output_words(), 3);
    assert_eq!(plan.edge_storage_words(), 2);
    assert_eq!(plan.program().workgroup_size, MOTIF_WORKGROUP_SIZE);
    let bindings = plan
        .program()
        .buffers
        .iter()
        .map(|buffer| buffer.binding)
        .collect::<Vec<_>>();
    assert!(bindings.contains(&MOTIF_HITS_BUFFER));
    assert!(bindings.contains(&MOTIF_WITNESS_OUT_BUFFER));

    let empty_edge_plan = plan_motif_dispatch(1, &[0, 0], &[], &[], &[], "witness_out")
        .expect("Fix: zero-edge motif graph should still have padded edge storage");
    assert_eq!(empty_edge_plan.layout().edge_count, 0);
    assert_eq!(empty_edge_plan.edge_storage_words(), 1);
}

#[test]
fn witness_participant_count_uses_primitive_contract() {
    assert_eq!(reduce_count_non_zero_witness(&[1, 0, 2, 0]), 2);
}

#[test]
fn validate_csr_inputs_rejects_malformed_csr() {
    let err = validate_csr_inputs(2, &[0, 1, 1], &[1], &[]).unwrap_err();
    assert!(err.contains("edge_targets.len() == edge_kind_mask.len()"));

    let err = validate_csr_inputs(2, &[0, 2, 1], &[1], &[1]).unwrap_err();
    assert!(err.contains("offsets must be monotonic"));

    let err = validate_csr_inputs(2, &[0, 1, 1], &[5], &[1]).unwrap_err();
    assert!(err.contains("outside node_count"));
}
