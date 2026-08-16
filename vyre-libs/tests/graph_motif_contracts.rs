//! Contracts for the motif primitive.

use vyre_libs::graph::motif::{
    count_witness_participants, cpu_ref, cpu_ref_into, cpu_ref_matches,
    cpu_ref_participation_count, plan_motif_dispatch, plan_motif_launch, try_cpu_ref_into,
    try_cpu_ref_participation_count, try_cpu_ref_participation_count_with_scratch,
    validate_csr_inputs, validate_motif_inputs, MotifCpuScratch, MotifEdge, MotifLayout,
    MotifProgramCacheKey, MOTIF_DISPATCH_GRID, MOTIF_HITS_BUFFER, MOTIF_WITNESS_OUT_BUFFER,
    MOTIF_WORKGROUP_SIZE, TWO_EDGE_PATH_MOTIF,
};

#[test]
fn try_cpu_ref_into_rejects_bad_motif_endpoint_without_clobbering_witness() {
    let mut witness = vec![9, 8, 7];
    let motif = [MotifEdge {
        from: 0,
        kind_mask: 1,
        to: 3,
    }];

    let err = try_cpu_ref_into(3, &[0, 1, 1, 1], &[1], &[1], &motif, &mut witness)
        .expect_err("motif endpoint beyond node_count must fail validation");

    assert!(
        err.contains("motif_edges[0].to=3 is outside node_count 3"),
        "Fix: motif endpoint errors must identify the bad endpoint, got: {err}"
    );
    assert_eq!(
        witness,
        vec![9, 8, 7],
        "failed motif preflight must preserve the previous witness vector"
    );
}

#[test]
fn try_participation_count_rejects_bad_motif_endpoint() {
    let motif = [MotifEdge {
        from: 4,
        kind_mask: 1,
        to: 0,
    }];

    let err = try_cpu_ref_participation_count(3, &[0, 0, 0, 0], &[], &[], &motif)
        .expect_err("motif participation count must validate pattern endpoints");

    assert!(
        err.contains("motif_edges[0].from=4 is outside node_count 3"),
        "Fix: motif participation count must surface endpoint shape errors, got: {err}"
    );
}

#[test]
fn try_participation_count_with_scratch_reuses_endpoint_storage() {
    let mut endpoints = Vec::with_capacity(8);
    endpoints.extend_from_slice(&[99, 98, 97]);
    let mut scratch = MotifCpuScratch { endpoints };
    let capacity = scratch.endpoints.capacity();
    let motif = TWO_EDGE_PATH_MOTIF;

    let count = try_cpu_ref_participation_count_with_scratch(
        3,
        &[0, 1, 2, 2],
        &[1, 2],
        &[1, 1],
        &motif,
        &mut scratch,
    )
    .expect("Fix: valid motif count must run with reusable endpoint scratch.");

    assert_eq!(count, 3);
    assert_eq!(scratch.endpoints.capacity(), capacity);
    assert_eq!(
        scratch.endpoints,
        vec![0, 1, 2],
        "Fix: endpoint scratch must be sorted and deduplicated for the live motif."
    );

    let count = try_cpu_ref_participation_count_with_scratch(
        3,
        &[0, 1, 1, 1],
        &[1],
        &[1],
        &motif,
        &mut scratch,
    )
    .expect("Fix: valid graph with missing motif must return zero without stale endpoints.");

    assert_eq!(count, 0);
    assert_eq!(scratch.endpoints.capacity(), capacity);
    assert!(
        scratch.endpoints.is_empty(),
        "Fix: missing motif must clear stale endpoint scratch."
    );
}

#[test]
fn try_participation_count_with_scratch_validates_before_mutating_storage() {
    let mut scratch = MotifCpuScratch {
        endpoints: vec![0xCAFE_BABE, 0xDEAD_BEEF],
    };
    let motif = [MotifEdge {
        from: 4,
        kind_mask: 1,
        to: 0,
    }];

    let err = try_cpu_ref_participation_count_with_scratch(
        3,
        &[0, 0, 0, 0],
        &[],
        &[],
        &motif,
        &mut scratch,
    )
    .expect_err("Fix: motif endpoint validation must run before scratch reuse.");

    assert!(
        err.contains("motif_edges[0].from=4 is outside node_count 3"),
        "Fix: motif participation count must surface endpoint shape errors, got: {err}"
    );
    assert_eq!(
        scratch.endpoints,
        vec![0xCAFE_BABE, 0xDEAD_BEEF],
        "Fix: validation failure must not clear reusable endpoint scratch."
    );
}

#[test]
fn generated_participation_count_matches_witness_count() {
    for node_count in 2u32..=7 {
        let mut offsets = Vec::with_capacity(node_count as usize + 1);
        let mut targets = Vec::new();
        let mut masks = Vec::new();
        offsets.push(0);
        for node in 0..node_count {
            targets.push((node + 1) % node_count);
            masks.push(1);
            offsets.push(targets.len() as u32);
        }
        let motif = [MotifEdge {
            from: 0,
            kind_mask: 1,
            to: 1,
        }];
        let witness = cpu_ref(node_count, &offsets, &targets, &masks, &motif);
        let witness_count =
            count_witness_participants(&witness).expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - generated witness count must fit u32");
        let count =
            try_cpu_ref_participation_count(node_count, &offsets, &targets, &masks, &motif)
                .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - generated motif participation count must pass validation");

        assert_eq!(
            count, witness_count,
            "participation count diverged from witness count at node_count={node_count}"
        );
    }
}

#[test]
fn three_node_chain_motif_marks_every_participant() {
    let witness = cpu_ref(3, &[0, 1, 2, 2], &[1, 2], &[1, 1], &TWO_EDGE_PATH_MOTIF);
    assert_eq!(witness, vec![1, 1, 1]);
}

#[test]
fn missing_motif_edge_clears_all_participants() {
    let witness = cpu_ref(3, &[0, 1, 1, 1], &[1], &[1], &TWO_EDGE_PATH_MOTIF);
    assert_eq!(witness, vec![0, 0, 0]);
}

#[test]
fn cpu_ref_into_reuses_witness_storage() {
    let mut witness = Vec::with_capacity(8);
    cpu_ref_into(
        3,
        &[0, 1, 2, 2],
        &[1, 2],
        &[1, 1],
        &TWO_EDGE_PATH_MOTIF,
        &mut witness,
    );
    let capacity = witness.capacity();
    assert_eq!(witness, vec![1, 1, 1]);

    cpu_ref_into(
        3,
        &[0, 1, 1, 1],
        &[1],
        &[1],
        &[MotifEdge {
            from: 1,
            kind_mask: 1,
            to: 2,
        }],
        &mut witness,
    );
    assert_eq!(witness.capacity(), capacity);
    assert_eq!(witness, vec![0, 0, 0]);
}

#[test]
fn cpu_ref_into_validates_before_clearing_witness_storage() {
    let mut witness = vec![0xCAFE_BABEu32, 0xDEAD_BEEF];
    let ptr = witness.as_ptr();
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cpu_ref_into(
            2,
            &[0, 1, 1],
            &[1],
            &[],
            &[MotifEdge {
                from: 0,
                kind_mask: 1,
                to: 1,
            }],
            &mut witness,
        );
    }));
    std::panic::set_hook(previous_hook);

    assert!(err.is_err(), "mismatched CSR edge arrays must be rejected");
    assert_eq!(
        witness,
        vec![0xCAFE_BABEu32, 0xDEAD_BEEF],
        "Fix: motif CPU oracle must validate before clearing caller witness storage."
    );
    assert_eq!(witness.as_ptr(), ptr);
}

#[test]
fn generated_try_cpu_ref_into_and_count_match_witness() {
    for node_count in 1u32..=64 {
        let mut scratch = MotifCpuScratch::new();
        let edge_offsets: Vec<u32> = (0..=node_count).collect();
        let edge_targets: Vec<u32> = (0..node_count)
            .map(|node| (node + 1) % node_count)
            .collect();
        let edge_kind_mask = vec![1u32; node_count as usize];
        for motif_len in 0usize..64 {
            let motif_edges: Vec<MotifEdge> = (0..motif_len)
                .map(|index| {
                    let from = (index as u32) % node_count;
                    MotifEdge {
                        from,
                        kind_mask: 1,
                        to: (from + 1) % node_count,
                    }
                })
                .collect();
            let mut witness = vec![0xCAFE_BABEu32; 3];
            try_cpu_ref_into(
                node_count,
                &edge_offsets,
                &edge_targets,
                &edge_kind_mask,
                &motif_edges,
                &mut witness,
            )
            .unwrap();
            let count = try_cpu_ref_participation_count_with_scratch(
                node_count,
                &edge_offsets,
                &edge_targets,
                &edge_kind_mask,
                &motif_edges,
                &mut scratch,
            )
            .unwrap();
            assert_eq!(witness.len(), node_count as usize);
            assert_eq!(
                count,
                witness.iter().filter(|&&value| value != 0).count() as u32
            );
        }
    }
}

#[test]
fn allocation_free_predicates_match_witness_contract() {
    let motif = TWO_EDGE_PATH_MOTIF;
    assert!(cpu_ref_matches(&[0, 1, 2, 2], &[1, 2], &[1, 1], &motif));
    assert_eq!(
        cpu_ref_participation_count(3, &[0, 1, 2, 2], &[1, 2], &[1, 1], &motif),
        3
    );
    assert!(!cpu_ref_matches(&[0, 1, 1, 1], &[1], &[1], &motif));
    assert_eq!(
        cpu_ref_participation_count(3, &[0, 1, 1, 1], &[1], &[1], &motif),
        0
    );
    assert!(
        cpu_ref_matches(&[0, 1, 2, 2], &[1, 2], &[1, 1], &[]),
        "empty motif has no missing edges"
    );
    assert_eq!(
        cpu_ref_participation_count(3, &[0, 1, 2, 2], &[1, 2], &[1, 1], &[]),
        0,
        "empty motif has no participating nodes"
    );
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
fn dispatch_plan_owns_shape_grid_buffers_and_readback_words() {
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
    assert_eq!(launch.dispatch_grid(), MOTIF_DISPATCH_GRID);
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
    assert_eq!(plan.dispatch_grid(), MOTIF_DISPATCH_GRID);
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
    assert_eq!(count_witness_participants(&[1, 0, 2, 0]).unwrap(), 2);
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
