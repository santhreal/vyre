use super::*;

use vyre_test_support::exploded_ifds_cases::{arm_coverage, declared_groups, ExplodedIfdsCase};

/// Every declared exploded-IFDS group has a primitive arm, and every case in it
/// holds.
///
/// The ledger reads the shared table at run time, so a group declared with no
/// branch below fails here by name rather than silently going unrun. Which cases
/// exist and what a correct CSR looks like belong to
/// `vyre_test_support::exploded_ifds_cases`; this test pins only what the
/// primitive CPU reference owes for them.
#[test]
fn primitive_exploded_ifds_arms_cover_every_declared_case_group() {
    let mut coverage = arm_coverage();
    for group in declared_groups() {
        match group.name {
            "mixed_flow_stream" => assert_reusable_oracle_matches_allocating(&group.cases),
            "dense_chain" => assert_allocating_oracle_emits_valid_csr(&group.cases),
            "flow_rule_edges" => assert_panicking_wrapper_agrees_with_try(&group.cases),
            "empty_domain" => assert_domain_rejected(&group.cases),
            _ => continue,
        }
        coverage.record(group.name, group.cases.len());
    }
    coverage.assert_covers_declared_table();
}

/// The workspace-reusing oracle produces exactly what the allocating one does,
/// case after case through one scratch.
fn assert_reusable_oracle_matches_allocating(cases: &[ExplodedIfdsCase]) {
    let mut row_ptr = Vec::new();
    let mut col_idx = Vec::new();
    let mut scratch = ExplodedIfdsCpuScratch::new();

    for case in cases {
        let expected = try_build_cpu_reference(
            case.num_procs,
            case.blocks_per_proc,
            case.facts_per_proc,
            &case.intra_edges,
            &case.inter_edges,
            &case.flow_gen,
            &case.flow_kill,
        )
        .expect("Fix: declared exploded IFDS case must build through the allocating oracle.");
        try_build_cpu_reference_into(
            case.num_procs,
            case.blocks_per_proc,
            case.facts_per_proc,
            &case.intra_edges,
            &case.inter_edges,
            &case.flow_gen,
            &case.flow_kill,
            &mut row_ptr,
            &mut col_idx,
            &mut scratch,
        )
        .expect("Fix: declared exploded IFDS case must build through the reusable oracle.");

        assert_eq!(
            (row_ptr.clone(), col_idx.clone()),
            expected,
            "Fix: reusable exploded IFDS oracle diverged at {}.",
            case.label
        );
        case.assert_csr("try_build_cpu_reference_into", &row_ptr, &col_idx);
    }
}

/// The allocating oracle emits a well-formed CSR for every declared shape.
fn assert_allocating_oracle_emits_valid_csr(cases: &[ExplodedIfdsCase]) {
    for case in cases {
        let (row_ptr, col_idx) = try_build_cpu_reference(
            case.num_procs,
            case.blocks_per_proc,
            case.facts_per_proc,
            &case.intra_edges,
            &case.inter_edges,
            &case.flow_gen,
            &case.flow_kill,
        )
        .expect("Fix: declared exploded IFDS case must build through the allocating oracle.");
        case.assert_csr("try_build_cpu_reference", &row_ptr, &col_idx);
    }
}

/// The panicking wrapper answers exactly what the `Result` form answers, and
/// both carry the declared rule semantics.
fn assert_panicking_wrapper_agrees_with_try(cases: &[ExplodedIfdsCase]) {
    for case in cases {
        let built = build_cpu_reference(
            case.num_procs,
            case.blocks_per_proc,
            case.facts_per_proc,
            &case.intra_edges,
            &case.inter_edges,
            &case.flow_gen,
            &case.flow_kill,
        );
        let tried = try_build_cpu_reference(
            case.num_procs,
            case.blocks_per_proc,
            case.facts_per_proc,
            &case.intra_edges,
            &case.inter_edges,
            &case.flow_gen,
            &case.flow_kill,
        )
        .expect("Fix: declared exploded IFDS case must build through the allocating oracle.");
        assert_eq!(
            built, tried,
            "Fix: panicking and fallible exploded IFDS references disagree at {}.",
            case.label
        );
        case.assert_csr("build_cpu_reference", &built.0, &built.1);
    }
}

/// An invalid domain is a reported error, not a fabricated empty CSR: parity
/// needs a real exploded-supergraph domain.
fn assert_domain_rejected(cases: &[ExplodedIfdsCase]) {
    for case in cases {
        let err = try_build_cpu_reference(
            case.num_procs,
            case.blocks_per_proc,
            case.facts_per_proc,
            &case.intra_edges,
            &case.inter_edges,
            &case.flow_gen,
            &case.flow_kill,
        )
        .expect_err("Fix: an empty exploded IFDS domain must be rejected.");
        assert!(
            err.contains("nonzero"),
            "Fix: empty-domain rejection must stay explicit at {}, got: {err}",
            case.label
        );
    }
}

#[test]
fn try_build_cpu_reference_into_reuses_output_and_workspace() {
    let mut row_ptr = Vec::with_capacity(32);
    row_ptr.extend_from_slice(&[9, 8, 7]);
    let mut col_idx = Vec::with_capacity(32);
    col_idx.extend_from_slice(&[6, 5, 4]);
    let mut scratch = ExplodedIfdsCpuScratch {
        edges_flat: Vec::with_capacity(32),
        killed: Vec::with_capacity(32),
        gen_offsets: Vec::with_capacity(16),
        gen_cursor: Vec::with_capacity(16),
        gen_facts: Vec::with_capacity(16),
        cursor: Vec::with_capacity(32),
    };
    scratch.edges_flat.extend_from_slice(&[(99, 98), (97, 96)]);
    scratch.killed.extend_from_slice(&[true, true]);
    scratch.gen_offsets.extend_from_slice(&[11, 12]);
    scratch.gen_cursor.extend_from_slice(&[13, 14]);
    scratch.gen_facts.extend_from_slice(&[15, 16]);
    scratch.cursor.extend_from_slice(&[17, 18]);
    let capacities = (
        row_ptr.capacity(),
        col_idx.capacity(),
        scratch.edges_flat.capacity(),
        scratch.killed.capacity(),
        scratch.gen_offsets.capacity(),
        scratch.gen_cursor.capacity(),
        scratch.gen_facts.capacity(),
        scratch.cursor.capacity(),
    );

    let expected = build_cpu_reference(1, 2, 4, &[(0, 0, 1)], &[], &[(0, 0, 2)], &[(0, 0, 3)]);
    try_build_cpu_reference_into(
        1,
        2,
        4,
        &[(0, 0, 1)],
        &[],
        &[(0, 0, 2)],
        &[(0, 0, 3)],
        &mut row_ptr,
        &mut col_idx,
        &mut scratch,
    )
    .expect("Fix: valid exploded IFDS graph must build with reusable workspace.");

    assert_eq!((row_ptr.clone(), col_idx.clone()), expected);
    assert_eq!(
        (
            row_ptr.capacity(),
            col_idx.capacity(),
            scratch.edges_flat.capacity(),
            scratch.killed.capacity(),
            scratch.gen_offsets.capacity(),
            scratch.gen_cursor.capacity(),
            scratch.gen_facts.capacity(),
            scratch.cursor.capacity(),
        ),
        capacities
    );

    try_build_cpu_reference_into(
        1,
        1,
        1,
        &[],
        &[],
        &[],
        &[],
        &mut row_ptr,
        &mut col_idx,
        &mut scratch,
    )
    .expect("Fix: smaller exploded IFDS graph must reuse the same workspace.");

    assert_eq!(row_ptr, vec![0, 0]);
    assert!(col_idx.is_empty());
    assert_eq!(
        (
            row_ptr.capacity(),
            col_idx.capacity(),
            scratch.edges_flat.capacity(),
            scratch.killed.capacity(),
            scratch.gen_offsets.capacity(),
            scratch.gen_cursor.capacity(),
            scratch.gen_facts.capacity(),
            scratch.cursor.capacity(),
        ),
        capacities
    );
}

#[test]
fn try_build_cpu_reference_into_validates_before_mutating_storage() {
    let mut row_ptr = vec![9, 8, 7];
    let mut col_idx = vec![6, 5, 4];
    let mut scratch = ExplodedIfdsCpuScratch {
        edges_flat: vec![(1, 2)],
        killed: vec![true],
        gen_offsets: vec![3],
        gen_cursor: vec![4],
        gen_facts: vec![5],
        cursor: vec![6],
    };

    let err = try_build_cpu_reference_into(
        0,
        0,
        0,
        &[],
        &[],
        &[],
        &[],
        &mut row_ptr,
        &mut col_idx,
        &mut scratch,
    )
    .expect_err("Fix: empty exploded IFDS domain must be rejected.");

    assert!(err.contains("nonzero"));
    assert_eq!(row_ptr, vec![9, 8, 7]);
    assert_eq!(col_idx, vec![6, 5, 4]);
    assert_eq!(scratch.edges_flat, vec![(1, 2)]);
    assert_eq!(scratch.killed, vec![true]);
    assert_eq!(scratch.gen_offsets, vec![3]);
    assert_eq!(scratch.gen_cursor, vec![4]);
    assert_eq!(scratch.gen_facts, vec![5]);
    assert_eq!(scratch.cursor, vec![6]);
}
