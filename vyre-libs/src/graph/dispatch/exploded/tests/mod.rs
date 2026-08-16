use super::*;
use crate::dispatch_buffers::u32_slice_to_le_bytes;
use crate::graph::dispatch::cpu_oracle::CpuOracleDispatcher;
use std::sync::Mutex;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};
use vyre_primitives::graph::exploded::build_cpu_reference;
use vyre_test_support::exploded_ifds_cases::{arm_coverage, declared_groups, ExplodedIfdsCase};

mod ifds_doubles;

use crate::test_parity_oracles::StaticOutputs;
use ifds_doubles::{canonical_expected, RecordingIfdsOracle};

/// The readback shapes below are deliberately malformed, so the dispatcher
/// asserts nothing about its inputs; what is under test is the decoder's
/// validation. Production parity uses `CpuOracleDispatcher`.
const MALFORMED_IFDS_CONTRACT: &str = "malformed exploded-IFDS readback";

/// Every declared exploded-IFDS group has a substrate arm, and every case in it
/// holds.
///
/// The ledger reads the shared table at run time, so a group declared with no
/// branch below fails here by name rather than silently going unrun. Which cases
/// exist and what a correct CSR looks like belong to
/// `vyre_test_support::exploded_ifds_cases`; this test pins only what the
/// substrate consumer owes for them.
#[test]
fn substrate_exploded_ifds_arms_cover_every_declared_case_group() {
    let mut coverage = arm_coverage();
    for group in declared_groups() {
        match group.name {
            "mixed_flow_stream" => assert_via_dispatch_matches_reference(&group.cases),
            "dense_chain" | "flow_rule_edges" => assert_reference_matches_primitive(&group.cases),
            "empty_domain" => assert_domain_rejected(&group.cases),
            _ => continue,
        }
        coverage.record(group.name, group.cases.len());
    }
    coverage.assert_covers_declared_table();
}

/// The dispatched path decodes to exactly what the host reference builds.
fn assert_via_dispatch_matches_reference(cases: &[ExplodedIfdsCase]) {
    let dispatcher = CpuOracleDispatcher::new();
    for case in cases {
        let expected = canonical_expected(
            case.num_procs,
            case.blocks_per_proc,
            case.facts_per_proc,
            &case.intra_edges,
            &case.inter_edges,
            &case.flow_gen,
            &case.flow_kill,
        );
        let actual = build_ifds_csr_via(
            &dispatcher,
            case.num_procs,
            case.blocks_per_proc,
            case.facts_per_proc,
            &case.intra_edges,
            &case.inter_edges,
            &case.flow_gen,
            &case.flow_kill,
        )
        .unwrap_or_else(|error| {
            panic!(
                "Fix: declared IFDS case must dispatch through the CPU oracle at {}: {error:?}",
                case.label
            )
        });
        assert_eq!(
            actual, expected,
            "Fix: CPU oracle via path diverged from the substrate reference at {}.",
            case.label
        );
        case.assert_csr("build_ifds_csr_via", &actual.0, &actual.1);
    }
}

/// Closure bar: the substrate reference is the primitive reference, and the node
/// count helper agrees with the CSR it sizes.
fn assert_reference_matches_primitive(cases: &[ExplodedIfdsCase]) {
    for case in cases {
        let via_substrate = reference_build_ifds_csr(
            case.num_procs,
            case.blocks_per_proc,
            case.facts_per_proc,
            &case.intra_edges,
            &case.inter_edges,
            &case.flow_gen,
            &case.flow_kill,
        );
        let via_primitive = build_cpu_reference(
            case.num_procs,
            case.blocks_per_proc,
            case.facts_per_proc,
            &case.intra_edges,
            &case.inter_edges,
            &case.flow_gen,
            &case.flow_kill,
        );
        assert_eq!(
            via_substrate, via_primitive,
            "Fix: substrate exploded IFDS reference diverged from the primitive owner at {}.",
            case.label
        );
        assert_eq!(
            ifds_node_count(case.num_procs, case.blocks_per_proc, case.facts_per_proc) as usize,
            case.node_count(),
            "Fix: substrate node-count helper disagrees with the CSR row count at {}.",
            case.label
        );
        case.assert_csr(
            "reference_build_ifds_csr",
            &via_substrate.0,
            &via_substrate.1,
        );
    }
}

/// An invalid domain is a reported error, not a fabricated host-side empty CSR:
/// parity needs a real exploded-supergraph domain.
fn assert_domain_rejected(cases: &[ExplodedIfdsCase]) {
    for case in cases {
        let message = try_reference_build_ifds_csr(
            case.num_procs,
            case.blocks_per_proc,
            case.facts_per_proc,
            &case.intra_edges,
            &case.inter_edges,
            &case.flow_gen,
            &case.flow_kill,
        )
        .expect_err("Fix: an empty IFDS reference domain must fail.");
        assert!(
            message.contains("exploded IFDS CPU reference dimensions must be nonzero"),
            "Fix: empty-domain rejection must remain explicit at {}, got: {message}",
            case.label
        );
    }
}

/// Round-trip dense to encoded must be identity for valid indices.
#[test]
fn round_trip_dense_is_identity() {
    let blocks_per_proc = 4;
    let facts_per_proc = 8;
    for dense in 0..32 {
        assert_eq!(
            round_trip_dense(dense, blocks_per_proc, facts_per_proc),
            Some(dense)
        );
    }
}

#[test]
fn via_decodes_exact_csr_outputs_into_reused_buffers() {
    let dispatcher = CpuOracleDispatcher::new();
    let intra = [(0, 0, 1)];
    let expected = canonical_expected(1, 2, 1, &intra, &[], &[], &[]);
    let mut row_ptr = Vec::with_capacity(4);
    let mut col_idx = Vec::with_capacity(4);
    let row_ptr_ptr = row_ptr.as_ptr();
    let col_idx_ptr = col_idx.as_ptr();
    build_ifds_csr_via_into(
        &dispatcher,
        1,
        2,
        1,
        &intra,
        &[],
        &[],
        &[],
        &mut row_ptr,
        &mut col_idx,
    )
    .expect("Fix: CPU oracle IFDS dispatch succeeds");
    assert_eq!((row_ptr.clone(), col_idx.clone()), expected);
    assert_eq!(row_ptr.as_ptr(), row_ptr_ptr);
    assert_eq!(col_idx.as_ptr(), col_idx_ptr);
}

#[test]
fn via_refreshes_static_rule_inputs_for_same_shape_rule_content_change() {
    let dispatcher = RecordingIfdsOracle {
        inner: CpuOracleDispatcher::new(),
        intra_src_blocks: Mutex::new(Vec::new()),
    };
    let mut scratch = IfdsCsrGpuScratch::default();
    let mut row_ptr = Vec::new();
    let mut col_idx = Vec::new();

    build_ifds_csr_via_with_scratch_into(
        &dispatcher,
        1,
        2,
        1,
        &[(0, 0, 1)],
        &[],
        &[],
        &[],
        &mut scratch,
        &mut row_ptr,
        &mut col_idx,
    )
    .expect("Fix: first IFDS same-shape dispatch should succeed");
    build_ifds_csr_via_with_scratch_into(
        &dispatcher,
        1,
        2,
        1,
        &[(0, 1, 0)],
        &[],
        &[],
        &[],
        &mut scratch,
        &mut row_ptr,
        &mut col_idx,
    )
    .expect("Fix: second IFDS same-shape dispatch should refresh rule columns");

    let recorded = dispatcher
        .intra_src_blocks
        .lock()
        .expect("Fix: IFDS recording mutex should not be poisoned");
    assert_eq!(recorded.as_slice(), &[vec![0], vec![1]]);
    assert_eq!(
        scratch.program_builds(),
        1,
        "Fix: same-count IFDS rule changes should refresh static rule inputs without rebuilding the generated Program."
    );
}

#[test]
fn via_with_scratch_reuses_split_dispatch_decode_and_output_storage() {
    let dispatcher = CpuOracleDispatcher::new();
    let mut scratch = IfdsCsrGpuScratch::default();
    let mut row_ptr = Vec::with_capacity(3);
    let mut col_idx = Vec::with_capacity(1);
    let first_intra = [(0, 0, 1)];
    let second_intra = [(0, 1, 0)];
    let two_edge_intra = [(0, 0, 1), (0, 1, 0)];

    build_ifds_csr_via_with_scratch_into(
        &dispatcher,
        1,
        2,
        1,
        &first_intra,
        &[],
        &[],
        &[],
        &mut scratch,
        &mut row_ptr,
        &mut col_idx,
    )
    .expect("Fix: dispatch succeeds");

    let input_capacities = scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>();
    let intra_proc_capacity = scratch.rule_columns.intra_proc.capacity();
    let row_cursor_capacity = scratch.row_cursor.capacity();
    let col_len_capacity = scratch.col_len_words.capacity();
    let row_ptr_capacity = row_ptr.capacity();
    let col_idx_capacity = col_idx.capacity();
    assert_eq!(
        scratch.program_builds(),
        1,
        "first non-empty IFDS dispatch should materialize one primitive Program"
    );

    build_ifds_csr_via_with_scratch_into(
        &dispatcher,
        1,
        2,
        1,
        &first_intra,
        &[],
        &[],
        &[],
        &mut scratch,
        &mut row_ptr,
        &mut col_idx,
    )
    .expect("Fix: dispatch succeeds");

    assert_eq!(
        scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>(),
        input_capacities
    );
    assert_eq!(
        scratch.rule_columns.intra_proc.capacity(),
        intra_proc_capacity
    );
    assert_eq!(scratch.row_cursor.capacity(), row_cursor_capacity);
    assert_eq!(scratch.col_len_words.capacity(), col_len_capacity);
    assert_eq!(row_ptr.capacity(), row_ptr_capacity);
    assert_eq!(col_idx.capacity(), col_idx_capacity);
    assert_eq!(
        (row_ptr.clone(), col_idx.clone()),
        canonical_expected(1, 2, 1, &first_intra, &[], &[], &[])
    );
    assert_eq!(
        scratch.program_builds(),
        1,
        "same IFDS program shape should reuse the primitive generated Program"
    );

    build_ifds_csr_via_with_scratch_into(
        &dispatcher,
        1,
        2,
        1,
        &second_intra,
        &[],
        &[],
        &[],
        &mut scratch,
        &mut row_ptr,
        &mut col_idx,
    )
    .expect("Fix: same-shape dispatch with different rule values succeeds");
    assert_eq!(
        scratch.program_builds(),
        1,
        "IFDS program cache key must depend on primitive shape, not rule values"
    );

    build_ifds_csr_via_with_scratch_into(
        &dispatcher,
        1,
        2,
        1,
        &two_edge_intra,
        &[],
        &[],
        &[],
        &mut scratch,
        &mut row_ptr,
        &mut col_idx,
    )
    .expect("Fix: changed IFDS rule count should still dispatch");
    assert_eq!(
        scratch.program_builds(),
        2,
        "changed IFDS program shape must materialize a new primitive Program"
    );
}

#[test]
fn empty_via_path_does_not_materialize_program_or_dispatch() {
    let dispatcher = CpuOracleDispatcher::new();
    let mut scratch = IfdsCsrGpuScratch::default();
    let mut row_ptr = vec![99];
    let mut col_idx = vec![88];

    build_ifds_csr_via_with_scratch_into(
        &dispatcher,
        0,
        0,
        0,
        &[],
        &[],
        &[],
        &[],
        &mut scratch,
        &mut row_ptr,
        &mut col_idx,
    )
    .expect("Fix: empty no-rule IFDS dispatch should complete without backend work");

    assert_eq!(row_ptr, vec![0]);
    assert!(col_idx.is_empty());
    assert_eq!(
        scratch.program_builds(),
        0,
        "empty IFDS plan should not build a generated Program"
    );
    assert!(
        scratch.inputs.is_empty(),
        "empty IFDS plan should not prepare upload buffers"
    );
}

#[test]
fn via_rejects_extra_outputs() {
    let dispatcher = StaticOutputs::new(
        MALFORMED_IFDS_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0, 0]),
            u32_slice_to_le_bytes(&[0]),
            u32_slice_to_le_bytes(&[0]),
            u32_slice_to_le_bytes(&[0]),
            u32_slice_to_le_bytes(&[0]),
            u32_slice_to_le_bytes(&[0]),
        ],
    );
    let err = build_ifds_csr_via(&dispatcher, 1, 1, 1, &[], &[], &[], &[])
        .expect_err("extra outputs must be rejected");
    assert!(
        matches!(err, DispatchError::BackendError(_)),
        "unexpected error: {err:?}"
    );
}

#[test]
fn via_rejects_trailing_col_len_bytes() {
    let dispatcher = StaticOutputs::new(
        MALFORMED_IFDS_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[0, 0]),
            u32_slice_to_le_bytes(&[0]),
            u32_slice_to_le_bytes(&[0]),
            vec![0, 0, 0, 0, 1],
        ],
    );
    let err = build_ifds_csr_via(&dispatcher, 1, 1, 1, &[], &[], &[], &[])
        .expect_err("trailing col_len bytes must be rejected");
    assert!(
        matches!(err, DispatchError::BackendError(_)),
        "unexpected error: {err:?}"
    );
}

#[test]
fn via_rejects_inconsistent_row_ptr_readback() {
    let dispatcher = StaticOutputs::new(
        MALFORMED_IFDS_CONTRACT,
        vec![
            u32_slice_to_le_bytes(&[1, 1]),
            u32_slice_to_le_bytes(&[0]),
            u32_slice_to_le_bytes(&[0]),
            u32_slice_to_le_bytes(&[0]),
        ],
    );
    let err = build_ifds_csr_via(&dispatcher, 1, 1, 1, &[], &[], &[], &[])
        .expect_err("row_ptr[0] drift must be rejected");
    assert!(
        matches!(err, DispatchError::BackendError(_)),
        "unexpected error: {err:?}"
    );
}
