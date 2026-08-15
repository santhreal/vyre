//! The substrate arm of the declared dense byte-tile Four-Russians matvec table.
//!
//! The primitive-level dense Four-Russians matvec is only useful if the
//! self-substrate exposes it as a reusable transform for graph/dataflow
//! schedulers. This file keeps that wiring broad and deterministic.
//!
//! Which cases exist, and what the answer is, belong to
//! `tests/support/dense_matvec_cases.rs`. This file pins only what
//! `vyre_libs::encoding::bitset_transform_pipeline` owes for those cases: its
//! frontier and LUT sizing, its LUT builder, its CPU parity oracle, and the
//! Program it composes.

use vyre_libs::encoding::bitset_transform_pipeline::{
    dense_boolean_matvec_lut, dense_matvec_frontier_words, dense_matvec_lut_words,
    four_russians_dense_matvec_program, reference_dense_boolean_matvec,
};

#[path = "../../tests/support/dense_matvec_cases.rs"]
mod dense_matvec_cases;

use dense_matvec_cases::{
    arm_coverage, assert_program_overwrites_dirty_output, declared_groups, DenseMatvecCase,
    LutCache,
};

/// Every declared group has a substrate arm, and every case in it holds.
///
/// The ledger reads the table at run time, so a group declared with no branch
/// below fails here by name rather than silently going unrun.
#[test]
fn substrate_dense_matvec_arms_cover_every_declared_case_group() {
    let mut coverage = arm_coverage();
    for group in declared_groups() {
        match group.name {
            "frontier_sweep" | "single_tile_active_byte" | "saturated_frontier" => {
                assert_transform_matches_naive(&group.cases);
            }
            "dirty_output_overwrite" => {
                assert_program_overwrites_dirty_output(
                    "self-substrate",
                    &group.cases,
                    dense_boolean_matvec_lut,
                    four_russians_dense_matvec_program,
                );
            }
            _ => continue,
        }
        coverage.record(group.name, group.cases.len());
    }
    coverage.assert_covers_declared_table();
}

/// The substrate sizing helpers, LUT builder and parity oracle agree with the
/// naive boolean-semiring oracle on every case.
fn assert_transform_matches_naive(cases: &[DenseMatvecCase]) {
    let mut cache = LutCache::new();
    for case in cases {
        let (columns, lut) = cache.get(case, dense_boolean_matvec_lut);
        assert_eq!(
            dense_matvec_lut_words(case.tile_count, case.dst_words) as usize,
            lut.len(),
            "Fix: self-substrate LUT sizing drifted for {}.",
            case.label()
        );
        let frontier = case.frontier();
        assert_eq!(
            dense_matvec_frontier_words(case.tile_count) as usize,
            frontier.len(),
            "Fix: self-substrate frontier sizing drifted for {}.",
            case.label()
        );
        assert_eq!(
            reference_dense_boolean_matvec(&frontier, lut, case.tile_count, case.dst_words),
            case.naive(columns, &frontier),
            "Fix: self-substrate dense matvec transform drifted for {}.",
            case.label()
        );
    }
}
