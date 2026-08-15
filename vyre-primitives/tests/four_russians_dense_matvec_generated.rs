//! The primitive arm of the declared dense byte-tile Four-Russians matvec table.
//!
//! This primitive is the packed graph/dataflow building block behind dense
//! frontier waves: eight source-column tests collapse into one LUT load per
//! destination word, then tiles are OR-reduced across the active frontier.
//!
//! Which cases exist, and what the answer is, belong to
//! `tests/support/dense_matvec_cases.rs`. This file pins only what
//! `vyre_primitives::bitset::four_russians` owes for those cases: the byte-LUT
//! builder, its word-count helper, the CPU reference, and the dispatch Program.

use vyre_primitives::bitset::four_russians::{
    dense_matvec_byte_lut, dense_matvec_byte_lut_words, dense_matvec_cpu_ref,
    four_russians_dense_matvec_byte_lut,
};
use vyre_primitives::wire::pack_u32_slice as u32_bytes;
use vyre_reference::value::Value;

#[path = "../../tests/support/dense_matvec_cases.rs"]
mod dense_matvec_cases;

use dense_matvec_cases::{arm_coverage, declared_groups, DenseMatvecCase, LutCache};

/// Every declared group has a primitive arm, and every case in it holds.
///
/// The ledger reads the table at run time, so a group declared with no branch
/// below fails here by name rather than silently going unrun.
#[test]
fn primitive_dense_matvec_arms_cover_every_declared_case_group() {
    let mut coverage = arm_coverage();
    for group in declared_groups() {
        match group.name {
            "frontier_sweep" | "single_tile_active_byte" | "saturated_frontier" => {
                assert_lut_reduction_matches_naive(&group.cases);
            }
            "dirty_output_overwrite" => {
                assert_program_overwrites_dirty_output(&group.cases);
            }
            _ => continue,
        }
        coverage.record(group.name, group.cases.len());
    }
    coverage.assert_covers_declared_table();
}

/// The LUT builder, its word-count helper and the CPU reference agree with the
/// naive boolean-semiring oracle on every case.
fn assert_lut_reduction_matches_naive(cases: &[DenseMatvecCase]) {
    let mut cache = LutCache::new();
    for case in cases {
        let (columns, lut) = cache.get(case, dense_matvec_byte_lut);
        assert_eq!(
            lut.len() as u32,
            dense_matvec_byte_lut_words(case.tile_count, case.dst_words),
            "Fix: LUT word-count helper drifted for {}.",
            case.label()
        );
        let frontier = case.frontier();
        assert_eq!(
            dense_matvec_cpu_ref(&frontier, lut, case.tile_count, case.dst_words),
            case.naive(columns, &frontier),
            "Fix: dense Four-Russians matvec drifted for {}.",
            case.label()
        );
    }
}

/// The dispatch Program overwrites a dirty output buffer instead of accumulating
/// into it.
fn assert_program_overwrites_dirty_output(cases: &[DenseMatvecCase]) {
    for case in cases {
        let columns = case.columns();
        let lut = dense_matvec_byte_lut(&columns, case.tile_count, case.dst_words);
        let frontier = case.frontier();
        let expected = case.naive(&columns, &frontier);
        let program = four_russians_dense_matvec_byte_lut(
            "frontier",
            "tile_lut",
            "out",
            case.tile_count,
            case.dst_words,
        );
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(u32_bytes(&frontier)),
                Value::from(u32_bytes(&lut)),
                Value::from(u32_bytes(&vec![u32::MAX; case.dst_words as usize])),
            ],
        )
        .expect("Fix: dense Four-Russians matvec Program must execute in the reference oracle.");

        assert_eq!(
            outputs[0].to_bytes(),
            u32_bytes(&expected),
            "Fix: dense Four-Russians matvec Program must overwrite dirty output with the LUT-reduced boolean matvec result for {}.",
            case.label()
        );
    }
}
