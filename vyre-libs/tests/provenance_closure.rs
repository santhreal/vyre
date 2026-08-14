//! P-MEAS-7: provenance-closure correctness corpus.
//!
//! Computes the standard provenance-closure corpus and checks its expected
//! lineage directly.
#![allow(missing_docs)]

use vyre_libs::encoding::scallop_provenance;

#[test]
fn provenance_closure_matches_expected_lineage() {
    let mut state = vec![0u32; 9];
    state[1] = 0b001;
    state[5] = 0b010;

    let join_rules = state.clone();
    let closure = scallop_provenance::reference_provenance_closure(&state, &join_rules, 3, 8);

    assert_eq!(
        closure,
        vec![0, 0b001, 0b011, 0, 0, 0b010, 0, 0, 0],
        "provenance closure must match the expected 0->1->2 lineage chain"
    );
}
