//! Neutral stability corpus well-formedness contracts.

use vyre_lower::program_stability_corpus::cases;

/// WHY: two goldens key their sections on these ids. A duplicate id would
/// silently overwrite one case's pinned section with another's.
#[test]
fn every_case_id_is_unique() {
    let mut ids = cases().into_iter().map(|case| case.id).collect::<Vec<_>>();
    let count = ids.len();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(
        ids.len(),
        count,
        "Fix: give every neutral stability case a distinct id."
    );
}

/// WHY: a case whose declared input buffers outnumber its supplied byte
/// vectors cannot be executed by the oracle, so it would pin nothing.
#[test]
fn every_case_supplies_bytes_for_every_read_buffer() {
    for case in cases() {
        let reads = case
            .program
            .buffers()
            .iter()
            .filter(|buffer| !buffer.is_output() && buffer.name() != "scratch")
            .count();
        assert_eq!(
            case.inputs.len(),
            reads,
            "Fix: case `{}` must supply one byte vector per read buffer.",
            case.id
        );
    }
}
