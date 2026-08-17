//! Contracts for GPU-resident region dedup survivor flags.
//! Uses the shared sequential witness to validate survivor-flag semantics.

#![cfg(feature = "pattern")]

use vyre_libs::pattern::{dedup_regions_flag_program, RegionTriple};
use vyre_reference::composition_witness::{
    dedup_regions_survivor_flags_witness, dedup_regions_witness,
};
fn reference_dedup_regions(input: Vec<RegionTriple>) -> Vec<RegionTriple> {
    let tuples: Vec<(u32, u32, u32)> = input.iter().map(|r| (r.pid, r.start, r.end)).collect();
    let deduped = dedup_regions_witness(tuples);
    deduped
        .into_iter()
        .map(|(pid, start, end)| RegionTriple::new(pid, start, end))
        .collect()
}

fn reference_flags(sorted: &[RegionTriple]) -> Vec<u32> {
    let tuples: Vec<(u32, u32, u32)> = sorted.iter().map(|r| (r.pid, r.start, r.end)).collect();
    dedup_regions_survivor_flags_witness(&tuples)
}

#[test]
fn flag_program_emitted_with_expected_buffer_count() {
    let prog = dedup_regions_flag_program("p", "s", "e", "f", 8);
    assert!(!prog.entry().is_empty());
    assert_eq!(prog.workgroup_size[1], 1);
    assert_eq!(prog.workgroup_size[2], 1);
    assert_eq!(prog.buffers.len(), 4);
    assert_eq!(prog.buffers[0].count, 8);
}

#[test]
fn flag_predicate_matches_reference_on_canonical_inputs() {
    let scenarios: &[Vec<RegionTriple>] = &[
        vec![],
        vec![RegionTriple::new(0, 5, 10)],
        vec![RegionTriple::new(0, 5, 10), RegionTriple::new(0, 5, 10)],
        vec![RegionTriple::new(0, 5, 10), RegionTriple::new(0, 7, 12)],
        vec![RegionTriple::new(0, 5, 10), RegionTriple::new(0, 10, 15)],
        vec![RegionTriple::new(0, 5, 10), RegionTriple::new(1, 5, 10)],
        vec![RegionTriple::new(0, 5, 5), RegionTriple::new(1, 5, 5)],
    ];
    for scenario in scenarios {
        let mut sorted = scenario.clone();
        sorted.sort_unstable();
        let flags = reference_flags(&sorted);
        let reference_dedup = reference_dedup_regions(scenario.clone());
        let expected_survivors = flags.iter().filter(|&&f| f == 1).count();
        assert_eq!(
            expected_survivors,
            reference_dedup.len(),
            "Fix: flag-program survivor count must match reference dedup output count for {scenario:?}"
        );
    }
}
