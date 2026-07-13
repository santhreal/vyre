//! End-to-end parity for `data::matching_diagnostic_compaction`'s three `_via` entry points
//! (`bracket_pairs_via`, `sort_regions_via`, `dedup_region_survivor_flags_via`) through the shared
//! faithful [`common::ReferenceEvalDispatcher`].
//!
//! Closes another mock-dispatcher-coherence family (see BACKLOG
//! `SWEEP-via-consumer-input-output-contract-audit`). The consumer's in-file `MatchingDispatcher`
//! mock hand-returns only the intended result buffer as `outputs[0]`, so it never modelled the real
//! backend's dispatch contract, which surfaced TWO real latent GPU-breaking bugs the audit found
//! and this test locks fixed:
//!
//!  * **`bracket_pairs_via` double-bug.** `bracket_match` binds `kinds` ReadOnly(0), `stack` plain
//!    ReadWrite(1, InputOutput), `match_pairs` `BufferDecl::output`(2, backend-allocated). So only
//!    TWO buffers are input-consuming (kinds + stack), yet the consumer passed THREE inputs (feeding
//!    a dead `MATCH_NONE` seed for the output-allocated `match_pairs`, whose entries the kernel
//!    initializes itself at bracket_match.rs's `store(match_pairs, i, MATCH_NONE)`) → OVER-FEED, a
//!    hard "expected 2, received 3" on a real backend. AND the writable buffers returned in binding
//!    order are `[stack, match_pairs]`, so `match_pairs` is `outputs[1]`: but the decode read
//!    `outputs[0]` (= the `stack` scratch), i.e. the wrong buffer entirely.
//!  * **`dedup_region_survivor_flags_via` over-feed.** `dedup_regions_flag_program` binds
//!    pids/starts/ends ReadOnly(0-2) + `survivors` `BufferAccess::WriteOnly`(3, backend-allocated) =
//!    THREE input-consuming, yet the consumer passed FOUR (a zero slot for the WriteOnly survivors)
//!    → OVER-FEED, "expected 3, received 4".
//!
//! `sort_regions_via` (region_sort_program: RO×3 + ReadWrite outs×3 = 6 input-consuming, passes 6,
//! `outputs[0]=pids_out`) was audited CLEAN and is covered here as a durable guard.
#![cfg(feature = "cpu-parity")]

use vyre_primitives::matching::bracket_match::{
    cpu_ref as bracket_cpu_ref, CLOSE_BRACE, OPEN_BRACE,
};
use vyre_primitives::matching::region::RegionTriple;
use vyre_self_substrate::data::matching_diagnostic_compaction::{
    bracket_pairs_via, dedup_region_survivor_flags_via, reference_dedup_regions,
    reference_sort_regions, sort_regions_via,
};

mod common;
use common::ReferenceEvalDispatcher;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// A random brace-token stream: `OPEN_BRACE` / `CLOSE_BRACE` / `OTHER(0)` in balanced-ish mix.
fn random_kinds(state: &mut u32, len: usize) -> Vec<u32> {
    (0..len)
        .map(|_| match xorshift(state) % 4 {
            0 => OPEN_BRACE,
            1 => CLOSE_BRACE,
            _ => 0, // OTHER
        })
        .collect()
}

#[test]
fn bracket_pairs_via_matches_primitive_cpu_oracle_over_generated_streams() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x0BEE_F00Du32;
    let mut nontrivial = 0u32;
    for case in 0..400u32 {
        let len = 1 + (case as usize % 48);
        let max_depth = 4 + (case % 12);
        let kinds = random_kinds(&mut state, len);

        let pairs = bracket_pairs_via(&dispatcher, &kinds, max_depth)
            .expect("bracket_pairs_via must dispatch the bracket-match kernel");
        let expected = bracket_cpu_ref(&kinds, max_depth);

        assert_eq!(
            pairs, expected,
            "case {case}: bracket_pairs_via must return match_pairs (outputs[1]), not the stack \
             scratch; kinds={kinds:?}"
        );
        // A case with at least one real pairing exercises the non-sentinel write path.
        if expected.iter().any(|&m| m != u32::MAX) {
            nontrivial += 1;
        }
    }
    assert!(
        nontrivial > 200,
        "expected >200 streams with a real brace pairing, got {nontrivial}"
    );
}

#[test]
fn bracket_pairs_via_matches_known_nested_pairs() {
    let dispatcher = ReferenceEvalDispatcher;
    // "( ( ) ( ) )" → outer 0-5, inner 1-2, inner 3-4.
    let kinds = vec![
        OPEN_BRACE,
        OPEN_BRACE,
        CLOSE_BRACE,
        OPEN_BRACE,
        CLOSE_BRACE,
        CLOSE_BRACE,
    ];
    let pairs = bracket_pairs_via(&dispatcher, &kinds, 8).unwrap();
    assert_eq!(pairs, vec![5, 2, 1, 4, 3, 0]);
}

#[test]
fn sort_regions_via_matches_cpu_sort_over_generated_batches() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x5057_A17Eu32;
    for case in 0..300u32 {
        let count = 1 + (case as usize % 96);
        let regions = (0..count)
            .map(|_| {
                let pid = xorshift(&mut state) % 5;
                let start = xorshift(&mut state) % 512;
                let width = xorshift(&mut state) % 32;
                RegionTriple::new(pid, start, start + width)
            })
            .collect::<Vec<_>>();

        let sorted = sort_regions_via(&dispatcher, &regions)
            .expect("sort_regions_via must dispatch the region-sort kernel");
        assert_eq!(
            sorted,
            reference_sort_regions(regions.clone()),
            "case {case}: region sort must match the CPU (pid,start,end) order"
        );
    }
}

#[test]
fn dedup_survivor_flags_via_marks_same_cluster_starts_as_cpu_over_generated_batches() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0xD1CE_C0DEu32;
    let mut nontrivial = 0u32;
    for case in 0..400u32 {
        let count = 1 + (case as usize % 128);
        let mut regions = (0..count)
            .map(|_| {
                let pid = xorshift(&mut state) % 6;
                let start = xorshift(&mut state) % 256;
                let width = xorshift(&mut state) % 48;
                RegionTriple::new(pid, start, start + width)
            })
            .collect::<Vec<_>>();
        regions.sort();

        let flags = dedup_region_survivor_flags_via(&dispatcher, &regions)
            .expect("dedup_region_survivor_flags_via must dispatch the region-dedup kernel");
        assert_eq!(
            flags.len(),
            regions.len(),
            "case {case}: one flag per region"
        );

        let actual_starts = regions
            .iter()
            .zip(flags.iter())
            .filter_map(|(r, f)| (*f != 0).then_some((r.pid, r.start)))
            .collect::<Vec<_>>();
        let expected_starts = reference_dedup_regions(regions.clone())
            .into_iter()
            .map(|r| (r.pid, r.start))
            .collect::<Vec<_>>();
        assert_eq!(
            actual_starts, expected_starts,
            "case {case}: survivor flags must mark the same cluster starts as CPU dedup"
        );
        if actual_starts.len() < regions.len() {
            nontrivial += 1;
        }
    }
    assert!(
        nontrivial > 100,
        "expected >100 batches with a real dedup (clustered regions), got {nontrivial}"
    );
}
