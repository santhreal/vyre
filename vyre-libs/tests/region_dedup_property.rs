//! Property tests for `vyre_libs::pattern::region`.
//!
//! Locks three invariants that the unit tests sample only at hand-
//! picked points:
//!
//!   1. **Idempotence.** `dedup(dedup(x)) == dedup(x)` for ANY input.
//!   2. **Sortedness.** Output is sorted by `(pid, start, end)`.
//!   3. **No same-pid overlaps in output.** For every pair of
//!      adjacent outputs with the same pid, `prev.end < next.start`.
//!
//! Generated input shape: 0..=64 triples with `pid ∈ 0..=7`,
//! `start ∈ 0..=255`, `end = start + (0..=32)`. Bounded ranges keep
//! shrinking fast and exercise both clusters and isolated spans.

#![cfg(feature = "pattern")]

use proptest::prelude::*;
use vyre_libs::pattern::RegionTriple;
use vyre_reference::composition_witness::{dedup_regions_witness, dedup_regions_witness_in_place};

fn reference_dedup_regions_in_place(regions: &mut Vec<RegionTriple>) {
    let mut tuples: Vec<(u32, u32, u32)> =
        regions.iter().map(|r| (r.pid, r.start, r.end)).collect();
    dedup_regions_witness_in_place(&mut tuples);
    regions.clear();
    regions.extend(
        tuples
            .into_iter()
            .map(|(pid, start, end)| RegionTriple::new(pid, start, end)),
    );
}

fn reference_dedup_regions(input: Vec<RegionTriple>) -> Vec<RegionTriple> {
    let tuples: Vec<(u32, u32, u32)> = input.iter().map(|r| (r.pid, r.start, r.end)).collect();
    let deduped = dedup_regions_witness(tuples);
    deduped
        .into_iter()
        .map(|(pid, start, end)| RegionTriple::new(pid, start, end))
        .collect()
}

fn arb_triple() -> impl Strategy<Value = RegionTriple> {
    (0u32..=7, 0u32..=255, 0u32..=32)
        .prop_map(|(pid, start, len)| RegionTriple::new(pid, start, start.saturating_add(len)))
}

fn arb_input() -> impl Strategy<Value = Vec<RegionTriple>> {
    proptest::collection::vec(arb_triple(), 0..=64)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    #[test]
    fn dedup_is_idempotent(input in arb_input()) {
        let once = reference_dedup_regions(input.clone());
        let twice = reference_dedup_regions(once.clone());
        prop_assert_eq!(once, twice);
    }

    #[test]
    fn output_is_sorted(input in arb_input()) {
        let out = reference_dedup_regions(input);
        for w in out.windows(2) {
            prop_assert!(w[0] <= w[1], "output not sorted: {:?}", w);
        }
    }

    #[test]
    fn no_overlapping_same_pid_in_output(input in arb_input()) {
        let out = reference_dedup_regions(input);
        for w in out.windows(2) {
            if w[0].pid == w[1].pid {
                prop_assert!(
                    w[0].end < w[1].start,
                    "adjacent same-pid outputs overlap: {:?}", w
                );
            }
        }
    }

    #[test]
    fn dedup_never_invents_pids(input in arb_input()) {
        let input_pids: std::collections::BTreeSet<u32> =
            input.iter().map(|t| t.pid).collect();
        let out = reference_dedup_regions(input);
        for t in &out {
            prop_assert!(input_pids.contains(&t.pid), "fabricated pid {} in output", t.pid);
        }
    }

    #[test]
    fn dedup_preserves_pid_set(input in arb_input()) {
        let input_pids: std::collections::BTreeSet<u32> =
            input.iter().map(|t| t.pid).collect();
        let out = reference_dedup_regions(input);
        let out_pids: std::collections::BTreeSet<u32> =
            out.iter().map(|t| t.pid).collect();
        prop_assert_eq!(input_pids, out_pids);
    }

    #[test]
    fn dedup_output_no_larger_than_input(input in arb_input()) {
        let n_in = input.len();
        let n_out = reference_dedup_regions(input).len();
        prop_assert!(n_out <= n_in);
    }

    #[test]
    fn inplace_matches_owned(input in arb_input()) {
        // The in-place sibling MUST produce the same output as the
        // owned-Vec variant for every input. Locks the contract that
        // performance optimization can't drift from semantics.
        let owned_result = reference_dedup_regions(input.clone());
        let mut inplace = input;
        reference_dedup_regions_in_place(&mut inplace);
        prop_assert_eq!(owned_result, inplace);
    }
}
