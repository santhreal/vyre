//! End-to-end parity for `data::reduction_metrics::*_via` through the shared faithful
//! [`common::ReferenceEvalDispatcher`], across every reduction the consumer exposes
//! (sum / max / min / count-non-zero / any / all scalar reduces, per-segment CSR sum, and the
//! atomic-scatter histogram).
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! the reduce / segment-reduce / histogram IRs are not run through a faithful dispatch boundary by any
//! `vyre-primitives/tests/*` file, and the consumer's only coverage is its own in-file dispatcher. This
//! is the FIRST-EVER execution of these kernels through a dispatch boundary that models the real
//! backend, for each reduction's distinct combine/scatter lowering.
//!
//! Contracts (audited CLEAN, batch-2): reduce_* bind values RO(0) + out RW(1) = 2 IC; segment_reduce
//! binds input RO(0) + segment_offsets RO(1) + output RW(2) = 3 IC; histogram binds input RO(0) +
//! output RW(1) = 2 IC. All decode outputs[0] = the sole writable buffer. Every op is exact integer
//! arithmetic → BIT-EXACT (no tolerance).
#![cfg(feature = "cpu-parity")]

use vyre_self_substrate::data::reduction_metrics::{
    histogram_atomic_scatter_via, reduce_all_via, reduce_any_via, reduce_count_non_zero_via,
    reduce_max_via, reduce_min_via, reduce_sum_via, reference_histogram_atomic_scatter,
    reference_segment_reduce_sum, segment_reduce_sum_via,
};

mod common;
use common::ReferenceEvalDispatcher;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
fn scalar_reduces_via_match_independent_oracles() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x2E_D0_C0_01u32;
    let mut saw_all_true = 0u32;
    let mut saw_some_zero = 0u32;
    for case in 0..400u32 {
        let n = 1 + (case % 64) as usize;
        // Bounded values so the sum never overflows u32 (n<=64, v<10_000 → sum < 640_000). ~1/8
        // entries are forced 0 to exercise count-non-zero / any / all against a mixed vector.
        let values: Vec<u32> = (0..n)
            .map(|_| {
                if xorshift(&mut state) % 8 == 0 {
                    0
                } else {
                    xorshift(&mut state) % 10_000
                }
            })
            .collect();

        let want_sum: u32 = values.iter().copied().sum();
        let want_max = values.iter().copied().max().unwrap();
        let want_min = values.iter().copied().min().unwrap();
        let want_count = values.iter().filter(|&&v| v != 0).count() as u32;
        let want_any = values.iter().any(|&v| v != 0);
        let want_all = values.iter().all(|&v| v != 0);

        assert_eq!(
            reduce_sum_via(&dispatcher, &values).unwrap(),
            want_sum,
            "case {case}: sum mismatch; values={values:?}"
        );
        assert_eq!(
            reduce_max_via(&dispatcher, &values).unwrap(),
            want_max,
            "case {case}: max mismatch; values={values:?}"
        );
        assert_eq!(
            reduce_min_via(&dispatcher, &values).unwrap(),
            want_min,
            "case {case}: min mismatch; values={values:?}"
        );
        assert_eq!(
            reduce_count_non_zero_via(&dispatcher, &values).unwrap(),
            want_count,
            "case {case}: count-non-zero mismatch; values={values:?}"
        );
        assert_eq!(
            reduce_any_via(&dispatcher, &values).unwrap(),
            want_any,
            "case {case}: any mismatch; values={values:?}"
        );
        assert_eq!(
            reduce_all_via(&dispatcher, &values).unwrap(),
            want_all,
            "case {case}: all mismatch; values={values:?}"
        );

        if want_all {
            saw_all_true += 1;
        }
        if !want_all {
            saw_some_zero += 1;
        }
    }
    // Both the all-nonzero branch and the has-a-zero branch must be genuinely exercised.
    assert!(
        saw_all_true > 20 && saw_some_zero > 100,
        "reduce sweep must exercise both all-nonzero and has-zero vectors: all_true={saw_all_true} some_zero={saw_some_zero}"
    );
}

#[test]
fn segment_reduce_sum_via_matches_cpu_ref_over_random_csr_partitions() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x5E_60_00_01u32;
    let mut nonempty_and_empty = 0u32;
    for case in 0..300u32 {
        let n = 1 + (case % 40) as usize;
        let values: Vec<u32> = (0..n).map(|_| xorshift(&mut state) % 5_000).collect();

        // Build a valid CSR partition: cut points sorted in [0, n], with 0 and n as the ends.
        let num_segments = 1 + (case % 5) as usize; // 1..5 segments
        let mut cuts: Vec<u32> = (0..num_segments.saturating_sub(1))
            .map(|_| xorshift(&mut state) % (n as u32 + 1))
            .collect();
        cuts.push(0);
        cuts.push(n as u32);
        cuts.sort_unstable();
        let offsets = cuts; // len = num_segments+1, non-decreasing, ends 0..n → valid CSR

        let got = segment_reduce_sum_via(&dispatcher, &values, &offsets)
            .expect("segment_reduce_sum_via must dispatch");
        let want = reference_segment_reduce_sum(&values, &offsets);
        assert_eq!(
            got, want,
            "case {case}: per-segment sum must match cpu_ref; values={values:?} offsets={offsets:?}"
        );

        // A partition with both an empty segment (equal cuts) and a nonempty one is the interesting case.
        if want.iter().any(|&s| s == 0) && want.iter().any(|&s| s != 0) {
            nonempty_and_empty += 1;
        }
    }
    assert!(
        nonempty_and_empty > 30,
        "segment sweep must exercise mixed empty/nonempty partitions, got {nonempty_and_empty}"
    );
}

#[test]
fn histogram_via_matches_cpu_ref_over_random_bin_indices() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x41_57_00_01u32;
    let mut saw_multi_count_bin = 0u32;
    for case in 0..300u32 {
        let num_bins = 1 + (case % 8); // 1..8
        let n = 1 + (case % 50) as usize;
        // Bin indices in [0, num_bins) so every input lands in a real bin.
        let input: Vec<u32> = (0..n).map(|_| xorshift(&mut state) % num_bins).collect();

        let got = histogram_atomic_scatter_via(&dispatcher, &input, num_bins)
            .expect("histogram_atomic_scatter_via must dispatch");
        let want = reference_histogram_atomic_scatter(&input, num_bins);
        assert_eq!(
            got, want,
            "case {case}: histogram counts must match cpu_ref; num_bins={num_bins} input={input:?}"
        );
        assert_eq!(
            want.iter().sum::<u32>(),
            n as u32,
            "case {case}: histogram total must equal input length"
        );
        if want.iter().any(|&c| c >= 2) {
            saw_multi_count_bin += 1;
        }
    }
    assert!(
        saw_multi_count_bin > 100,
        "histogram sweep must exercise bins with multiple atomic-scatter hits, got {saw_multi_count_bin}"
    );
}

#[test]
fn reduction_via_hand_checked_cases() {
    let d = ReferenceEvalDispatcher;

    let v = vec![3, 1, 4, 1, 5, 9, 2, 6];
    assert_eq!(reduce_sum_via(&d, &v).unwrap(), 31, "sum of the vector");
    assert_eq!(reduce_max_via(&d, &v).unwrap(), 9, "max");
    assert_eq!(reduce_min_via(&d, &v).unwrap(), 1, "min");
    assert_eq!(
        reduce_count_non_zero_via(&d, &v).unwrap(),
        8,
        "all nonzero → count 8"
    );
    assert!(reduce_all_via(&d, &v).unwrap(), "all nonzero");

    let with_zero = vec![0, 7, 0, 3];
    assert_eq!(
        reduce_count_non_zero_via(&d, &with_zero).unwrap(),
        2,
        "two nonzero"
    );
    assert!(reduce_any_via(&d, &with_zero).unwrap(), "some nonzero");
    assert!(!reduce_all_via(&d, &with_zero).unwrap(), "not all nonzero");
    assert_eq!(
        reduce_min_via(&d, &with_zero).unwrap(),
        0,
        "min includes the zero"
    );

    // Segments: [10,20 | 30 | (empty) | 40,50] via CSR offsets [0,2,3,3,5].
    let vals = vec![10, 20, 30, 40, 50];
    let offsets = vec![0, 2, 3, 3, 5];
    assert_eq!(
        segment_reduce_sum_via(&d, &vals, &offsets).unwrap(),
        vec![30, 30, 0, 90],
        "per-segment sums with an empty middle segment"
    );

    // Histogram: indices into 3 bins → counts.
    let idx = vec![0, 2, 2, 1, 2, 0];
    assert_eq!(
        histogram_atomic_scatter_via(&d, &idx, 3).unwrap(),
        vec![2, 1, 3],
        "bin counts: two 0s, one 1, three 2s"
    );
}
