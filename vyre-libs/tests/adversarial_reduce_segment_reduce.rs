//! Adversarial oracle tests for `reduce::segment_reduce`.

#![allow(unused_imports, dead_code, clippy::identity_op)]

use vyre_libs::reduce::segment_reduce::*;
use vyre_reference::composition_witness::segment_reduce_sum_witness as reference_segment_reduce_sum;

#[test]
fn segment_reduce_hostile_corpus() {
    let cases: &[(&[u32], &[u32], &[u32])] = &[
        (&[], &[0], &[]),
        (&[1, 2, 3], &[0, 3], &[6]),
        (&[10, 20, 30, 40], &[0, 2, 4], &[30, 70]),
        (&[0xffff_ffff, 1], &[0, 1, 2], &[0xffff_ffff, 1]),
    ];
    for (idx, (input, offsets, expected)) in cases.iter().enumerate() {
        assert_eq!(
            reference_segment_reduce_sum(input, offsets),
            *expected,
            "Fix: segment_reduce oracle mismatch on case {idx}"
        );
    }
}

#[test]
#[should_panic(expected = "monotonic segment offsets")]
fn segment_reduce_rejects_non_monotonic_offsets() {
    let _ = reference_segment_reduce_sum(&[1, 2, 3], &[0, 3, 2]);
}
