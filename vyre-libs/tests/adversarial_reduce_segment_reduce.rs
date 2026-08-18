//! Adversarial oracle tests for `reduce::segment_reduce`.
//!
//! WHY: Closes the defect class where non-monotonic segment offsets were lumped into a
//! generic "malformed segment offsets" error rather than specifically rejected as
//! non-monotonic. Ensures exact error classification between non-monotonic offsets and
//! out-of-bounds segment offsets, protects the caller output buffer across failure paths,
//! and validates adversarial input shapes.

#![allow(unused_imports, dead_code, clippy::identity_op)]

use vyre_libs::reduce::segment_reduce::*;
use vyre_reference::composition_witness::{
    segment_reduce_sum_witness as reference_segment_reduce_sum,
    segment_reduce_sum_witness_into as reference_segment_reduce_sum_into,
    try_segment_reduce_sum_witness_into as try_reference_segment_reduce_sum_into,
};

#[test]
fn segment_reduce_hostile_corpus() {
    let cases: &[(&[u32], &[u32], &[u32])] = &[
        (&[], &[], &[]),
        (&[], &[0], &[]),
        (&[], &[0, 0], &[0]),
        (&[], &[0, 0, 0], &[0, 0]),
        (&[1, 2, 3], &[0, 3], &[6]),
        (&[10, 20, 30, 40], &[0, 2, 4], &[30, 70]),
        (&[0xffff_ffff, 1], &[0, 1, 2], &[0xffff_ffff, 1]),
        (&[0xffff_ffff, 2], &[0, 2], &[1]),
        (&[5, 10, 15], &[0, 0, 2, 2, 3], &[0, 15, 0, 15]),
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

#[test]
#[should_panic(expected = "monotonic segment offsets")]
fn segment_reduce_rejects_immediate_descending_pair() {
    let _ = reference_segment_reduce_sum(&[1, 2, 3], &[2, 1]);
}

#[test]
#[should_panic(expected = "monotonic segment offsets")]
fn segment_reduce_rejects_non_monotonic_in_multi_segment() {
    let _ = reference_segment_reduce_sum(&[1, 2, 3, 4], &[0, 1, 3, 2, 4]);
}

#[test]
#[should_panic(expected = "monotonic segment offsets")]
fn segment_reduce_rejects_non_monotonic_at_tail() {
    let _ = reference_segment_reduce_sum(&[1, 2, 3, 4], &[0, 2, 4, 3]);
}

#[test]
fn segment_reduce_fallible_error_classification_contracts() {
    let mut out = vec![0xDEAD_BEEF, 0xCAFE_BABE];
    let snapshot = out.clone();

    // Non-monotonic offsets must return specific non-monotonic error class
    let err_non_mono = try_reference_segment_reduce_sum_into(&[1, 2, 3], &[0, 3, 2], &mut out)
        .expect_err("non-monotonic offsets must fail");
    assert!(
        err_non_mono.contains("monotonic segment offsets"),
        "expected non-monotonic error class, got `{err_non_mono}`"
    );
    assert_eq!(out, snapshot, "output buffer must not be mutated on failure");

    // Out-of-bounds offset (monotonic start <= end, but end > input.len()) must return malformed error
    let err_oob = try_reference_segment_reduce_sum_into(&[1, 2, 3], &[0, 4], &mut out)
        .expect_err("out-of-bounds offset must fail");
    assert!(
        err_oob.contains("malformed segment offsets"),
        "expected malformed segment error class, got `{err_oob}`"
    );
    assert_eq!(out, snapshot, "output buffer must not be mutated on failure");

    // Single out-of-bounds offset
    let err_single_oob = try_reference_segment_reduce_sum_into(&[1, 2, 3], &[5], &mut out)
        .expect_err("single out-of-bounds offset must fail");
    assert!(
        err_single_oob.contains("malformed segment offsets"),
        "expected malformed segment error class, got `{err_single_oob}`"
    );
    assert_eq!(out, snapshot, "output buffer must not be mutated on failure");

    // Valid case populates output and clears previous values
    try_reference_segment_reduce_sum_into(&[10, 20, 30], &[0, 2, 3], &mut out)
        .expect("valid offsets must succeed");
    assert_eq!(out, vec![30, 30]);
}
