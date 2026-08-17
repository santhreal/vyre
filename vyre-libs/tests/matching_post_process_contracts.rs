//! Contracts for canonical match post-processing and engine pipelines.

#![cfg(all(feature = "pattern", feature = "cpu-parity"))]
#![allow(deprecated)]
use vyre_foundation::match_result::ByteRange;
use vyre_libs::pattern::{
    shannon_entropy_bits_per_byte, try_reference_post_process, PostProcessError,
};

#[test]
fn empty_matches_yields_empty_output() {
    assert!(try_reference_post_process(&[], b"hello")
        .expect("Fix: empty post-process input must be valid")
        .is_empty());
}

#[test]
fn invalid_match_range_is_an_error_not_silent_drop() {
    let matches = [ByteRange::new(0, 0, 100)];
    let error = try_reference_post_process(&matches, b"hi")
        .expect_err("Fix: corrupt hit ranges must surface a contract error");
    let message = error.to_string();
    assert!(
        message.contains("pattern_id=0") && message.contains("haystack_len=2"),
        "Fix: invalid range diagnostics must identify the corrupt hit and haystack length: {message}"
    );
}

#[test]
fn corrupt_match_range_uses_result_error_contract() {
    let matches = [ByteRange::new(0, 0, 100)];
    let error = try_reference_post_process(&matches, b"hi")
        .expect_err("Fix: corrupt hit ranges must surface a Result error");
    assert!(matches!(error, PostProcessError::InvalidRange { .. }));
}

#[test]
fn entropy_zero_for_constant_input() {
    let bytes = vec![b'A'; 64];
    assert_eq!(shannon_entropy_bits_per_byte(&bytes), 0.0);
}

#[test]
fn entropy_eight_for_uniform_byte_distribution() {
    let bytes: Vec<u8> = (0..=255).cycle().take(2048).collect();
    let h = shannon_entropy_bits_per_byte(&bytes);
    assert!(
        (h - 8.0).abs() < 1e-3,
        "Fix: uniform-byte distribution must hit 8 bits/byte, got {h}"
    );
}

#[test]
fn dedup_collapses_same_pid_overlap_in_post_process() {
    let matches = [
        ByteRange::new(0, 0, 5),
        ByteRange::new(0, 3, 7),
        ByteRange::new(1, 10, 12),
    ];
    let haystack = b"abcdefghijkl";
    let out = try_reference_post_process(&matches, haystack)
        .expect("Fix: canonical in-bounds ranges must post-process");
    assert_eq!(
        out.len(),
        2,
        "Fix: same-pid overlap collapses and distinct pid stays"
    );
    assert!(out
        .iter()
        .any(|m| m.pattern_id == 0 && m.start == 0 && m.end == 7));
}

#[test]
fn confidence_increases_with_high_entropy_long_input() {
    let high_input: Vec<u8> = (0..16).collect();
    let m_long = ByteRange::new(0, 0, 16);
    let out = try_reference_post_process(&[m_long], &high_input)
        .expect("Fix: high-entropy fixture range must be valid");
    assert!(
        out[0].confidence > 0.4,
        "Fix: high-entropy 16 bytes should carry a high confidence signal"
    );

    let low_input = b"aaaa";
    let m_short = ByteRange::new(0, 0, 4);
    let out_low = try_reference_post_process(&[m_short], low_input)
        .expect("Fix: low-entropy fixture range must be valid");
    assert!((out_low[0].confidence - 0.0).abs() < 1e-6);
}

#[test]
fn output_is_sorted_by_pid_then_start_then_end() {
    let matches = [
        ByteRange::new(2, 5, 7),
        ByteRange::new(0, 1, 4),
        ByteRange::new(1, 0, 3),
    ];
    let haystack = b"abcdefghij";
    let out = try_reference_post_process(&matches, haystack)
        .expect("Fix: sortedness fixture ranges must be valid");
    for w in out.windows(2) {
        assert!(
            (w[0].pattern_id, w[0].start, w[0].end) <= (w[1].pattern_id, w[1].start, w[1].end),
            "Fix: post_process output must be sorted by (pid, start, end)"
        );
    }
}
