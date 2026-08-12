//! Real-GPU parity and reuse coverage for [`ResidentScanSession`].
//!
//! The resident session uploads the NFA tables once, reuses one authenticated
//! artifact across submissions, and must match the backend-neutral reference scan.

use vyre_driver_wgpu as _;
use vyre_scan::{build_scan_session, ResidentScanSession};

/// A pattern set large enough that the lane-major transition table is non-trivial
/// (this is the table the resident path avoids re-uploading every scan).
const PATTERNS: &[&str] = &[
    "abc", "abd", "bcd", "cde", "def", "key", "token", "secret", "passwd", "AKIA",
];

const MAX_MATCHES: u32 = 10_000;

#[test]
fn resident_rule_pipeline_matches_reference_on_real_gpu() {
    let haystacks: &[&[u8]] = &[
        b"zabcd",
        b"the api key=secret and the token=AKIAEXAMPLE passwd=def",
        b"no matches here at all, just prose without any pattern bytes",
        b"",
        b"abcabcabc def def secret secret token",
    ];

    // Size the resident haystack buffer to the largest haystack under test.
    let capacity = haystacks.iter().map(|h| h.len()).max().unwrap_or(0).max(1);
    let pipeline = build_scan_session(PATTERNS, "input", "hits", capacity as u32);

    let session: ResidentScanSession = pipeline
        .prepare_resident("wgpu", capacity, MAX_MATCHES)
        .expect("Fix: WGPU must materialize the resident scan artifact");

    let mut scratch = Vec::new();
    let mut resident_matches = Vec::new();

    for haystack in haystacks {
        let expected = pipeline.reference_scan(haystack);

        session
            .scan_into(haystack, &mut resident_matches, &mut scratch)
            .expect("resident artifact submission");

        assert_eq!(
            resident_matches,
            expected,
            "resident scan diverged from borrowed/reference for {:?}",
            String::from_utf8_lossy(haystack)
        );
    }

    // Stability: re-scan the busiest haystack many times on the same session and
    // confirm the resident tables + counter reset keep producing the same set.
    let busy: &[u8] = b"abcabcabc def def secret secret token AKIA passwd key";
    let expected = pipeline.reference_scan(busy);
    for round in 0..32 {
        session
            .scan_into(busy, &mut resident_matches, &mut scratch)
            .expect("resident artifact re-submission");
        assert_eq!(
            resident_matches, expected,
            "resident scan drifted on reuse round {round}"
        );
    }

    session.free().expect("free resident artifact resources");
}
