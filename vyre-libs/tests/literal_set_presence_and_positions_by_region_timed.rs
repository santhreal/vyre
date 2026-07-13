//! W3-3 (attribution) gate: `GpuLiteralSet::scan_presence_and_positions_by_region_timed`
//! returns backend-owned timing WITHOUT changing EITHER fused output.
//!
//! The fused path emits BOTH the per-region presence bitmap AND the
//! `(pid, start, end)` match triples in one dispatch. The timed twin locks that
//! both outputs are byte-for-byte identical to the untimed
//! `scan_presence_and_positions_by_region` (same program, same inputs, only
//! `dispatch_borrowed_timed` vs `dispatch_borrowed` differs) and that the
//! returned `TimedDispatchResult` reports honest timing (`device_ns` is `None`,
//! a loud absence rather than a fabricated zero, on the CPU reference backend).

use vyre_driver_reference::CpuRefBackend;
use vyre_foundation::match_result::Match;
use vyre_libs::scan::GpuLiteralSet;

const MAX_MATCHES: u32 = 256;

/// Two regions: region 0 contains `alpha` (twice) and `kilo`; region 1 contains
/// `tango`. So the presence bitmap distinguishes the rows AND there are several
/// positioned matches (incl. a repeated one), a non-vacuous fixture for both
/// outputs.
fn fixture() -> (GpuLiteralSet, Vec<u8>, Vec<u32>) {
    let patterns: [&[u8]; 3] = [b"alpha", b"kilo", b"tango"];
    let set = GpuLiteralSet::compile(&patterns);
    let region0 = b"__alpha__kilo__alpha__".to_vec();
    let region1 = b"..tango..".to_vec();
    let region_starts = vec![0u32, region0.len() as u32];
    let mut haystack = region0;
    haystack.extend_from_slice(&region1);
    (set, haystack, region_starts)
}

#[test]
fn timed_fused_equals_untimed_and_reports_timing() {
    let backend = CpuRefBackend;
    let (set, haystack, region_starts) = fixture();

    let mut plain_matches: Vec<Match> = Vec::new();
    let plain_presence = set
        .scan_presence_and_positions_by_region(
            &backend,
            &haystack,
            &region_starts,
            0,
            MAX_MATCHES,
            &mut plain_matches,
        )
        .expect("untimed fused scan");

    let mut timed_matches: Vec<Match> = Vec::new();
    let (timed_presence, timed) = set
        .scan_presence_and_positions_by_region_timed(
            &backend,
            &haystack,
            &region_starts,
            0,
            MAX_MATCHES,
            &mut timed_matches,
        )
        .expect("timed fused scan");

    // Correctness: neither output may change by a single bit.
    assert_eq!(
        timed_presence, plain_presence,
        "timed fused presence bitmap must equal the untimed one"
    );
    assert_eq!(
        timed_matches, plain_matches,
        "timed fused match triples must equal the untimed ones"
    );

    // Non-vacuous: the fixture must produce present bits AND positioned matches.
    assert!(
        timed_presence.iter().any(|&w| w != 0),
        "fixture must set at least one presence bit"
    );
    assert!(
        !timed_matches.is_empty(),
        "fixture must produce at least one positioned match"
    );

    // Attribution: raw outputs carried, honest device_ns.
    assert!(
        timed.outputs.len() >= 3 && !timed.outputs[0].is_empty(),
        "TimedDispatchResult must carry presence + count + matches output buffers"
    );
    assert!(
        timed.device_ns.is_none(),
        "CPU reference backend has no device timer; device_ns must be None, not a fabricated zero"
    );
}

#[test]
fn timed_fused_clears_stale_matches_and_is_stable() {
    let backend = CpuRefBackend;
    let (set, haystack, region_starts) = fixture();

    // Seed the buffer with stale entries; the timed scan must clear them.
    let mut matches: Vec<Match> = vec![Match::new(99, 7, 11); 5];
    let (first, _) = set
        .scan_presence_and_positions_by_region_timed(
            &backend,
            &haystack,
            &region_starts,
            0,
            MAX_MATCHES,
            &mut matches,
        )
        .expect("first timed fused scan");
    let first_matches = matches.clone();
    assert!(
        !first_matches.iter().any(|m| m.pattern_id == 99),
        "stale pre-seeded matches must be cleared, not accumulated"
    );

    let (second, _) = set
        .scan_presence_and_positions_by_region_timed(
            &backend,
            &haystack,
            &region_starts,
            0,
            MAX_MATCHES,
            &mut matches,
        )
        .expect("second timed fused scan");
    assert_eq!(
        first, second,
        "repeated timed fused presence must be deterministic"
    );
    assert_eq!(
        first_matches, matches,
        "repeated timed fused matches must be deterministic"
    );
}
