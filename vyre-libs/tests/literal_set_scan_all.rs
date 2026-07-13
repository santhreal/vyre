//! W1-2 gate: `GpuLiteralSet::scan_all` returns EVERY match with no fixed cap
//! and no consumer-side paging, auto-resizing to the exact device count and
//! never silently truncating.
//!
//! Runs on the CPU reference backend (`CpuRefBackend`), so it exercises the real
//! dispatch + auto-resize control flow everywhere with no GPU. The oracle is
//! `GpuLiteralSet::reference_scan` (an independent plain-Rust DFA walk).

use vyre_driver_reference::CpuRefBackend;
use vyre_libs::scan::{GpuLiteralSet, LiteralMatch};

/// The fixed-cap default `scan_all` starts at; a corpus with more matches than
/// this forces the auto-resize path. Mirrors `LITERAL_SET_DEFAULT_MAX_MATCHES`.
const DEFAULT_CAP: u32 = 10_000;

fn sorted_triples(matches: &[LiteralMatch]) -> Vec<(u32, u32, u32)> {
    let mut v: Vec<(u32, u32, u32)> = matches
        .iter()
        .map(|m| (m.pattern_id, m.start, m.end))
        .collect();
    v.sort_unstable();
    v
}

#[test]
fn scan_all_returns_everything_past_the_fixed_cap() {
    let backend = CpuRefBackend;
    let literals: &[&[u8]] = &[b"a"];
    let matcher = GpuLiteralSet::compile(literals);

    // 25_000 'a' bytes => 25_000 single-byte matches, far past the 10_000 cap.
    let n = 25_000usize;
    let haystack = vec![b'a'; n];

    let all = matcher
        .scan_all(&backend, &haystack)
        .expect("scan_all auto-resizes and completes");
    assert_eq!(
        all.len(),
        n,
        "scan_all must return every match past the fixed cap, got {} of {n}",
        all.len()
    );

    // Independent oracle: exact triple-set equality, not just cardinality.
    let oracle = matcher.reference_scan(&haystack);
    assert_eq!(
        sorted_triples(&all),
        sorted_triples(&oracle),
        "scan_all triples must equal the reference DFA oracle"
    );

    // The fixed-cap scan on the SAME input must fail closed (never truncate).
    let capped = matcher.scan(&backend, &haystack, DEFAULT_CAP);
    assert!(
        capped.is_err(),
        "fixed-cap scan must fail closed when matches ({n}) exceed the cap ({DEFAULT_CAP}), not silently truncate"
    );
}

#[test]
fn scan_all_single_dispatch_common_case() {
    let backend = CpuRefBackend;
    let literals: &[&[u8]] = &[b"abc", b"bc", b"xyz"];
    let matcher = GpuLiteralSet::compile(literals);
    let haystack = b"__abc__bc__xyz__abc__".to_vec();

    let all = matcher
        .scan_all(&backend, &haystack)
        .expect("scan_all completes for a small corpus in one dispatch");
    let oracle = matcher.reference_scan(&haystack);
    assert_eq!(
        sorted_triples(&all),
        sorted_triples(&oracle),
        "scan_all triples must equal the reference DFA oracle on the common (unsaturated) path"
    );
    assert!(!all.is_empty(), "expected matches for abc/bc/xyz");
}

#[test]
fn scan_all_dense_multi_pattern_past_cap() {
    let backend = CpuRefBackend;
    let literals: &[&[u8]] = &[b"a", b"aa"];
    let matcher = GpuLiteralSet::compile(literals);

    // 8_000 'a' bytes: 8_000 "a" matches + 7_999 "aa" matches = 15_999 > cap.
    let n = 8_000usize;
    let haystack = vec![b'a'; n];

    let all = matcher
        .scan_all(&backend, &haystack)
        .expect("scan_all auto-resizes for dense multi-pattern");
    let expected = n + (n - 1);
    assert_eq!(
        all.len(),
        expected,
        "dense multi-pattern scan_all must return all {expected} matches, got {}",
        all.len()
    );
    let oracle = matcher.reference_scan(&haystack);
    assert_eq!(
        sorted_triples(&all),
        sorted_triples(&oracle),
        "dense multi-pattern scan_all triples must equal the reference oracle"
    );
}

#[test]
fn scan_all_empty_haystack_and_no_matches() {
    let backend = CpuRefBackend;
    let matcher = GpuLiteralSet::compile(&[b"needle".as_slice()]);

    let empty = matcher
        .scan_all(&backend, b"")
        .expect("scan_all handles an empty haystack");
    assert!(empty.is_empty(), "empty haystack yields no matches");

    let clean = matcher
        .scan_all(&backend, b"nothing to find here")
        .expect("scan_all handles a clean haystack");
    assert!(clean.is_empty(), "clean haystack yields no matches");
}

/// Law-10 regression: the CPU reference backend must cover the WHOLE haystack,
/// including positions past the buffer-shape-inferred grid.
///
/// `CpuRefBackend` infers its dispatch grid from buffer SHAPES, which cannot see
/// the runtime scan length, so a byte-scan over a haystack LARGER than that
/// inferred grid (observed ~41185 for the packed-4-bytes/u32 layout) used to be
/// under-dispatched and SILENTLY drop every match in the tail while the GPU (and
/// the independent `reference_scan` DFA oracle) found them. The fix threads the
/// caller's true element coverage (`DispatchConfig::dispatch_elements`, set by
/// `byte_scan_dispatch_config`) into the interpreter's dispatch floor. The
/// existing tests here scan 25k/8k bytes. UNDER the threshold, so they never
/// exercised the tail; this one deliberately places needles PAST it.
#[test]
fn cpuref_scan_covers_tail_past_buffer_inferred_grid() {
    let backend = CpuRefBackend;
    let matcher = GpuLiteralSet::compile(&[b"needle".as_slice()]);

    // 64 KiB (comfortably larger than the ~41185-byte buffer-inferred grid).
    let mut haystack = vec![b'.'; 64 * 1024];
    let needle = b"needle";
    // Two needles under the old threshold and two well past it (the tail that was
    // silently dropped before dispatch_elements was threaded through).
    let starts = [100usize, 20_000, 45_000, 60_000];
    for &pos in &starts {
        haystack[pos..pos + needle.len()].copy_from_slice(needle);
    }

    let all = matcher
        .scan_all(&backend, &haystack)
        .expect("scan_all completes on the CPU reference backend");

    // Independent oracle (full-haystack DFA walk) (exact triple-set equality).
    let oracle = matcher.reference_scan(&haystack);
    assert_eq!(
        sorted_triples(&all),
        sorted_triples(&oracle),
        "CpuRefBackend must cover the whole haystack (incl. the tail past the buffer grid), matching the reference oracle"
    );

    // Concrete anchor: EVERY placed needle, including the two past ~41185, is
    // reported, so this can never pass by both sides being empty.
    for &pos in &starts {
        assert!(
            all.iter().any(|m| m.start == pos as u32),
            "needle at offset {pos} must be found by CpuRefBackend (tail offsets were silently dropped before the dispatch_elements fix)"
        );
    }
    assert_eq!(
        all.len(),
        starts.len(),
        "exactly the four placed needles match"
    );
}
