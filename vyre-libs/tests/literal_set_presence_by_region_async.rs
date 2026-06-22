//! Real-GPU proof for the ASYNC region-presence scan method
//! (`GpuLiteralSet::scan_presence_by_region_async`).
//!
//! The async entry submits the dispatch and hands back a
//! `PendingPresenceByRegion` so a caller can overlap host work with the
//! in-flight GPU scan, then decode via `await_words`. This test exercises that
//! plumbing on the real wgpu backend (the RTX 5090 here) and asserts:
//!   (1) the async bitmap equals `scan_presence_by_region`'s, WORD FOR WORD —
//!       the async path must be a correctness-equivalent of the sync path, and
//!   (2) the decoded per-region bit SETS are exactly the planted hits (real
//!       values, not "non-empty"): region 0 = {key,token,secret,AKIA,api},
//!       region 1 = {ghp_,sk_live_,password}, region 2 = {} (no anchors).
//!   (3) `is_ready()` is a callable non-blocking probe and `await_words`
//!       returns the full `region_count * words` bitmap.
//!
//! Skips cleanly when no GPU is available.
//!
//! Run:
//!   cargo test -p vyre-libs --test literal_set_presence_by_region_async --release -- --nocapture

use std::collections::BTreeSet;

use vyre_driver_wgpu::WgpuBackend;
use vyre_libs::scan::GpuLiteralSet;

// pattern_id order: key=0 token=1 secret=2 AKIA=3 ghp_=4 sk_live_=5 password=6 api=7
const LITERALS: &[&[u8]] = &[
    b"key", b"token", b"secret", b"AKIA", b"ghp_", b"sk_live_", b"password", b"api",
];

/// A "file" carrying a known subset of literal hits, terminated by a separator
/// byte (newline) that is in NO literal, so no match spans the region boundary —
/// exactly keyhog's coalesced-batch layout.
fn file_with(hits: &str) -> Vec<u8> {
    let mut v = hits.as_bytes().to_vec();
    v.push(b'\n');
    v
}

/// Decode one region's presence row into the set of pattern ids whose bit is set.
fn present_ids(row: &[u32], pattern_count: u32) -> BTreeSet<u32> {
    (0..pattern_count)
        .filter(|&p| {
            let w = (p >> 5) as usize;
            let b = p & 31;
            row.get(w).is_some_and(|word| (word >> b) & 1 == 1)
        })
        .collect()
}

#[test]
fn async_region_presence_matches_sync_and_planted_hits_on_gpu() {
    let backend = match WgpuBackend::shared() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("no wgpu backend ({e}); skipping async region-presence GPU test");
            return;
        }
    };
    let matcher = GpuLiteralSet::compile(LITERALS);
    let pattern_count = LITERALS.len() as u32;
    let words = pattern_count.div_ceil(32).max(1) as usize;

    // Three coalesced files with distinct, known hit sets.
    let files = [
        file_with("api key here AKIA token secret"), // {api,key,AKIA,token,secret} = {7,0,3,1,2}
        file_with("ghp_abc sk_live_xyz password"),    // {ghp_,sk_live_,password} = {4,5,6}
        file_with("plain prose with no anchors here"), // {} (no literal occurs)
    ];
    let mut haystack = Vec::new();
    let mut region_starts = Vec::new();
    for f in &files {
        region_starts.push(haystack.len() as u32);
        haystack.extend_from_slice(f);
    }
    let region_count = region_starts.len();

    // ---- Sync reference ----
    let sync_words = matcher
        .scan_presence_by_region(backend.as_ref(), &haystack, &region_starts)
        .expect("sync gpu presence-by-region scan");

    // ---- Async: submit, probe, decode ----
    let pending = matcher
        .scan_presence_by_region_async(backend.as_ref(), &haystack, &region_starts, 0)
        .expect("async gpu presence-by-region submit");
    // Exercise the non-blocking probe (value is backend/timing-dependent, so we
    // only require it be callable without blocking — not a fixed bool).
    let _ready_before_await = pending.is_ready();
    let async_words = pending.await_words().expect("async gpu presence-by-region await");

    // (1) Async bitmap == sync bitmap, WORD FOR WORD.
    assert_eq!(
        async_words, sync_words,
        "async region-presence bitmap must equal scan_presence_by_region word-for-word"
    );

    // (3) Full bitmap shape: region_count rows of `words` words each.
    assert_eq!(
        async_words.len(),
        region_count * words,
        "async bitmap must be region_count ({region_count}) * words ({words}) = {} u32s",
        region_count * words
    );

    // (2) Exact planted bit sets per region (real values, not non-empty).
    let row = |r: usize| &async_words[r * words..(r + 1) * words];
    assert_eq!(
        present_ids(row(0), pattern_count),
        BTreeSet::from([0, 1, 2, 3, 7]),
        "region 0 must carry exactly {{key,token,secret,AKIA,api}}"
    );
    assert_eq!(
        present_ids(row(1), pattern_count),
        BTreeSet::from([4, 5, 6]),
        "region 1 must carry exactly {{ghp_,sk_live_,password}}"
    );
    assert_eq!(
        present_ids(row(2), pattern_count),
        BTreeSet::new(),
        "region 2 has no literal occurrence and must be all-zero"
    );
}
