//! Proof for the ASYNC region-presence scan method
//! (`GpuLiteralSet::scan_presence_by_region_async`).
//!
//! The async entry submits the dispatch and hands back a
//! `PendingPresenceByRegion` so a caller can overlap host work with the
//! in-flight scan, then decode via `await_words`. Two tests pin it:
//!
//!   * `async_region_presence_matches_sync_and_planted_hits_on_gpu` — REAL GPU
//!     (wgpu, the RTX 5090 here): the async bitmap equals
//!     `scan_presence_by_region`'s word-for-word AND the decoded per-region bit
//!     SETS are exactly the planted hits. Skips cleanly with no GPU.
//!
//!   * `async_region_presence_equals_sync_on_cpu_reference` — CPU reference
//!     backend, runs EVERYWHERE (incl. GPU-less CI, where the GPU test skips).
//!     On a backend that does not pipeline host/device work, `dispatch_async`
//!     uses the synchronous default and the handle is trivially ready: this
//!     asserts `is_ready()` is `true`, the async words equal the sync words
//!     (NO silent change of result on the degraded path — Law 10), and the same
//!     exact planted bit sets.
//!
//! Run:
//!   cargo test -p vyre-libs --test literal_set_presence_by_region_async --release -- --nocapture

use std::collections::BTreeSet;

use vyre_driver_reference::CpuRefBackend;
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

/// Three coalesced files with distinct, KNOWN hit sets, returned as a coalesced
/// haystack + ascending region starts (the keyhog phase-1 layout).
fn planted_corpus() -> (Vec<u8>, Vec<u32>) {
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
    (haystack, region_starts)
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

/// Assert the full `region_count * words` bitmap carries EXACTLY the planted hit
/// sets per region (real values, not "non-empty").
fn assert_planted_bits(words_per_region: usize, pattern_count: u32, bitmap: &[u32]) {
    let row = |r: usize| &bitmap[r * words_per_region..(r + 1) * words_per_region];
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
    let (haystack, region_starts) = planted_corpus();
    let region_count = region_starts.len();

    let sync_words = matcher
        .scan_presence_by_region(backend.as_ref(), &haystack, &region_starts)
        .expect("sync gpu presence-by-region scan");

    let pending = matcher
        .scan_presence_by_region_async(backend.as_ref(), &haystack, &region_starts, 0)
        .expect("async gpu presence-by-region submit");
    // Exercise the non-blocking probe (value is backend/timing-dependent on a
    // pipelining backend, so we only require it be callable without blocking).
    let _ready_before_await = pending.is_ready();
    let async_words = pending.await_words().expect("async gpu presence-by-region await");

    // (1) Async bitmap == sync bitmap, WORD FOR WORD.
    assert_eq!(
        async_words, sync_words,
        "async region-presence bitmap must equal scan_presence_by_region word-for-word"
    );
    // (2) Full bitmap shape.
    assert_eq!(
        async_words.len(),
        region_count * words,
        "async bitmap must be region_count ({region_count}) * words ({words}) u32s"
    );
    // (3) Exact planted bit sets per region.
    assert_planted_bits(words, pattern_count, &async_words);
}

#[test]
fn async_region_presence_equals_sync_on_cpu_reference() {
    // CPU reference backend: no GPU, runs everywhere. dispatch_async here uses the
    // synchronous default (the backend does not pipeline host/device work), so the
    // pending handle is trivially ready and MUST yield the same bitmap as the sync
    // entry — the degraded path changes nothing about the result (Law 10).
    let matcher = GpuLiteralSet::compile(LITERALS);
    let pattern_count = LITERALS.len() as u32;
    let words = pattern_count.div_ceil(32).max(1) as usize;
    let (haystack, region_starts) = planted_corpus();
    let region_count = region_starts.len();

    let sync_words = matcher
        .scan_presence_by_region(&CpuRefBackend, &haystack, &region_starts)
        .expect("sync cpu-reference presence-by-region scan");

    let pending = matcher
        .scan_presence_by_region_async(&CpuRefBackend, &haystack, &region_starts, 0)
        .expect("async cpu-reference presence-by-region submit");
    // Non-pipelining backend -> trivially-ready handle: is_ready is deterministically true.
    assert!(
        pending.is_ready(),
        "a non-pipelining backend must return a trivially-ready handle (is_ready == true)"
    );
    let async_words = pending.await_words().expect("async cpu-reference await");

    assert_eq!(
        async_words, sync_words,
        "async region-presence on the CPU reference backend must equal the sync result \
         word-for-word — the non-overlapping path must not change the bitmap"
    );
    assert_eq!(
        async_words.len(),
        region_count * words,
        "async bitmap must be region_count ({region_count}) * words ({words}) u32s"
    );
    assert_planted_bits(words, pattern_count, &async_words);
}
