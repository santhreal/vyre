//! Proof for the ASYNC global-presence scan method
//! (`GpuLiteralSet::scan_presence_async`).
//!
//! The async entry submits the dispatch and hands back a `PendingPresence` so a
//! caller can overlap host work with the in-flight scan, then decode via
//! `await_words`. The async bitmap must equal `scan_presence`'s word-for-word on
//! BOTH the real GPU (wgpu) and the CPU reference backend. On a backend that does
//! not pipeline host/device work (`CpuRefBackend`), `dispatch_async` uses the
//! synchronous default and the handle is trivially ready: the degraded path must
//! not change the bitmap by a single bit (Law 10).
//!
//! Run:
//!   cargo test -p vyre-libs --test literal_set_presence_async --release -- --nocapture

use vyre_driver_reference::CpuRefBackend;
use vyre_driver_wgpu::WgpuBackend;
use vyre_libs::scan::GpuLiteralSet;

/// Three patterns; two occur (`alpha`, `tango`), one does not (`kilo`), so the
/// bitmap has some set and some clear bits (non-vacuous equality).
const LITERALS: &[&[u8]] = &[b"alpha", b"kilo", b"tango"];
fn haystack() -> Vec<u8> {
    b"__alpha__and__tango__here__no_k1lo__alpha_again".to_vec()
}

#[test]
fn async_global_presence_matches_sync_on_gpu() {
    let backend = match WgpuBackend::shared() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("no wgpu backend ({e}); skipping async global-presence GPU test");
            return;
        }
    };
    let matcher = GpuLiteralSet::compile(LITERALS);
    let hay = haystack();

    let sync_words = matcher
        .scan_presence(backend.as_ref(), &hay)
        .expect("sync gpu global-presence scan");

    let pending = matcher
        .scan_presence_async(backend.as_ref(), &hay)
        .expect("async gpu global-presence submit");
    let _ready_before_await = pending.is_ready();
    let async_words = pending
        .await_words()
        .expect("async gpu global-presence await");

    assert_eq!(
        async_words, sync_words,
        "async global-presence bitmap must equal scan_presence word-for-word"
    );
    assert!(
        async_words.iter().any(|&w| w != 0),
        "fixture must set at least one presence bit (non-vacuous)"
    );
}

#[test]
fn async_global_presence_equals_sync_on_cpu_reference() {
    // CPU reference backend: no GPU, runs everywhere. dispatch_async uses the
    // synchronous default, so the handle is trivially ready and MUST yield the
    // same bitmap as the sync entry (the degraded path changes nothing (Law 10)).
    let matcher = GpuLiteralSet::compile(LITERALS);
    let hay = haystack();

    let sync_words = matcher
        .scan_presence(&CpuRefBackend, &hay)
        .expect("sync cpu-reference global-presence scan");

    let pending = matcher
        .scan_presence_async(&CpuRefBackend, &hay)
        .expect("async cpu-reference global-presence submit");
    assert!(
        pending.is_ready(),
        "a non-pipelining backend must return a trivially-ready handle (is_ready == true)"
    );
    let async_words = pending.await_words().expect("async cpu-reference await");

    assert_eq!(
        async_words, sync_words,
        "async global-presence on the CPU reference backend must equal the sync result \
         word-for-word: the non-overlapping path must not change the bitmap"
    );
    assert!(
        async_words.iter().any(|&w| w != 0),
        "fixture must set at least one presence bit (non-vacuous)"
    );
}
