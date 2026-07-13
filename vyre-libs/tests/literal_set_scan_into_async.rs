//! Proof for the ASYNC position scan method
//! (`GpuLiteralSet::scan_into_async`).
//!
//! The async entry submits the match dispatch and hands back a `PendingMatches`
//! so a caller can overlap host work with the in-flight scan, then decode the
//! `(pattern_id, start, end)` triples via `await_into` / `await_matches`. The
//! async triples must equal the sync `scan_into`'s exactly on BOTH the real GPU
//! (wgpu) and the CPU reference backend. On a non-pipelining backend
//! (`CpuRefBackend`) the handle is trivially ready and the degraded synchronous
//! path must not change the triples by a single field (Law 10).
//!
//! Run:
//!   cargo test -p vyre-libs --test literal_set_scan_into_async --release -- --nocapture

use vyre_driver_reference::CpuRefBackend;
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::match_result::Match;
use vyre_libs::scan::GpuLiteralSet;

const LITERALS: &[&[u8]] = &[b"alpha", b"kilo", b"tango"];
const MAX_MATCHES: u32 = 64;
fn haystack() -> Vec<u8> {
    // Several positioned matches, including repeats of `alpha` and `tango`.
    b"alpha__kilo__tango__alpha__tango__alpha".to_vec()
}

#[test]
fn async_scan_into_matches_sync_on_gpu() {
    let backend = match WgpuBackend::shared() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("no wgpu backend ({e}); skipping async scan_into GPU test");
            return;
        }
    };
    let matcher = GpuLiteralSet::compile(LITERALS);
    let hay = haystack();

    let mut sync_matches: Vec<Match> = Vec::new();
    matcher
        .scan_into(backend.as_ref(), &hay, MAX_MATCHES, &mut sync_matches)
        .expect("sync gpu scan_into");

    let pending = matcher
        .scan_into_async(backend.as_ref(), &hay, MAX_MATCHES)
        .expect("async gpu scan_into submit");
    let _ready_before_await = pending.is_ready();
    let async_matches = pending.await_matches().expect("async gpu scan_into await");

    assert_eq!(
        async_matches, sync_matches,
        "async scan_into triples must equal sync scan_into triples"
    );
    assert!(
        !async_matches.is_empty(),
        "fixture must produce positioned matches (non-vacuous)"
    );
}

#[test]
fn async_scan_into_equals_sync_on_cpu_reference() {
    // CPU reference backend: dispatch_async uses the synchronous default, so the
    // handle is trivially ready and MUST yield the same triples (Law 10).
    let matcher = GpuLiteralSet::compile(LITERALS);
    let hay = haystack();

    let mut sync_matches: Vec<Match> = Vec::new();
    matcher
        .scan_into(&CpuRefBackend, &hay, MAX_MATCHES, &mut sync_matches)
        .expect("sync cpu-reference scan_into");

    let pending = matcher
        .scan_into_async(&CpuRefBackend, &hay, MAX_MATCHES)
        .expect("async cpu-reference scan_into submit");
    assert!(
        pending.is_ready(),
        "a non-pipelining backend must return a trivially-ready handle (is_ready == true)"
    );

    // Exercise await_into (the caller-owned-buffer entry) with a pre-seeded stale
    // buffer: it must be cleared, not accumulated.
    let mut async_matches: Vec<Match> = vec![Match::new(123, 0, 1); 4];
    pending
        .await_into(&mut async_matches)
        .expect("async cpu-reference await_into");

    assert!(
        !async_matches.iter().any(|m| m.pattern_id == 123),
        "stale pre-seeded matches must be cleared by await_into, not accumulated"
    );
    assert_eq!(
        async_matches, sync_matches,
        "async scan_into on the CPU reference backend must equal the sync triples \
The non-overlapping path must not change the result"
    );
    assert!(
        !async_matches.is_empty(),
        "fixture must produce positioned matches (non-vacuous)"
    );
}
