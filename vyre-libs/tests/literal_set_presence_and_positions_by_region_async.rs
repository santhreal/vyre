//! Proof for the ASYNC fused presence+positions scan method
//! (`GpuLiteralSet::scan_presence_and_positions_by_region_async`).
//!
//! One submitted dispatch yields BOTH the per-region presence bitmap AND the
//! `(pattern_id, start, end)` match triples. The async outputs must equal the
//! sync `scan_presence_and_positions_by_region`'s exactly on BOTH the real GPU
//! (wgpu) and the CPU reference backend. On a non-pipelining backend
//! (`CpuRefBackend`) the handle is trivially ready and the degraded synchronous
//! path must not change either output (Law 10).
//!
//! Run:
//!   cargo test -p vyre-libs --test literal_set_presence_and_positions_by_region_async --release -- --nocapture

use vyre_driver_reference::CpuRefBackend;
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::match_result::Match;
use vyre_libs::scan::GpuLiteralSet;

const LITERALS: &[&[u8]] = &[b"alpha", b"kilo", b"tango"];
const MAX_MATCHES: u32 = 256;

/// Two regions: region 0 has `alpha` (twice) + `kilo`; region 1 has `tango`.
fn fixture() -> (GpuLiteralSet, Vec<u8>, Vec<u32>) {
    let set = GpuLiteralSet::compile(LITERALS);
    let region0 = b"__alpha__kilo__alpha__".to_vec();
    let region1 = b"..tango..".to_vec();
    let region_starts = vec![0u32, region0.len() as u32];
    let mut haystack = region0;
    haystack.extend_from_slice(&region1);
    (set, haystack, region_starts)
}

#[test]
fn async_fused_matches_sync_on_gpu() {
    let backend = match WgpuBackend::shared() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("no wgpu backend ({e}); skipping async fused GPU test");
            return;
        }
    };
    let (set, haystack, region_starts) = fixture();

    let mut sync_matches: Vec<Match> = Vec::new();
    let sync_presence = set
        .scan_presence_and_positions_by_region(
            backend.as_ref(),
            &haystack,
            &region_starts,
            0,
            MAX_MATCHES,
            &mut sync_matches,
        )
        .expect("sync gpu fused scan");

    let pending = set
        .scan_presence_and_positions_by_region_async(
            backend.as_ref(),
            &haystack,
            &region_starts,
            0,
            MAX_MATCHES,
        )
        .expect("async gpu fused submit");
    let _ready = pending.is_ready();
    let mut async_matches: Vec<Match> = Vec::new();
    let async_presence = pending
        .await_into(&mut async_matches)
        .expect("async gpu fused await");

    assert_eq!(
        async_presence, sync_presence,
        "async fused presence bitmap must equal the sync one word-for-word"
    );
    assert_eq!(
        async_matches, sync_matches,
        "async fused match triples must equal the sync ones"
    );
    assert!(async_presence.iter().any(|&w| w != 0));
    assert!(!async_matches.is_empty());
}

#[test]
fn async_fused_equals_sync_on_cpu_reference() {
    let (set, haystack, region_starts) = fixture();

    let mut sync_matches: Vec<Match> = Vec::new();
    let sync_presence = set
        .scan_presence_and_positions_by_region(
            &CpuRefBackend,
            &haystack,
            &region_starts,
            0,
            MAX_MATCHES,
            &mut sync_matches,
        )
        .expect("sync cpu-reference fused scan");

    let pending = set
        .scan_presence_and_positions_by_region_async(
            &CpuRefBackend,
            &haystack,
            &region_starts,
            0,
            MAX_MATCHES,
        )
        .expect("async cpu-reference fused submit");
    assert!(
        pending.is_ready(),
        "a non-pipelining backend must return a trivially-ready handle"
    );

    // Pre-seed the match buffer; await_into must clear it.
    let mut async_matches: Vec<Match> = vec![Match::new(77, 3, 5); 4];
    let async_presence = pending
        .await_into(&mut async_matches)
        .expect("async cpu-reference fused await");

    assert!(
        !async_matches.iter().any(|m| m.pattern_id == 77),
        "stale pre-seeded matches must be cleared by await_into, not accumulated"
    );
    assert_eq!(
        async_presence, sync_presence,
        "async fused presence on the CPU reference backend must equal the sync bitmap"
    );
    assert_eq!(
        async_matches, sync_matches,
        "async fused matches on the CPU reference backend must equal the sync triples"
    );
    assert!(async_presence.iter().any(|&w| w != 0));
    assert!(!async_matches.is_empty());
}
