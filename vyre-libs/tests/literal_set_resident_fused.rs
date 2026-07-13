//! GPU parity proof for the RESIDENT FUSED per-region presence + positions
//! pipeline (`GpuLiteralSet::prepare_resident_fused_scan` →
//! `ResidentFusedRegionScan`).
//!
//! The fused resident pipeline uploads the seven immutable literal-match tables
//! ONCE, then re-dispatches the fused program across a corpus re-uploading only
//! the per-batch haystack, region controls, and two zeroed accumulators (the
//! per-region presence prefix + the 4-byte match counter). This test pins, on
//! REAL GPU hardware (wgpu, the RTX 5090 here), that BOTH outputs, the per-region
//! presence bitmap AND the `(pattern_id, start, end)` triples, are byte-identical
//! to the borrowed `scan_presence_and_positions_by_region` across repeated scans.
//! The resident table-residency optimization must not change a single result bit
//! (Law 10). Skips cleanly with no GPU.
//!
//! Run:
//!   cargo test -p vyre-libs --test literal_set_resident_fused --release -- --nocapture

mod presence_corpus;

use presence_corpus::{assert_planted_bits, planted_corpus, LITERALS};
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::match_result::Match;
use vyre_libs::scan::GpuLiteralSet;

#[test]
fn resident_fused_presence_and_positions_match_borrowed_on_gpu() {
    let backend = match WgpuBackend::shared() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("no wgpu backend ({e}); skipping resident fused GPU test");
            return;
        }
    };
    let matcher = GpuLiteralSet::compile(LITERALS);
    let pattern_count = LITERALS.len() as u32;
    let words = pattern_count.div_ceil(32).max(1) as usize;
    let (haystack, region_starts) = planted_corpus();
    let region_count = region_starts.len();
    let max_matches = 8_192u32;

    // Ground truth: the borrowed fused scan produces BOTH the presence bitmap and
    // the positioned matches in one dispatch.
    let mut borrowed_matches: Vec<Match> = Vec::new();
    let borrowed_presence = matcher
        .scan_presence_and_positions_by_region(
            backend.as_ref(),
            &haystack,
            &region_starts,
            0,
            max_matches,
            &mut borrowed_matches,
        )
        .expect("borrowed gpu fused scan");
    // Non-vacuous: the fixture must exercise both outputs.
    assert!(
        !borrowed_matches.is_empty(),
        "the planted corpus must produce positioned matches"
    );
    assert!(
        borrowed_presence.iter().any(|&w| w != 0),
        "the planted corpus must set presence bits"
    );

    // Prepare a resident session with region head room (max_regions > region_count
    // proves the dynamic-region-count path).
    let session = matcher
        .prepare_resident_fused_scan(
            backend.as_ref(),
            haystack.len() + 64,
            region_count as u32 + 2,
            max_matches,
        )
        .expect("prepare resident fused session");
    assert_eq!(session.max_regions(), region_count as u32 + 2);
    assert_eq!(session.max_matches(), max_matches);

    // Re-dispatch the SAME corpus several times: the immutable tables stay resident
    // (uploaded once at prepare), and every scan must reproduce BOTH borrowed
    // outputs bit-for-bit.
    let mut out: Vec<u32> = Vec::new();
    let mut matches: Vec<Match> = Vec::new();
    let mut scratch: Vec<u8> = Vec::new();
    for iter in 0..4 {
        session
            .scan_into(
                backend.as_ref(),
                &haystack,
                &region_starts,
                0,
                &mut out,
                &mut matches,
                &mut scratch,
            )
            .expect("resident fused scan");
        assert_eq!(
            out, borrowed_presence,
            "iteration {iter}: resident presence must equal the borrowed fused presence word-for-word"
        );
        assert_eq!(
            matches, borrowed_matches,
            "iteration {iter}: resident matches must equal the borrowed fused matches"
        );
        assert_eq!(
            out.len(),
            region_count * words,
            "iteration {iter}: presence bitmap must be region_count ({region_count}) * words ({words}) u32s"
        );
        assert_planted_bits(words, pattern_count, &out);
    }

    session
        .free(backend.as_ref())
        .expect("free resident fused session");
}

#[test]
fn resident_fused_serves_smaller_batches_under_the_cap_on_gpu() {
    // One resident session sized for the full corpus must also correctly scan a
    // SMALLER batch (fewer regions than max_regions) for BOTH outputs, proving the
    // kernel reads the live region count from buf_len(region_starts), not the cap.
    let backend = match WgpuBackend::shared() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("no wgpu backend ({e}); skipping resident fused sub-batch GPU test");
            return;
        }
    };
    let matcher = GpuLiteralSet::compile(LITERALS);
    let words = (LITERALS.len() as u32).div_ceil(32).max(1) as usize;
    let (haystack, region_starts) = planted_corpus();
    let max_matches = 8_192u32;

    let session = matcher
        .prepare_resident_fused_scan(
            backend.as_ref(),
            haystack.len() + 64,
            region_starts.len() as u32,
            max_matches,
        )
        .expect("prepare resident fused session");

    // A single-region sub-batch: just the first coalesced file.
    let single_region_start = [0u32];
    let first_file_end = region_starts[1] as usize;
    let first_file = &haystack[..first_file_end];

    let mut borrowed_matches: Vec<Match> = Vec::new();
    let borrowed_presence = matcher
        .scan_presence_and_positions_by_region(
            backend.as_ref(),
            first_file,
            &single_region_start,
            0,
            max_matches,
            &mut borrowed_matches,
        )
        .expect("borrowed single-region fused scan");

    let mut out: Vec<u32> = Vec::new();
    let mut matches: Vec<Match> = Vec::new();
    let mut scratch: Vec<u8> = Vec::new();
    session
        .scan_into(
            backend.as_ref(),
            first_file,
            &single_region_start,
            0,
            &mut out,
            &mut matches,
            &mut scratch,
        )
        .expect("resident single-region fused scan under the cap");

    assert_eq!(
        out, borrowed_presence,
        "a 1-region sub-batch presence must match the borrowed fused scan"
    );
    assert_eq!(
        matches, borrowed_matches,
        "a 1-region sub-batch matches must match the borrowed fused scan"
    );
    assert_eq!(
        out.len(),
        words,
        "single region -> exactly `words` presence u32s"
    );

    session.free(backend.as_ref()).expect("free session");
}
