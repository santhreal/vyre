//! GPU parity proof for the RESIDENT region-presence pipeline
//! (`GpuLiteralSet::prepare_resident_presence` → `ResidentPresencePipeline`).
//!
//! The resident pipeline uploads the seven immutable region-presence tables (DFA
//! transition / output-offset / output-record / pattern-length tables + the three
//! suffix-prefilter masks) into backend-resident resources ONCE, then re-dispatches
//! across a corpus transferring only the per-file haystack and a presence-prefix
//! reset. This test pins, on REAL GPU hardware (wgpu, the RTX 5090 here), that the
//! resident bitmap is byte-identical to the borrowed `scan_presence_by_region`
//! across repeated scans AND carries the exact planted per-region hit sets —
//! the resident table-residency optimization must not change a single result bit
//! (Law 10). Skips cleanly with no GPU.
//!
//! Run:
//!   cargo test -p vyre-libs --test literal_set_resident_presence --release -- --nocapture

mod presence_corpus;

use presence_corpus::{assert_planted_bits, planted_corpus, LITERALS};
use vyre_driver_wgpu::WgpuBackend;
use vyre_libs::scan::GpuLiteralSet;

#[test]
fn resident_region_presence_matches_borrowed_and_planted_hits_on_gpu() {
    let backend = match WgpuBackend::shared() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("no wgpu backend ({e}); skipping resident region-presence GPU test");
            return;
        }
    };
    let matcher = GpuLiteralSet::compile(LITERALS);
    let pattern_count = LITERALS.len() as u32;
    let words = pattern_count.div_ceil(32).max(1) as usize;
    let (haystack, region_starts) = planted_corpus();
    let region_count = region_starts.len();

    // Ground truth: the borrowed per-region presence scan.
    let borrowed_words = matcher
        .scan_presence_by_region(backend.as_ref(), &haystack, &region_starts)
        .expect("borrowed gpu presence-by-region scan");

    // Prepare a resident session sized for this corpus (a couple regions of head
    // room proves the dynamic-region-count path: max_regions > region_count).
    let session = matcher
        .prepare_resident_presence(backend.as_ref(), haystack.len() + 64, region_count as u32 + 2)
        .expect("prepare resident region-presence session");
    assert_eq!(session.max_regions(), region_count as u32 + 2);
    assert_eq!(session.presence_words(), words as u32);
    assert_eq!(session.pattern_count(), pattern_count);

    // Re-dispatch the SAME corpus several times through the resident session: the
    // immutable tables stay resident (uploaded once at prepare), and every scan
    // must reproduce the borrowed bitmap word-for-word.
    let mut out = Vec::new();
    let mut scratch = Vec::new();
    for iter in 0..4 {
        session
            .scan_into(
                backend.as_ref(),
                &haystack,
                &region_starts,
                0,
                &mut out,
                &mut scratch,
            )
            .expect("resident region-presence scan");
        assert_eq!(
            out, borrowed_words,
            "iteration {iter}: resident bitmap must equal scan_presence_by_region word-for-word"
        );
        assert_eq!(
            out.len(),
            region_count * words,
            "iteration {iter}: bitmap must be region_count ({region_count}) * words ({words}) u32s"
        );
        assert_planted_bits(words, pattern_count, &out);
    }

    session
        .free(backend.as_ref())
        .expect("free resident region-presence session");
}

#[test]
fn resident_region_presence_serves_smaller_batches_under_the_cap_on_gpu() {
    // One resident session sized for the full corpus must also correctly scan a
    // SMALLER batch (fewer regions than max_regions) — proving the kernel reads the
    // live region count from buf_len(region_starts), not the compiled cap.
    let backend = match WgpuBackend::shared() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("no wgpu backend ({e}); skipping resident sub-batch GPU test");
            return;
        }
    };
    let matcher = GpuLiteralSet::compile(LITERALS);
    let pattern_count = LITERALS.len() as u32;
    let words = pattern_count.div_ceil(32).max(1) as usize;
    let (haystack, region_starts) = planted_corpus();

    let session = matcher
        .prepare_resident_presence(backend.as_ref(), haystack.len() + 64, region_starts.len() as u32)
        .expect("prepare resident region-presence session");

    // A two-region sub-batch: just the first file (regions {0} start, terminated by
    // the second file's start). Scan only region 0's bytes with a single region.
    let single_region_start = [0u32];
    let first_file_end = region_starts[1] as usize; // start of file 1 == end of file 0
    let first_file = &haystack[..first_file_end];

    let borrowed = matcher
        .scan_presence_by_region(backend.as_ref(), first_file, &single_region_start)
        .expect("borrowed single-region scan");

    let mut out = Vec::new();
    let mut scratch = Vec::new();
    session
        .scan_into(
            backend.as_ref(),
            first_file,
            &single_region_start,
            0,
            &mut out,
            &mut scratch,
        )
        .expect("resident single-region scan under the cap");

    assert_eq!(
        out, borrowed,
        "a 1-region sub-batch on a session capped for the full corpus must match the borrowed scan"
    );
    assert_eq!(out.len(), words, "single region -> exactly `words` u32s");
    // Region 0 of the planted corpus carries {key,token,secret,AKIA,api} = {0,1,2,3,7}.
    let present: std::collections::BTreeSet<u32> = presence_corpus::present_ids(&out, pattern_count);
    assert_eq!(
        present,
        std::collections::BTreeSet::from([0, 1, 2, 3, 7]),
        "the first planted file carries exactly {{key,token,secret,AKIA,api}}"
    );

    session.free(backend.as_ref()).expect("free session");
}
