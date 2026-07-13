//! GPU parity proof for the RESIDENT POSITION-scan pipeline
//! (`GpuLiteralSet::prepare_resident_scan` → `ResidentLiteralScan`).
//!
//! The resident position pipeline uploads the seven immutable literal-match tables
//! (DFA transition / output-offset / output-record / pattern-length tables + the
//! three suffix-prefilter masks) into backend-resident resources ONCE, then
//! re-dispatches across a corpus transferring only the per-file haystack + a 4-byte
//! match-counter reset. This test pins, on REAL GPU hardware (wgpu, the RTX 5090
//! here), that the resident `(pattern_id, start, end)` triples are byte-identical to
//! the borrowed `scan_into` across repeated scans, the resident table-residency
//! optimization must not change a single match (Law 10). Skips cleanly with no GPU.
//!
//! Run:
//!   cargo test -p vyre-libs --test literal_set_resident_scan --release -- --nocapture

use std::time::Instant;

use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::match_result::Match;
use vyre_libs::scan::GpuLiteralSet;

// pattern_id order matches the compile order: key=0 .. api=7.
const LITERALS: &[&[u8]] = &[
    b"key",
    b"token",
    b"secret",
    b"AKIA",
    b"ghp_",
    b"sk_live_",
    b"password",
    b"api",
];

/// A haystack that plants every detector several times at known offsets, so the
/// borrowed scan produces a non-trivial multi-match set the resident scan must
/// reproduce exactly.
fn planted_haystack() -> Vec<u8> {
    let mut h = Vec::new();
    // Repeat a mixed block so multiple patterns fire at multiple offsets.
    for _ in 0..64 {
        h.extend_from_slice(b"key api token secret AKIA ghp_ sk_live_ password key api\n");
    }
    h
}

#[test]
fn resident_position_scan_matches_borrowed_on_gpu() {
    let backend = match WgpuBackend::shared() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("no wgpu backend ({e}); skipping resident position-scan GPU test");
            return;
        }
    };
    let matcher = GpuLiteralSet::compile(LITERALS);
    let haystack = planted_haystack();
    let max_matches = 8_192u32;

    // Ground truth: the borrowed position scan.
    let mut borrowed: Vec<Match> = Vec::new();
    matcher
        .scan_into(backend.as_ref(), &haystack, max_matches, &mut borrowed)
        .expect("borrowed gpu position scan");
    assert!(
        !borrowed.is_empty(),
        "the planted corpus must produce matches (fixture sanity)"
    );

    // Prepare a resident session sized for this corpus with a little head room.
    let session = matcher
        .prepare_resident_scan(backend.as_ref(), haystack.len() + 64, max_matches)
        .expect("prepare resident position-scan session");
    assert_eq!(session.max_matches(), max_matches);

    // Re-dispatch the SAME corpus several times through the resident session: the
    // immutable tables stay resident (uploaded once at prepare), and every scan
    // must reproduce the borrowed triples match-for-match.
    let mut out: Vec<Match> = Vec::new();
    let mut scratch: Vec<u8> = Vec::new();
    for iter in 0..4 {
        session
            .scan_into(backend.as_ref(), &haystack, &mut out, &mut scratch)
            .expect("resident position scan");
        assert_eq!(
            out, borrowed,
            "iteration {iter}: resident triples must equal the borrowed scan_into match-for-match"
        );
    }

    session
        .free(backend.as_ref())
        .expect("free resident position-scan session");
}

#[test]
fn resident_position_scan_serves_smaller_haystacks_under_the_capacity_on_gpu() {
    // One resident session sized for the full corpus must also correctly scan a
    // SHORTER haystack, the kernel bounds its cursor with the per-scan haystack_len,
    // not the resident buffer capacity.
    let backend = match WgpuBackend::shared() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("no wgpu backend ({e}); skipping resident sub-capacity GPU test");
            return;
        }
    };
    let matcher = GpuLiteralSet::compile(LITERALS);
    let full = planted_haystack();
    let max_matches = 8_192u32;

    let session = matcher
        .prepare_resident_scan(backend.as_ref(), full.len() + 64, max_matches)
        .expect("prepare resident position-scan session");

    // A short haystack, far under the resident capacity.
    let short = b"api key token AKIA api";
    let mut borrowed: Vec<Match> = Vec::new();
    matcher
        .scan_into(backend.as_ref(), short, max_matches, &mut borrowed)
        .expect("borrowed short scan");

    let mut out: Vec<Match> = Vec::new();
    let mut scratch: Vec<u8> = Vec::new();
    session
        .scan_into(backend.as_ref(), short, &mut out, &mut scratch)
        .expect("resident short scan under capacity");

    assert_eq!(
        out, borrowed,
        "a short haystack on a session sized for the full corpus must match the borrowed scan"
    );

    session.free(backend.as_ref()).expect("free session");
}

/// PERF (opt-in via `--ignored`): the resident position pipeline must be FASTER
/// end-to-end than the borrowed `scan_into` across a multi-batch corpus, because it
/// uploads the multi-MiB DFA tables (and builds the program) ONCE instead of on
/// every batch. Correctness is the hard gate on every batch; the timing is sized so
/// the table-residency win dominates GPU noise. Prints the measured speedup.
///
/// Run:
///   cargo test -p vyre-libs --test literal_set_resident_scan --release \
///     -- --ignored --nocapture resident_position_throughput
#[test]
#[ignore = "perf measurement; needs a GPU and runs a timed multi-batch loop"]
fn resident_position_throughput_beats_borrowed_across_a_corpus_gpu() {
    let backend = match WgpuBackend::shared() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("no wgpu backend ({e}); skipping resident position throughput test");
            return;
        }
    };
    // A large detector set => a large DFA transition table: exactly the multi-MiB
    // per-scan re-upload the resident path eliminates.
    let detectors: Vec<Vec<u8>> = (0..400)
        .map(|i| format!("detector_token_{i:04}").into_bytes())
        .collect();
    let detector_refs: Vec<&[u8]> = detectors.iter().map(Vec::as_slice).collect();
    let matcher = GpuLiteralSet::compile(&detector_refs);
    let max_matches = 16_384u32;

    // 32 heterogeneous batches: each plants a rotating subset of the detectors.
    const BATCHES: usize = 32;
    const ITERS: usize = 6;
    let corpus: Vec<Vec<u8>> = (0..BATCHES)
        .map(|seed| {
            let mut h = Vec::new();
            for r in 0..48 {
                let d = &detectors[(seed * 7 + r) % detectors.len()];
                h.extend_from_slice(b"prefix ");
                h.extend_from_slice(d);
                h.extend_from_slice(b" suffix\n");
            }
            h
        })
        .collect();
    let cap_bytes = corpus.iter().map(Vec::len).max().unwrap() + 64;

    let session = matcher
        .prepare_resident_scan(backend.as_ref(), cap_bytes, max_matches)
        .expect("prepare resident throughput session");

    // Correctness gate (hard): every batch's resident triples equal the borrowed set.
    let mut out: Vec<Match> = Vec::new();
    let mut scratch: Vec<u8> = Vec::new();
    let mut borrowed: Vec<Match> = Vec::new();
    for h in &corpus {
        matcher
            .scan_into(backend.as_ref(), h, max_matches, &mut borrowed)
            .expect("borrowed scan");
        session
            .scan_into(backend.as_ref(), h, &mut out, &mut scratch)
            .expect("resident scan");
        assert_eq!(
            out, borrowed,
            "resident triples must equal borrowed for every batch before timing"
        );
    }

    // Warm up both paths (compile caches, device queues) before timing.
    for h in &corpus {
        let _ = matcher.scan_into(backend.as_ref(), h, max_matches, &mut borrowed);
        let _ = session.scan_into(backend.as_ref(), h, &mut out, &mut scratch);
    }

    // Timed: borrowed re-uploads tables + rebuilds the program every batch.
    let t_borrowed = Instant::now();
    for _ in 0..ITERS {
        for h in &corpus {
            matcher
                .scan_into(backend.as_ref(), h, max_matches, &mut borrowed)
                .expect("timed borrowed scan");
        }
    }
    let borrowed_time = t_borrowed.elapsed();

    // Timed: resident transfers only the haystack + counter reset per batch.
    let t_resident = Instant::now();
    for _ in 0..ITERS {
        for h in &corpus {
            session
                .scan_into(backend.as_ref(), h, &mut out, &mut scratch)
                .expect("timed resident scan");
        }
    }
    let resident_time = t_resident.elapsed();

    let scans = BATCHES * ITERS;
    let speedup = borrowed_time.as_secs_f64() / resident_time.as_secs_f64();
    eprintln!(
        "resident position throughput ({} detectors, {scans} scans): \
         borrowed {borrowed_time:?} vs resident {resident_time:?} = {speedup:.2}x faster",
        detectors.len()
    );

    assert!(
        resident_time < borrowed_time,
        "resident table-residency must beat re-uploading the DFA + rebuilding the program every batch \
         (borrowed {borrowed_time:?}, resident {resident_time:?})"
    );
    session.free(backend.as_ref()).expect("free session");
}
