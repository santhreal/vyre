//! CUDA parity for the authenticated resident region-presence route.
//!
//! The test materializes one canonical artifact, reuses its resident tables
//! across submissions, and checks exact planted per-region hit sets.

use std::collections::BTreeSet;

use vyre_driver_cuda as _;
use vyre_scan::GpuLiteralSet;

// pattern_id order: key=0 token=1 secret=2 AKIA=3 ghp_=4 sk_live_=5 password=6 api=7
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

/// Three coalesced "files" with KNOWN hit sets (keyhog's phase-1 layout): a
/// coalesced haystack + ascending region starts beginning at 0, each file
/// terminated by a newline that is in no literal. Mirrors the planted corpus in
/// `vyre-libs/tests/literal_set_resident_presence.rs` so the two backends assert
/// against identical ground truth.
fn planted_corpus() -> (Vec<u8>, Vec<u32>) {
    let files: [&[u8]; 3] = [
        b"api key here AKIA token secret\n", // {api,key,AKIA,token,secret} = {7,0,3,1,2}
        b"ghp_abc sk_live_xyz password\n",   // {ghp_,sk_live_,password} = {4,5,6}
        b"plain prose with no anchors here\n", // {} (no literal occurs)
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

fn assert_planted_bits(words: usize, pattern_count: u32, bitmap: &[u32]) {
    let row = |r: usize| &bitmap[r * words..(r + 1) * words];
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
fn resident_region_presence_matches_planted_hits_on_cuda() {
    let matcher = GpuLiteralSet::compile(LITERALS);
    let pattern_count = LITERALS.len() as u32;
    let words = pattern_count.div_ceil(32).max(1) as usize;
    let (haystack, region_starts) = planted_corpus();
    let region_count = region_starts.len();

    let expected = vec![0b1000_1111, 0b0111_0000, 0];
    assert_eq!(expected.len(), region_count * words);
    assert_planted_bits(words, pattern_count, &expected);

    // Resident session sized with a region of head room (max_regions > region_count
    // exercises the dynamic-region-count path on CUDA too).
    let session = matcher
        .prepare_resident_presence("cuda", haystack.len() + 64, region_count as u32 + 1)
        .expect("prepare authenticated resident region-presence artifact on CUDA");

    // Re-dispatch several times: the authenticated artifact and tables stay resident.
    let mut out = Vec::new();
    let mut scratch = Vec::new();
    for iter in 0..4 {
        session
            .scan_into(&haystack, &region_starts, 0, &mut out, &mut scratch)
            .expect("resident CUDA region-presence artifact submission");
        assert_eq!(
            out, expected,
            "iteration {iter}: CUDA resident bitmap must equal the planted result"
        );
        assert_planted_bits(words, pattern_count, &out);
    }

    session
        .free()
        .expect("free resident region-presence artifact resources on CUDA");
}
