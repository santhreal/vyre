//! Real-GPU (RTX 5090, wgpu) dispatch parity for anchored-window regex
//! extraction (plan W2-3, line 179, layer 3).
//!
//! The inline `reference_eval` gate
//! (`regex_anchored_window::tests::extract_program_reference_eval_matches_cpu_oracle`)
//! proves the emitted PROGRAM's *semantics*. This test proves the borrowed
//! backend DISPATCH plumbing on real hardware, the binding-order the program
//! declares, real device atomics on the shared `match_count`, and the
//! candidate-sized grid, reproduces the [`AnchoredWindowValidator`] CPU oracle
//! exactly. Skips cleanly when no GPU is available.
//!
//! Run:
//!   cargo test -p vyre-libs --features matching-regex,matching-dfa \
//!     --test regex_anchored_window_gpu --release -- --nocapture
#![cfg(all(feature = "matching-regex", feature = "matching-dfa"))]

use std::collections::BTreeSet;

use vyre::{DispatchConfig, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;
use vyre_libs::scan::{
    anchored_window_extract_program, build_regex_dfa_pipeline, pack_haystack_u32, pack_u32_slice,
    regex_admission_by_region_program, regex_admission_by_region_reference,
    regex_admission_presence_words, unpack_match_triples, AnchoredWindowValidator,
    RegionEvidencePipeline,
};
use vyre_primitives::matching::CompiledDfa;

type Triple = (u32, u32, u32);

/// Build + dispatch the anchored-window extraction program on the GPU and return
/// its `(pid, start, end)` triples. ONE source of truth for the binding order
/// (buffers 0..=8), the candidate-sized grid, and the shared-atomic `match_count`
/// readback, both parity tests below drive the identical plumbing, differing
/// only in the pattern set, haystack, and the assertions they make on the result.
fn gpu_extract_triples(
    backend: &WgpuBackend,
    dfa: &CompiledDfa,
    haystack: &[u8],
    candidates: &[u32],
) -> BTreeSet<Triple> {
    let num_candidates = candidates.len() as u32;
    let max_matches = 4096u32;
    let program = anchored_window_extract_program(
        "haystack",
        "transitions",
        "output_offsets",
        "output_records",
        "candidates",
        "candidate_count",
        "haystack_len",
        "match_count",
        "matches",
        dfa.state_count,
        dfa.output_records.len() as u32,
        num_candidates,
        max_matches,
        dfa.max_pattern_len,
    );
    // Borrowed input bytes in the program's binding order (0..=8). `match_count`
    // starts at 0 (one shared atomic); `matches` is zero-filled scratch.
    let inputs: Vec<Vec<u8>> = vec![
        pack_haystack_u32(haystack),
        pack_u32_slice(&dfa.transitions),
        pack_u32_slice(&dfa.output_offsets),
        pack_u32_slice(&dfa.output_records),
        pack_u32_slice(candidates),
        pack_u32_slice(&[num_candidates]),
        pack_u32_slice(&[haystack.len() as u32]),
        pack_u32_slice(&[0]),
        vec![0u8; max_matches as usize * 3 * 4],
    ];
    let borrowed: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();
    // One invocation per candidate: grid = ceil(candidates / workgroup-128).
    // `DispatchConfig` is `#[non_exhaustive]`, so build via Default + field set.
    let mut config = DispatchConfig::default();
    config.grid_override = Some([num_candidates.div_ceil(128).max(1), 1, 1]);
    let outputs = backend
        .dispatch_borrowed(&program, &borrowed, &config)
        .expect("wgpu dispatch of anchored-window extract program");
    // Writable buffers are returned in binding order: outputs[0] = match_count,
    // outputs[1] = matches (shadow-conformance guarantees this ordering).
    let count = u32::from_le_bytes(
        outputs[0][0..4]
            .try_into()
            .expect("match_count output must carry at least one u32"),
    );
    unpack_match_triples(&outputs[1], count)
        .iter()
        .map(|m| (m.pattern_id, m.start, m.end))
        .collect()
}

#[test]
fn anchored_window_extract_on_gpu_matches_cpu_oracle() {
    let backend = match WgpuBackend::shared() {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("no wgpu backend ({error}); skipping anchored-window GPU parity test");
            return;
        }
    };

    // A regex set with distinct fixed-length patterns (so the CPU oracle and the
    // GPU kernel agree without depending on the bounded-repetition window quirk
    // tracked in BACKLOG). `abc`/`abcde` share a prefix (multi-length at one
    // origin); `bcd` starts mid-word; the rest are literal detectors.
    let patterns = ["abc", "abcde", "bcd", "AKIA", "token"];
    let pipeline = build_regex_dfa_pipeline(&patterns, 4096, 16_384)
        .expect("Fix: fixed-length regex set must compile to an anchored DFA");
    let dfa = &pipeline.dfa;

    let haystack: &[u8] = b"zabcde AKIA bcd token abc AKIA1 abcde bcd";
    // Exhaustive candidate set: EVERY origin. The kernel must extract at true
    // starts and reject every non-start (the anchoring contract on hardware).
    let candidates: Vec<u32> = (0..haystack.len() as u32).collect();

    // CPU oracle.
    let oracle: BTreeSet<Triple> = AnchoredWindowValidator::new(dfa)
        .validate_candidates(haystack, &candidates)
        .iter()
        .map(|m| (m.pattern_id, m.start, m.end))
        .collect();
    assert!(
        !oracle.is_empty(),
        "vacuous test: the oracle extracted no matches for this corpus"
    );

    // Build + dispatch the extraction program on the GPU (shared plumbing).
    let gpu = gpu_extract_triples(&backend, dfa, haystack, &candidates);

    assert_eq!(
        gpu, oracle,
        "GPU anchored-window extraction must equal the CPU oracle (binding order / atomics / grid)"
    );
}

/// Variable-length `{n,m}` extraction parity on real hardware, the direct
/// hardware proof of the BACKLOG items 18/27 fix. The FIRST test deliberately
/// uses only fixed-length patterns "without depending on the bounded-repetition
/// window quirk"; that quirk is now FIXED (`build_repetition` records the MAX
/// length so `max_pattern_len` covers the full range), so this exercises a mix
/// of variable `{n,m}` and fixed detectors and asserts BOTH parities:
///   (1) raw all-ends: GPU triples == CPU `validate_candidates` (the kernel sees
///       every admissible end for a variable body, exactly like the CPU walk);
///   (2) leftmost-longest: the GPU all-ends coalesced host-side (max end per
///       `(pid, start)`) == CPU `validate_candidates_leftmost_longest`: the
///       scanner-correct one-match-per-token semantics, agreeing CPU↔GPU.
#[test]
fn anchored_window_variable_repeat_on_gpu_matches_cpu_oracle() {
    let backend = match WgpuBackend::shared() {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("no wgpu backend ({error}); skipping variable-repeat GPU parity test");
            return;
        }
    };

    // Mix variable-length `{n,m}` bodies with fixed detectors. `k[0-9]{2,4}` and
    // `v[a-z]{3,5}` accept at several ends per origin; `AKIA` is fixed.
    let patterns = ["k[0-9]{2,4}", "AKIA", "v[a-z]{3,5}"];
    let pipeline = build_regex_dfa_pipeline(&patterns, 4096, 16_384)
        .expect("Fix: variable-length regex set must compile to an anchored DFA");
    let dfa = &pipeline.dfa;

    // Bodies of every interesting length: below-min (`k1`, `v99`), exact-min,
    // interior, exact-max, and above-max (`vabcdef`: munch caps at 5).
    let haystack: &[u8] = b"k12 k1234 AKIA vabc vabcdef k1 v99 k123";
    let candidates: Vec<u32> = (0..haystack.len() as u32).collect();

    let validator = AnchoredWindowValidator::new(dfa);
    let oracle: BTreeSet<Triple> = validator
        .validate_candidates(haystack, &candidates)
        .iter()
        .map(|m| (m.pattern_id, m.start, m.end))
        .collect();
    assert!(
        oracle.len() > patterns.len(),
        "vacuous test: variable bodies must produce multiple ends per token"
    );

    let gpu = gpu_extract_triples(&backend, dfa, haystack, &candidates);

    // (1) Raw all-ends parity.
    assert_eq!(
        gpu, oracle,
        "GPU must surface every admissible {{n,m}} end, equal to the CPU all-ends oracle"
    );

    // (2) Leftmost-longest parity: coalesce the GPU all-ends to the MAX end per
    // (pid, start), then compare to the CPU leftmost-longest extraction.
    let mut best: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
    for (pid, start, end) in &gpu {
        let slot = best.entry((*pid, *start)).or_insert(0);
        if *end > *slot {
            *slot = *end;
        }
    }
    let gpu_ll: BTreeSet<Triple> = best
        .iter()
        .map(|(&(pid, start), &end)| (pid, start, end))
        .collect();
    let cpu_ll: BTreeSet<Triple> = validator
        .validate_candidates_leftmost_longest(haystack, &candidates)
        .iter()
        .map(|m| (m.pattern_id, m.start, m.end))
        .collect();
    assert_eq!(
        gpu_ll, cpu_ll,
        "GPU-coalesced leftmost-longest must equal the CPU leftmost-longest extraction"
    );
}

#[test]
fn regex_admission_by_region_on_gpu_matches_cpu_oracle() {
    let backend = match WgpuBackend::shared() {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("no wgpu backend ({error}); skipping regex admission GPU parity test");
            return;
        }
    };

    let patterns = ["abc", "AKIA", "token", "bcd", "secret"];
    let pipeline = build_regex_dfa_pipeline(&patterns, 4096, 16_384)
        .expect("Fix: regex set must compile to an anchored DFA");
    let dfa = &pipeline.dfa;
    let pattern_count = patterns.len() as u32;

    // Coalesced batch: three regions separated by '\n' (in no pattern).
    let haystack: &[u8] = b"xx abc AKIA\nsecret token\nbcd abc\n";
    let region_starts = [0u32, 12, 25];
    let region_count = region_starts.len() as u32;
    let words = regex_admission_presence_words(pattern_count);
    let log2_max_regions = (32 - (region_count.max(2) - 1).leading_zeros()).max(1);

    // CPU oracle bitmap.
    let expected =
        regex_admission_by_region_reference(dfa, haystack, &region_starts, 0, pattern_count);
    assert!(
        expected.iter().any(|&w| w != 0),
        "vacuous test: the oracle admitted no patterns"
    );

    let program = regex_admission_by_region_program(
        "haystack",
        "transitions",
        "output_offsets",
        "output_records",
        "region_starts",
        "region_base",
        "haystack_len",
        "presence",
        dfa.state_count,
        dfa.output_records.len() as u32,
        region_count,
        words,
        dfa.max_pattern_len,
        log2_max_regions,
    );
    let inputs: Vec<Vec<u8>> = vec![
        pack_haystack_u32(haystack),
        pack_u32_slice(&dfa.transitions),
        pack_u32_slice(&dfa.output_offsets),
        pack_u32_slice(&dfa.output_records),
        pack_u32_slice(&region_starts),
        pack_u32_slice(&[0]),
        pack_u32_slice(&[haystack.len() as u32]),
        vec![0u8; expected.len() * 4],
    ];
    let borrowed: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();
    // One invocation per haystack byte.
    let mut config = DispatchConfig::default();
    config.grid_override = Some([(haystack.len() as u32).div_ceil(128).max(1), 1, 1]);
    let outputs = backend
        .dispatch_borrowed(&program, &borrowed, &config)
        .expect("wgpu dispatch of regex admission-by-region program");

    // The sole writable buffer (presence) is outputs[0].
    let got: Vec<u32> = outputs[0]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .take(expected.len())
        .collect();

    assert_eq!(
        got, expected,
        "GPU regex admission bitmap must equal the CPU oracle (region search / atomics / grid)"
    );
}

/// Plan W2-2, line 158: the `RegionEvidencePipeline` successor's BOTH GPU
/// strategies, the fast two-dispatch `scan` and the single-launch `scan_fused`
/// (line 153's capability), must reproduce the CPU oracle bundle bit-for-bit on
/// real hardware. One bundle definition, three ways to compute it; here we prove
/// all three agree on the RTX 5090.
#[test]
fn region_evidence_pipeline_both_strategies_match_oracle_on_gpu() {
    let backend = match WgpuBackend::shared() {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("no wgpu backend ({error}); skipping region-evidence pipeline GPU test");
            return;
        }
    };

    let patterns = ["abc", "abcde", "AKIA", "token", "bcd", "secret"];
    let pipeline_dfa = build_regex_dfa_pipeline(&patterns, 4096, 16_384)
        .expect("Fix: regex set must compile to an anchored DFA");
    let pattern_count = patterns.len() as u32;
    // positions for {abc, abcde, bcd}; admission for {AKIA, token, secret}.
    let position_mask = vec![1u32, 1, 0, 0, 1, 0];
    let admission_mask = vec![0u32, 0, 1, 1, 0, 1];
    let pipeline = RegionEvidencePipeline::new(
        pipeline_dfa.dfa.clone(),
        pattern_count,
        position_mask,
        admission_mask,
    )
    .expect("masks cover every pattern");

    // Three coalesced regions separated by '\n' (in no pattern).
    let haystack: &[u8] = b"zabcde AKIA bcd\nsecret token abc\nAKIA1 abcde bcd\n";
    let region_starts = [0u32, 16, 32];
    let region_base = 0u32;
    let max_matches = 4096u32;

    let oracle = pipeline.reference_scan(haystack, &region_starts, region_base);
    assert!(
        oracle.presence.iter().any(|&w| w != 0) && !oracle.positions.is_empty(),
        "vacuous test: the oracle produced no presence/positions"
    );

    let fast = pipeline
        .scan(
            backend.as_ref(),
            haystack,
            &region_starts,
            region_base,
            max_matches,
        )
        .expect("fast-path scan must dispatch on the GPU");
    let fused = pipeline
        .scan_fused(
            backend.as_ref(),
            haystack,
            &region_starts,
            region_base,
            max_matches,
        )
        .expect("single-launch fused scan must dispatch on the GPU");

    assert_eq!(
        fast, oracle,
        "fast two-dispatch path must equal the CPU oracle bundle (presence/positions/admission)"
    );
    assert_eq!(
        fused, oracle,
        "single-launch fused path must equal the CPU oracle bundle (presence/positions/admission)"
    );
}
