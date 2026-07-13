//! INDEPENDENT ground-truth soundness gate for the region-presence program
//! (`try_build_ac_bounded_ranges_suffix3_presence_by_region_program`), on the CPU
//! REFERENCE backend (runs everywhere, no GPU).
//!
//! Why this exists: the sibling `literal_set_presence_and_positions_reference`
//! test compares the FUSED region program against the SEPARATE region + position
//! programs. All three share the identical suffix3 prefilter cascade, DFA walk,
//! and region binary search, so a bug in ANY of those shared components passes
//! that test (all three under-fire together and still agree). It cannot catch a
//! region-presence UNDER-FIRE, which is exactly the open W1-1 correctness debt:
//! keyhog observed the GPU region-presence path missing real `(chunk, detector)`
//! pairs its CPU recall floor recovered.
//!
//! This gate checks the program's output against a genuinely independent oracle
//! (`presence_oracle::oracle_presence`, a plain-Rust DFA walk that touches none of
//! the program's machinery). The GPU twin
//! (`literal_set_presence_by_region_gpu_ground_truth`) runs the SAME oracle
//! against the real wgpu backend, the actual path keyhog dispatches, so a
//! subgroup-divergence under-fire that only manifests on device is caught there.

mod presence_oracle;

use presence_oracle::{
    assert_presence_matches, edge_cases, random_haystack, random_literals, random_region_starts,
    scale_cases, Lcg,
};
use vyre_libs::scan::classic_ac::{
    classic_ac_candidate_end_byte_mask_words, classic_ac_candidate_suffix2_mask_words,
    classic_ac_candidate_suffix3_bloom_words, classic_ac_compile, presence_by_region_words,
    try_build_ac_bounded_ranges_suffix3_presence_by_region_program,
};
use vyre_libs::scan::{pack_haystack_u32, pack_u32_slice};

fn decode_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Evaluate the region-presence GPU program on the CPU reference backend and
/// return the decoded per-region presence bitmap.
fn program_presence(literals: &[Vec<u8>], haystack: &[u8], region_starts: &[u32]) -> Vec<u32> {
    let pattern_refs: Vec<&[u8]> = literals.iter().map(Vec::as_slice).collect();
    let ac = classic_ac_compile(&pattern_refs);
    let pattern_count = literals.len() as u32;
    let region_count = region_starts.len() as u32;

    let end_mask = classic_ac_candidate_end_byte_mask_words(&ac.dfa);
    let suffix2_mask = classic_ac_candidate_suffix2_mask_words(&ac.dfa);
    let suffix3_bloom = classic_ac_candidate_suffix3_bloom_words(&pattern_refs);
    let lengths: Vec<u32> = literals.iter().map(|l| l.len() as u32).collect();

    let haystack_packed = pack_haystack_u32(haystack);
    let transitions = pack_u32_slice(&ac.dfa.transitions);
    let output_offsets = pack_u32_slice(&ac.dfa.output_offsets);
    let output_records = pack_u32_slice(&ac.dfa.output_records);
    let lengths_packed = pack_u32_slice(&lengths);
    let hay_len = pack_u32_slice(&[haystack.len() as u32]);
    let end_mask_packed = pack_u32_slice(&end_mask);
    let suffix2_packed = pack_u32_slice(&suffix2_mask);
    let suffix3_packed = pack_u32_slice(&suffix3_bloom);
    let region_starts_packed = pack_u32_slice(region_starts);
    let zero = pack_u32_slice(&[0u32]);
    let total_presence_words = presence_by_region_words(pattern_count, region_count) as usize;
    let presence_zeroed = pack_u32_slice(&vec![0u32; total_presence_words]);

    let val = vyre_reference::value::Value::from;
    let program = try_build_ac_bounded_ranges_suffix3_presence_by_region_program(
        &ac.dfa,
        pattern_count,
        region_count,
    )
    .expect("region-presence program builds");
    let inputs = vec![
        val(haystack_packed),
        val(transitions),
        val(output_offsets),
        val(output_records),
        val(lengths_packed),
        val(hay_len),
        val(presence_zeroed),
        val(end_mask_packed),
        val(suffix2_packed),
        val(suffix3_packed),
        val(region_starts_packed),
        val(zero),
    ];
    // Force the interpreter's grid to cover one invocation per haystack BYTE.
    // Buffer-shape inference alone under-covers a byte-scan program (the haystack
    // is packed 4 bytes/u32), silently skipping high positions and under-firing 
    // proven a reference-interpreter artifact, not a kernel bug, by the GPU gate
    // (`literal_set_presence_by_region_gpu_ground_truth`) passing the same fixtures.
    let out =
        vyre_reference::reference_eval_with_dispatch(&program, &inputs, haystack.len() as u32)
            .expect("region-presence program evaluates");
    decode_u32(&out[0].to_bytes())
}

fn check(literals: &[Vec<u8>], haystack: &[u8], region_starts: &[u32], label: &str) {
    let produced = program_presence(literals, haystack, region_starts);
    assert_presence_matches(literals, haystack, region_starts, &produced, label);
}

#[test]
fn region_presence_matches_independent_dfa_oracle_high_volume() {
    let cases: usize = std::env::var("VYRE_PRESENCE_GROUND_TRUTH_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);

    let mut rng = Lcg::new(0x7072_6573_656eu64);
    for case in 0..cases {
        let literals = random_literals(&mut rng);
        let haystack = random_haystack(&mut rng);
        let region_starts = random_region_starts(&mut rng, haystack.len());
        check(
            &literals,
            &haystack,
            &region_starts,
            &format!("case {case}"),
        );
    }
}

/// Targeted W1-1 edge classes: literals straddling every offset of a region
/// boundary, matches at region start/end, `\0` separator adjacency, dense
/// saturation, non-ASCII adjacency, single-byte early-offset cascade boundaries.
#[test]
fn region_presence_edge_classes() {
    for (label, literals, haystack, region_starts) in edge_cases() {
        check(&literals, &haystack, &region_starts, &label);
    }
}

/// keyhog-shaped scale: multi-word presence rows (>32/>64 patterns), many small
/// regions, and full byte-range patterns, the coverage the edge cases never
/// reach and the shape W6-1 names as the consumer spec.
#[test]
fn region_presence_scale_classes() {
    for (label, literals, haystack, region_starts) in scale_cases() {
        check(&literals, &haystack, &region_starts, &label);
    }
}
