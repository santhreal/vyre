//! High-volume DIFFERENTIAL soundness gate for the FUSED region-presence +
//! match-positions program, evaluated on the CPU REFERENCE backend (no GPU, runs
//! everywhere).
//!
//! The fold's correctness claim is "recall-identical by construction": one
//! suffix3-gated walk that emits BOTH the per-region presence bitmap AND the match
//! triples must produce EXACTLY what the two separate programs produce
//! `scan_presence_by_region` for the bitmap and the suffix3 prefilter (positions)
//! program for the triples. This test proves that empirically across thousands of
//! random (literal set, multi-region haystack) cases: for each case it
//! `reference_eval`s all three programs and asserts the fused bitmap equals the
//! separate presence bitmap word-for-word AND the fused triple set equals the
//! separate position set. A divergence here is a recall bug in the fold.

#![cfg(feature = "pattern-substring")]

mod wire_words;
use wire_words::decode_u32_words as decode_u32;

mod presence_oracle;
use presence_oracle::{random_haystack, random_literals, random_region_starts, Lcg, PackedCase};
use std::collections::BTreeSet;

use vyre_libs::pattern::classic_ac::{
    try_build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce,
    try_build_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program,
    try_build_ac_bounded_ranges_suffix3_presence_by_region_program,
};
use vyre_reference::value::Value;

const MAX_MATCHES: u32 = 4096;
fn decode_triples(count_words: &[u32], match_words: &[u32]) -> BTreeSet<(u32, u32, u32)> {
    let count = *count_words.first().unwrap_or(&0) as usize;
    match_words
        .chunks_exact(3)
        .take(count)
        .map(|c| (c[0], c[1], c[2]))
        .collect()
}

#[test]
fn fused_presence_and_positions_equals_separate_scans_high_volume() {
    // Each case evaluates THREE programs in the reference backend (~0.1 s/case), so
    // the always-on gate defaults to 1000 cases (~2 min). VYRE_FUSED_CASES scales it
    // up for thorough/nightly runs (10k+ exercises the contract's proptest depth).
    let cases: usize = std::env::var("VYRE_FUSED_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);

    let mut rng = Lcg(0x6675_7365_645fu64);
    let mut checked = 0usize;
    let mut nonempty_presence = 0usize;
    let mut nonempty_matches = 0usize;
    let mut multi_region = 0usize;

    for case in 0..cases {
        let literals = random_literals(&mut rng);
        let haystack = random_haystack(&mut rng);
        let region_starts = random_region_starts(&mut rng, haystack.len());
        let packed = PackedCase::new(&literals, &haystack, &region_starts);
        let region_count = packed.region_count;

        // --- Separate presence-by-region program (bindings 0-11) ---
        let sep_presence_program = try_build_ac_bounded_ranges_suffix3_presence_by_region_program(
            &packed.ac.dfa,
            packed.pattern_count,
            region_count,
        )
        .expect("separate presence-by-region program builds");
        let sep_presence_out =
            vyre_reference::reference_eval(&sep_presence_program, &packed.presence_inputs())
                .expect("separate presence-by-region program evaluates");
        let sep_presence = decode_u32(&sep_presence_out[0].to_bytes());

        // --- Separate positions (suffix3 prefilter) program (bindings 0-10) ---
        // `use_subgroup_coalesce = false`: the reference backend can't lower
        // subgroup ops, and this is the exact non-subgroup form the production consumer's position
        // scan uses (`try_build_literal_set_program`). The fused program likewise
        // uses plain `append_match`, so both append paths match bit-for-bit.
        let sep_positions_program =
            try_build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce(
                &packed.ac.dfa,
                packed.pattern_count,
                MAX_MATCHES,
                false,
            )
            .expect("separate positions program builds");
        let sep_positions_inputs = vec![
            Value::from(packed.haystack.clone()),
            Value::from(packed.transitions.clone()),
            Value::from(packed.output_offsets.clone()),
            Value::from(packed.output_records.clone()),
            Value::from(packed.lengths.clone()),
            Value::from(packed.hay_len.clone()),
            Value::from(packed.zero.clone()), // 6: match_count
            Value::from(packed.end_mask.clone()),
            Value::from(packed.suffix2.clone()),
            Value::from(packed.suffix3.clone()),
        ];
        let sep_positions_out =
            vyre_reference::reference_eval(&sep_positions_program, &sep_positions_inputs)
                .expect("separate positions program evaluates");
        let sep_count = decode_u32(&sep_positions_out[0].to_bytes());
        let sep_matches = decode_u32(&sep_positions_out[1].to_bytes());
        let sep_triples = decode_triples(&sep_count, &sep_matches);

        // --- Fused program (bindings 0-13) ---
        let fused_program =
            try_build_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program(
                &packed.ac.dfa,
                packed.pattern_count,
                region_count,
                MAX_MATCHES,
            )
            .expect("fused program builds");
        let mut fused_inputs = packed.presence_inputs();
        // Binding 12 is the fused program's match counter, which the presence
        // program does not declare.
        fused_inputs.push(Value::from(packed.zero.clone()));
        let fused_out = vyre_reference::reference_eval(&fused_program, &fused_inputs)
            .expect("fused program evaluates");
        let fused_presence = decode_u32(&fused_out[0].to_bytes());
        let fused_count = decode_u32(&fused_out[1].to_bytes());
        let fused_matches = decode_u32(&fused_out[2].to_bytes());
        let fused_triples = decode_triples(&fused_count, &fused_matches);

        // The fold's two outputs must EXACTLY equal the two separate scans.
        assert_eq!(
            fused_presence,
            sep_presence,
            "case {case}: fused per-region presence differs from scan_presence_by_region \
             (literals={:?}, regions={region_starts:?})",
            literals
                .iter()
                .map(|l| String::from_utf8_lossy(l).into_owned())
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            fused_triples,
            sep_triples,
            "case {case}: fused match triples differ from the separate positions scan \
             (literals={:?})",
            literals
                .iter()
                .map(|l| String::from_utf8_lossy(l).into_owned())
                .collect::<Vec<_>>(),
        );

        // Independent linear-AC oracle cross-check: the fused triple SET (pid,end)
        // must equal the bounded-ranges AC oracle's, so the differential isn't two
        // programs sharing the same bug.
        let oracle: BTreeSet<(u32, u32)> = literals
            .iter()
            .enumerate()
            .flat_map(|(pattern_id, literal)| {
                haystack
                    .windows(literal.len())
                    .enumerate()
                    .filter(move |(_, window)| *window == literal.as_slice())
                    .map(move |(start, _)| (pattern_id as u32, (start + literal.len()) as u32))
            })
            .collect();
        let fused_pid_end: BTreeSet<(u32, u32)> =
            fused_triples.iter().map(|&(pid, _s, e)| (pid, e)).collect();
        assert_eq!(
            fused_pid_end, oracle,
            "case {case}: fused (pid,end) set diverges from the linear AC oracle"
        );

        if fused_presence.iter().any(|&w| w != 0) {
            nonempty_presence += 1;
        }
        if !fused_triples.is_empty() {
            nonempty_matches += 1;
        }
        if region_count > 1 {
            multi_region += 1;
        }
        checked += 1;
    }

    assert_eq!(checked, cases);
    // The corpus must actually exercise the present-pattern, match-emitting, and
    // multi-region paths, or the differential is vacuous.
    assert!(
        nonempty_presence * 4 > cases,
        "only {nonempty_presence}/{cases} cases had any present pattern; corpus too sparse"
    );
    assert!(
        nonempty_matches * 4 > cases,
        "only {nonempty_matches}/{cases} cases emitted any match; corpus too sparse"
    );
    assert!(
        multi_region * 2 > cases,
        "only {multi_region}/{cases} cases had >1 region; multi-region attribution under-tested"
    );
    eprintln!(
        "fused vs separate parity: {checked} cases, {nonempty_presence} present, \
         {nonempty_matches} matching, {multi_region} multi-region"
    );
}
