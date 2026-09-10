//! The regex whole-buffer program emits the same matches with subgroup
//! coalescing on as with it off.
//!
//! `build_regex_dfa_pipeline` selects `use_subgroup_coalesce = true`, and the
//! coalesced emit runs a subgroup shuffle. A collective reads a value out of a
//! peer lane, so every lane of the subgroup has to reach it with that value
//! bound. Three things made the regex walk diverge before this was proved: a
//! lane whose origin is past the live length skipped the walk, a lane near the
//! end replayed a shorter window, and a position with no output records never
//! entered the record loop. The reference run did not merely disagree, it had
//! nothing to read, so the shipped strategy was unjudgeable.

#![cfg(all(feature = "pattern-regex", feature = "pattern-dfa"))]

use vyre_libs_pattern::pattern::pack_haystack_u32;
use vyre_libs_pattern::pattern::{
    build_regex_dfa_pipeline_with_policy_and_subgroup_coalesce, RegexReplayPolicy,
};
use vyre_primitives::wire::pack_u32_slice;
use vyre_test_support::test_parity_oracles::{bytes_to_u32, eval_bytes};

const PATTERNS: [&str; 4] = ["alpha", "beta", "gamma", "al"];
const MAX_MATCHES: u32 = 512;
const MAX_DFA_STATES: usize = 4096;

/// A haystack whose hits cluster, so lanes in one subgroup carry different
/// record spans and different replay windows. A uniform haystack would hide
/// the divergence this file exists to catch.
const HAYSTACK: &[u8] = b"alpha al beta alpha gamma al alphabeta  gamma";

fn triples(outputs: &[Vec<u8>]) -> Vec<(u32, u32, u32)> {
    let count = bytes_to_u32(&outputs[0])[0] as usize;
    let words = bytes_to_u32(&outputs[1]);
    let mut decoded: Vec<(u32, u32, u32)> = words[..count.saturating_mul(3)]
        .chunks_exact(3)
        .map(|chunk| (chunk[0], chunk[1], chunk[2]))
        .collect();
    decoded.sort_unstable();
    decoded
}

fn scan(coalesce: bool) -> Vec<(u32, u32, u32)> {
    let pipeline = build_regex_dfa_pipeline_with_policy_and_subgroup_coalesce(
        &PATTERNS,
        MAX_MATCHES,
        MAX_DFA_STATES,
        RegexReplayPolicy::default(),
        coalesce,
    )
    .expect("the regex pipeline compiles this literal set");
    let inputs = vec![
        pack_haystack_u32(HAYSTACK),
        pack_u32_slice(&pipeline.dfa.transitions),
        pack_u32_slice(&pipeline.dfa.output_offsets),
        pack_u32_slice(&pipeline.dfa.output_records),
        pack_u32_slice(&pipeline.pattern_lengths),
        pack_u32_slice(&[HAYSTACK.len() as u32]),
        pack_u32_slice(&[0]),
    ];
    triples(&eval_bytes(
        "regex_exact_coalesce_parity",
        &pipeline.program,
        inputs,
    ))
}

#[test]
fn the_coalesced_regex_program_emits_what_the_serial_one_emits() {
    assert_eq!(
        scan(true),
        scan(false),
        "subgroup coalescing reserves hit-buffer slots per subgroup instead of per lane; it \
         decides where a triple lands, never whether one exists"
    );
}

#[test]
fn the_coalesced_regex_program_finds_every_literal() {
    let found = scan(true);
    assert!(
        !found.is_empty(),
        "a haystack containing every pattern must produce matches; an empty result means the \
         uniform walk admitted nothing rather than agreeing with the serial emit"
    );
}
