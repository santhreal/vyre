//! Shared test helpers for classic-AC program conformance tests.

use crate::pattern::classic_ac::{classic_ac_bounded_ranges_scan, classic_ac_compile};
use crate::pattern::CompiledDfa;
use vyre_foundation::ir::Program;
use vyre_primitives::wire::pack_u32_slice;

use crate::fixture_bytes::bytes_to_u32;
use crate::fixture_bytes::eval_bytes;
use crate::pattern::haystack::pack_haystack_u32;

/// A u32 slice as one reference-backend input value.
pub(crate) fn u32_input(words: &[u32]) -> Vec<u8> {
    pack_u32_slice(words)
}

/// Bindings 0-2 of any AC dispatch: the packed haystack, the dense
/// `state * 256 + byte` transition table, and the flat output-link offsets.
///
/// Ordered to match `classic_ac_dfa_buffer_decls`, which is the declaration side
/// of the same ABI, so a binding reordered there fails these tests rather than
/// silently feeding a program the wrong buffer.
pub(crate) fn ac_dfa_table_inputs(dfa: &CompiledDfa, haystack: &[u8]) -> Vec<Vec<u8>> {
    vec![
        pack_haystack_u32(haystack),
        u32_input(&dfa.transitions),
        u32_input(&dfa.output_offsets),
    ]
}

/// Bindings 0-5 of a bounded-RANGES AC dispatch: [`ac_dfa_table_inputs`] plus the
/// flat output records, the pattern-length table, and the live haystack length.
///
/// Ordered to match `AcInputBindings::decls`. A caller appends only what its own
/// program shape adds past binding 5: a zeroed match counter, a presence bitmap,
/// prefilter mask words. A count program binds a shorter prefix and uses
/// [`ac_dfa_table_inputs`] instead of truncating this.
pub(crate) fn ac_ranges_inputs(
    dfa: &CompiledDfa,
    haystack: &[u8],
    lengths: &[u32],
) -> Vec<Vec<u8>> {
    let mut inputs = ac_dfa_table_inputs(dfa, haystack);
    inputs.reserve(3);
    inputs.push(u32_input(&dfa.output_records));
    inputs.push(u32_input(lengths));
    inputs.push(u32_input(&[haystack.len() as u32]));
    inputs
}

/// Assert an infallible AC builder wired the REAL DFA rather than a degenerate
/// empty rejecting program, and that it agrees with its fallible partner.
///
/// The two programs stay INDEPENDENTLY built at the call site and arrive here as
/// arguments. That is the whole point of the check: two separate entry points
/// must produce the same dispatch shape, so this compares them and never builds
/// either. `shape` names the builder in the failure message.
///
/// It checks the two bindings a degenerate fallback would betray, binding 1's
/// transition-table width and binding 3's record count, not the whole buffer
/// list. A full binding-layout assertion belongs with the builder that owns that
/// layout.
pub(crate) fn assert_infallible_matches_try(
    shape: &str,
    via_infallible: &Program,
    via_try: &Program,
    dfa: &CompiledDfa,
) {
    let records = via_infallible.buffers()[3].count;
    assert_eq!(records as usize, dfa.output_records.len());
    assert!(
        records > 0,
        "infallible {shape} builder must not emit a degenerate empty rejecting program"
    );
    assert_eq!(
        via_infallible.buffers()[1].count,
        dfa.state_count.saturating_mul(256)
    );
    assert_eq!(via_infallible.buffers().len(), via_try.buffers().len());
    assert_eq!(
        via_infallible.buffers()[3].count,
        via_try.buffers()[3].count
    );
}

/// Rewrite the `match_count` buffer to `lanes` output words so the reference
/// backend materializes one count slot per dispatched lane.
pub(crate) fn with_reference_dispatch_lanes(program: Program, lanes: u32) -> Program {
    let buffers = program
        .buffers()
        .iter()
        .cloned()
        .map(|buffer| {
            if buffer.name() == "match_count" {
                buffer.with_count(lanes.max(1)).with_output_byte_range(0..4)
            } else {
                buffer
            }
        })
        .collect();
    program.with_rewritten_buffers(buffers)
}

/// Pattern byte-lengths as u32 (the `pattern_lengths` buffer contents).
pub(crate) fn pattern_lengths(patterns: &[&[u8]]) -> Vec<u32> {
    patterns
        .iter()
        .map(|pattern| pattern.len() as u32)
        .collect()
}

/// Decode `(pattern_id, start, end)` triples from a `match_count` + `matches`
/// reference-output pair.
pub(crate) fn decode_match_triples(outputs: &[Vec<u8>]) -> Vec<(u32, u32, u32)> {
    let count = bytes_to_u32(&outputs[0])[0] as usize;
    let words = bytes_to_u32(&outputs[1]);
    words[..count.saturating_mul(3)]
        .chunks_exact(3)
        .map(|chunk| (chunk[0], chunk[1], chunk[2]))
        .collect()
}

/// Reference evaluation of an AC bounded-ranges program and assertions against expected match triples.
pub(crate) fn evaluate_and_assert_ranges_matches(
    program: &Program,
    inputs: &[Vec<u8>],
    expected: &[(u32, u32, u32)],
) {
    let outputs = eval_bytes("test_dispatch_and_decode", program, inputs.to_vec());
    let mut decoded = decode_match_triples(&outputs);
    decoded.sort_unstable();
    let mut expected_sorted = expected.to_vec();
    expected_sorted.sort_unstable();
    assert_eq!(decoded, expected_sorted);
}

/// Assert a bounded-ranges PREFILTER program reproduces the host scan oracle.
///
/// Every prefilter width binds the same prefix (`ac_ranges_inputs` plus a zeroed
/// match counter) and then its own mask words, so the width supplies two
/// closures and nothing else: `build` for the program under test and
/// `prefilter_words` for the bindings past the counter, in declaration order. A
/// width that assembles that prefix itself can bind the counter and its masks in
/// the wrong order and still agree with a copy of the same mistake.
pub(crate) fn assert_ranges_prefilter_matches_oracle(
    patterns: &[&[u8]],
    haystack: &[u8],
    build: impl FnOnce(&CompiledDfa, u32) -> Program,
    prefilter_words: impl FnOnce(&CompiledDfa, &[&[u8]]) -> Vec<Vec<u32>>,
) {
    let ac = classic_ac_compile(patterns);
    let lengths = pattern_lengths(patterns);
    let expected = classic_ac_bounded_ranges_scan(&ac, &lengths, haystack);
    let program = build(&ac.dfa, patterns.len() as u32);
    let mut inputs = ac_ranges_inputs(&ac.dfa, haystack, &lengths);
    inputs.push(u32_input(&[0]));
    inputs.extend(
        prefilter_words(&ac.dfa, patterns)
            .iter()
            .map(|words| u32_input(words)),
    );
    evaluate_and_assert_ranges_matches(&program, &inputs, &expected);
}
