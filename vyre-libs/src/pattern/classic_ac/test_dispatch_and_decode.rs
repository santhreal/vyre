//! Shared test helpers for classic-AC program conformance tests.

use crate::pattern::CompiledDfa;
use vyre_foundation::ir::Program;
use vyre_primitives::wire::pack_u32_slice;
use vyre_reference::value::Value;

use crate::fixture_bytes::bytes_to_u32;
use crate::pattern::haystack::pack_haystack_u32;

/// A u32 slice as one reference-backend input value.
pub(crate) fn u32_input(words: &[u32]) -> Value {
    Value::from(pack_u32_slice(words))
}

/// Bindings 0-2 of any AC dispatch: the packed haystack, the dense
/// `state * 256 + byte` transition table, and the flat output-link offsets.
///
/// Ordered to match `classic_ac_dfa_buffer_decls`, which is the declaration side
/// of the same ABI, so a binding reordered there fails these tests rather than
/// silently feeding a program the wrong buffer.
pub(crate) fn ac_dfa_table_inputs(dfa: &CompiledDfa, haystack: &[u8]) -> Vec<Value> {
    vec![
        Value::from(pack_haystack_u32(haystack)),
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
pub(crate) fn ac_ranges_inputs(dfa: &CompiledDfa, haystack: &[u8], lengths: &[u32]) -> Vec<Value> {
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
pub(crate) fn decode_match_triples(
    outputs: &[vyre_reference::value::Value],
) -> Vec<(u32, u32, u32)> {
    let count = bytes_to_u32(&outputs[0].to_bytes())[0] as usize;
    let words = bytes_to_u32(&outputs[1].to_bytes());
    words[..count.saturating_mul(3)]
        .chunks_exact(3)
        .map(|chunk| (chunk[0], chunk[1], chunk[2]))
        .collect()
}
