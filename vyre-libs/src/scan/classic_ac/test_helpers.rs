//! Shared test helpers for classic-AC program conformance tests.

use vyre::ir::Program;

use crate::test_support::byte_pack::bytes_to_u32;

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
