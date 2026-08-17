//! Sequential mathematical witnesses for pattern matching, bracket pairing, and match post-processing.

use super::text::shannon_entropy_bits_per_byte_witness;

/// Match bounded nested brackets and return bidirectional partner indices.
#[must_use]
pub fn bracket_match_witness(kinds: &[u32], max_depth: u32) -> Vec<u32> {
    let mut output = Vec::new();
    let mut stack = Vec::new();
    bracket_match_witness_into(kinds, max_depth, &mut output, &mut stack);
    output
}

/// Match bounded nested brackets into caller-owned output and stack storage.
///
/// Kinds `1` and `2` denote open and close tokens. Unmatched entries contain
/// `u32::MAX`; `stack` retains only unmatched opens admitted by `max_depth`.
pub fn bracket_match_witness_into(
    kinds: &[u32],
    max_depth: u32,
    output: &mut Vec<u32>,
    stack: &mut Vec<u32>,
) {
    output.clear();
    output.resize(kinds.len(), u32::MAX);
    stack.clear();
    for (index, &kind) in kinds.iter().enumerate() {
        match kind {
            1 if stack.len() < max_depth as usize => stack.push(index as u32),
            2 => {
                if let Some(open) = stack.pop() {
                    output[open as usize] = index as u32;
                    output[index] = open;
                }
            }
            _ => {}
        }
    }
}

/// Canonical match post-processing witness.
pub fn try_match_post_process_witness(
    matches: &[vyre_foundation::match_result::ByteRange],
    haystack: &[u8],
) -> Result<Vec<vyre_libs::pattern::PostProcessedMatch>, vyre_libs::pattern::PostProcessError> {
    use vyre_libs::pattern::{PostProcessError, PostProcessedMatch};
    for range in matches {
        if range.start > range.end || range.end as usize > haystack.len() {
            return Err(PostProcessError::InvalidRange {
                pattern_id: range.tag,
                start: range.start,
                end: range.end,
                haystack_len: haystack.len(),
            });
        }
    }
    let mut ranges = matches.to_vec();
    ranges.sort_unstable_by_key(|range| (range.tag, range.start, range.end));
    let mut merged = Vec::<vyre_foundation::match_result::ByteRange>::new();
    for range in ranges {
        if let Some(previous) = merged.last_mut() {
            if previous.tag == range.tag && range.start <= previous.end {
                previous.end = previous.end.max(range.end);
                continue;
            }
        }
        merged.push(range);
    }
    Ok(merged
        .into_iter()
        .map(|range| {
            let entropy = shannon_entropy_bits_per_byte_witness(
                &haystack[range.start as usize..range.end as usize],
            );
            let length_factor = (range.len() as f32 / 16.0).min(1.0);
            PostProcessedMatch {
                pattern_id: range.tag,
                start: range.start,
                end: range.end,
                entropy_bits_per_byte: entropy,
                confidence: length_factor * (entropy / 8.0),
            }
        })
        .collect())
}

/// Canonical match post-processing witness into caller-owned scratch.
pub fn try_match_post_process_witness_into(
    matches: &[vyre_foundation::match_result::ByteRange],
    haystack: &[u8],
    triples: &mut Vec<vyre_libs::pattern::RegionTriple>,
    output: &mut Vec<vyre_libs::pattern::PostProcessedMatch>,
) -> Result<(), vyre_libs::pattern::PostProcessError> {
    let processed = try_match_post_process_witness(matches, haystack)?;
    triples.clear();
    triples.extend(
        processed
            .iter()
            .map(|entry| vyre_libs::pattern::RegionTriple::new(entry.pattern_id, entry.start, entry.end)),
    );
    output.clear();
    output.extend(processed);
    Ok(())
}

/// Infallible canonical match post-processing witness.
#[must_use]
pub fn match_post_process_witness(
    matches: &[vyre_foundation::match_result::ByteRange],
    haystack: &[u8],
) -> Vec<vyre_libs::pattern::PostProcessedMatch> {
    try_match_post_process_witness(matches, haystack)
        .unwrap_or_else(|error| panic!("post-process contract failed: {error}"))
}
