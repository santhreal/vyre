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

/// Structured error from reference match post-processing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WitnessPostProcessError {
    /// Range exceeds haystack boundaries or is inverted.
    InvalidRange {
        /// Tag / pattern id.
        pattern_id: u32,
        /// Half-open start offset.
        start: u32,
        /// Half-open end offset.
        end: u32,
        /// Haystack length in bytes.
        haystack_len: usize,
    },
}

impl std::fmt::Display for WitnessPostProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRange {
                pattern_id,
                start,
                end,
                haystack_len,
            } => write!(
                f,
                "post-process input contains an invalid byte range [{start}, {end}) for pattern_id={pattern_id} (haystack_len={haystack_len})"
            ),
        }
    }
}

impl std::error::Error for WitnessPostProcessError {}

/// Post-processed match record produced by the reference witness.
#[derive(Debug, Clone, PartialEq)]
pub struct WitnessPostProcessedMatch {
    /// Matched pattern identifier.
    pub pattern_id: u32,
    /// Half-open start byte offset in haystack.
    pub start: u32,
    /// Half-open end byte offset in haystack.
    pub end: u32,
    /// Shannon entropy in bits per byte of the matched span.
    pub entropy_bits_per_byte: f32,
    /// Calibrated match confidence in `[0.0, 1.0]`.
    pub confidence: f32,
}

/// Canonical match post-processing witness.
pub fn try_match_post_process_witness(
    matches: &[vyre_foundation::match_result::ByteRange],
    haystack: &[u8],
) -> Result<Vec<WitnessPostProcessedMatch>, WitnessPostProcessError> {
    for range in matches {
        if range.start > range.end || range.end as usize > haystack.len() {
            return Err(WitnessPostProcessError::InvalidRange {
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
            WitnessPostProcessedMatch {
                pattern_id: range.tag,
                start: range.start,
                end: range.end,
                entropy_bits_per_byte: entropy,
                confidence: length_factor * (entropy / 8.0),
            }
        })
        .collect())
}

/// Canonical match post-processing records into caller callback.
pub fn try_match_post_process_records_into(
    matches: &[vyre_foundation::match_result::ByteRange],
    haystack: &[u8],
    mut on_record: impl FnMut(u32, u32, u32, f32, f32),
) -> Result<(), WitnessPostProcessError> {
    for range in matches {
        if range.start > range.end || range.end as usize > haystack.len() {
            return Err(WitnessPostProcessError::InvalidRange {
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
    for range in merged {
        let entropy = shannon_entropy_bits_per_byte_witness(
            &haystack[range.start as usize..range.end as usize],
        );
        let length_factor = (range.len() as f32 / 16.0).min(1.0);
        let confidence = length_factor * (entropy / 8.0);
        on_record(range.tag, range.start, range.end, entropy, confidence);
    }
    Ok(())
}

/// Canonical match post-processing witness into caller-owned scratch.
pub fn try_match_post_process_witness_into(
    matches: &[vyre_foundation::match_result::ByteRange],
    haystack: &[u8],
    triples: &mut Vec<(u32, u32, u32)>,
    output: &mut Vec<WitnessPostProcessedMatch>,
) -> Result<(), WitnessPostProcessError> {
    triples.clear();
    output.clear();
    try_match_post_process_records_into(
        matches,
        haystack,
        |pattern_id, start, end, entropy_bits_per_byte, confidence| {
            triples.push((pattern_id, start, end));
            output.push(WitnessPostProcessedMatch {
                pattern_id,
                start,
                end,
                entropy_bits_per_byte,
                confidence,
            });
        },
    )
}

/// Infallible canonical match post-processing witness.
#[must_use]
pub fn match_post_process_witness(
    matches: &[vyre_foundation::match_result::ByteRange],
    haystack: &[u8],
) -> Vec<WitnessPostProcessedMatch> {
    try_match_post_process_witness(matches, haystack)
        .unwrap_or_else(|error| panic!("post-process contract failed: {error}"))
}

/// Stable lexicographic sort of diagnostic region triples `(pid, start, end)`.
#[must_use]
pub fn sort_regions_witness(mut regions: Vec<(u32, u32, u32)>) -> Vec<(u32, u32, u32)> {
    regions.sort_unstable();
    regions
}

/// Sort and merge same-pattern region triples whose spans overlap or touch.
#[must_use]
pub fn dedup_regions_witness(mut regions: Vec<(u32, u32, u32)>) -> Vec<(u32, u32, u32)> {
    regions.sort_unstable();
    let mut merged = Vec::<(u32, u32, u32)>::with_capacity(regions.len());
    for (pid, start, end) in regions {
        if let Some((prev_pid, _prev_start, prev_end)) = merged.last_mut() {
            if *prev_pid == pid && start <= *prev_end {
                *prev_end = (*prev_end).max(end);
                continue;
            }
        }
        merged.push((pid, start, end));
    }
    merged
}

/// Compute binary survivor flags for sorted `(pid, start, end)` region triples.
#[must_use]
pub fn dedup_regions_survivor_flags_witness(sorted: &[(u32, u32, u32)]) -> Vec<u32> {
    let mut flags = vec![0_u32; sorted.len()];
    if sorted.is_empty() {
        return flags;
    }
    flags[0] = 1;
    for i in 1..sorted.len() {
        let (cur_pid, cur_start, _cur_end) = sorted[i];
        let (prv_pid, _prv_start, prv_end) = sorted[i - 1];
        let different_pid = cur_pid != prv_pid;
        let no_overlap = cur_start > prv_end;
        flags[i] = if different_pid || no_overlap { 1 } else { 0 };
    }
    flags
}

/// Sequential Thompson-NFA transition and epsilon-closure step into caller-owned storage.
pub fn subgroup_nfa_step_witness_into(
    state: &[u32],
    byte: u8,
    transitions: &[u32],
    epsilon: &[u32],
    num_states: usize,
    output: &mut Vec<u32>,
    scratch: &mut Vec<u32>,
) {
    const WORDS: usize = 32;
    assert!(num_states <= 1024, "NFA state count exceeds one subgroup");
    assert_eq!(state.len(), WORDS, "complete NFA subgroup state");
    assert_eq!(
        transitions.len(),
        num_states * 256 * WORDS,
        "complete NFA transition table"
    );
    assert_eq!(
        epsilon.len(),
        num_states * WORDS,
        "complete NFA epsilon table"
    );
    if output.capacity() < WORDS {
        output.reserve(WORDS.saturating_sub(output.len()));
    }
    output.clear();
    output.resize(WORDS, 0);
    for source in 0..num_states {
        if state
            .get(source / 32)
            .is_some_and(|word| word & (1_u32 << (source % 32)) != 0)
        {
            let start = source * 256 * WORDS + usize::from(byte) * WORDS;
            for (destination, value) in output.iter_mut().enumerate() {
                *value |= transitions.get(start + destination).copied().unwrap_or(0);
            }
        }
    }
    if scratch.capacity() < WORDS {
        scratch.reserve(WORDS.saturating_sub(scratch.len()));
    }
    for _ in 0..num_states.min(1024) {
        scratch.clear();
        scratch.extend_from_slice(output);
        for source in 0..num_states {
            if scratch
                .get(source / 32)
                .is_some_and(|word| word & (1_u32 << (source % 32)) != 0)
            {
                let start = source * WORDS;
                for (destination, value) in output.iter_mut().enumerate() {
                    *value |= epsilon.get(start + destination).copied().unwrap_or(0);
                }
            }
        }
        if output == scratch {
            break;
        }
    }
}

/// Sequential Thompson-NFA transition and epsilon-closure step.
#[must_use]
pub fn subgroup_nfa_step_witness(
    state: &[u32],
    byte: u8,
    transitions: &[u32],
    epsilon: &[u32],
    num_states: usize,
) -> Vec<u32> {
    let mut output = Vec::with_capacity(32);
    let mut scratch = Vec::with_capacity(32);
    subgroup_nfa_step_witness_into(
        state,
        byte,
        transitions,
        epsilon,
        num_states,
        &mut output,
        &mut scratch,
    );
    output
}

/// Greedy row-major non-overlapping planar rewrite schedule.
#[must_use]
pub fn planar_rewrite_schedule_witness(
    candidates: &[u32],
    height: u32,
    width: u32,
    exclusion: u32,
) -> Vec<u32> {
    let mut chosen = vec![0_u32; (height * width) as usize];
    for row in 0..height {
        for column in 0..width {
            let index = (row * width + column) as usize;
            if candidates.get(index).copied().unwrap_or(0) == 0 {
                continue;
            }
            let conflict = (0..exclusion).any(|dr| {
                (0..exclusion).any(|dc| {
                    row >= dr
                        && column >= dc
                        && chosen[((row - dr) * width + column - dc) as usize] != 0
                        && (dr != 0 || dc != 0)
                })
            });
            if !conflict {
                chosen[index] = 1;
            }
        }
    }
    chosen
}
/// Sort diagnostic regions in place by their canonical ordering.
pub fn sort_regions_witness_in_place(regions: &mut [(u32, u32, u32)]) {
    regions.sort_unstable();
}

/// Sort and merge diagnostic regions in caller-owned storage.
pub fn dedup_regions_witness_in_place(regions: &mut Vec<(u32, u32, u32)>) {
    let merged = dedup_regions_witness(std::mem::take(regions));
    *regions = merged;
}

/// Mark the first `limit` occurrences of each pattern id.
#[must_use]
pub fn cap_regions_per_pattern_survivors_witness(pattern_ids: &[u32], limit: u32) -> Vec<u32> {
    let mut counts = std::collections::HashMap::<u32, u32>::new();
    pattern_ids
        .iter()
        .map(|&pattern| {
            let count = counts.entry(pattern).or_default();
            let keep = *count < limit;
            *count = count.saturating_add(1);
            u32::from(keep)
        })
        .collect()
}

/// Mark the first occurrence of each `(region, pattern)` pair.
#[must_use]
pub fn compact_first_per_region_pattern_survivors_witness(
    regions: &[u32],
    pattern_ids: &[u32],
) -> Vec<u32> {
    let mut seen = std::collections::HashSet::<(u32, u32)>::new();
    regions
        .iter()
        .zip(pattern_ids)
        .map(|(&region, &pattern)| u32::from(seen.insert((region, pattern))))
        .collect()
}

/// Walk a classic Aho-Corasick automaton and emit every matching pattern id.
#[must_use]
pub fn classic_ac_scan_witness(
    transitions: &[u32],
    output_offsets: &[u32],
    output_records: &[u32],
    haystack: &[u8],
) -> Vec<(u32, u32)> {
    let mut state = 0_u32;
    let mut matches = Vec::new();
    for (position, &byte) in haystack.iter().enumerate() {
        state = transitions[state as usize * 256 + byte as usize];
        let start = output_offsets[state as usize] as usize;
        let end = output_offsets[state as usize + 1] as usize;
        matches.extend(
            output_records[start..end]
                .iter()
                .map(|&pattern| (pattern, position as u32)),
        );
    }
    matches
}

/// Walk a classic Aho-Corasick automaton and count matches per byte position.
#[must_use]
pub fn classic_ac_scan_counts_witness(
    transitions: &[u32],
    output_offsets: &[u32],
    haystack: &[u8],
) -> Vec<u32> {
    let mut state = 0_u32;
    haystack
        .iter()
        .map(|&byte| {
            state = transitions[state as usize * 256 + byte as usize];
            output_offsets[state as usize + 1] - output_offsets[state as usize]
        })
        .collect()
}

/// Walk a classic Aho-Corasick automaton and emit bounded match ranges.
#[must_use]
pub fn classic_ac_bounded_ranges_scan_witness(
    transitions: &[u32],
    output_offsets: &[u32],
    output_records: &[u32],
    pattern_lengths: &[u32],
    haystack: &[u8],
) -> Vec<(u32, u32, u32)> {
    classic_ac_scan_witness(transitions, output_offsets, output_records, haystack)
        .into_iter()
        .map(|(pattern, end)| {
            let end = end.saturating_add(1);
            let start = end.saturating_sub(pattern_lengths[pattern as usize]);
            (pattern, start, end)
        })
        .collect()
}

/// Walk a compiled DFA byte-by-byte and emit the accept bitmask at each byte position.
#[must_use]
pub fn dfa_scan_accept_witness(transitions: &[u32], accept: &[u32], haystack: &[u8]) -> Vec<u32> {
    let mut state = 0_u32;
    let mut out = Vec::with_capacity(haystack.len());
    for &b in haystack {
        state = transitions[(state as usize) * 256 + b as usize];
        out.push(accept[state as usize]);
    }
    out
}

/// Return the index of the region whose start is the greatest value not exceeding `position`.
#[must_use]
pub fn region_of_witness(position: u32, region_starts: &[u32]) -> usize {
    match region_starts.binary_search(&position) {
        Ok(exact) => exact,
        Err(0) => 0,
        Err(insertion) => insertion - 1,
    }
}

/// Derive the 8-word candidate-end-byte bitset for an AC DFA.
#[must_use]
pub fn classic_ac_candidate_end_byte_mask_words_witness(
    transitions: &[u32],
    output_offsets: &[u32],
    state_count: u32,
) -> [u32; 8] {
    let mut mask = [0_u32; 8];
    let states = (state_count as usize)
        .min(output_offsets.len().saturating_sub(1))
        .min(transitions.len() / 256);
    for state in 0..states {
        let row = state * 256;
        for byte in 0..256 {
            let next = transitions[row + byte] as usize;
            if next + 1 < output_offsets.len() && output_offsets[next] != output_offsets[next + 1] {
                mask[byte / 32] |= 1_u32 << (byte % 32);
            }
        }
    }
    mask
}

/// Derive the 65,536-bit (2048 u32 word) two-byte suffix mask for an AC DFA.
#[must_use]
pub fn classic_ac_candidate_suffix2_mask_words_witness(
    transitions: &[u32],
    output_offsets: &[u32],
    state_count: u32,
) -> [u32; 2048] {
    let mut mask = [0_u32; 2048];
    let states = (state_count as usize)
        .min(output_offsets.len().saturating_sub(1))
        .min(transitions.len() / 256);
    for state in 0..states {
        let row = state * 256;
        for previous in 0..256 {
            let mid = transitions[row + previous] as usize;
            if mid >= states {
                continue;
            }
            let mid_row = mid * 256;
            for byte in 0..256 {
                let next = transitions[mid_row + byte] as usize;
                if next + 1 < output_offsets.len()
                    && output_offsets[next] != output_offsets[next + 1]
                {
                    let suffix = (previous << 8) | byte;
                    mask[suffix / 32] |= 1_u32 << (suffix % 32);
                }
            }
        }
    }
    mask
}

/// ASCII-case-aware variants of a byte: returns `([byte, _], 1)` or `([lower, upper], 2)`.
#[must_use]
pub fn ascii_case_variants_witness(byte: u8, case_insensitive: bool) -> ([u8; 2], usize) {
    if case_insensitive && byte.is_ascii_alphabetic() {
        ([byte.to_ascii_lowercase(), byte.to_ascii_uppercase()], 2)
    } else {
        ([byte, 0], 1)
    }
}

const SUFFIX3_BLOOM_WORDS: usize = 8192;
const SUFFIX3_BLOOM_BITS: u32 = (SUFFIX3_BLOOM_WORDS as u32) * 32;
const SUFFIX3_BLOOM_INDEX_MASK: u32 = SUFFIX3_BLOOM_BITS - 1;

fn suffix3_bloom_hash_witness(suffix: u32) -> u32 {
    let mixed = (suffix ^ (suffix >> 11)).wrapping_mul(0x9E37_79B1);
    mixed ^ (mixed >> 15)
}

fn classic_ac_suffix3_bloom_bit_index_witness(previous2: u8, previous: u8, current: u8) -> usize {
    let suffix = (u32::from(previous2) << 16) | (u32::from(previous) << 8) | u32::from(current);
    (suffix3_bloom_hash_witness(suffix) & SUFFIX3_BLOOM_INDEX_MASK) as usize
}

fn set_suffix3_bloom_bit_witness(mask: &mut [u32], previous2: u8, previous: u8, current: u8) {
    let bit_index = classic_ac_suffix3_bloom_bit_index_witness(previous2, previous, current);
    mask[bit_index / 32] |= 1_u32 << (bit_index % 32);
}

/// Derive the 8192-word hashed three-byte suffix mask for a pattern set.
#[must_use]
pub fn classic_ac_candidate_suffix3_bloom_words_witness(patterns: &[&[u8]]) -> Vec<u32> {
    classic_ac_candidate_suffix3_bloom_words_ci_witness(patterns, false)
}

/// ASCII-case-aware variant of [`classic_ac_candidate_suffix3_bloom_words_witness`].
#[must_use]
pub fn classic_ac_candidate_suffix3_bloom_words_ci_witness(
    patterns: &[&[u8]],
    case_insensitive: bool,
) -> Vec<u32> {
    let mut mask = vec![0_u32; SUFFIX3_BLOOM_WORDS];
    for pattern in patterns
        .iter()
        .copied()
        .filter(|pattern| !pattern.is_empty())
    {
        match pattern.len() {
            1 => {
                let (cv, cn) = ascii_case_variants_witness(pattern[0], case_insensitive);
                for previous2 in 0..=u8::MAX {
                    for previous in 0..=u8::MAX {
                        for &c in &cv[..cn] {
                            set_suffix3_bloom_bit_witness(&mut mask, previous2, previous, c);
                        }
                    }
                }
            }
            2 => {
                let (bv, bn) = ascii_case_variants_witness(pattern[0], case_insensitive);
                let (cv, cn) = ascii_case_variants_witness(pattern[1], case_insensitive);
                for previous2 in 0..=u8::MAX {
                    for &b in &bv[..bn] {
                        for &c in &cv[..cn] {
                            set_suffix3_bloom_bit_witness(&mut mask, previous2, b, c);
                        }
                    }
                }
            }
            len => {
                let (av, an) = ascii_case_variants_witness(pattern[len - 3], case_insensitive);
                let (bv, bn) = ascii_case_variants_witness(pattern[len - 2], case_insensitive);
                let (cv, cn) = ascii_case_variants_witness(pattern[len - 1], case_insensitive);
                for &a in &av[..an] {
                    for &b in &bv[..bn] {
                        for &c in &cv[..cn] {
                            set_suffix3_bloom_bit_witness(&mut mask, a, b, c);
                        }
                    }
                }
            }
        }
    }
    mask
}

/// Return whether the hashed suffix3 mask admits this candidate triple.
#[must_use]
pub fn classic_ac_suffix3_bloom_contains_witness(
    mask: &[u32],
    previous2: u8,
    previous: u8,
    current: u8,
) -> bool {
    let bit_index = classic_ac_suffix3_bloom_bit_index_witness(previous2, previous, current);
    let word = bit_index / 32;
    mask.get(word)
        .is_some_and(|word_value| (word_value & (1_u32 << (bit_index % 32))) != 0)
}
