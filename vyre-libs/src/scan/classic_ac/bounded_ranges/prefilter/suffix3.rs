//! The suffix3-gated bounded-ranges shapes: match triples, a global presence
//! bitmap, a per-region bitmap, and the fused presence-and-positions program.
//!
//! All four bind the IDENTICAL suffix3 gate and inputs and differ only in the
//! result buffer, the trailing sink and the replay, so each is one call to the
//! family's assembler with those three supplied. The gate width itself, the
//! shared input ABI and the fail-closed path have their owners elsewhere; this
//! file restates none of them.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Program};

use crate::matching::CompiledDfa;

use super::super::{
    ac_ranges_output_records_len, bounded_ranges_presence_and_positions_by_region_nodes,
    bounded_ranges_presence_by_region_nodes, bounded_ranges_presence_nodes, AcInputBindings,
};
use super::{
    build_ranges_scan, gated_ranges_program, try_build_ranges_scan, PrefilterGate, PrefilterWidth,
    FIRST_GATE_BINDING,
};

/// The binding the per-region attribution table starts at: immediately after the
/// suffix3 gate's three mask buffers.
const FIRST_REGION_BINDING: u32 = FIRST_GATE_BINDING + PrefilterWidth::Suffix3.mask_count();


/// Number of u32 words a presence bitmap needs for `pattern_count` patterns.
#[must_use]
pub fn presence_bitmap_words(pattern_count: u32) -> u32 {
    pattern_count.div_ceil(32).max(1)
}

/// Build a suffix3-prefiltered bounded-ranges AC PRESENCE program: same candidate
/// gating + DFA replay as the match-emitting scan, but each accepted pattern sets
/// one idempotent bit in a `presence_bitmap_words(pattern_count)`-word read-write
/// bitmap (binding 6, replacing the `match_count` + `matches` buffers) via
/// `atomic_or`. The inputs at bindings 0-5 and 7-9 (haystack, DFA tables, prefilter
/// masks) are byte-identical to the scan program, so a resident integration can
/// share the uploaded static tables. There is NO match-triple output and the entire
/// readback is the small bitmap (removing the dense-workload output bottleneck).
#[must_use]
#[allow(clippy::too_many_arguments)]
fn classic_ac_bounded_ranges_suffix3_presence_program(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    output_records: &str,
    pattern_lengths: &str,
    haystack_len: &str,
    presence: &str,
    candidate_end_mask: &str,
    candidate_suffix2_mask: &str,
    candidate_suffix3_bloom: &str,
    state_count: u32,
    output_records_len: u32,
    pattern_count: u32,
    max_pattern_len: u32,
) -> Program {
    gated_ranges_program(
        PrefilterGate::suffix3(
            candidate_end_mask,
            candidate_suffix2_mask,
            candidate_suffix3_bloom,
        ),
        AcInputBindings::new(
            [
                haystack,
                transitions,
                output_offsets,
                output_records,
                pattern_lengths,
                haystack_len,
            ],
            state_count,
            output_records_len,
            pattern_count,
        ),
        BufferDecl::read_write(presence, 6, DataType::U32)
            .with_count(presence_bitmap_words(pattern_count)),
        Vec::new(),
        "vyre-libs::matching::classic_ac_bounded_ranges_suffix3_presence",
        bounded_ranges_presence_nodes(
            haystack,
            transitions,
            output_offsets,
            output_records,
            presence,
            max_pattern_len,
        ),
    )
}

/// Build the suffix3-prefiltered PRESENCE program for a compiled DFA.
///
/// # Errors
/// Returns an actionable error when DFA output-record metadata exceeds the u32
/// GPU buffer-count ABI.
pub fn try_build_ac_bounded_ranges_suffix3_presence_program(
    dfa: &CompiledDfa,
    pattern_count: u32,
) -> Result<Program, String> {
    let output_records_len = ac_ranges_output_records_len(dfa, "bounded-ranges suffix3 presence")?;
    Ok(classic_ac_bounded_ranges_suffix3_presence_program(
        "haystack",
        "transitions",
        "output_offsets",
        "output_records",
        "pattern_lengths",
        "haystack_len",
        "presence",
        "candidate_end_mask",
        "candidate_suffix2_mask",
        "candidate_suffix3_bloom",
        dfa.state_count,
        output_records_len,
        pattern_count,
        dfa.max_pattern_len,
    ))
}

/// `ceil(log2(n))` for the binary-search iteration count, with a floor of 1 so a
/// 1- or 2-region program still runs one narrowing step.
#[must_use]
fn ceil_log2(n: u32) -> u32 {
    match n {
        0 | 1 => 1,
        _ => (32 - (n - 1).leading_zeros()).max(1),
    }
}

/// Number of u32 words a per-region presence bitmap needs for `region_count`
/// regions of `pattern_count` patterns each: `region_count × presence_words`.
#[must_use]
pub fn presence_by_region_words(pattern_count: u32, max_regions: u32) -> u32 {
    presence_bitmap_words(pattern_count).saturating_mul(max_regions.max(1))
}

/// The per-region attribution table, bound immediately after the suffix3 gate:
/// `region_starts` (the ascending file start offsets of the coalesced buffer,
/// with `region_starts[0] == 0`) then the shard's single-element `region_base`.
///
/// Both region-attributed builders bind exactly these, which is what keeps
/// bindings 0-11 byte-identical between the presence-only program and the fused
/// presence+positions program that appends its own sink after them.
fn region_table_decls(region_starts: &str, region_base: &str) -> Vec<BufferDecl> {
    vec![
        BufferDecl::storage(
            region_starts,
            FIRST_REGION_BINDING,
            BufferAccess::ReadOnly,
            DataType::U32,
        ),
        BufferDecl::storage(
            region_base,
            FIRST_REGION_BINDING + 1,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(1),
    ]
}

/// Region-attributed variant of [`classic_ac_bounded_ranges_suffix3_presence_program`]:
/// the presence bitmap (binding 6) is `max_regions × presence_bitmap_words(pattern_count)`
/// words, and a `region_starts` table (binding 10, the ascending file start
/// offsets of the coalesced buffer with `region_starts[0] == 0`) maps each hit to
/// its region row. Same candidate gating + DFA replay + idempotent `atomic_or` as
/// the global presence program, so it keeps the dense-input scan ceiling; the only
/// added per-hit work is a `ceil(log2(max_regions))`-iteration binary search. One
/// compiled program serves any batch with `region_count <= max_regions` (the live
/// count is read from `buf_len(region_starts)`).
#[must_use]
#[allow(clippy::too_many_arguments)]
fn classic_ac_bounded_ranges_suffix3_presence_by_region_program(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    output_records: &str,
    pattern_lengths: &str,
    haystack_len: &str,
    presence: &str,
    candidate_end_mask: &str,
    candidate_suffix2_mask: &str,
    candidate_suffix3_bloom: &str,
    region_starts: &str,
    region_base: &str,
    state_count: u32,
    output_records_len: u32,
    pattern_count: u32,
    max_pattern_len: u32,
    max_regions: u32,
) -> Program {
    gated_ranges_program(
        PrefilterGate::suffix3(
            candidate_end_mask,
            candidate_suffix2_mask,
            candidate_suffix3_bloom,
        ),
        AcInputBindings::new(
            [
                haystack,
                transitions,
                output_offsets,
                output_records,
                pattern_lengths,
                haystack_len,
            ],
            state_count,
            output_records_len,
            pattern_count,
        ),
        BufferDecl::read_write(presence, 6, DataType::U32)
            .with_count(presence_by_region_words(pattern_count, max_regions)),
        region_table_decls(region_starts, region_base),
        "vyre-libs::matching::classic_ac_bounded_ranges_suffix3_presence_by_region",
        bounded_ranges_presence_by_region_nodes(
            haystack,
            transitions,
            output_offsets,
            output_records,
            presence,
            region_starts,
            region_base,
            max_pattern_len,
            presence_bitmap_words(pattern_count),
            ceil_log2(max_regions),
        ),
    )
}

/// Build the region-attributed suffix3 PRESENCE program for a compiled DFA, sized
/// for up to `max_regions` coalesced files.
///
/// # Errors
/// Returns an actionable error when DFA output-record metadata exceeds the u32
/// GPU buffer-count ABI.
pub fn try_build_ac_bounded_ranges_suffix3_presence_by_region_program(
    dfa: &CompiledDfa,
    pattern_count: u32,
    max_regions: u32,
) -> Result<Program, String> {
    let output_records_len =
        ac_ranges_output_records_len(dfa, "bounded-ranges suffix3 region-presence")?;
    Ok(
        classic_ac_bounded_ranges_suffix3_presence_by_region_program(
            "haystack",
            "transitions",
            "output_offsets",
            "output_records",
            "pattern_lengths",
            "haystack_len",
            "presence",
            "candidate_end_mask",
            "candidate_suffix2_mask",
            "candidate_suffix3_bloom",
            "region_starts",
            "region_base",
            dfa.state_count,
            output_records_len,
            pattern_count,
            dfa.max_pattern_len,
            max_regions,
        ),
    )
}

/// FUSED region-presence + match-positions program: one suffix3-gated bounded-ranges
/// scan that writes BOTH the per-region presence bitmap (binding 6, `atomic_or`, like
/// [`classic_ac_bounded_ranges_suffix3_presence_by_region_program`]) AND the
/// `(pattern_id, start, end)` match triples (bindings 12 `match_count` + 13 `matches`,
/// [`build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce`]).
///
/// Bindings 0-11 are byte-identical to the presence-by-region program (so an
/// integration can share every uploaded static table and the `region_starts`/
/// `region_base` inputs); bindings 12-13 add the match counter + triple output. A
/// consumer that today dispatches the presence-by-region scan AND a separate
/// position scan over the same haystack collapses both into ONE dispatch with this
/// program, the expensive candidate gate + DFA replay runs once. The two outputs
/// are recall-identical to the separate scans by construction (same candidate set,
/// same `output_records` iteration).
#[must_use]
#[allow(clippy::too_many_arguments)]
fn classic_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    output_records: &str,
    pattern_lengths: &str,
    haystack_len: &str,
    presence: &str,
    candidate_end_mask: &str,
    candidate_suffix2_mask: &str,
    candidate_suffix3_bloom: &str,
    region_starts: &str,
    region_base: &str,
    match_count: &str,
    matches: &str,
    state_count: u32,
    output_records_len: u32,
    pattern_count: u32,
    max_pattern_len: u32,
    max_regions: u32,
    max_matches: u32,
) -> Program {
    classic_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program_filtered(
        haystack,
        transitions,
        output_offsets,
        output_records,
        pattern_lengths,
        haystack_len,
        presence,
        candidate_end_mask,
        candidate_suffix2_mask,
        candidate_suffix3_bloom,
        region_starts,
        region_base,
        match_count,
        matches,
        state_count,
        output_records_len,
        pattern_count,
        max_pattern_len,
        max_regions,
        max_matches,
        0,
    )
}

/// Fused region presence with positioned output restricted to pattern IDs at
/// or above `first_positioned_pattern_id`. Presence remains complete for every
/// pattern; only the atomic triple append is filtered.
#[must_use]
#[allow(clippy::too_many_arguments)]
fn classic_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program_filtered(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    output_records: &str,
    pattern_lengths: &str,
    haystack_len: &str,
    presence: &str,
    candidate_end_mask: &str,
    candidate_suffix2_mask: &str,
    candidate_suffix3_bloom: &str,
    region_starts: &str,
    region_base: &str,
    match_count: &str,
    matches: &str,
    state_count: u32,
    output_records_len: u32,
    pattern_count: u32,
    max_pattern_len: u32,
    max_regions: u32,
    max_matches: u32,
    first_positioned_pattern_id: u32,
) -> Program {
    let mut trailing = region_table_decls(region_starts, region_base);
    // Match counter + triple output: the position half of the fused output.
    // `append_match` bounds writes to `buf_len(matches) / 3 == max_matches`.
    trailing.push(
        BufferDecl::read_write(match_count, FIRST_REGION_BINDING + 2, DataType::U32).with_count(1),
    );
    trailing.push(
        BufferDecl::output(matches, FIRST_REGION_BINDING + 3, DataType::U32)
            .with_count(max_matches.saturating_mul(3)),
    );

    gated_ranges_program(
        PrefilterGate::suffix3(
            candidate_end_mask,
            candidate_suffix2_mask,
            candidate_suffix3_bloom,
        ),
        AcInputBindings::new(
            [
                haystack,
                transitions,
                output_offsets,
                output_records,
                pattern_lengths,
                haystack_len,
            ],
            state_count,
            output_records_len,
            pattern_count,
        ),
        BufferDecl::read_write(presence, 6, DataType::U32)
            .with_count(presence_by_region_words(pattern_count, max_regions)),
        trailing,
        "vyre-libs::matching::classic_ac_bounded_ranges_suffix3_presence_and_positions_by_region",
        bounded_ranges_presence_and_positions_by_region_nodes(
            haystack,
            transitions,
            output_offsets,
            output_records,
            pattern_lengths,
            presence,
            region_starts,
            region_base,
            match_count,
            matches,
            max_pattern_len,
            presence_bitmap_words(pattern_count),
            ceil_log2(max_regions),
            first_positioned_pattern_id,
        ),
    )
}

/// Build the fused region-presence + match-positions suffix3 program for a compiled
/// DFA, sized for up to `max_regions` coalesced files and `max_matches` triples.
///
/// # Errors
/// Returns an actionable error when DFA output-record metadata exceeds the u32 GPU
/// buffer-count ABI.
pub fn try_build_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program(
    dfa: &CompiledDfa,
    pattern_count: u32,
    max_regions: u32,
    max_matches: u32,
) -> Result<Program, String> {
    try_build_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program_filtered(
        dfa,
        pattern_count,
        max_regions,
        max_matches,
        0,
    )
}

/// Build a fused program whose presence bitmap covers all patterns while match
/// triples are emitted only for IDs at or above the supplied boundary.
///
/// # Errors
/// Returns an actionable error when the boundary exceeds the pattern count, or
/// when DFA output-record metadata exceeds the u32 GPU buffer-count ABI.
pub fn try_build_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program_filtered(
    dfa: &CompiledDfa,
    pattern_count: u32,
    max_regions: u32,
    max_matches: u32,
    first_positioned_pattern_id: u32,
) -> Result<Program, String> {
    if first_positioned_pattern_id > pattern_count {
        return Err(format!(
            "AC bounded-ranges fused positioned pattern boundary {first_positioned_pattern_id} exceeds pattern count {pattern_count}. Fix: pass a boundary in 0..={pattern_count}."
        ));
    }
    let output_records_len =
        ac_ranges_output_records_len(dfa, "bounded-ranges suffix3 region-presence+positions")?;
    Ok(
        classic_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program_filtered(
            "haystack",
            "transitions",
            "output_offsets",
            "output_records",
            "pattern_lengths",
            "haystack_len",
            "presence",
            "candidate_end_mask",
            "candidate_suffix2_mask",
            "candidate_suffix3_bloom",
            "region_starts",
            "region_base",
            "match_count",
            "matches",
            dfa.state_count,
            output_records_len,
            pattern_count,
            dfa.max_pattern_len,
            max_regions,
            max_matches,
            first_positioned_pattern_id,
        ),
    )
}

/// Build the suffix-prefiltered bounded-ranges AC scan for a compiled DFA.
#[must_use]
pub fn build_ac_bounded_ranges_suffix3_prefilter_program(
    dfa: &CompiledDfa,
    pattern_count: u32,
    max_matches: u32,
) -> Program {
    build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce(
        dfa,
        pattern_count,
        max_matches,
        true,
    )
}

/// Variant of [`build_ac_bounded_ranges_suffix3_prefilter_program`] that
/// exposes the match-append coalescing selector.
///
/// # Panics
/// Panics when the suffix3 prefilter exceeds the GPU ABI limits, through the
/// crate's shared fail-closed wrapper. Callers that must recover use
/// [`try_build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce`].
#[must_use]
pub fn build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce(
    dfa: &CompiledDfa,
    pattern_count: u32,
    max_matches: u32,
    use_subgroup_coalesce: bool,
) -> Program {
    build_ranges_scan(
        PrefilterWidth::Suffix3,
        dfa,
        pattern_count,
        max_matches,
        use_subgroup_coalesce,
    )
}

/// Fallible variant of [`build_ac_bounded_ranges_suffix3_prefilter_program`].
///
/// # Errors
///
/// Returns an actionable error when DFA metadata cannot fit the GPU program's
/// u32 buffer-count ABI.
pub fn try_build_ac_bounded_ranges_suffix3_prefilter_program(
    dfa: &CompiledDfa,
    pattern_count: u32,
    max_matches: u32,
) -> Result<Program, String> {
    try_build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce(
        dfa,
        pattern_count,
        max_matches,
        true,
    )
}

/// Fallible variant of [`build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce`].
///
/// # Errors
///
/// Returns an actionable error when DFA metadata cannot fit the GPU program's
/// u32 buffer-count ABI.
pub fn try_build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce(
    dfa: &CompiledDfa,
    pattern_count: u32,
    max_matches: u32,
    use_subgroup_coalesce: bool,
) -> Result<Program, String> {
    try_build_ranges_scan(
        PrefilterWidth::Suffix3,
        dfa,
        pattern_count,
        max_matches,
        use_subgroup_coalesce,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scan::classic_ac::test_dispatch_and_decode::{
        ac_ranges_inputs, assert_infallible_matches_try, decode_match_triples, pattern_lengths,
        u32_input,
    };
    use crate::scan::classic_ac::{
        classic_ac_bounded_ranges_scan, classic_ac_candidate_end_byte_mask_words,
        classic_ac_candidate_suffix2_mask_words, classic_ac_candidate_suffix3_bloom_words,
        classic_ac_compile, CLASSIC_AC_SUFFIX2_MASK_WORDS, CLASSIC_AC_SUFFIX3_BLOOM_WORDS,
    };

    /// Behavioral Law-10 regression guard: the infallible suffix3 prefilter builder
    /// must wire the REAL DFA (delegating to the `try_` Ok program), never a
    /// degenerate empty rejecting program (state_count=1, output_records_len=0) that
    /// silently drops every match.
    #[test]
    fn infallible_suffix3_prefilter_uses_real_dfa_not_empty_fallback() {
        let ac = classic_ac_compile(&[b"abc", b"de", b"abcd"]);
        let via_infallible =
            build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce(
                &ac.dfa, 3, 128, false,
            );
        let via_try = try_build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce(
            &ac.dfa, 3, 128, false,
        )
        .expect("valid DFA must build");
        assert_infallible_matches_try(
            "bounded-ranges suffix3 prefilter",
            &via_infallible,
            &via_try,
            &ac.dfa,
        );
    }

    #[test]
    fn bounded_ranges_suffix3_prefilter_reference_eval_matches_cpu_oracle() {
        let patterns: [&[u8]; 6] = [b"a", b"bc", b"ab", b"abcd", b"BEGIN", b"token"];
        let haystack = b"zabcd a bc BEGIN token abcdbc";
        let ac = classic_ac_compile(&patterns);
        let lengths = pattern_lengths(&patterns);
        let mut expected = classic_ac_bounded_ranges_scan(&ac, &lengths, haystack);
        expected.sort_unstable();
        let program = build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce(
            &ac.dfa,
            patterns.len() as u32,
            128,
            false,
        );
        let mut inputs = ac_ranges_inputs(&ac.dfa, haystack, &lengths);
        inputs.push(u32_input(&[0]));
        inputs.push(u32_input(&classic_ac_candidate_end_byte_mask_words(
            &ac.dfa,
        )));
        inputs.push(u32_input(&classic_ac_candidate_suffix2_mask_words(&ac.dfa)));
        inputs.push(u32_input(&classic_ac_candidate_suffix3_bloom_words(
            &patterns,
        )));
        let outputs = vyre_reference::reference_eval(&program, &inputs).expect(
            "Fix: suffix3 prefiltered AC bounded-ranges program should evaluate in reference backend.",
        );
        let mut actual = decode_match_triples(&outputs);
        actual.sort_unstable();

        assert_eq!(actual, expected);
    }

    #[test]
    fn bounded_ranges_suffix3_prefilter_program_has_compact_stable_shape() {
        let ac = classic_ac_compile(&[b"Authorization: Bearer ", b"token", b"tok"]);
        let program = build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce(
            &ac.dfa, 3, 1024, false,
        );

        assert_eq!(program.workgroup_size(), [128, 1, 1]);
        assert_eq!(program.buffers().len(), 11);
        assert_eq!(program.buffers()[6].name(), "match_count");
        assert_eq!(program.buffers()[6].count, 1);
        assert_eq!(program.buffers()[7].name(), "candidate_end_mask");
        assert_eq!(program.buffers()[7].count, 8);
        assert_eq!(program.buffers()[8].name(), "candidate_suffix2_mask");
        assert_eq!(
            program.buffers()[8].count,
            CLASSIC_AC_SUFFIX2_MASK_WORDS as u32
        );
        assert_eq!(program.buffers()[9].name(), "candidate_suffix3_bloom");
        assert_eq!(
            program.buffers()[9].count,
            CLASSIC_AC_SUFFIX3_BLOOM_WORDS as u32
        );
        assert_eq!(program.buffers()[10].name(), "matches");
        assert_eq!(program.buffers()[10].count, 1024 * 3);
    }

    #[test]
    fn region_presence_program_has_region_attributed_shape() {
        let ac = classic_ac_compile(&[b"token", b"tok", b"secret"]);
        let pattern_count = 3u32;
        let max_regions = 8u32;
        let program = try_build_ac_bounded_ranges_suffix3_presence_by_region_program(
            &ac.dfa,
            pattern_count,
            max_regions,
        )
        .expect("Fix: region-presence program must build for a small DFA");

        // Bindings 0-9 match the global presence program; the per-region variant
        // adds `region_starts` at binding 10 and a `region_base` shard offset at
        // binding 11, and widens `presence` to a per-region bitmap (row stride ×
        // max_regions) instead of a single global row.
        assert_eq!(program.workgroup_size(), [128, 1, 1]);
        assert_eq!(program.buffers().len(), 12);
        assert_eq!(program.buffers()[6].name(), "presence");
        let words = presence_bitmap_words(pattern_count);
        assert_eq!(program.buffers()[6].count, words * max_regions);
        assert_eq!(
            program.buffers()[6].count,
            presence_by_region_words(pattern_count, max_regions)
        );
        assert_eq!(program.buffers()[10].name(), "region_starts");
        assert_eq!(program.buffers()[11].name(), "region_base");
        assert_eq!(program.buffers()[11].count, 1);
    }

    #[test]
    fn ceil_log2_bounds_binary_search_iterations() {
        // ceil(log2(n)) with a floor of 1: the number of narrowing steps a
        // binary search over n regions needs (n region rows → 1 index).
        assert_eq!(ceil_log2(0), 1);
        assert_eq!(ceil_log2(1), 1);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(8), 3);
        assert_eq!(ceil_log2(9), 4);
        assert_eq!(ceil_log2(16), 4);
        assert_eq!(ceil_log2(65536), 16);
    }
}
