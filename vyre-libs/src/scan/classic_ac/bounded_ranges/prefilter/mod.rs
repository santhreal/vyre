use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Program};

use crate::region::wrap_anonymous;
use vyre_primitives::matching::CompiledDfa;

use super::{
    ac_ranges_output_records_len, ac_ranges_program_or_fail_closed, bounded_ranges_scan_nodes,
    candidate_end_gate_nodes, AcInputBindings,
};

mod suffix3;

pub use suffix3::{
    build_ac_bounded_ranges_suffix3_prefilter_program,
    build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce,
    classic_ac_bounded_ranges_suffix3_prefilter_program,
    classic_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce,
    classic_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program,
    classic_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program_filtered,
    classic_ac_bounded_ranges_suffix3_presence_by_region_program,
    classic_ac_bounded_ranges_suffix3_presence_program, presence_bitmap_words,
    presence_by_region_words, try_build_ac_bounded_ranges_suffix3_prefilter_program,
    try_build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce,
    try_build_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program,
    try_build_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program_filtered,
    try_build_ac_bounded_ranges_suffix3_presence_by_region_program,
    try_build_ac_bounded_ranges_suffix3_presence_program,
};

/// Build a bounded-window AC ranges program with an exact candidate-end-byte
/// prefilter.
///
/// `candidate_end_mask` is an 8-word bitset where bit `b` is set when byte `b`
/// can terminate at least one accepted DFA state. Non-candidate lanes skip the
/// bounded replay window and match append path entirely.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn classic_ac_bounded_ranges_prefilter_program(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    output_records: &str,
    pattern_lengths: &str,
    haystack_len: &str,
    match_count: &str,
    candidate_end_mask: &str,
    matches: &str,
    state_count: u32,
    output_records_len: u32,
    pattern_count: u32,
    max_matches: u32,
    max_pattern_len: u32,
) -> Program {
    classic_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce(
        haystack,
        transitions,
        output_offsets,
        output_records,
        pattern_lengths,
        haystack_len,
        match_count,
        candidate_end_mask,
        matches,
        state_count,
        output_records_len,
        pattern_count,
        max_matches,
        max_pattern_len,
        true,
    )
}

/// Variant of [`classic_ac_bounded_ranges_prefilter_program`] with explicit
/// control over subgroup match-append coalescing.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn classic_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    output_records: &str,
    pattern_lengths: &str,
    haystack_len: &str,
    match_count: &str,
    candidate_end_mask: &str,
    matches: &str,
    state_count: u32,
    output_records_len: u32,
    pattern_count: u32,
    max_matches: u32,
    max_pattern_len: u32,
    use_subgroup_coalesce: bool,
) -> Program {
    let body = candidate_end_gate_nodes(
        haystack,
        haystack_len,
        candidate_end_mask,
        bounded_ranges_scan_nodes(
            haystack,
            transitions,
            output_offsets,
            output_records,
            pattern_lengths,
            match_count,
            matches,
            max_pattern_len,
            use_subgroup_coalesce,
        ),
    );

    let mut buffers = AcInputBindings {
        haystack,
        transitions,
        output_offsets,
        output_records,
        pattern_lengths,
        haystack_len,
        state_count,
        output_records_len,
        pattern_count,
    }
    .decls_with_match_count(match_count);
    buffers.extend([
        BufferDecl::storage(candidate_end_mask, 7, BufferAccess::ReadOnly, DataType::U32)
            .with_count(8),
        BufferDecl::output(matches, 8, DataType::U32).with_count(max_matches.saturating_mul(3)),
    ]);

    Program::wrapped(
        buffers,
        [128, 1, 1],
        vec![wrap_anonymous(
            "vyre-libs::matching::classic_ac_bounded_ranges_prefilter",
            body,
        )],
    )
}

/// Build the candidate-end prefiltered bounded-ranges AC scan for a compiled
/// DFA.
#[must_use]
pub fn build_ac_bounded_ranges_prefilter_program(
    dfa: &CompiledDfa,
    pattern_count: u32,
    max_matches: u32,
) -> Program {
    build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce(
        dfa,
        pattern_count,
        max_matches,
        true,
    )
}

/// Variant of [`build_ac_bounded_ranges_prefilter_program`] that exposes the
/// match-append coalescing selector.
///
/// # Panics
/// Panics when the prefilter program exceeds the GPU ABI limits, through
/// [`ac_ranges_program_or_fail_closed`]. Callers that must recover use
/// [`try_build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce`].
#[must_use]
pub fn build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce(
    dfa: &CompiledDfa,
    pattern_count: u32,
    max_matches: u32,
    use_subgroup_coalesce: bool,
) -> Program {
    ac_ranges_program_or_fail_closed(
        try_build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce(
            dfa,
            pattern_count,
            max_matches,
            use_subgroup_coalesce,
        ),
        "bounded-ranges prefilter",
        "try_build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce",
    )
}

/// Fallible variant of [`build_ac_bounded_ranges_prefilter_program`].
///
/// # Errors
///
/// Returns an actionable error when DFA metadata cannot fit the GPU program's
/// u32 buffer-count ABI.
pub fn try_build_ac_bounded_ranges_prefilter_program(
    dfa: &CompiledDfa,
    pattern_count: u32,
    max_matches: u32,
) -> Result<Program, String> {
    try_build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce(
        dfa,
        pattern_count,
        max_matches,
        true,
    )
}

/// Fallible variant of [`build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce`].
///
/// # Errors
///
/// Returns an actionable error when DFA metadata cannot fit the GPU program's
/// u32 buffer-count ABI.
pub fn try_build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce(
    dfa: &CompiledDfa,
    pattern_count: u32,
    max_matches: u32,
    use_subgroup_coalesce: bool,
) -> Result<Program, String> {
    let output_records_len = ac_ranges_output_records_len(dfa, "bounded-ranges prefilter")?;
    Ok(
        classic_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce(
            "haystack",
            "transitions",
            "output_offsets",
            "output_records",
            "pattern_lengths",
            "haystack_len",
            "match_count",
            "candidate_end_mask",
            "matches",
            dfa.state_count,
            output_records_len,
            pattern_count,
            max_matches,
            dfa.max_pattern_len,
            use_subgroup_coalesce,
        ),
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
        classic_ac_compile,
    };

    #[test]
    fn bounded_ranges_prefilter_reference_eval_matches_cpu_oracle() {
        let patterns: [&[u8]; 5] = [b"a", b"bc", b"abcd", b"BEGIN", b"token"];
        let haystack = b"zabcd BEGIN token abcdbc";
        let ac = classic_ac_compile(&patterns);
        let lengths = pattern_lengths(&patterns);
        let mut expected = classic_ac_bounded_ranges_scan(&ac, &lengths, haystack);
        expected.sort_unstable();
        let program = build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce(
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
        let outputs = vyre_reference::reference_eval(&program, &inputs).expect(
            "Fix: prefiltered AC bounded-ranges program should evaluate in reference backend.",
        );
        let mut actual = decode_match_triples(&outputs);
        actual.sort_unstable();

        assert_eq!(actual, expected);
    }

    /// Behavioral regression guard: the infallible prefilter builder must wire the
    /// REAL DFA (delegating to the `try_` Ok program), never a degenerate empty-mask
    /// program (state_count=1, output_records_len=0) that suppresses every candidate.
    #[test]
    fn infallible_prefilter_uses_real_dfa_not_empty_fallback() {
        let ac = classic_ac_compile(&[b"abc", b"de", b"abcd"]);
        let via_infallible = build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce(
            &ac.dfa, 3, 128, false,
        );
        let via_try = try_build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce(
            &ac.dfa, 3, 128, false,
        )
        .expect("valid DFA must build");
        assert_infallible_matches_try(
            "bounded-ranges prefilter",
            &via_infallible,
            &via_try,
            &ac.dfa,
        );
    }

    #[test]
    fn bounded_ranges_prefilter_program_has_compact_stable_shape() {
        let ac = classic_ac_compile(&[b"Authorization: Bearer ", b"token", b"tok"]);
        let program = build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce(
            &ac.dfa, 3, 1024, false,
        );

        assert_eq!(program.workgroup_size(), [128, 1, 1]);
        assert_eq!(program.buffers().len(), 9);
        assert_eq!(program.buffers()[6].name(), "match_count");
        assert_eq!(program.buffers()[6].count, 1);
        assert_eq!(program.buffers()[7].name(), "candidate_end_mask");
        assert_eq!(program.buffers()[7].count, 8);
        assert_eq!(program.buffers()[8].name(), "matches");
        assert_eq!(program.buffers()[8].count, 1024 * 3);
    }
}
