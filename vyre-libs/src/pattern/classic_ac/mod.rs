//! Classic Aho-Corasick scanner with flat `output_links`.
//!
//! Unlike the single-accept DFA in `super::dfa::dfa_compile`, this
//! module precomputes **all** pattern ids that match at each state
//! (including patterns reachable via failure links) into a flat
//! `output_offsets` + `output_records` array.  At scan time the
//! match loop is **O(matches)**, not **O(states × n)**.
//!
//! Build-time complexity is still dominated by the dense transition
//! table (O(states × alphabet)), but the per-state pattern list is
//! built in one BFS pass using dynamic programming on the failure
//! links.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::pattern::{dfa_compile, CompiledDfa};

/// THE Aho-Corasick walk. Scan-level builders reach it directly rather than
/// through a second re-export path in this module.
pub(in crate::pattern) mod bounded_ranges;
mod count_program;

#[cfg(test)]
pub(crate) mod test_dispatch_and_decode;

/// The `build_*`/`try_build_*` form is the published entry for each program.
/// The buffer-name form each one wraps stays inside `crate::pattern`: the names it
/// takes are the ABI every gate, prefilter row and parity test pins, so a
/// caller passing its own names would build a program no gate here judges, and
/// publishing both put two paths to one program in the surface.
pub use bounded_ranges::{
    build_ac_bounded_ranges_prefilter_program,
    build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce,
    build_ac_bounded_ranges_program, build_ac_bounded_ranges_program_with_subgroup_coalesce,
    build_ac_bounded_ranges_suffix3_prefilter_program,
    build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce,
    presence_bitmap_words, presence_by_region_words, try_build_ac_bounded_ranges_prefilter_program,
    try_build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce,
    try_build_ac_bounded_ranges_program,
    try_build_ac_bounded_ranges_program_with_subgroup_coalesce,
    try_build_ac_bounded_ranges_suffix3_prefilter_program,
    try_build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce,
    try_build_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program,
    try_build_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program_filtered,
    try_build_ac_bounded_ranges_suffix3_presence_by_region_program,
    try_build_ac_bounded_ranges_suffix3_presence_program,
};

#[cfg(all(feature = "pattern-regex", feature = "pattern-dfa"))]
pub(in crate::pattern) use bounded_ranges::regex_exact_ranges_program;
pub use count_program::{
    build_ac_bounded_count_prefilter_program, build_ac_bounded_count_program,
    build_ac_bounded_count_suffix2_prefilter_program,
    build_ac_bounded_count_suffix3_prefilter_program, CLASSIC_AC_SUFFIX2_MASK_WORDS,
    CLASSIC_AC_SUFFIX3_BLOOM_WORDS,
};

#[cfg(test)]
pub(crate) use count_program::{
    classic_ac_candidate_end_byte_mask_words, classic_ac_candidate_suffix2_mask_words,
    classic_ac_candidate_suffix3_bloom_words,
};

/// THE Aho-Corasick walk lives in [`bounded_ranges`]. Only the state-advance
/// step is re-exported, for `dfa::aho_corasick`; every other scan-level
/// builder reaches `bounded_ranges` directly rather than through a second
/// path.
pub(in crate::pattern) use bounded_ranges::ac_advance_state_node;
use bounded_ranges::ac_output_span_nodes;

/// A classic AC automaton with precomputed flat output links.
///
/// Wraps a [`CompiledDfa`] and exposes the Reference oracle scan plus
/// a vyre IR Program for GPU dispatch.
#[derive(Debug, Clone)]
pub struct ClassicAcAutomaton {
    /// Underlying compiled DFA (transitions + flat output links).
    pub dfa: CompiledDfa,
}

/// Build a [`ClassicAcAutomaton`] from a list of byte patterns.
///
/// # Panics
///
/// Panics with an actionable message on DFA budget exhaustion.
/// Use `super::dfa::dfa_compile_with_budget` for structured
/// error handling.
#[must_use]
pub fn classic_ac_compile(patterns: &[&[u8]]) -> ClassicAcAutomaton {
    let dfa = dfa_compile(patterns);
    ClassicAcAutomaton { dfa }
}

#[cfg(test)]
pub(crate) fn classic_ac_scan(automaton: &ClassicAcAutomaton, haystack: &[u8]) -> Vec<(u32, u32)> {
    vyre_reference::composition_witness::classic_ac_scan_witness(
        &automaton.dfa.transitions,
        &automaton.dfa.output_offsets,
        &automaton.dfa.output_records,
        haystack,
    )
}

#[cfg(test)]
pub(crate) fn classic_ac_scan_counts(automaton: &ClassicAcAutomaton, haystack: &[u8]) -> Vec<u32> {
    vyre_reference::composition_witness::classic_ac_scan_counts_witness(
        &automaton.dfa.transitions,
        &automaton.dfa.output_offsets,
        haystack,
    )
}

#[cfg(test)]
pub(crate) fn classic_ac_bounded_ranges_scan(
    automaton: &ClassicAcAutomaton,
    pattern_lengths: &[u32],
    haystack: &[u8],
) -> Vec<(u32, u32, u32)> {
    vyre_reference::composition_witness::classic_ac_bounded_ranges_scan_witness(
        &automaton.dfa.transitions,
        &automaton.dfa.output_offsets,
        &automaton.dfa.output_records,
        pattern_lengths,
        haystack,
    )
}

/// Build a vyre `Program` that scans `haystack` and appends every
/// matching pattern id to `matches` via an atomic slot counter.
///
/// Buffers (bindings 0..5):
///
/// | binding | name | access | count |
/// |---|---|---|---|
/// | 0 | `haystack` | ReadOnly | `haystack_len` |
/// | 1 | `transitions` | ReadOnly | `state_count * 256` |
/// | 2 | `output_offsets` | ReadOnly | `state_count + 1` |
/// | 3 | `output_records` | ReadOnly | `output_records_len` |
/// | 4 | `match_count` | ReadWrite | 1 (atomic) |
/// | 5 | `matches` | ReadWrite | `max_matches` |
///
/// Each invocation `i` replays the DFA from state 0 through
/// `haystack[0..=i]`, then atomically claims slots in `matches` and
/// writes every pattern id for the final state.
///
/// The serial replay is still O(n²) total work  -  the fix here is
/// the **match-emission loop**, which is O(matches) thanks to the
/// flat `output_links`.
#[must_use]
pub fn classic_ac_program(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    output_records: &str,
    match_count: &str,
    matches: &str,
    haystack_len: u32,
    state_count: u32,
    output_records_len: u32,
    max_matches: u32,
) -> Program {
    let i = Expr::var("i");

    // body executed per invocation i:
    //   walk DFA from 0 through haystack[0..=i]
    //   for each pattern in output_links[state]:
    //       slot = atomic_add(match_count, 0, 1)
    //       if slot < max_matches:
    //           matches[slot] = pattern_id
    //
    // The transition step and the output-link span come from the shared AC walk
    // owner. This builder differs from the bounded family in one respect: it
    // reads the haystack UNPACKED, one byte per u32 element, so it hands the
    // owner a direct element load instead of an unpacked-word byte.
    let mut per_position = vec![
        Node::let_bind("state", Expr::u32(0)),
        Node::loop_for(
            "step",
            Expr::u32(0),
            Expr::add(Expr::var("i"), Expr::u32(1)),
            vec![ac_advance_state_node(
                transitions,
                Expr::load(haystack, Expr::var("step")),
            )],
        ),
    ];
    per_position.extend(ac_output_span_nodes(output_offsets));
    per_position.push(bounded_ranges::output_record_loop_node(
        output_records,
        vec![
            Node::let_bind(
                "slot",
                Expr::atomic_add(match_count, Expr::u32(0), Expr::u32(1)),
            ),
            Node::if_then(
                Expr::lt(Expr::var("slot"), Expr::u32(max_matches)),
                vec![Node::Store {
                    buffer: matches.into(),
                    index: Expr::var("slot"),
                    value: Expr::var("pattern_id"),
                }],
            ),
        ],
    ));

    let walk_body = vec![
        Node::let_bind("i", Expr::InvocationId { axis: 0 }),
        Node::if_then(Expr::lt(i.clone(), Expr::buf_len(haystack)), per_position),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(haystack, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(haystack_len),
            BufferDecl::storage(transitions, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(state_count.saturating_mul(256)),
            BufferDecl::storage(output_offsets, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(state_count.saturating_add(1)),
            BufferDecl::storage(output_records, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(output_records_len),
            BufferDecl::storage(match_count, 4, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
            BufferDecl::storage(matches, 5, BufferAccess::ReadWrite, DataType::U32)
                .with_count(max_matches),
        ],
        [64, 1, 1],
        vec![wrap_anonymous_region(
            "vyre-libs::matching::classic_ac",
            walk_body,
        )],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::bytes_to_u32 as decode_u32_words;
    use crate::fixture_bytes::eval_bytes;

    #[test]
    fn single_pattern_matches() {
        let ac = classic_ac_compile(&[b"abc"]);
        let matches = classic_ac_scan(&ac, b"xxabcxx");
        assert_eq!(matches, vec![(0, 4)]);
    }

    #[test]
    fn overlapping_patterns_report_all() {
        // Patterns "he", "she", "his", "hers" on "ushers".
        let ac = classic_ac_compile(&[b"he", b"she", b"his", b"hers"]);
        let matches = classic_ac_scan(&ac, b"ushers");
        // "she" ends at position 3, "he" ends at position 3,
        // "hers" ends at position 5.
        assert!(matches.contains(&(1, 3)), "must match she");
        assert!(matches.contains(&(0, 3)), "must match he");
        assert!(matches.contains(&(3, 5)), "must match hers");
    }

    #[test]
    fn nested_suffix_patterns_all_reported() {
        // a, aa, aaa, aaaa on "aaaa".
        let ac = classic_ac_compile(&[b"a", b"aa", b"aaa", b"aaaa"]);
        let matches = classic_ac_scan(&ac, b"aaaa");
        // Position 0: "a" (1 char)
        // Position 1: "a", "aa"
        // Position 2: "a", "aa", "aaa"
        // Position 3: "a", "aa", "aaa", "aaaa"
        let expected = vec![
            (0, 0),
            (0, 1),
            (1, 1),
            (0, 2),
            (1, 2),
            (2, 2),
            (0, 3),
            (1, 3),
            (2, 3),
            (3, 3),
        ];
        assert_eq!(matches, expected);
    }

    #[test]
    fn adversarial_failure_chain_is_linear_in_matches() {
        // Patterns a, aa, aaa, ... up to length 128.
        // This creates a long failure-link chain.
        let patterns: Vec<Vec<u8>> = (1..=128).map(|i| vec![b'a'; i]).collect();
        let pattern_refs: Vec<&[u8]> = patterns.iter().map(|p| p.as_slice()).collect();
        let ac = classic_ac_compile(&pattern_refs);
        let haystack = vec![b'a'; 128];
        let matches = classic_ac_scan(&ac, &haystack);

        // At position i (0-based) there are i+1 matches.
        // Total matches = 128 * 129 / 2 = 8256.
        assert_eq!(matches.len(), 8256);

        // Verify the last position has all 128 patterns.
        let last_pos_matches: Vec<u32> = matches
            .iter()
            .filter(|&&(_, pos)| pos == 127)
            .map(|&(pid, _)| pid)
            .collect();
        assert_eq!(last_pos_matches.len(), 128);
        for (expected_pid, &actual_pid) in last_pos_matches.iter().enumerate() {
            assert_eq!(actual_pid, expected_pid as u32);
        }
    }

    #[test]
    fn empty_haystack_yields_no_matches() {
        let ac = classic_ac_compile(&[b"abc"]);
        assert!(classic_ac_scan(&ac, b"").is_empty());
    }

    #[test]
    fn empty_pattern_list_yields_no_matches() {
        let ac = classic_ac_compile(&[]);
        assert!(classic_ac_scan(&ac, b"anything").is_empty());
    }

    #[test]
    fn gpu_emit_matches_reference_oracle() {
        let patterns: [&[u8]; 4] = [b"he", b"she", b"his", b"hers"];
        let ac = classic_ac_compile(&patterns);
        let haystack = b"ushers";
        let reference_matches = classic_ac_scan(&ac, haystack);
        // Build the vyre IR program.
        let program = classic_ac_program(
            "haystack",
            "transitions",
            "output_offsets",
            "output_records",
            "match_count",
            "matches",
            haystack.len() as u32,
            ac.dfa.state_count,
            ac.dfa.output_records.len() as u32,
            1024,
        );

        let inputs = vec![
            crate::fixture_bytes::u32_bytes(
                &haystack.iter().map(|&b| u32::from(b)).collect::<Vec<_>>(),
            ),
            crate::fixture_bytes::u32_bytes(&ac.dfa.transitions),
            crate::fixture_bytes::u32_bytes(&ac.dfa.output_offsets),
            crate::fixture_bytes::u32_bytes(&ac.dfa.output_records),
            crate::fixture_bytes::u32_bytes(&[0u32]),
            vec![0u8; 1024 * 4],
        ];

        let outputs = eval_bytes("classic_ac", &program, inputs.clone());

        let match_count = decode_u32_words(&outputs[0])[0];
        let gpu_matches_raw = decode_u32_words(&outputs[1]);
        let gpu_matches: Vec<u32> = gpu_matches_raw[..match_count as usize].to_vec();

        // Reference scan gives (pattern_id, end_pos); GPU gives just pattern_id
        // because each invocation writes its own patterns.  The order
        // is nondeterministic (atomic scheduling), so sort both.
        let mut reference_ids: Vec<u32> = reference_matches.iter().map(|&(pid, _)| pid).collect();
        reference_ids.sort_unstable();

        let mut gpu_ids = gpu_matches;
        gpu_ids.sort_unstable();

        assert_eq!(
            reference_ids, gpu_ids,
            "GPU emit must agree with Reference oracle on pattern ids"
        );
    }

    #[test]
    fn gpu_emit_does_not_overflow_when_max_matches_is_small() {
        let ac = classic_ac_compile(&[b"a", b"aa", b"aaa"]);
        let haystack = b"aaa";
        let program = classic_ac_program(
            "haystack",
            "transitions",
            "output_offsets",
            "output_records",
            "match_count",
            "matches",
            haystack.len() as u32,
            ac.dfa.state_count,
            ac.dfa.output_records.len() as u32,
            2, // only allow 2 stored matches
        );

        let inputs = vec![
            crate::fixture_bytes::u32_bytes(
                &haystack.iter().map(|&b| u32::from(b)).collect::<Vec<_>>(),
            ),
            crate::fixture_bytes::u32_bytes(&ac.dfa.transitions),
            crate::fixture_bytes::u32_bytes(&ac.dfa.output_offsets),
            crate::fixture_bytes::u32_bytes(&ac.dfa.output_records),
            crate::fixture_bytes::u32_bytes(&[0u32]),
            vec![0u8; 2 * 4],
        ];

        let outputs = eval_bytes("classic_ac", &program, inputs);

        let match_count = decode_u32_words(&outputs[0])[0];
        // Total matches = 1 + 2 + 3 = 6, but only 2 slots.
        assert_eq!(match_count, 6, "match_count must reflect total discoveries");
        let stored = decode_u32_words(&outputs[1]);
        assert_eq!(stored.len(), 2, "only 2 slots allocated");
    }

    #[test]
    fn case_insensitive_classic_ac_automaton_matches() {
        use crate::pattern::dfa_compile_case_insensitive;

        let dfa = dfa_compile_case_insensitive(&[b"he", b"she"]);
        let ac = ClassicAcAutomaton { dfa };
        let matches = classic_ac_scan(&ac, b"UsHeRs");
        assert!(matches.contains(&(1, 3)), "must match sHe");
        assert!(matches.contains(&(0, 3)), "must match He");
    }
}
