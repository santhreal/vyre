use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Program};

use crate::pattern::CompiledDfa;

use super::count_scan_nodes;

/// Number of u32 words in the hashed three-byte suffix mask.
pub const CLASSIC_AC_SUFFIX3_BLOOM_WORDS: usize = 8192;

const CLASSIC_AC_SUFFIX3_BLOOM_BITS: u32 = (CLASSIC_AC_SUFFIX3_BLOOM_WORDS as u32) * 32;
const CLASSIC_AC_SUFFIX3_BLOOM_INDEX_MASK: u32 = CLASSIC_AC_SUFFIX3_BLOOM_BITS - 1;

/// Build a bounded-window AC count program with byte, suffix2, and suffix3 filters.
///
/// The suffix3 mask is a compact hashed set keyed by
/// `(byte[i-2] << 16) | (byte[i-1] << 8) | byte[i]`. It is checked only after
/// the exact end-byte and suffix2 masks, so false positives still take the safe
/// bounded DFA replay path while true matches cannot be filtered out.
#[must_use]
#[allow(clippy::too_many_arguments)]
fn classic_ac_bounded_count_suffix3_prefilter_program(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    candidate_end_mask: &str,
    candidate_suffix2_mask: &str,
    candidate_suffix3_bloom: &str,
    haystack_len: &str,
    match_count: &str,
    state_count: u32,
    max_pattern_len: u32,
) -> Program {
    let scan_nodes = count_scan_nodes(
        haystack,
        transitions,
        output_offsets,
        match_count,
        max_pattern_len,
    );
    let suffix3_match_nodes =
        super::suffix3_prefilter_match_nodes(haystack, candidate_suffix3_bloom, scan_nodes.clone());
    let body = super::count_suffix2_prefilter_body(
        haystack,
        candidate_end_mask,
        candidate_suffix2_mask,
        haystack_len,
        scan_nodes,
        suffix3_match_nodes,
    );
    let mut buffers = super::count_suffix2_prefilter_buffers(
        haystack,
        transitions,
        output_offsets,
        candidate_end_mask,
        candidate_suffix2_mask,
        state_count,
    );
    buffers.push(
        BufferDecl::storage(
            candidate_suffix3_bloom,
            5,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(CLASSIC_AC_SUFFIX3_BLOOM_WORDS as u32),
    );
    buffers.push(
        BufferDecl::storage(haystack_len, 6, BufferAccess::ReadOnly, DataType::U32).with_count(1),
    );
    buffers.push(BufferDecl::read_write(match_count, 7, DataType::U32).with_count(1));
    Program::wrapped(
        buffers,
        [128, 1, 1],
        vec![wrap_anonymous_region(
            "vyre-libs::matching::classic_ac_bounded_count_suffix3_prefilter",
            body,
        )],
    )
}

#[cfg(test)]
pub(crate) fn classic_ac_candidate_suffix3_bloom_words(patterns: &[&[u8]]) -> Vec<u32> {
    vyre_reference::composition_witness::classic_ac_candidate_suffix3_bloom_words_witness(patterns)
}

#[cfg(test)]
pub(crate) fn ascii_case_variants(byte: u8, case_insensitive: bool) -> ([u8; 2], usize) {
    vyre_reference::composition_witness::ascii_case_variants_witness(byte, case_insensitive)
}

#[cfg(test)]
pub(crate) fn classic_ac_candidate_suffix3_bloom_words_ci(
    patterns: &[&[u8]],
    case_insensitive: bool,
) -> Vec<u32> {
    vyre_reference::composition_witness::classic_ac_candidate_suffix3_bloom_words_ci_witness(
        patterns,
        case_insensitive,
    )
}

#[cfg(test)]
pub(crate) fn classic_ac_suffix3_bloom_contains(
    mask: &[u32],
    previous2: u8,
    previous: u8,
    current: u8,
) -> bool {
    vyre_reference::composition_witness::classic_ac_suffix3_bloom_contains_witness(
        mask, previous2, previous, current,
    )
}
/// Build the three-byte-suffix prefiltered AC count-only program for a compiled DFA.
#[must_use]
pub fn build_ac_bounded_count_suffix3_prefilter_program(dfa: &CompiledDfa) -> Program {
    classic_ac_bounded_count_suffix3_prefilter_program(
        "haystack",
        "transitions",
        "output_offsets",
        "candidate_end_mask",
        "candidate_suffix2_mask",
        "candidate_suffix3_bloom",
        "haystack_len",
        "match_count",
        dfa.state_count,
        dfa.max_pattern_len,
    )
}

pub(in crate::pattern::classic_ac) fn suffix3_bloom_bit_index_expr(suffix: Expr) -> Expr {
    let mixed = Expr::mul(
        Expr::bitxor(suffix.clone(), Expr::shr(suffix, Expr::u32(11))),
        Expr::u32(0x9E37_79B1),
    );
    Expr::bitand(
        Expr::bitxor(mixed.clone(), Expr::shr(mixed, Expr::u32(15))),
        Expr::u32(CLASSIC_AC_SUFFIX3_BLOOM_INDEX_MASK),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::bytes_to_u32 as decode_u32;
    use crate::pattern::classic_ac::test_dispatch_and_decode::{
        ac_dfa_table_inputs, u32_input, with_reference_dispatch_lanes,
    };
    use crate::pattern::classic_ac::{
        classic_ac_candidate_end_byte_mask_words, classic_ac_candidate_suffix2_mask_words,
        classic_ac_compile, classic_ac_scan_counts, CLASSIC_AC_SUFFIX2_MASK_WORDS,
    };

    #[test]
    fn suffix3_bloom_marks_inserted_short_and_long_pattern_suffixes() {
        let patterns: [&[u8]; 4] = [b"z", b"ab", b"token", b"BEGIN"];
        let mask = classic_ac_candidate_suffix3_bloom_words(&patterns);

        assert_eq!(mask.len(), CLASSIC_AC_SUFFIX3_BLOOM_WORDS);
        assert!(classic_ac_suffix3_bloom_contains(&mask, b'x', b'y', b'z'));
        assert!(classic_ac_suffix3_bloom_contains(&mask, b'x', b'a', b'b'));
        assert!(classic_ac_suffix3_bloom_contains(&mask, b'k', b'e', b'n'));
        assert!(classic_ac_suffix3_bloom_contains(&mask, b'G', b'I', b'N'));
        assert!(!classic_ac_suffix3_bloom_contains(&mask, b'n', b'e', b'k'));
    }

    #[test]
    fn suffix3_prefilter_reference_eval_matches_reference_count() {
        let patterns: [&[u8]; 5] = [b"a", b"bc", b"ab", b"abcd", b"BEGIN"];
        let haystack = b"abcd a bc BEGIN zabcda";
        let ac = classic_ac_compile(&patterns);
        let expected = classic_ac_scan_counts(&ac, haystack).iter().sum::<u32>();
        let program = with_reference_dispatch_lanes(
            build_ac_bounded_count_suffix3_prefilter_program(&ac.dfa),
            haystack.len() as u32,
        );
        let mut inputs = ac_dfa_table_inputs(&ac.dfa, haystack);
        inputs.push(u32_input(&classic_ac_candidate_end_byte_mask_words(
            &ac.dfa,
        )));
        inputs.push(u32_input(&classic_ac_candidate_suffix2_mask_words(&ac.dfa)));
        inputs.push(u32_input(&classic_ac_candidate_suffix3_bloom_words(
            &patterns,
        )));
        inputs.push(u32_input(&[haystack.len() as u32]));
        inputs.push(vyre_reference::value::Value::from(vec![
            0_u8;
            haystack.len() * 4
        ]));
        let outputs = vyre_reference::reference_eval(&program, &inputs).expect(
            "Fix: suffix3 prefiltered AC bounded count program should evaluate in reference backend.",
        );

        assert_eq!(decode_u32(&outputs[0].to_bytes()), vec![expected]);
    }

    #[test]
    fn suffix3_prefilter_program_has_compact_stable_shape() {
        let ac = classic_ac_compile(&[b"Authorization: Bearer ", b"token", b"tok"]);
        let program = build_ac_bounded_count_suffix3_prefilter_program(&ac.dfa);

        assert_eq!(program.workgroup_size(), [128, 1, 1]);
        assert_eq!(program.buffers().len(), 8);
        assert_eq!(program.buffers()[4].name(), "candidate_suffix2_mask");
        assert_eq!(
            program.buffers()[4].count,
            CLASSIC_AC_SUFFIX2_MASK_WORDS as u32
        );
        assert_eq!(program.buffers()[5].name(), "candidate_suffix3_bloom");
        assert_eq!(
            program.buffers()[5].count,
            CLASSIC_AC_SUFFIX3_BLOOM_WORDS as u32
        );
        assert_eq!(program.buffers()[7].name(), "match_count");
        assert_eq!(program.buffers()[7].count, 1);
    }

    #[test]
    fn ascii_case_variants_expands_only_when_requested() {
        assert_eq!(ascii_case_variants(b'a', false), ([b'a', 0], 1));
        assert_eq!(ascii_case_variants(b'a', true), ([b'a', b'A'], 2));
        assert_eq!(ascii_case_variants(b'Z', true), ([b'z', b'Z'], 2));
        assert_eq!(ascii_case_variants(b'1', true), ([b'1', 0], 1));
        assert_eq!(ascii_case_variants(b'_', true), ([b'_', 0], 1));
    }

    #[test]
    fn suffix3_bloom_ci_expands_all_case_permutations() {
        let patterns: [&[u8]; 1] = [b"cat"];
        let mask_cs = classic_ac_candidate_suffix3_bloom_words_ci(&patterns, false);
        let mask_ci = classic_ac_candidate_suffix3_bloom_words_ci(&patterns, true);

        assert!(classic_ac_suffix3_bloom_contains(
            &mask_cs, b'c', b'a', b't'
        ));
        assert!(!classic_ac_suffix3_bloom_contains(
            &mask_cs, b'C', b'A', b'T'
        ));
        assert!(!classic_ac_suffix3_bloom_contains(
            &mask_cs, b'c', b'A', b't'
        ));

        for &c in &[b'c', b'C'] {
            for &a in &[b'a', b'A'] {
                for &t in &[b't', b'T'] {
                    assert!(
                        classic_ac_suffix3_bloom_contains(&mask_ci, c, a, t),
                        "candidate triple ({}, {}, {}) must be admitted in CI mode",
                        c as char,
                        a as char,
                        t as char,
                    );
                }
            }
        }
    }
}
