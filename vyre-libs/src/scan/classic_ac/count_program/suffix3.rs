use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::region::wrap_anonymous;
use crate::scan::builders::load_packed_byte_expr;
use crate::scan::dfa::CompiledDfa;

use super::count_scan_nodes;
use super::suffix2::CLASSIC_AC_SUFFIX2_MASK_WORDS;

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
pub fn classic_ac_bounded_count_suffix3_prefilter_program(
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
    let i = Expr::var("i");
    let candidate_byte = load_packed_byte_expr(haystack, i.clone());
    let previous_byte =
        load_packed_byte_expr(haystack, Expr::saturating_sub(i.clone(), Expr::u32(1)));
    let previous2_byte =
        load_packed_byte_expr(haystack, Expr::saturating_sub(i.clone(), Expr::u32(2)));
    let suffix2_index = Expr::bitor(
        Expr::shl(Expr::var("previous_byte"), Expr::u32(8)),
        Expr::var("candidate_byte"),
    );
    let suffix3_index = Expr::bitor(
        Expr::bitor(
            Expr::shl(Expr::var("previous2_byte"), Expr::u32(16)),
            Expr::shl(Expr::var("previous_byte"), Expr::u32(8)),
        ),
        Expr::var("candidate_byte"),
    );
    let suffix3_bit_index = suffix3_bloom_bit_index_expr(Expr::var("suffix3_index"));
    let scan_nodes = count_scan_nodes(
        haystack,
        transitions,
        output_offsets,
        match_count,
        max_pattern_len,
    );

    let body = vec![
        Node::let_bind("i", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(i.clone(), Expr::load(haystack_len, Expr::u32(0))),
            vec![
                Node::let_bind("candidate_byte", candidate_byte),
                Node::let_bind(
                    "candidate_word",
                    Expr::load(
                        candidate_end_mask,
                        Expr::shr(Expr::var("candidate_byte"), Expr::u32(5)),
                    ),
                ),
                Node::let_bind(
                    "candidate_bit",
                    Expr::shl(
                        Expr::u32(1),
                        Expr::bitand(Expr::var("candidate_byte"), Expr::u32(31)),
                    ),
                ),
                Node::if_then(
                    Expr::ne(
                        Expr::bitand(Expr::var("candidate_word"), Expr::var("candidate_bit")),
                        Expr::u32(0),
                    ),
                    vec![Node::if_then_else(
                        Expr::eq(i.clone(), Expr::u32(0)),
                        scan_nodes.clone(),
                        vec![
                            Node::let_bind("previous_byte", previous_byte),
                            Node::let_bind("suffix2_index", suffix2_index),
                            Node::let_bind(
                                "suffix2_word",
                                Expr::load(
                                    candidate_suffix2_mask,
                                    Expr::shr(Expr::var("suffix2_index"), Expr::u32(5)),
                                ),
                            ),
                            Node::let_bind(
                                "suffix2_bit",
                                Expr::shl(
                                    Expr::u32(1),
                                    Expr::bitand(Expr::var("suffix2_index"), Expr::u32(31)),
                                ),
                            ),
                            Node::if_then(
                                Expr::ne(
                                    Expr::bitand(
                                        Expr::var("suffix2_word"),
                                        Expr::var("suffix2_bit"),
                                    ),
                                    Expr::u32(0),
                                ),
                                vec![Node::if_then_else(
                                    Expr::eq(i.clone(), Expr::u32(1)),
                                    scan_nodes.clone(),
                                    vec![
                                        Node::let_bind("previous2_byte", previous2_byte),
                                        Node::let_bind("suffix3_index", suffix3_index),
                                        Node::let_bind("suffix3_bit_index", suffix3_bit_index),
                                        Node::let_bind(
                                            "suffix3_word",
                                            Expr::load(
                                                candidate_suffix3_bloom,
                                                Expr::shr(
                                                    Expr::var("suffix3_bit_index"),
                                                    Expr::u32(5),
                                                ),
                                            ),
                                        ),
                                        Node::let_bind(
                                            "suffix3_bit",
                                            Expr::shl(
                                                Expr::u32(1),
                                                Expr::bitand(
                                                    Expr::var("suffix3_bit_index"),
                                                    Expr::u32(31),
                                                ),
                                            ),
                                        ),
                                        Node::if_then(
                                            Expr::ne(
                                                Expr::bitand(
                                                    Expr::var("suffix3_word"),
                                                    Expr::var("suffix3_bit"),
                                                ),
                                                Expr::u32(0),
                                            ),
                                            scan_nodes,
                                        ),
                                    ],
                                )],
                            ),
                        ],
                    )],
                ),
            ],
        ),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(haystack, 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(transitions, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(state_count.saturating_mul(256)),
            BufferDecl::storage(output_offsets, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(state_count.saturating_add(1)),
            BufferDecl::storage(candidate_end_mask, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(8),
            BufferDecl::storage(
                candidate_suffix2_mask,
                4,
                BufferAccess::ReadOnly,
                DataType::U32,
            )
            .with_count(CLASSIC_AC_SUFFIX2_MASK_WORDS as u32),
            BufferDecl::storage(
                candidate_suffix3_bloom,
                5,
                BufferAccess::ReadOnly,
                DataType::U32,
            )
            .with_count(CLASSIC_AC_SUFFIX3_BLOOM_WORDS as u32),
            BufferDecl::storage(haystack_len, 6, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1),
            BufferDecl::read_write(match_count, 7, DataType::U32).with_count(1),
        ],
        [128, 1, 1],
        vec![wrap_anonymous(
            "vyre-libs::matching::classic_ac_bounded_count_suffix3_prefilter",
            body,
        )],
    )
}

/// Derive the hashed three-byte suffix mask consumed by the suffix3 prefilter.
#[must_use]
pub fn classic_ac_candidate_suffix3_bloom_words(patterns: &[&[u8]]) -> Vec<u32> {
    classic_ac_candidate_suffix3_bloom_words_ci(patterns, false)
}

/// The RAW haystack bytes a case-`ci` scan must treat as equal to `byte`:
/// `([byte, _], 1)` normally, or `([lower, upper], 2)` for an ASCII letter under
/// case-insensitive matching. Returned as a fixed array + count so callers can
/// iterate the cartesian product with plain nested loops (no closures capturing
/// the mutable mask). The ONE owner of the case fold shared by the end / suffix2
/// / suffix3 mask builders.
#[must_use]
pub(crate) fn ascii_case_variants(byte: u8, case_insensitive: bool) -> ([u8; 2], usize) {
    if case_insensitive && byte.is_ascii_alphabetic() {
        ([byte.to_ascii_lowercase(), byte.to_ascii_uppercase()], 2)
    } else {
        ([byte, 0], 1)
    }
}

/// ASCII-case-aware variant of [`classic_ac_candidate_suffix3_bloom_words`]: when
/// `case_insensitive`, every ASCII-letter byte of the 3-byte suffix is expanded
/// to BOTH cases (a `2^k`-way cartesian product for `k` letters among the three),
/// so a raw uppercase candidate triple passes the bloom. The prefilter reads the
/// unfolded haystack, so the mask (not the haystack (must carry both cases)).
#[must_use]
pub fn classic_ac_candidate_suffix3_bloom_words_ci(
    patterns: &[&[u8]],
    case_insensitive: bool,
) -> Vec<u32> {
    let mut mask = vec![0_u32; CLASSIC_AC_SUFFIX3_BLOOM_WORDS];
    for pattern in patterns
        .iter()
        .copied()
        .filter(|pattern| !pattern.is_empty())
    {
        match pattern.len() {
            1 => {
                let (cv, cn) = ascii_case_variants(pattern[0], case_insensitive);
                for previous2 in 0..=u8::MAX {
                    for previous in 0..=u8::MAX {
                        for &c in &cv[..cn] {
                            set_suffix3_bloom_bit(&mut mask, previous2, previous, c);
                        }
                    }
                }
            }
            2 => {
                let (bv, bn) = ascii_case_variants(pattern[0], case_insensitive);
                let (cv, cn) = ascii_case_variants(pattern[1], case_insensitive);
                for previous2 in 0..=u8::MAX {
                    for &b in &bv[..bn] {
                        for &c in &cv[..cn] {
                            set_suffix3_bloom_bit(&mut mask, previous2, b, c);
                        }
                    }
                }
            }
            len => {
                let (av, an) = ascii_case_variants(pattern[len - 3], case_insensitive);
                let (bv, bn) = ascii_case_variants(pattern[len - 2], case_insensitive);
                let (cv, cn) = ascii_case_variants(pattern[len - 1], case_insensitive);
                for &a in &av[..an] {
                    for &b in &bv[..bn] {
                        for &c in &cv[..cn] {
                            set_suffix3_bloom_bit(&mut mask, a, b, c);
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
pub fn classic_ac_suffix3_bloom_contains(
    mask: &[u32],
    previous2: u8,
    previous: u8,
    current: u8,
) -> bool {
    let bit_index = classic_ac_suffix3_bloom_bit_index(previous2, previous, current);
    let word = bit_index / 32;
    mask.get(word)
        .is_some_and(|word_value| (word_value & (1_u32 << (bit_index % 32))) != 0)
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

fn set_suffix3_bloom_bit(mask: &mut [u32], previous2: u8, previous: u8, current: u8) {
    let bit_index = classic_ac_suffix3_bloom_bit_index(previous2, previous, current);
    mask[bit_index / 32] |= 1_u32 << (bit_index % 32);
}

fn classic_ac_suffix3_bloom_bit_index(previous2: u8, previous: u8, current: u8) -> usize {
    let suffix = (u32::from(previous2) << 16) | (u32::from(previous) << 8) | u32::from(current);
    (suffix3_bloom_hash(suffix) & CLASSIC_AC_SUFFIX3_BLOOM_INDEX_MASK) as usize
}

fn suffix3_bloom_hash(suffix: u32) -> u32 {
    let mixed = (suffix ^ (suffix >> 11)).wrapping_mul(0x9E37_79B1);
    mixed ^ (mixed >> 15)
}

pub(in crate::scan::classic_ac) fn suffix3_bloom_bit_index_expr(suffix: Expr) -> Expr {
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
    use crate::scan::classic_ac::test_helpers::with_reference_dispatch_lanes;
    use crate::scan::classic_ac::{
        classic_ac_candidate_end_byte_mask_words, classic_ac_candidate_suffix2_mask_words,
        classic_ac_compile, classic_ac_scan_counts,
    };
    use crate::scan::{pack_haystack_u32, pack_u32_slice};
    use crate::test_support::byte_pack::bytes_to_u32 as decode_u32;

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
    fn suffix3_prefilter_reference_eval_matches_cpu_count() {
        let patterns: [&[u8]; 5] = [b"a", b"bc", b"ab", b"abcd", b"BEGIN"];
        let haystack = b"abcd a bc BEGIN zabcda";
        let ac = classic_ac_compile(&patterns);
        let expected = classic_ac_scan_counts(&ac, haystack).iter().sum::<u32>();
        let program = with_reference_dispatch_lanes(
            build_ac_bounded_count_suffix3_prefilter_program(&ac.dfa),
            haystack.len() as u32,
        );
        let inputs = vec![
            vyre_reference::value::Value::from(pack_haystack_u32(haystack)),
            vyre_reference::value::Value::from(pack_u32_slice(&ac.dfa.transitions)),
            vyre_reference::value::Value::from(pack_u32_slice(&ac.dfa.output_offsets)),
            vyre_reference::value::Value::from(pack_u32_slice(
                &classic_ac_candidate_end_byte_mask_words(&ac.dfa),
            )),
            vyre_reference::value::Value::from(pack_u32_slice(
                &classic_ac_candidate_suffix2_mask_words(&ac.dfa),
            )),
            vyre_reference::value::Value::from(pack_u32_slice(
                &classic_ac_candidate_suffix3_bloom_words(&patterns),
            )),
            vyre_reference::value::Value::from(pack_u32_slice(&[haystack.len() as u32])),
            vyre_reference::value::Value::from(vec![0_u8; haystack.len() * 4]),
        ];
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
}
