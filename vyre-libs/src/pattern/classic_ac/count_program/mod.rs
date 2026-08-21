//! Bounded-window Aho-Corasick counting, and the prefilter variants that skip
//! windows a candidate cannot end in.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::pattern::builders::load_packed_byte_expr;

use super::bounded_ranges::{
    bounded_walk_prologue_nodes, candidate_end_gate_nodes, classic_ac_dfa_buffer_decls,
};
use crate::pattern::CompiledDfa;

mod suffix2;
mod suffix3;

pub use suffix2::{
    build_ac_bounded_count_suffix2_prefilter_program, CLASSIC_AC_SUFFIX2_MASK_WORDS,
};
pub use suffix3::{
    build_ac_bounded_count_suffix3_prefilter_program, CLASSIC_AC_SUFFIX3_BLOOM_WORDS,
};

#[cfg(test)]
pub(crate) use suffix2::classic_ac_candidate_suffix2_mask_words;
#[cfg(test)]
pub(crate) use suffix3::classic_ac_candidate_suffix3_bloom_words;

pub(in crate::pattern::classic_ac) use suffix3::suffix3_bloom_bit_index_expr;

fn count_scan_nodes(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    match_count: &str,
    max_pattern_len: u32,
) -> Vec<Node> {
    let mut nodes =
        bounded_walk_prologue_nodes(haystack, transitions, output_offsets, max_pattern_len);
    nodes.push(Node::let_bind(
        "out_count",
        Expr::sub(Expr::var("out_end"), Expr::var("out_begin")),
    ));
    nodes.push(Node::if_then(
        Expr::ne(Expr::var("out_count"), Expr::u32(0)),
        vec![Node::let_bind(
            "_count_old",
            Expr::atomic_add(match_count, Expr::u32(0), Expr::var("out_count")),
        )],
    ));
    nodes
}

/// The candidate-end gate followed by the suffix2 gate: at offset 0 there is no
/// preceding byte to form a bigram from, so the walk runs unconditionally
/// (`offset_zero_scan_nodes`); everywhere else the preceding byte and the
/// candidate byte index the 64Ki-bit `candidate_suffix2_mask` and only a set bit
/// reaches `suffix2_match_nodes`. The outer byte gate comes from the shared AC
/// walk owner, so the count, ranges and presence programs cannot drift in what
/// they admit.
pub(in crate::pattern) fn count_suffix2_prefilter_body(
    haystack: &str,
    candidate_end_mask: &str,
    candidate_suffix2_mask: &str,
    haystack_len: &str,
    offset_zero_scan_nodes: Vec<Node>,
    suffix2_match_nodes: Vec<Node>,
) -> Vec<Node> {
    let previous_byte =
        load_packed_byte_expr(haystack, Expr::saturating_sub(Expr::var("i"), Expr::u32(1)));
    let suffix2_index = Expr::bitor(
        Expr::shl(Expr::var("previous_byte"), Expr::u32(8)),
        Expr::var("candidate_byte"),
    );
    candidate_end_gate_nodes(
        haystack,
        haystack_len,
        candidate_end_mask,
        vec![Node::if_then_else(
            Expr::eq(Expr::var("i"), Expr::u32(0)),
            offset_zero_scan_nodes,
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
                        Expr::bitand(Expr::var("suffix2_word"), Expr::var("suffix2_bit")),
                        Expr::u32(0),
                    ),
                    suffix2_match_nodes,
                ),
            ],
        )],
    )
}

pub(in crate::pattern) fn suffix3_prefilter_match_nodes(
    haystack: &str,
    candidate_suffix3_bloom: &str,
    replay_nodes: Vec<Node>,
) -> Vec<Node> {
    let i = Expr::var("i");
    let previous2_byte =
        load_packed_byte_expr(haystack, Expr::saturating_sub(i.clone(), Expr::u32(2)));
    let suffix3_index = Expr::bitor(
        Expr::bitor(
            Expr::shl(Expr::var("previous2_byte"), Expr::u32(16)),
            Expr::shl(Expr::var("previous_byte"), Expr::u32(8)),
        ),
        Expr::var("candidate_byte"),
    );
    vec![Node::if_then_else(
        Expr::eq(i, Expr::u32(1)),
        replay_nodes.clone(),
        vec![
            Node::let_bind("previous2_byte", previous2_byte),
            Node::let_bind("suffix3_index", suffix3_index),
            Node::let_bind(
                "suffix3_bit_index",
                suffix3_bloom_bit_index_expr(Expr::var("suffix3_index")),
            ),
            Node::let_bind(
                "suffix3_word",
                Expr::load(
                    candidate_suffix3_bloom,
                    Expr::shr(Expr::var("suffix3_bit_index"), Expr::u32(5)),
                ),
            ),
            Node::let_bind(
                "suffix3_bit",
                Expr::shl(
                    Expr::u32(1),
                    Expr::bitand(Expr::var("suffix3_bit_index"), Expr::u32(31)),
                ),
            ),
            Node::if_then(
                Expr::ne(
                    Expr::bitand(Expr::var("suffix3_word"), Expr::var("suffix3_bit")),
                    Expr::u32(0),
                ),
                replay_nodes,
            ),
        ],
    )]
}

pub(in crate::pattern::classic_ac) fn wrap_count_program(
    region_name: &'static str,
    buffers: Vec<BufferDecl>,
    body: Vec<Node>,
) -> Program {
    Program::wrapped(
        buffers,
        [128, 1, 1],
        vec![wrap_anonymous_region(region_name, body)],
    )
}

pub(in crate::pattern::classic_ac) fn count_suffix2_prefilter_buffers(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    candidate_end_mask: &str,
    candidate_suffix2_mask: &str,
    state_count: u32,
) -> Vec<BufferDecl> {
    let mut buffers =
        classic_ac_dfa_buffer_decls(haystack, transitions, output_offsets, state_count);
    buffers.extend([
        BufferDecl::storage(candidate_end_mask, 3, BufferAccess::ReadOnly, DataType::U32)
            .with_count(8),
        BufferDecl::storage(
            candidate_suffix2_mask,
            4,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(CLASSIC_AC_SUFFIX2_MASK_WORDS as u32),
    ]);
    buffers
}

/// Build a bounded-window AC program that returns only the total match count.
///
/// This is the GPU preflight shape for irregular scans: one pass over the
/// packed haystack, no match-triple output allocation, and a four-byte readback.
#[must_use]
fn classic_ac_bounded_count_program(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    haystack_len: &str,
    match_count: &str,
    state_count: u32,
    max_pattern_len: u32,
) -> Program {
    let i = Expr::var("i");

    let body = vec![
        Node::let_bind("i", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(i.clone(), Expr::load(haystack_len, Expr::u32(0))),
            count_scan_nodes(
                haystack,
                transitions,
                output_offsets,
                match_count,
                max_pattern_len,
            ),
        ),
    ];

    wrap_count_program(
        "vyre-libs::matching::classic_ac_bounded_count",
        vec![
            BufferDecl::storage(haystack, 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(transitions, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(state_count.saturating_mul(256)),
            BufferDecl::storage(output_offsets, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(state_count.saturating_add(1)),
            BufferDecl::storage(haystack_len, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1),
            BufferDecl::read_write(match_count, 4, DataType::U32).with_count(1),
        ],
        body,
    )
}

/// Build a bounded-window AC count program with a candidate-end-byte prefilter.
///
/// `candidate_end_mask` is an 8-word bitset where bit `b` is set when byte `b`
/// can terminate at least one accepted pattern in the DFA. Non-candidate lanes
/// skip the bounded DFA replay entirely, which is the common case on noisy
/// security/parser scans with a small literal set.
#[must_use]
#[allow(clippy::too_many_arguments)]
fn classic_ac_bounded_count_prefilter_program(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    candidate_end_mask: &str,
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
    let body = candidate_end_gate_nodes(haystack, haystack_len, candidate_end_mask, scan_nodes);

    wrap_count_program(
        "vyre-libs::matching::classic_ac_bounded_count_prefilter",
        vec![
            BufferDecl::storage(haystack, 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(transitions, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(state_count.saturating_mul(256)),
            BufferDecl::storage(output_offsets, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(state_count.saturating_add(1)),
            BufferDecl::storage(candidate_end_mask, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(8),
            BufferDecl::storage(haystack_len, 4, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1),
            BufferDecl::read_write(match_count, 5, DataType::U32).with_count(1),
        ],
        body,
    )
}

#[cfg(test)]
pub(crate) fn classic_ac_candidate_end_byte_mask_words(dfa: &CompiledDfa) -> [u32; 8] {
    vyre_reference::composition_witness::classic_ac_candidate_end_byte_mask_words_witness(
        &dfa.transitions,
        &dfa.output_offsets,
        dfa.state_count,
    )
}

/// Build a bounded-window AC count-only program for a compiled DFA.
#[must_use]
pub fn build_ac_bounded_count_program(dfa: &CompiledDfa) -> Program {
    classic_ac_bounded_count_program(
        "haystack",
        "transitions",
        "output_offsets",
        "haystack_len",
        "match_count",
        dfa.state_count,
        dfa.max_pattern_len,
    )
}

/// Build the candidate-end-byte prefiltered AC count-only program for a
/// compiled DFA.
#[must_use]
pub fn build_ac_bounded_count_prefilter_program(dfa: &CompiledDfa) -> Program {
    classic_ac_bounded_count_prefilter_program(
        "haystack",
        "transitions",
        "output_offsets",
        "candidate_end_mask",
        "haystack_len",
        "match_count",
        dfa.state_count,
        dfa.max_pattern_len,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::bytes_to_u32 as decode_u32;
    use crate::fixture_bytes::eval_bytes;
    use crate::pattern::classic_ac::classic_ac_compile;
    use crate::pattern::classic_ac::classic_ac_scan_counts;
    use crate::pattern::classic_ac::test_dispatch_and_decode::{
        ac_dfa_table_inputs, u32_input, with_reference_dispatch_lanes,
    };

    #[test]
    fn bounded_count_program_reference_eval_matches_reference_count() {
        let patterns: [&[u8]; 4] = [b"a", b"aa", b"she", b"he"];
        let haystack = b"aaashehe";
        let ac = classic_ac_compile(&patterns);
        let expected = classic_ac_scan_counts(&ac, haystack).iter().sum::<u32>();
        let program = with_reference_dispatch_lanes(
            build_ac_bounded_count_program(&ac.dfa),
            haystack.len() as u32,
        );
        let mut inputs = ac_dfa_table_inputs(&ac.dfa, haystack);
        inputs.push(u32_input(&[haystack.len() as u32]));
        inputs.push(vec![0_u8; haystack.len() * 4]);
        let outputs = eval_bytes("mod", &program, inputs.clone());

        assert_eq!(decode_u32(&outputs[0]), vec![expected]);
    }

    #[test]
    fn candidate_end_byte_mask_marks_only_bytes_that_can_finish_matches() {
        let ac = classic_ac_compile(&[b"ab", b"cab", b"tool"]);
        let mask = classic_ac_candidate_end_byte_mask_words(&ac.dfa);
        let byte_is_candidate =
            |byte: u8| (mask[byte as usize / 32] & (1_u32 << (byte as usize % 32))) != 0;

        assert!(byte_is_candidate(b'b'));
        assert!(byte_is_candidate(b'l'));
        assert!(!byte_is_candidate(b'a'));
        assert!(!byte_is_candidate(b'c'));
        assert_eq!(mask.iter().map(|word| word.count_ones()).sum::<u32>(), 2);
    }

    #[test]
    fn bounded_count_prefilter_reference_eval_matches_reference_count() {
        let patterns: [&[u8]; 4] = [b"ab", b"cab", b"token", b"BEGIN"];
        let haystack = b"zzzzab zzzzcab zzzBEGIN zztoken zzz";
        let ac = classic_ac_compile(&patterns);
        let expected = classic_ac_scan_counts(&ac, haystack).iter().sum::<u32>();
        let program = with_reference_dispatch_lanes(
            build_ac_bounded_count_prefilter_program(&ac.dfa),
            haystack.len() as u32,
        );
        let mut inputs = ac_dfa_table_inputs(&ac.dfa, haystack);
        inputs.push(u32_input(&classic_ac_candidate_end_byte_mask_words(
            &ac.dfa,
        )));
        inputs.push(u32_input(&[haystack.len() as u32]));
        inputs.push(vec![0_u8; haystack.len() * 4]);
        let outputs = eval_bytes("mod", &program, inputs);

        assert_eq!(decode_u32(&outputs[0]), vec![expected]);
    }

    #[test]
    fn bounded_count_program_has_compact_stable_shape() {
        let ac = classic_ac_compile(&[b"Authorization: Bearer ", b"token", b"tok"]);
        let program = build_ac_bounded_count_program(&ac.dfa);

        assert_eq!(program.workgroup_size(), [128, 1, 1]);
        assert_eq!(program.buffers().len(), 5);
        assert_eq!(program.buffers()[4].name(), "match_count");
        assert_eq!(program.buffers()[4].count, 1);
    }

    #[test]
    fn bounded_count_prefilter_program_has_compact_stable_shape() {
        let ac = classic_ac_compile(&[b"Authorization: Bearer ", b"token", b"tok"]);
        let program = build_ac_bounded_count_prefilter_program(&ac.dfa);

        assert_eq!(program.workgroup_size(), [128, 1, 1]);
        assert_eq!(program.buffers().len(), 6);
        assert_eq!(program.buffers()[3].name(), "candidate_end_mask");
        assert_eq!(program.buffers()[3].count, 8);
        assert_eq!(program.buffers()[5].name(), "match_count");
        assert_eq!(program.buffers()[5].count, 1);
    }
}
