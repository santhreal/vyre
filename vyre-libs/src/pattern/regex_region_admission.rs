//! Regex-DFA per-region admission (presence), plan W2-2, line 153's third
//! evidence family, delivered as a SEPARATE efficient pass.
//!
//! # Why separate, not fused
//!
//! 153 asks for "a single launch" producing literal presence + literal positions
//! + regex-DFA admission bits. But the existing two-family fusion
//! (`GpuLiteralSet::scan_presence_and_positions_by_region`) is measured **~20x
//! SLOWER** than the two separate scans (measured on the release profile of one discrete device and its secondary text backend), occupancy
//! collapse from a 3×-inlined replay in a kernel that grows with each fused
//! family (see that method's source + `tests/literal_set_presence_and_positions_gpu.rs`,
//! whose own conclusion is "the lever is segmentation, not fusion"). Fusing a
//! THIRD family into that kernel compounds the pessimization (Law 7). So this
//! ships the regex-DFA admission family as its own specialized, occupancy-cheap
//! pass (the same evidence, without the refuted fusion).
//!
//! # Admission semantics
//!
//! For a coalesced batch (files separated by a byte in no pattern, so no match
//! spans a region boundary, the coalesced-batch layout), "pattern `p` is admitted in
//! region `r`" == "`p` has a match STARTING at some byte of `r`". Each invocation
//! `i` (a haystack byte) replays the ANCHORED regex DFA forward from `i`
//! (identical walk to [`crate::pattern::regex_anchored_window`]); every pattern the
//! DFA accepts starts at `i`, so its presence bit is OR'd into the row of the
//! region that owns `i`. The result is bit-for-bit the literal-presence bitmap's
//! regex counterpart, and its CPU oracle simply attributes
//! [`AnchoredWindowValidator`] extractions to regions, one source of truth for
//! the walk semantics.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::pattern::CompiledDfa;

use crate::pattern::classic_ac::bounded_ranges::{
    ac_output_span_nodes, ac_transition_step_nodes, classic_ac_dfa_buffer_decls,
    output_record_loop_node, presence_bit_write_node, region_search_prologue_nodes,
};
use crate::pattern::regex_anchored_window::AnchoredWindowValidator;

/// Presence-bitmap word count per region for `pattern_count` patterns
/// (`ceil(pattern_count / 32)`, min 1). One owner so the program, the CPU
/// oracle, and consumers agree on row width.
#[must_use]
pub fn regex_admission_presence_words(pattern_count: u32) -> u32 {
    pattern_count.div_ceil(32).max(1)
}

/// Largest region index `r` with `region_starts[r] <= pos`. `region_starts` is
/// ascending with `region_starts[0] == 0`; every `pos` therefore lands in a
/// region. Shared by the CPU oracle, the GPU program (as IR), and the fused
/// evidence oracle in [`crate::pattern::fused_region_evidence`]. ONE owner.
#[must_use]
pub fn region_of(pos: u32, region_starts: &[u32]) -> usize {
    match region_starts.binary_search(&pos) {
        Ok(exact) => exact,
        // `Err(insert)` is the count of starts `<= pos` is `insert`; the owning
        // region is the one before the insertion point (>= 1 since starts[0]=0).
        Err(insert) => insert - 1,
    }
}

/// CPU reference for regex-DFA per-region admission (the GPU parity oracle).
///
/// Returns a `region_starts.len() * regex_admission_presence_words(pattern_count)`
/// word bitmap: bit `p & 31` of word `region * words + (p >> 5)` is set iff
/// pattern `p` starts a match within that region. Reuses
/// [`AnchoredWindowValidator`] for the walk (ONE source of truth) and attributes
/// each extracted match's `start` to its region.
#[must_use]
pub fn regex_admission_by_region_reference(
    dfa: &CompiledDfa,
    haystack: &[u8],
    region_starts: &[u32],
    region_base: u32,
    pattern_count: u32,
) -> Vec<u32> {
    let words = regex_admission_presence_words(pattern_count) as usize;
    let mut presence = vec![0u32; region_starts.len() * words];
    if haystack.is_empty() {
        return presence;
    }
    let validator = AnchoredWindowValidator::new(dfa);
    let origins: Vec<u32> = (0..haystack.len() as u32).collect();
    for m in validator.validate_candidates_leftmost_longest(haystack, &origins) {
        let region = region_of(m.start + region_base, region_starts);
        let word = region * words + (m.tag >> 5) as usize;
        presence[word] |= 1u32 << (m.tag & 31);
    }
    presence
}

/// Buffer names and window shape for the anchored per-region walk, shared by the
/// admission program here and the fused evidence program in
/// [`crate::pattern::fused_region_evidence`]. One value instead of nine positional
/// names threaded through both.
#[derive(Clone, Copy)]
pub(crate) struct AnchoredRegionWalk<'a> {
    /// Packed u32 haystack, four bytes per word.
    pub haystack: &'a str,
    /// Dense `state * 256 + byte` transition table.
    pub transitions: &'a str,
    /// Flat output-link offsets, `state_count + 1` entries.
    pub output_offsets: &'a str,
    /// Ascending region start offsets, `region_starts[0] == 0`.
    pub region_starts: &'a str,
    /// Single-element buffer holding this shard's global base offset.
    pub region_base: &'a str,
    /// Single-element buffer holding the live haystack byte length.
    pub haystack_len: &'a str,
    /// Per-region presence-row stride in u32 words.
    pub presence_words: u32,
    /// Forward window cap; no pattern is longer than this.
    pub max_pattern_len: u32,
    /// Fixed binary-search iteration count, `ceil(log2(max_regions))`.
    pub log2_max_regions: u32,
}

/// One invocation per haystack byte `i`: find the region owning
/// `i + region_base`, then replay the ANCHORED DFA forward over
/// `[i, min(i + max_pattern_len, haystack_len))`, running `emit_loop` after each
/// step so every accepted pattern is attributed to the anchor `origin == i`.
///
/// The region lookup, the transition step and the output-link span all come from
/// the AC walk owner in `scan/classic_ac/bounded_ranges`. What is local to this
/// walk is the forward (rather than suffix) window and the `origin` bind the
/// emission reads.
pub(crate) fn anchored_region_walk_body(
    walk: AnchoredRegionWalk<'_>,
    emit_loop: Node,
) -> Vec<Node> {
    let max_pattern_len = walk.max_pattern_len.max(1);
    let haystack_len = Expr::load(walk.haystack_len, Expr::u32(0));

    let mut walk_step =
        ac_transition_step_nodes(walk.haystack, walk.transitions, Expr::var("step"));
    walk_step.extend(ac_output_span_nodes(walk.output_offsets));
    walk_step.push(emit_loop);

    let mut per_position = vec![Node::let_bind("origin", Expr::var("i"))];
    per_position.extend(region_search_prologue_nodes(
        walk.region_starts,
        walk.region_base,
        walk.presence_words,
        walk.log2_max_regions,
    ));
    per_position.push(Node::let_bind("state", Expr::u32(0)));
    let uncapped_end = Expr::add(Expr::var("i"), Expr::u32(max_pattern_len));
    per_position.push(Node::let_bind(
        "win_end",
        Expr::select(
            Expr::lt(uncapped_end.clone(), haystack_len.clone()),
            uncapped_end,
            haystack_len.clone(),
        ),
    ));
    per_position.push(Node::loop_for(
        "step",
        Expr::var("i"),
        Expr::var("win_end"),
        walk_step,
    ));

    vec![
        Node::let_bind("i", Expr::InvocationId { axis: 0 }),
        Node::if_then(Expr::lt(Expr::var("i"), haystack_len), per_position),
    ]
}

pub(crate) fn regex_region_scan_common_buffers(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    output_records: &str,
    region_starts: &str,
    region_base: &str,
    state_count: u32,
    output_records_len: u32,
    region_count: u32,
) -> Vec<BufferDecl> {
    let mut buffers =
        classic_ac_dfa_buffer_decls(haystack, transitions, output_offsets, state_count);
    buffers.extend([
        BufferDecl::storage(output_records, 3, BufferAccess::ReadOnly, DataType::U32)
            .with_count(output_records_len),
        BufferDecl::storage(region_starts, 4, BufferAccess::ReadOnly, DataType::U32)
            .with_count(region_count.max(1)),
        BufferDecl::storage(region_base, 5, BufferAccess::ReadOnly, DataType::U32).with_count(1),
    ]);
    buffers
}

/// Build the regex-DFA per-region admission GPU program.
///
/// One invocation per haystack byte `i`: binary-search `region_starts` for the
/// region owning `i + region_base`, replay the anchored DFA forward over
/// `[i, min(i + max_pattern_len, haystack_len))`, and `atomic_or` each accepted
/// pattern's bit into that region's presence row. Idempotent bit sets need no
/// per-hit counter, so this stays occupancy-cheap (unlike the refuted fused
/// triple path). Output is the `region_count * presence_words` bitmap.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn regex_admission_by_region_program(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    output_records: &str,
    region_starts: &str,
    region_base: &str,
    haystack_len: &str,
    presence: &str,
    state_count: u32,
    output_records_len: u32,
    region_count: u32,
    presence_words: u32,
    max_pattern_len: u32,
    log2_max_regions: u32,
) -> Program {
    let emit_loop = output_record_loop_node(
        output_records,
        vec![presence_bit_write_node(presence, Some("rs_base"))],
    );
    let walk_body = anchored_region_walk_body(
        AnchoredRegionWalk {
            haystack,
            transitions,
            output_offsets,
            region_starts,
            region_base,
            haystack_len,
            presence_words,
            max_pattern_len,
            log2_max_regions,
        },
        emit_loop,
    );
    let mut buffers = regex_region_scan_common_buffers(
        haystack,
        transitions,
        output_offsets,
        output_records,
        region_starts,
        region_base,
        state_count,
        output_records_len,
        region_count,
    );
    buffers.push(
        BufferDecl::storage(haystack_len, 6, BufferAccess::ReadOnly, DataType::U32).with_count(1),
    );
    buffers.push(
        BufferDecl::read_write(presence, 7, DataType::U32)
            .with_count(region_count.max(1).saturating_mul(presence_words)),
    );
    Program::wrapped(
        buffers,
        [128, 1, 1],
        vec![wrap_anonymous_region(
            "vyre-libs::matching::regex_admission_by_region",
            walk_body,
        )],
    )
}

#[cfg(all(test, feature = "pattern-regex", feature = "pattern-dfa"))]
mod tests {
    use super::*;
    use crate::pattern::haystack::pack_haystack_u32;
    use crate::pattern::regex_dfa::build_regex_dfa_pipeline;
    use vyre_primitives::wire::pack_u32_slice;

    const MAX_MATCHES: u32 = 4096;
    const MAX_DFA_STATES: usize = 16_384;

    fn dfa_for(patterns: &[&str]) -> CompiledDfa {
        build_regex_dfa_pipeline(patterns, MAX_MATCHES, MAX_DFA_STATES)
            .expect("Fix: test patterns must compile to an anchored regex DFA")
            .dfa
    }

    fn presence_bit(bitmap: &[u32], region: usize, words: usize, pid: u32) -> bool {
        (bitmap[region * words + (pid >> 5) as usize] >> (pid & 31)) & 1 == 1
    }

    /// `region_of` picks the region whose start is the greatest `<= pos`.
    #[test]
    fn region_of_attributes_positions_to_the_owning_region() {
        let starts = [0u32, 10, 25];
        assert_eq!(region_of(0, &starts), 0);
        assert_eq!(region_of(9, &starts), 0);
        assert_eq!(region_of(10, &starts), 1);
        assert_eq!(region_of(24, &starts), 1);
        assert_eq!(region_of(25, &starts), 2);
        assert_eq!(region_of(1000, &starts), 2);
    }

    /// The CPU oracle sets exactly the patterns that start in each region, and
    /// nothing in a region with no matches.
    #[test]
    fn cpu_oracle_admits_patterns_per_region() {
        // Two coalesced "files": region 0 = "abc AKIA\n", region 1 = "token bcd\n".
        let patterns = ["abc", "AKIA", "token", "bcd", "zzz"];
        let dfa = dfa_for(&patterns);
        let haystack = b"abc AKIA\ntoken bcd\n";
        let region_starts = [0u32, 9]; // region 1 begins after the first '\n'
        let words = regex_admission_presence_words(patterns.len() as u32) as usize;

        let bitmap = regex_admission_by_region_reference(
            &dfa,
            haystack,
            &region_starts,
            0,
            patterns.len() as u32,
        );

        assert!(presence_bit(&bitmap, 0, words, 0), "region 0 admits abc");
        assert!(presence_bit(&bitmap, 0, words, 1), "region 0 admits AKIA");
        assert!(presence_bit(&bitmap, 1, words, 2), "region 1 admits token");
        assert!(presence_bit(&bitmap, 1, words, 3), "region 1 admits bcd");
        // Cross-region leakage must not happen.
        assert!(
            !presence_bit(&bitmap, 0, words, 2),
            "abc-region must not admit token"
        );
        assert!(
            !presence_bit(&bitmap, 1, words, 0),
            "token-region must not admit abc"
        );
        // zzz (pid 4) occurs nowhere.
        assert!(!presence_bit(&bitmap, 0, words, 4) && !presence_bit(&bitmap, 1, words, 4));
    }

    /// GPU program ↔ CPU oracle parity via the reference backend: the emitted IR,
    /// evaluated by the reference interpreter, must produce the byte-identical
    /// per-region admission bitmap the CPU oracle defines.
    #[test]
    fn admission_program_reference_eval_matches_cpu_oracle() {
        let patterns = ["abc", "AKIA", "token", "bcd", "secret"];
        let dfa = dfa_for(&patterns);
        let haystack = b"xx abc AKIA\nsecret token\nbcd abc\n";
        let region_starts = [0u32, 12, 25];
        let pattern_count = patterns.len() as u32;
        let words = regex_admission_presence_words(pattern_count);
        let region_count = region_starts.len() as u32;
        // log2 ceil of region_count, min 1.
        let log2_max_regions = (32 - (region_count.max(2) - 1).leading_zeros()).max(1);

        let expected =
            regex_admission_by_region_reference(&dfa, haystack, &region_starts, 0, pattern_count);

        let program = regex_admission_by_region_program(
            "haystack",
            "transitions",
            "output_offsets",
            "output_records",
            "region_starts",
            "region_base",
            "haystack_len",
            "presence",
            dfa.state_count,
            dfa.output_records.len() as u32,
            region_count,
            words,
            dfa.max_pattern_len,
            log2_max_regions,
        );
        let inputs = vec![
            vyre_reference::value::Value::from(pack_haystack_u32(haystack)),
            vyre_reference::value::Value::from(pack_u32_slice(&dfa.transitions)),
            vyre_reference::value::Value::from(pack_u32_slice(&dfa.output_offsets)),
            vyre_reference::value::Value::from(pack_u32_slice(&dfa.output_records)),
            vyre_reference::value::Value::from(pack_u32_slice(&region_starts)),
            vyre_reference::value::Value::from(pack_u32_slice(&[0])),
            vyre_reference::value::Value::from(pack_u32_slice(&[haystack.len() as u32])),
            vyre_reference::value::Value::from(vec![0u8; expected.len() * 4]),
        ];
        let outputs = vyre_reference::reference_eval(&program, &inputs).expect(
            "Fix: regex admission-by-region program must evaluate in the reference backend",
        );

        let got: Vec<u32> = outputs[0]
            .to_bytes()
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .take(expected.len())
            .collect();

        assert_eq!(
            got, expected,
            "reference-eval admission bitmap must equal the CPU oracle's, word for word"
        );
        assert!(
            expected.iter().any(|&w| w != 0),
            "vacuous test: the oracle admitted no patterns"
        );
    }
}
