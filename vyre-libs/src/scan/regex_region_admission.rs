//! Regex-DFA per-region admission (presence), plan W2-2, line 153's third
//! evidence family, delivered as a SEPARATE efficient pass.
//!
//! # Why separate, not fused
//!
//! 153 asks for "a single launch" producing literal presence + literal positions
//! + regex-DFA admission bits. But the existing two-family fusion
//! (`GpuLiteralSet::scan_presence_and_positions_by_region`) is measured **~20x
//! SLOWER** than the two separate scans (RTX 5090/wgpu/release), occupancy
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
//! spans a region boundary, keyhog's layout), "pattern `p` is admitted in
//! region `r`" == "`p` has a match STARTING at some byte of `r`". Each invocation
//! `i` (a haystack byte) replays the ANCHORED regex DFA forward from `i`
//! (identical walk to [`crate::scan::regex_anchored_window`]); every pattern the
//! DFA accepts starts at `i`, so its presence bit is OR'd into the row of the
//! region that owns `i`. The result is bit-for-bit the literal-presence bitmap's
//! regex counterpart, and its CPU oracle simply attributes
//! [`AnchoredWindowValidator`] extractions to regions, one source of truth for
//! the walk semantics.

use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use vyre_primitives::matching::CompiledDfa;

use crate::region::wrap_anonymous;
use crate::scan::builders::load_packed_byte;
use crate::scan::regex_anchored_window::AnchoredWindowValidator;

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
/// evidence oracle in [`crate::scan::fused_region_evidence`]. ONE owner.
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
    for m in validator.validate_candidates(haystack, &origins) {
        let region = region_of(m.start + region_base, region_starts);
        let word = region * words + (m.pattern_id >> 5) as usize;
        presence[word] |= 1u32 << (m.pattern_id & 31);
    }
    presence
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
    let max_pattern_len = max_pattern_len.max(1);
    let (load_step_byte, step_byte) = load_packed_byte(haystack, Expr::var("step"));

    // Per accepted pattern at the current DFA state: set its presence bit in the
    // owning region's row. `rs_base = region * presence_words`.
    let emit_loop = Node::loop_for(
        "out_idx",
        Expr::var("out_begin"),
        Expr::var("out_end"),
        vec![
            Node::let_bind(
                "pattern_id",
                Expr::load(output_records, Expr::var("out_idx")),
            ),
            Node::let_bind(
                "_vyre_presence_prev",
                Expr::atomic_or(
                    presence,
                    Expr::add(
                        Expr::var("rs_base"),
                        Expr::shr(Expr::var("pattern_id"), Expr::u32(5)),
                    ),
                    Expr::shl(
                        Expr::u32(1),
                        Expr::bitand(Expr::var("pattern_id"), Expr::u32(31)),
                    ),
                ),
            ),
        ],
    );

    // Forward anchored DFA replay from origin `i`, emitting presence at each step.
    let walk_step = vec![
        load_step_byte,
        Node::assign(
            "state",
            Expr::load(
                transitions,
                Expr::add(Expr::mul(Expr::var("state"), Expr::u32(256)), step_byte),
            ),
        ),
        Node::let_bind("out_begin", Expr::load(output_offsets, Expr::var("state"))),
        Node::let_bind(
            "out_end",
            Expr::load(output_offsets, Expr::add(Expr::var("state"), Expr::u32(1))),
        ),
        emit_loop,
    ];

    // region = largest r with region_starts[r] <= (i + region_base); fixed-
    // iteration binary search (rs_lo converges to the owning region).
    let mut per_position = vec![
        Node::let_bind(
            "rs_pos",
            Expr::add(Expr::var("i"), Expr::load(region_base, Expr::u32(0))),
        ),
        Node::let_bind("rs_lo", Expr::u32(0)),
        Node::let_bind(
            "rs_hi",
            Expr::sub(Expr::buf_len(region_starts), Expr::u32(1)),
        ),
        Node::loop_for(
            "rs_step",
            Expr::u32(0),
            Expr::u32(log2_max_regions.max(1)),
            vec![
                Node::let_bind(
                    "rs_mid",
                    Expr::div(
                        Expr::add(
                            Expr::add(Expr::var("rs_lo"), Expr::var("rs_hi")),
                            Expr::u32(1),
                        ),
                        Expr::u32(2),
                    ),
                ),
                Node::let_bind(
                    "rs_cond",
                    Expr::le(
                        Expr::load(region_starts, Expr::var("rs_mid")),
                        Expr::var("rs_pos"),
                    ),
                ),
                Node::assign(
                    "rs_lo",
                    Expr::select(
                        Expr::var("rs_cond"),
                        Expr::var("rs_mid"),
                        Expr::var("rs_lo"),
                    ),
                ),
                Node::assign(
                    "rs_hi",
                    Expr::select(
                        Expr::var("rs_cond"),
                        Expr::var("rs_hi"),
                        Expr::sub(Expr::var("rs_mid"), Expr::u32(1)),
                    ),
                ),
            ],
        ),
        Node::let_bind(
            "rs_base",
            Expr::mul(Expr::var("rs_lo"), Expr::u32(presence_words)),
        ),
        Node::let_bind("state", Expr::u32(0)),
    ];
    // Forward window bound = min(i + max_pattern_len, haystack_len).
    let uncapped_end = Expr::add(Expr::var("i"), Expr::u32(max_pattern_len));
    let window_end = Expr::select(
        Expr::lt(uncapped_end.clone(), Expr::load(haystack_len, Expr::u32(0))),
        uncapped_end,
        Expr::load(haystack_len, Expr::u32(0)),
    );
    per_position.push(Node::let_bind("win_end", window_end));
    per_position.push(Node::loop_for(
        "step",
        Expr::var("i"),
        Expr::var("win_end"),
        walk_step,
    ));

    let walk_body = vec![
        Node::let_bind("i", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(Expr::var("i"), Expr::load(haystack_len, Expr::u32(0))),
            per_position,
        ),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(haystack, 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(transitions, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(state_count.saturating_mul(256)),
            BufferDecl::storage(output_offsets, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(state_count.saturating_add(1)),
            BufferDecl::storage(output_records, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(output_records_len),
            BufferDecl::storage(region_starts, 4, BufferAccess::ReadOnly, DataType::U32)
                .with_count(region_count.max(1)),
            BufferDecl::storage(region_base, 5, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1),
            BufferDecl::storage(haystack_len, 6, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1),
            BufferDecl::read_write(presence, 7, DataType::U32)
                .with_count(region_count.max(1).saturating_mul(presence_words)),
        ],
        [128, 1, 1],
        vec![wrap_anonymous(
            "vyre-libs::matching::regex_admission_by_region",
            walk_body,
        )],
    )
}

#[cfg(all(test, feature = "matching-regex", feature = "matching-dfa"))]
mod tests {
    use super::*;
    use crate::scan::regex_dfa::build_regex_dfa_pipeline;
    use crate::scan::{pack_haystack_u32, pack_u32_slice};

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
