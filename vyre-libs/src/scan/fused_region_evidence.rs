//! Fused single-launch phase-1 evidence (plan W2-2, line 153).
//!
//! ONE dispatch, ONE DFA walk per haystack byte, THREE evidence families a
//! coalesced-batch consumer (keyhog) otherwise assembles from three separate
//! dispatches:
//!
//! 1. **Per-region presence**: a bitmap `presence[region][pid]` = "pattern `pid`
//!    matches somewhere in region".
//! 2. **Positions**: `(pid, start, end)` match triples for a *designated
//!    subset* of patterns (`position_mask[pid] != 0`), the ones a consumer wants
//!    located, not just admitted.
//! 3. **Per-region admission**: a second bitmap `admission[region][pid]` for a
//!    (possibly different) designated subset (`admission_mask[pid] != 0`), the
//!    coarse "this detector could fire here" signal that gates a heavier
//!    verifier.
//!
//! All three fall out of the SAME anchored forward-walk this crate already uses
//! for extraction ([`crate::scan::regex_anchored_window`]) and admission
//! ([`crate::scan::regex_region_admission`]); the CPU oracle reuses those so
//! there is one source of truth for the walk.
//!
//! # Perf: a correctness primitive, not (yet) a win
//!
//! Fusing families into one kernel is MEASURED-REFUTED on this substrate: the
//! existing two-family fusion (`GpuLiteralSet::scan_presence_and_positions_by_region`)
//! is ~20x SLOWER than the separate passes (occupancy collapse as the kernel
//! grows), and its own test concludes "the lever is segmentation, not fusion".
//! This three-family launch ships for the same reason that one does, a
//! CORRECTNESS-equivalent primitive that produces the full bundle in one
//! dispatch, while the fast path stays the separate specialized passes
//! (`scan_presence_by_region` + `scan` + `regex_admission_by_region_program`).
//! Prefer those until a segmentation/occupancy redesign makes fusion pay.

use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::match_result::Match;
use vyre_primitives::matching::CompiledDfa;

use crate::region::wrap_anonymous;
use crate::scan::builders::{append_match, load_packed_byte};
use crate::scan::regex_anchored_window::AnchoredWindowValidator;
use crate::scan::regex_region_admission::{regex_admission_presence_words, region_of};

/// The three evidence families a fused launch produces, as a CPU value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusedRegionEvidence {
    /// `presence[region * words + (pid >> 5)]` bit `pid & 31` set iff `pid`
    /// matches in `region`. `words = regex_admission_presence_words(pattern_count)`.
    pub presence: Vec<u32>,
    /// `(pid, start, end)` triples for patterns with `position_mask[pid] != 0`,
    /// in canonical `(start, end, pid)` order.
    pub positions: Vec<Match>,
    /// Per-region bitmap (same shape as `presence`) for patterns with
    /// `admission_mask[pid] != 0`.
    pub admission: Vec<u32>,
}

/// CPU reference for the fused launch, the GPU parity oracle. Reuses
/// [`AnchoredWindowValidator`] for the walk (ONE source of truth) and routes each
/// extracted match into the three families by region and by role mask.
#[must_use]
pub fn fused_region_evidence_reference(
    dfa: &CompiledDfa,
    haystack: &[u8],
    region_starts: &[u32],
    region_base: u32,
    position_mask: &[u32],
    admission_mask: &[u32],
    pattern_count: u32,
) -> FusedRegionEvidence {
    let words = regex_admission_presence_words(pattern_count) as usize;
    let mut presence = vec![0u32; region_starts.len() * words];
    let mut admission = vec![0u32; region_starts.len() * words];
    let mut positions = Vec::new();
    if !haystack.is_empty() {
        let validator = AnchoredWindowValidator::new(dfa);
        let origins: Vec<u32> = (0..haystack.len() as u32).collect();
        for m in validator.validate_candidates(haystack, &origins) {
            let region = region_of(m.start + region_base, region_starts);
            let word = region * words + (m.pattern_id >> 5) as usize;
            let bit = 1u32 << (m.pattern_id & 31);
            presence[word] |= bit;
            if position_mask
                .get(m.pattern_id as usize)
                .copied()
                .unwrap_or(0)
                != 0
            {
                positions.push(m);
            }
            if admission_mask
                .get(m.pattern_id as usize)
                .copied()
                .unwrap_or(0)
                != 0
            {
                admission[word] |= bit;
            }
        }
    }
    positions.sort_unstable_by_key(|m| (m.start, m.end, m.pattern_id));
    positions.dedup();
    FusedRegionEvidence {
        presence,
        positions,
        admission,
    }
}

/// Writable-buffer binding indices, in the order the backend returns them, so a
/// host dispatch and its readback share one owner.
pub const FUSED_EVIDENCE_PRESENCE_BINDING: u32 = 9;
/// See [`FUSED_EVIDENCE_PRESENCE_BINDING`].
pub const FUSED_EVIDENCE_MATCH_COUNT_BINDING: u32 = 10;
/// See [`FUSED_EVIDENCE_PRESENCE_BINDING`].
pub const FUSED_EVIDENCE_MATCHES_BINDING: u32 = 11;
/// See [`FUSED_EVIDENCE_PRESENCE_BINDING`].
pub const FUSED_EVIDENCE_ADMISSION_BINDING: u32 = 12;

/// Build the fused single-launch phase-1 evidence program.
///
/// One invocation per haystack byte `i`: find the region owning `i + region_base`,
/// forward-walk the anchored DFA over `[i, min(i + max_pattern_len, haystack_len))`,
/// and at each accepted pattern set its presence bit, append its `(pid, i, end)`
/// triple when `position_mask[pid] != 0`, and set its admission bit when
/// `admission_mask[pid] != 0`. Outputs, in binding order: `presence`,
/// `match_count`, `matches`, `admission`.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn fused_region_evidence_program(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    output_records: &str,
    region_starts: &str,
    region_base: &str,
    position_mask: &str,
    admission_mask: &str,
    haystack_len: &str,
    presence: &str,
    match_count: &str,
    matches: &str,
    admission: &str,
    state_count: u32,
    output_records_len: u32,
    region_count: u32,
    pattern_count: u32,
    presence_words: u32,
    max_matches: u32,
    max_pattern_len: u32,
    log2_max_regions: u32,
) -> Program {
    let max_pattern_len = max_pattern_len.max(1);
    let (load_step_byte, step_byte) = load_packed_byte(haystack, Expr::var("step"));

    // For each pattern id the current DFA state accepts: presence bit (always),
    // a position triple (if position_mask), an admission bit (if admission_mask).
    let presence_word = Expr::add(
        Expr::var("rs_base"),
        Expr::shr(Expr::var("pattern_id"), Expr::u32(5)),
    );
    let presence_bit = Expr::shl(
        Expr::u32(1),
        Expr::bitand(Expr::var("pattern_id"), Expr::u32(31)),
    );
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
                Expr::atomic_or(presence, presence_word.clone(), presence_bit.clone()),
            ),
            // Positions for the designated subset.
            Node::if_then(
                Expr::ne(
                    Expr::load(position_mask, Expr::var("pattern_id")),
                    Expr::u32(0),
                ),
                vec![append_match(
                    matches,
                    match_count,
                    Expr::var("pattern_id"),
                    Expr::var("origin"),
                    Expr::add(Expr::var("step"), Expr::u32(1)),
                )],
            ),
            // Admission bits for the designated subset.
            Node::if_then(
                Expr::ne(
                    Expr::load(admission_mask, Expr::var("pattern_id")),
                    Expr::u32(0),
                ),
                vec![Node::let_bind(
                    "_vyre_admission_prev",
                    Expr::atomic_or(admission, presence_word.clone(), presence_bit.clone()),
                )],
            ),
        ],
    );

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

    // region binary search (rs_lo converges to owning region), then rs_base.
    let mut per_position = vec![
        Node::let_bind("origin", Expr::var("i")),
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

    let region_bitmap_len = region_count.max(1).saturating_mul(presence_words);
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
            BufferDecl::storage(position_mask, 6, BufferAccess::ReadOnly, DataType::U32)
                .with_count(pattern_count.max(1)),
            BufferDecl::storage(admission_mask, 7, BufferAccess::ReadOnly, DataType::U32)
                .with_count(pattern_count.max(1)),
            BufferDecl::storage(haystack_len, 8, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1),
            BufferDecl::read_write(presence, FUSED_EVIDENCE_PRESENCE_BINDING, DataType::U32)
                .with_count(region_bitmap_len),
            BufferDecl::read_write(
                match_count,
                FUSED_EVIDENCE_MATCH_COUNT_BINDING,
                DataType::U32,
            )
            .with_count(1),
            BufferDecl::output(matches, FUSED_EVIDENCE_MATCHES_BINDING, DataType::U32)
                .with_count(max_matches.saturating_mul(3)),
            BufferDecl::read_write(admission, FUSED_EVIDENCE_ADMISSION_BINDING, DataType::U32)
                .with_count(region_bitmap_len),
        ],
        [128, 1, 1],
        vec![wrap_anonymous(
            "vyre-libs::matching::fused_region_evidence",
            walk_body,
        )],
    )
}

#[cfg(all(test, feature = "matching-regex", feature = "matching-dfa"))]
mod tests {
    use super::*;
    use crate::scan::regex_dfa::build_regex_dfa_pipeline;
    use crate::scan::{pack_haystack_u32, pack_u32_slice};

    fn dfa_for(patterns: &[&str]) -> CompiledDfa {
        build_regex_dfa_pipeline(patterns, 4096, 16_384)
            .expect("Fix: test patterns must compile to an anchored regex DFA")
            .dfa
    }

    fn log2_regions(region_count: u32) -> u32 {
        (32 - (region_count.max(2) - 1).leading_zeros()).max(1)
    }

    /// The fused CPU oracle routes each match into presence (all), positions
    /// (position_mask subset), and admission (admission_mask subset) correctly.
    #[test]
    fn fused_reference_routes_three_families_by_region_and_mask() {
        let patterns = ["abc", "AKIA", "token", "bcd"];
        let dfa = dfa_for(&patterns);
        let haystack = b"abc AKIA\ntoken bcd\n";
        let region_starts = [0u32, 9];
        // positions for {abc, token}; admission for {AKIA, bcd}.
        let position_mask = [1u32, 0, 1, 0];
        let admission_mask = [0u32, 1, 0, 1];

        let ev = fused_region_evidence_reference(
            &dfa,
            haystack,
            &region_starts,
            0,
            &position_mask,
            &admission_mask,
            patterns.len() as u32,
        );

        let words = regex_admission_presence_words(patterns.len() as u32) as usize;
        let bit = |bm: &[u32], r: usize, pid: u32| {
            (bm[r * words + (pid >> 5) as usize] >> (pid & 31)) & 1 == 1
        };
        // presence: all four in their regions.
        assert!(bit(&ev.presence, 0, 0) && bit(&ev.presence, 0, 1));
        assert!(bit(&ev.presence, 1, 2) && bit(&ev.presence, 1, 3));
        // positions only for masked pids.
        let pids: Vec<u32> = ev.positions.iter().map(|m| m.pattern_id).collect();
        assert!(
            pids.contains(&0) && pids.contains(&2),
            "positions must include abc, token"
        );
        assert!(
            !pids.contains(&1) && !pids.contains(&3),
            "positions must exclude AKIA, bcd"
        );
        // admission only for masked pids.
        assert!(bit(&ev.admission, 0, 1) && bit(&ev.admission, 1, 3));
        assert!(!bit(&ev.admission, 0, 0) && !bit(&ev.admission, 1, 2));
    }

    /// The single fused PROGRAM, evaluated by the reference backend, must produce
    /// all three families byte-identical to the CPU oracle, one launch, one
    /// walk, three evidence outputs.
    #[test]
    fn fused_program_reference_eval_matches_cpu_oracle() {
        let patterns = ["abc", "AKIA", "token", "bcd", "secret"];
        let dfa = dfa_for(&patterns);
        let haystack = b"xx abc AKIA\nsecret token\nbcd abc\n";
        let region_starts = [0u32, 12, 25];
        let pattern_count = patterns.len() as u32;
        let position_mask = [1u32, 0, 1, 0, 1];
        let admission_mask = [0u32, 1, 0, 1, 0];
        let words = regex_admission_presence_words(pattern_count);
        let region_count = region_starts.len() as u32;
        let max_matches = 4096u32;

        let expected = fused_region_evidence_reference(
            &dfa,
            haystack,
            &region_starts,
            0,
            &position_mask,
            &admission_mask,
            pattern_count,
        );

        let program = fused_region_evidence_program(
            "haystack",
            "transitions",
            "output_offsets",
            "output_records",
            "region_starts",
            "region_base",
            "position_mask",
            "admission_mask",
            "haystack_len",
            "presence",
            "match_count",
            "matches",
            "admission",
            dfa.state_count,
            dfa.output_records.len() as u32,
            region_count,
            pattern_count,
            words,
            max_matches,
            dfa.max_pattern_len,
            log2_regions(region_count),
        );
        let bitmap_words = (region_count * words) as usize;
        let inputs = vec![
            vyre_reference::value::Value::from(pack_haystack_u32(haystack)),
            vyre_reference::value::Value::from(pack_u32_slice(&dfa.transitions)),
            vyre_reference::value::Value::from(pack_u32_slice(&dfa.output_offsets)),
            vyre_reference::value::Value::from(pack_u32_slice(&dfa.output_records)),
            vyre_reference::value::Value::from(pack_u32_slice(&region_starts)),
            vyre_reference::value::Value::from(pack_u32_slice(&[0])),
            vyre_reference::value::Value::from(pack_u32_slice(&position_mask)),
            vyre_reference::value::Value::from(pack_u32_slice(&admission_mask)),
            vyre_reference::value::Value::from(pack_u32_slice(&[haystack.len() as u32])),
            vyre_reference::value::Value::from(vec![0u8; bitmap_words * 4]),
            vyre_reference::value::Value::from(pack_u32_slice(&[0])),
            vyre_reference::value::Value::from(vec![0u8; max_matches as usize * 3 * 4]),
            vyre_reference::value::Value::from(vec![0u8; bitmap_words * 4]),
        ];
        let outputs = vyre_reference::reference_eval(&program, &inputs)
            .expect("Fix: fused region-evidence program must evaluate in the reference backend");

        let words_of = |v: &vyre_reference::value::Value| -> Vec<u32> {
            v.to_bytes()
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        // Writable buffers, binding order: presence, match_count, matches, admission.
        let presence: Vec<u32> = words_of(&outputs[0])
            .into_iter()
            .take(bitmap_words)
            .collect();
        let count = words_of(&outputs[1])[0] as usize;
        let match_words = words_of(&outputs[2]);
        let mut positions: Vec<Match> = match_words[..count * 3]
            .chunks_exact(3)
            .map(|c| Match::new(c[0], c[1], c[2]))
            .collect();
        positions.sort_unstable_by_key(|m| (m.start, m.end, m.pattern_id));
        positions.dedup();
        let admission: Vec<u32> = words_of(&outputs[3])
            .into_iter()
            .take(bitmap_words)
            .collect();

        assert_eq!(
            presence, expected.presence,
            "fused presence bitmap mismatch"
        );
        assert_eq!(
            positions, expected.positions,
            "fused position triples mismatch"
        );
        assert_eq!(
            admission, expected.admission,
            "fused admission bitmap mismatch"
        );
        assert!(
            expected.presence.iter().any(|&w| w != 0) && !expected.positions.is_empty(),
            "vacuous test"
        );
    }
}
