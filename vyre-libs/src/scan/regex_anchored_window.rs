//! Anchored-window regex validation: W2-3, plan line 179.
//!
//! **Admission vs extraction.** The fused literal scan (`W2-2`) tells a consumer
//! *a candidate exists at position `p`* (a literal prefilter fired). That is
//! *admission*: it does not prove the full regex actually matches, nor does it
//! locate the match extent. **Anchored-window matching closes that gap**: given
//! the candidate origins the positions pass emits and an **anchored** regex DFA
//! (`build_regex_dfa_pipeline`: matches only starting at the scan origin), it
//! replays the DFA seeded at *each* candidate origin and emits every
//! `(pattern_id, start = origin, end)` the DFA accepts within the pattern
//! window. That is *extraction*, confirm **and** locate, which is what makes a
//! GPU regex path useful to a consumer, not just a "maybe here" signal.
//!
//! # Why a windowed walk from the origin is exactly anchored
//!
//! The anchored DFA produced by subset construction
//! ([`vyre_primitives::matching::nfa_to_dfa`]) rejects into a **dedicated dead
//! state** that self-loops on every byte and never accepts, it does *not* fall
//! back to the start state. So once the anchored path from `origin` diverges
//! from every pattern, the walk enters the dead state and can never spuriously
//! re-accept later in the window. A forward replay from `origin` for at most
//! `max_pattern_len` bytes therefore yields precisely the matches that *start*
//! at `origin`, with no unanchored "match somewhere in the window" leakage.
//!
//! This module is the CPU reference/primitive. It is deliberately allocation-
//! light and side-effect free so it doubles as the **parity oracle** for the
//! GPU anchored-window extraction kernel (the sibling unit): the GPU kernel
//! seeds the same transition table at the same origins and must produce the
//! byte-identical match set this walk defines.

use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::match_result::Match;
use vyre_primitives::matching::CompiledDfa;

use crate::region::wrap_anonymous;
use crate::scan::builders::{append_match, load_packed_byte};

/// Validates candidate origins against an anchored regex [`CompiledDfa`],
/// extracting the full `(pattern_id, start, end)` match set that begins at each
/// origin.
///
/// Construct once per DFA (it precomputes the dead-state id for an O(1) early
/// out), then validate any number of candidate batches against different
/// haystacks.
#[derive(Debug, Clone, Copy)]
pub struct AnchoredWindowValidator<'dfa> {
    dfa: &'dfa CompiledDfa,
    /// Precomputed dead-sink state id, if the DFA has one. Reaching it ends a
    /// window walk early: it self-loops forever and never accepts, so no match
    /// can follow. Purely an optimization, correctness holds without it because
    /// the dead state never accepts.
    dead_state: Option<u32>,
}

impl<'dfa> AnchoredWindowValidator<'dfa> {
    /// Bind a validator to an anchored regex DFA (e.g.
    /// `build_regex_dfa_pipeline(..).dfa`).
    ///
    /// The DFA MUST be anchored (no implicit `.*` prefix): the walk treats each
    /// candidate origin as the scan origin. Passing an *unanchored* DFA
    /// (`build_regex_dfa_unanchored`) would report a match at `origin` whenever
    /// the pattern occurs anywhere at or after `origin`, defeating the anchoring
    /// contract.
    #[must_use]
    pub fn new(dfa: &'dfa CompiledDfa) -> Self {
        Self {
            dead_state: detect_dead_state(dfa),
            dfa,
        }
    }

    /// The longest byte window any single candidate can consume, the DFA's
    /// `max_pattern_len`. A consumer sizing a GPU per-candidate replay buffer
    /// reads this to bound the work per origin.
    #[must_use]
    pub fn window_len(&self) -> u32 {
        self.dfa.max_pattern_len
    }

    /// Replay the anchored DFA seeded at a single candidate `origin`, appending
    /// every `(pattern_id, origin, end)` it accepts to `out`.
    ///
    /// Emits one [`Match`] per `(accepting state, pattern id in that state's
    /// output set)`, so a variable-length pattern that accepts at several ends,
    /// and distinct overlapping patterns that accept at one end, all surface
    /// (mirrors the whole-buffer AC dispatch's `output_records` fan-out). Does
    /// not sort or deduplicate; call [`Self::validate_candidates`] for a
    /// canonical, deduplicated batch result. Out-of-range origins are ignored.
    pub fn validate_candidate(&self, haystack: &[u8], origin: u32, out: &mut Vec<Match>) {
        let origin_idx = origin as usize;
        if origin_idx >= haystack.len() {
            return;
        }
        let window = (self.dfa.max_pattern_len as usize).min(haystack.len() - origin_idx);
        let mut state = 0u32;
        for step in 0..window {
            let byte = haystack[origin_idx + step];
            state = self.dfa.transitions[state as usize * 256 + byte as usize];
            if Some(state) == self.dead_state {
                // Dead sink: self-loops forever, never accepts, no match can
                // follow, so stop replaying this origin.
                break;
            }
            let end = origin + step as u32 + 1;
            let lo = self.dfa.output_offsets[state as usize] as usize;
            let hi = self.dfa.output_offsets[state as usize + 1] as usize;
            for &pattern_id in &self.dfa.output_records[lo..hi] {
                out.push(Match::new(pattern_id, origin, end));
            }
        }
    }

    /// Validate a batch of candidate origins, returning the extracted match set
    /// in canonical `(start, end, pattern_id)` order with exact duplicates
    /// removed.
    ///
    /// Duplicate or overlapping origins that yield the same `(pid, start, end)`
    /// collapse to one entry, so the result is a set a consumer can union with
    /// other shards without double counting.
    #[must_use]
    pub fn validate_candidates(&self, haystack: &[u8], origins: &[u32]) -> Vec<Match> {
        let mut matches = Vec::new();
        for &origin in origins {
            self.validate_candidate(haystack, origin, &mut matches);
        }
        matches.sort_unstable_by_key(|m| (m.start, m.end, m.pattern_id));
        matches.dedup();
        matches
    }

    /// Replay the anchored DFA seeded at `origin` and append only the LONGEST
    /// match per pattern id, the leftmost-longest ("maximal munch") semantics a
    /// scanner wants (to `out`).
    ///
    /// [`Self::validate_candidate`] emits one [`Match`] per accepting end (the
    /// raw DFA fan-out); for a variable-length pattern (`{n,m}`, `+`, `*`) that
    /// is `m - n + 1` overlapping partial hits for a single token. A credential
    /// scanner wants exactly one finding covering the whole token, so this
    /// collapses each pattern's accepts to the maximal `end` reachable from
    /// `origin`. Because the walk is *seeded at* `origin`, the start is exact 
    /// there is no derive-`start`-from-a-fixed-length error (the flaw that makes
    /// the whole-buffer `bounded_ranges` `start = end - max_pattern_len` path
    /// unsound for variable lengths; see BACKLOG items 18/27).
    ///
    /// Scope: this resolves overlap *within* one origin (the longest wins). It
    /// does NOT suppress a match at origin `b` that falls inside a longer match
    /// from an earlier origin `a < b`: cross-origin non-overlap is the caller's
    /// policy (the prefilter supplies token-start origins, and
    /// [`Self::validate_candidates_leftmost_longest`] deduplicates identical
    /// triples). Each pattern surfaces at most once per origin here.
    pub fn validate_candidate_leftmost_longest(
        &self,
        haystack: &[u8],
        origin: u32,
        out: &mut Vec<Match>,
    ) {
        let origin_idx = origin as usize;
        if origin_idx >= haystack.len() {
            return;
        }
        let window = (self.dfa.max_pattern_len as usize).min(haystack.len() - origin_idx);
        let mut state = 0u32;
        // (pattern_id, longest end seen) for this origin. `step` increases
        // monotonically so a later accept for the same pid is strictly longer 
        // overwrite the slot rather than keep the shorter earlier end.
        let mut longest: Vec<(u32, u32)> = Vec::new();
        for step in 0..window {
            let byte = haystack[origin_idx + step];
            state = self.dfa.transitions[state as usize * 256 + byte as usize];
            if Some(state) == self.dead_state {
                // Dead sink: never accepts again, so no longer match can follow.
                break;
            }
            let end = origin + step as u32 + 1;
            let lo = self.dfa.output_offsets[state as usize] as usize;
            let hi = self.dfa.output_offsets[state as usize + 1] as usize;
            for &pattern_id in &self.dfa.output_records[lo..hi] {
                match longest.iter_mut().find(|(pid, _)| *pid == pattern_id) {
                    Some(slot) => slot.1 = end,
                    None => longest.push((pattern_id, end)),
                }
            }
        }
        for (pattern_id, end) in longest {
            out.push(Match::new(pattern_id, origin, end));
        }
    }

    /// Batch [`Self::validate_candidate_leftmost_longest`] over `origins`,
    /// returning the canonical `(start, end, pattern_id)`-ordered set with exact
    /// duplicates removed, the leftmost-longest analogue of
    /// [`Self::validate_candidates`].
    #[must_use]
    pub fn validate_candidates_leftmost_longest(
        &self,
        haystack: &[u8],
        origins: &[u32],
    ) -> Vec<Match> {
        let mut matches = Vec::new();
        for &origin in origins {
            self.validate_candidate_leftmost_longest(haystack, origin, &mut matches);
        }
        matches.sort_unstable_by_key(|m| (m.start, m.end, m.pattern_id));
        matches.dedup();
        matches
    }
}

/// Find the DFA's dead-sink state: a non-accepting state that self-loops on
/// every byte and owns no output records. Subset construction allocates at most
/// one such state (the image of the empty NFA-state set); returns its id, or
/// `None` if the automaton has no dead state (every state can still reach an
/// accept). Scans states once; O(state_count · 256) but run only at
/// construction.
fn detect_dead_state(dfa: &CompiledDfa) -> Option<u32> {
    for state in 0..dfa.state_count {
        let s = state as usize;
        if dfa.accept[s] != 0 {
            continue;
        }
        if dfa.output_offsets[s] != dfa.output_offsets[s + 1] {
            continue;
        }
        let base = s * 256;
        if dfa.transitions[base..base + 256]
            .iter()
            .all(|&next| next == state)
        {
            return Some(state);
        }
    }
    None
}

/// Standard match-buffer binding indices for the anchored-window program, so a
/// host dispatch and its readback agree on one owner (never two hand-kept
/// copies). The RW `match_count` (7) and output `matches` (8) are the writable
/// buffers the backend returns, in that order.
pub const ANCHORED_WINDOW_MATCH_COUNT_BINDING: u32 = 7;
/// See [`ANCHORED_WINDOW_MATCH_COUNT_BINDING`].
pub const ANCHORED_WINDOW_MATCHES_BINDING: u32 = 8;

/// Build the anchored-window extraction GPU program.
///
/// **One invocation per candidate origin.** Invocation `i` (guarded by
/// `i < candidate_count`) loads `origin = candidates[i]`, seeds the DFA at state
/// 0, and replays FORWARD over `[origin, min(origin + max_pattern_len,
/// haystack_len))`, appending every `(pattern_id, origin, end)` the DFA accepts
/// along the way, the GPU counterpart of [`AnchoredWindowValidator`], which is
/// its parity oracle. Emits the same `(id, start, end)` triple contract as the
/// literal-AC dispatch (`match_count` + `matches[max_matches * 3]`), so a
/// consumer reuses the existing hit-buffer readback.
///
/// This is a distinct kernel from the whole-buffer AC scan: that walks a suffix
/// window ENDING at each position `i` and emits one end; this walks FORWARD from
/// each candidate origin and emits at every accepting step, extraction at
/// prefilter-supplied origins, not a full-buffer sweep.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn anchored_window_extract_program(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    output_records: &str,
    candidates: &str,
    candidate_count: &str,
    haystack_len: &str,
    match_count: &str,
    matches: &str,
    state_count: u32,
    output_records_len: u32,
    max_candidates: u32,
    max_matches: u32,
    max_pattern_len: u32,
) -> Program {
    let max_pattern_len = max_pattern_len.max(1);
    let (load_step_byte, step_byte) = load_packed_byte(haystack, Expr::var("step"));

    // Per-step of the forward walk: advance the DFA one byte, then emit every
    // pattern id the (new) state accepts as a match ending at `step + 1`,
    // starting at the anchored `origin`.
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
        Node::loop_for(
            "out_idx",
            Expr::var("out_begin"),
            Expr::var("out_end"),
            vec![
                Node::let_bind(
                    "pattern_id",
                    Expr::load(output_records, Expr::var("out_idx")),
                ),
                append_match(
                    matches,
                    match_count,
                    Expr::var("pattern_id"),
                    Expr::var("origin"),
                    Expr::add(Expr::var("step"), Expr::u32(1)),
                ),
            ],
        ),
    ];

    // For one candidate: bound the forward window at
    // min(origin + max_pattern_len, haystack_len) and replay.
    let uncapped_end = Expr::add(Expr::var("origin"), Expr::u32(max_pattern_len));
    let window_end = Expr::select(
        Expr::lt(uncapped_end.clone(), Expr::load(haystack_len, Expr::u32(0))),
        uncapped_end,
        Expr::load(haystack_len, Expr::u32(0)),
    );
    let per_candidate = vec![
        Node::let_bind("origin", Expr::load(candidates, Expr::var("i"))),
        Node::if_then(
            Expr::lt(Expr::var("origin"), Expr::load(haystack_len, Expr::u32(0))),
            vec![
                Node::let_bind("state", Expr::u32(0)),
                Node::let_bind("win_end", window_end),
                Node::loop_for("step", Expr::var("origin"), Expr::var("win_end"), walk_step),
            ],
        ),
    ];

    let walk_body = vec![
        Node::let_bind("i", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(Expr::var("i"), Expr::load(candidate_count, Expr::u32(0))),
            per_candidate,
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
            BufferDecl::storage(candidates, 4, BufferAccess::ReadOnly, DataType::U32)
                .with_count(max_candidates),
            BufferDecl::storage(candidate_count, 5, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1),
            BufferDecl::storage(haystack_len, 6, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1),
            BufferDecl::read_write(
                match_count,
                ANCHORED_WINDOW_MATCH_COUNT_BINDING,
                DataType::U32,
            )
            .with_count(1),
            BufferDecl::output(matches, ANCHORED_WINDOW_MATCHES_BINDING, DataType::U32)
                .with_count(max_matches.saturating_mul(3)),
        ],
        [128, 1, 1],
        vec![wrap_anonymous(
            "vyre-libs::matching::regex_anchored_window",
            walk_body,
        )],
    )
}

#[cfg(all(test, feature = "matching-regex", feature = "matching-dfa"))]
mod tests {
    use super::*;
    use crate::scan::regex_dfa::build_regex_dfa_pipeline;

    const MAX_MATCHES: u32 = 4096;
    const MAX_DFA_STATES: usize = 16_384;

    fn validator_for<'p>(patterns: &[&str]) -> CompiledDfa {
        build_regex_dfa_pipeline(patterns, MAX_MATCHES, MAX_DFA_STATES)
            .expect("Fix: test patterns must compile to an anchored regex DFA")
            .dfa
    }

    /// THE anchoring contract: a match is emitted ONLY when the candidate origin
    /// is exactly where the pattern starts. A candidate one byte early or one
    /// byte late, even though the pattern is present in the window, yields
    /// nothing. This is precisely what distinguishes anchored-window extraction
    /// from an unanchored "match somewhere in the region" scan.
    #[test]
    fn matches_only_at_exact_candidate_origin() {
        let dfa = validator_for(&["abc"]);
        let validator = AnchoredWindowValidator::new(&dfa);
        let haystack = b"..abc..";

        assert_eq!(
            validator.validate_candidates(haystack, &[2]),
            vec![Match::new(0, 2, 5)],
            "candidate at the true start must extract the match with start==origin"
        );
        assert!(
            validator.validate_candidates(haystack, &[1]).is_empty(),
            "candidate one byte before the match start must NOT match (anchored, not unanchored)"
        );
        assert!(
            validator.validate_candidates(haystack, &[3]).is_empty(),
            "candidate one byte after the match start must NOT match"
        );
    }

    /// A short match at the origin must not let the DFA re-accept later in the
    /// window: after "abc" accepts at end 3, the trailing bytes drive the walk
    /// into the dead sink, which never accepts. Proves the dead-state stop is
    /// what keeps a long window anchored.
    #[test]
    fn short_match_does_not_re_accept_deeper_in_window() {
        let dfa = validator_for(&["abc"]);
        let validator = AnchoredWindowValidator::new(&dfa);
        // Long tail after the match, well past max_pattern_len were it unbounded.
        let haystack = b"abcabcabc";
        assert_eq!(
            validator.validate_candidates(haystack, &[0]),
            vec![Match::new(0, 0, 3)],
            "only the anchored match at the origin may surface; later 'abc's start at other origins"
        );
    }

    /// One origin, multiple accept lengths: two patterns sharing a prefix both
    /// accept at the same origin at their own end offsets, and both surface via
    /// the accepting states' output records.
    #[test]
    fn shared_prefix_patterns_emit_every_accept_length_at_one_origin() {
        let dfa = validator_for(&["abc", "abcde"]);
        let validator = AnchoredWindowValidator::new(&dfa);
        let haystack = b"abcde";
        let got = validator.validate_candidates(haystack, &[0]);
        assert_eq!(
            got,
            vec![Match::new(0, 0, 3), Match::new(1, 0, 5)],
            "both the length-3 and length-5 pattern must extract at the shared origin"
        );
    }

    /// A variable-length pattern (bounded repetition) extracts a faithful,
    /// anchored, non-vacuous match: whatever accept ends the compiled DFA
    /// carries, the validator surfaces them all, each starting exactly at the
    /// origin. We compute the expectation by walking the *same* DFA directly
    /// (faithfulness), never by assuming a particular multi-length semantics 
    /// vyre's AC-at-end DFA reports a bounded repetition at a single canonical
    /// length, not one match per length (recorded in BACKLOG for the regex-DFA
    /// owner). Asserting the direct-walk truth keeps this test correct
    /// regardless of that choice.
    #[test]
    fn bounded_repetition_pattern_extracts_faithful_anchored_matches() {
        let dfa = validator_for(&["a{2,4}"]);
        let validator = AnchoredWindowValidator::new(&dfa);
        let haystack = b"aaaaa";
        // Compare over the SAME candidate set the oracle walks (every origin),
        // else the sets legitimately differ (the oracle finds "aa" at each
        // origin, not just origin 0).
        let origins: Vec<u32> = (0..haystack.len() as u32).collect();
        let got = validator.validate_candidates(haystack, &origins);

        let expected = direct_walk_all_origins(&dfa, haystack);
        assert_eq!(
            got, expected,
            "validator must extract exactly what a direct walk of the same DFA accepts"
        );
        assert!(
            !got.is_empty(),
            "a bounded repetition anchored at a matching origin must extract at least one match"
        );
        // Every extracted match starts at one of the supplied origins and is a
        // genuine run of 'a's of the accepted length.
        assert!(
            got.iter()
                .all(|m| haystack[m.start as usize..m.end as usize]
                    .iter()
                    .all(|&b| b == b'a')),
            "every anchored-window match must be a real run of the repeated byte"
        );
        // The bounded-repetition lowering fix (BACKLOG items 18/27) records the
        // MAXIMUM match length, so the replay window now covers the full range:
        // `a{2,4}` has max_pattern_len == 4 and the raw fan-out surfaces every
        // admissible length 2..=4 (the ε skip edges make the fragment end
        // reachable after 2, 3, or 4 copies). Before the fix the window was
        // capped at the MINIMUM (2), so the longer accepts were never visited.
        assert_eq!(
            dfa.max_pattern_len, 4,
            "the {{n,m}} lowering must size the window to the MAX repetition (4), \
             not the min (2), so the windowed walk can reach the longer accepts"
        );
        // At origin 0 over "aaaa" the raw fan-out accepts at lengths 2, 3, AND 4.
        let origin0_ends: Vec<u32> = got
            .iter()
            .filter(|m| m.start == 0)
            .map(|m| m.end - m.start)
            .collect();
        assert_eq!(
            origin0_ends,
            vec![2, 3, 4],
            "raw fan-out must now surface every admissible {{2,4}} length at origin 0"
        );
        // Leftmost-longest extraction collapses those to the single maximal
        // match (the whole 4-'a' run, not three overlapping partial hits).
        assert_eq!(
            validator.validate_candidates_leftmost_longest(haystack, &[0]),
            vec![Match::new(0, 0, 4)],
            "leftmost-longest must emit exactly the longest {{2,4}} match at origin 0"
        );
    }

    /// Direct, un-optimized reference walk of `dfa` over every origin of
    /// `haystack` (no dead-state early-out), returning the canonical extracted
    /// set. This is the faithfulness oracle: the validator must equal it.
    fn direct_walk_all_origins(dfa: &CompiledDfa, haystack: &[u8]) -> Vec<Match> {
        let mut out = Vec::new();
        for origin in 0..haystack.len() {
            let window = (dfa.max_pattern_len as usize).min(haystack.len() - origin);
            let mut state = 0u32;
            for step in 0..window {
                state = dfa.transitions[state as usize * 256 + haystack[origin + step] as usize];
                let lo = dfa.output_offsets[state as usize] as usize;
                let hi = dfa.output_offsets[state as usize + 1] as usize;
                for &pid in &dfa.output_records[lo..hi] {
                    out.push(Match::new(pid, origin as u32, (origin + step + 1) as u32));
                }
            }
        }
        out.sort_unstable_by_key(|m| (m.start, m.end, m.pattern_id));
        out.dedup();
        out
    }

    /// Distinct patterns anchored at their own origins each extract exactly once;
    /// candidate origins are validated independently.
    #[test]
    fn distinct_patterns_extract_at_their_own_origins() {
        let dfa = validator_for(&["abc", "bcd"]);
        let validator = AnchoredWindowValidator::new(&dfa);
        let haystack = b"abcd";
        assert_eq!(
            validator.validate_candidates(haystack, &[0, 1]),
            vec![Match::new(0, 0, 3), Match::new(1, 1, 4)],
            "each pattern extracts at the origin where it starts"
        );
    }

    /// Boundary safety: an origin at or past EOF is ignored, and an origin whose
    /// window is truncated by EOF only extracts matches that fit, no panic, no
    /// out-of-bounds read.
    #[test]
    fn origins_at_and_past_eof_are_safe_and_windows_truncate() {
        let dfa = validator_for(&["abcd"]);
        let validator = AnchoredWindowValidator::new(&dfa);
        let haystack = b"xxabc"; // "abcd" does not fit starting at index 2 (only "abc" remains)
        assert!(
            validator.validate_candidates(haystack, &[2]).is_empty(),
            "a pattern that runs off the end of the haystack must not match"
        );
        assert!(
            validator
                .validate_candidates(haystack, &[haystack.len() as u32])
                .is_empty(),
            "origin == haystack.len() must be ignored, not indexed"
        );
        assert!(
            validator
                .validate_candidates(haystack, &[haystack.len() as u32 + 9])
                .is_empty(),
            "origin past EOF must be ignored"
        );
    }

    /// Duplicate origins collapse: validating the same origin twice yields the
    /// same match once, so the batch result is a clean set.
    #[test]
    fn duplicate_origins_collapse_to_a_set() {
        let dfa = validator_for(&["abc"]);
        let validator = AnchoredWindowValidator::new(&dfa);
        let haystack = b"abc";
        assert_eq!(
            validator.validate_candidates(haystack, &[0, 0, 0]),
            vec![Match::new(0, 0, 3)],
            "repeated origins must not duplicate the extracted match"
        );
    }

    #[test]
    fn empty_candidate_batch_is_empty() {
        let dfa = validator_for(&["abc"]);
        let validator = AnchoredWindowValidator::new(&dfa);
        assert!(validator.validate_candidates(b"abcabc", &[]).is_empty());
    }

    /// GPU program ↔ CPU oracle parity via the reference backend: the vyre IR
    /// `anchored_window_extract_program`, evaluated by the reference
    /// interpreter, must emit exactly the match set [`AnchoredWindowValidator`]
    /// defines for the same DFA, haystack and candidate origins. This proves the
    /// emitted kernel implements the anchored-window semantics, the same
    /// program dispatches on the GPU.
    /// Rewrite `match_count` to `lanes` slots (one per reference-backend lane)
    /// returning only its first word (the shared atomic counter). Minimal local
    /// copy of the classic-AC test glue, inlined because that module is
    /// `#[cfg(test)]`-private and reaching it would mean editing another agent's
    /// in-flight `classic_ac.rs`.
    fn with_reference_dispatch_lanes(program: Program, lanes: u32) -> Program {
        let buffers = program
            .buffers()
            .iter()
            .cloned()
            .map(|buffer| {
                if buffer.name() == "match_count" {
                    buffer.with_count(lanes.max(1)).with_output_byte_range(0..4)
                } else {
                    buffer
                }
            })
            .collect();
        program.with_rewritten_buffers(buffers)
    }

    /// Decode `(pattern_id, start, end)` triples from a `[match_count, matches]`
    /// reference-output pair (little-endian u32 words).
    fn decode_match_triples(outputs: &[vyre_reference::value::Value]) -> Vec<(u32, u32, u32)> {
        let words = |value: &vyre_reference::value::Value| -> Vec<u32> {
            value
                .to_bytes()
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let count = words(&outputs[0])[0] as usize;
        let matches = words(&outputs[1]);
        matches[..count.saturating_mul(3)]
            .chunks_exact(3)
            .map(|chunk| (chunk[0], chunk[1], chunk[2]))
            .collect()
    }

    #[test]
    fn extract_program_reference_eval_matches_cpu_oracle() {
        use crate::scan::{pack_haystack_u32, pack_u32_slice};

        let patterns = ["abc", "abcde", "bcd", "x"];
        let dfa = validator_for(&patterns);
        let validator = AnchoredWindowValidator::new(&dfa);
        let haystack = b"zabcdex bcd abc x abcde";
        // Candidate origins: a mix of real match starts, near-misses, and EOF-
        // adjacent positions (the program must reject the non-starts).
        let candidates: Vec<u32> = vec![0, 1, 2, 8, 12, 16, 18, haystack.len() as u32 - 1];

        // Oracle.
        let mut expected = validator.validate_candidates(haystack, &candidates);
        expected.sort_unstable_by_key(|m| (m.start, m.end, m.pattern_id));

        // Reference dispatch of the emitted program (one lane per candidate).
        let num_candidates = candidates.len() as u32;
        let max_matches = 4096u32;
        let program = with_reference_dispatch_lanes(
            anchored_window_extract_program(
                "haystack",
                "transitions",
                "output_offsets",
                "output_records",
                "candidates",
                "candidate_count",
                "haystack_len",
                "match_count",
                "matches",
                dfa.state_count,
                dfa.output_records.len() as u32,
                num_candidates,
                max_matches,
                dfa.max_pattern_len,
            ),
            num_candidates,
        );
        let inputs = vec![
            vyre_reference::value::Value::from(pack_haystack_u32(haystack)),
            vyre_reference::value::Value::from(pack_u32_slice(&dfa.transitions)),
            vyre_reference::value::Value::from(pack_u32_slice(&dfa.output_offsets)),
            vyre_reference::value::Value::from(pack_u32_slice(&dfa.output_records)),
            vyre_reference::value::Value::from(pack_u32_slice(&candidates)),
            vyre_reference::value::Value::from(pack_u32_slice(&[num_candidates])),
            vyre_reference::value::Value::from(pack_u32_slice(&[haystack.len() as u32])),
            vyre_reference::value::Value::from(vec![0_u8; num_candidates as usize * 4]),
        ];
        let outputs = vyre_reference::reference_eval(&program, &inputs)
            .expect("Fix: anchored-window extract program must evaluate in the reference backend");

        let mut actual: Vec<Match> = decode_match_triples(&outputs)
            .into_iter()
            .map(|(pid, start, end)| Match::new(pid, start, end))
            .collect();
        actual.sort_unstable_by_key(|m| (m.start, m.end, m.pattern_id));
        actual.dedup();

        assert_eq!(
            actual, expected,
            "reference-eval of the anchored-window program must equal the CPU oracle's extraction"
        );
        assert!(
            !expected.is_empty(),
            "parity test is vacuous: the oracle extracted no matches for these candidates"
        );
    }

    /// Rigorous differential: for literal patterns, the anchored-window
    /// extraction over EVERY position must equal an independent naive
    /// "does pattern P start exactly at position p" substring oracle. A
    /// deterministic LCG builds the haystack so the test is reproducible without
    /// an RNG dependency.
    #[test]
    fn differential_vs_naive_anchored_substring_oracle() {
        let patterns = ["ab", "abc", "bcx", "x", "cab"];
        let dfa = validator_for(&patterns);
        let validator = AnchoredWindowValidator::new(&dfa);

        // Deterministic haystack over a small alphabet that plants the patterns
        // densely (LCG (pure, reproducible, no rand crate)).
        let alphabet = b"abcx";
        let mut state: u32 = 0x1234_5678;
        let mut haystack = Vec::with_capacity(600);
        for _ in 0..600 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            haystack.push(alphabet[(state >> 24) as usize % alphabet.len()]);
        }

        // Candidates = every position (the extractor must be exact everywhere).
        let origins: Vec<u32> = (0..haystack.len() as u32).collect();
        let got = validator.validate_candidates(&haystack, &origins);

        // Independent oracle: naive anchored substring test per pattern.
        let mut oracle: Vec<Match> = Vec::new();
        for (pid, pat) in patterns.iter().enumerate() {
            let pb = pat.as_bytes();
            if pb.len() <= haystack.len() {
                for start in 0..=haystack.len() - pb.len() {
                    if &haystack[start..start + pb.len()] == pb {
                        oracle.push(Match::new(
                            pid as u32,
                            start as u32,
                            (start + pb.len()) as u32,
                        ));
                    }
                }
            }
        }
        oracle.sort_unstable_by_key(|m| (m.start, m.end, m.pattern_id));
        oracle.dedup();

        assert_eq!(
            got, oracle,
            "anchored-window extraction must equal the naive anchored-substring oracle at every position"
        );
        // Guard against a vacuous pass: the dense haystack must actually plant
        // matches for several distinct patterns.
        assert!(
            oracle.len() > 50,
            "differential is vacuous: oracle found only {} matches",
            oracle.len()
        );
        let distinct_pids: std::collections::BTreeSet<u32> =
            oracle.iter().map(|m| m.pattern_id).collect();
        assert!(
            distinct_pids.len() >= 4,
            "differential should exercise most patterns; saw pids {distinct_pids:?}"
        );
    }

    /// The dead-state early-out must not change results: a validator that stops
    /// at the dead sink extracts the same set as a full-window walk that never
    /// stops early. We reconstruct the un-optimized walk inline and compare.
    #[test]
    fn dead_state_early_out_equals_full_window_walk() {
        let patterns = ["abc", "abcde", "bx"];
        let dfa = validator_for(&patterns);
        let validator = AnchoredWindowValidator::new(&dfa);
        let haystack = b"abcdefabxabc";
        let origins: Vec<u32> = (0..haystack.len() as u32).collect();
        let optimized = validator.validate_candidates(haystack, &origins);

        // Reference: identical walk WITHOUT the dead-state break (shared helper).
        let full = direct_walk_all_origins(&dfa, haystack);

        assert_eq!(
            optimized, full,
            "dead-state early-out must be a pure optimization, identical extraction to the full walk"
        );
    }

    /// The DFA has exactly one detectable dead sink, and it is neither the start
    /// state nor an accepting state (guards the detector against misclassifying a
    /// live state).
    #[test]
    fn detected_dead_state_is_non_start_non_accepting_self_loop() {
        let dfa = validator_for(&["abc"]);
        let dead =
            detect_dead_state(&dfa).expect("an anchored DFA with a rejecting path has a dead sink");
        assert_ne!(dead, 0, "the start state must not be classified as dead");
        assert_eq!(dfa.accept[dead as usize], 0, "dead state must not accept");
        for byte in 0..=255u16 {
            assert_eq!(
                dfa.transitions[dead as usize * 256 + byte as usize],
                dead,
                "dead state must self-loop on every byte"
            );
        }
    }
}
