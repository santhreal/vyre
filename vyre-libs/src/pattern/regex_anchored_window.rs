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
//! ([`crate::pattern::nfa_to_dfa()`]) rejects into a **dedicated dead
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

#[cfg(test)]
use crate::pattern::CompiledDfa;
use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
#[cfg(test)]
use vyre_foundation::match_result::ByteRange;

use crate::pattern::builders::append_match;
use crate::pattern::classic_ac::bounded_ranges::{
    ac_output_span_nodes, ac_transition_step_nodes, output_record_loop_node,
};

/// Collapse raw accepting ends to one longest match per `(start, pattern_id)`.
///
/// GPU extraction intentionally emits accepting states without per-pattern
/// scratch. Every host result path applies this in-place contract before
/// exposing token findings.
#[cfg(test)]
pub(crate) fn canonicalize_leftmost_longest(matches: &mut Vec<ByteRange>) {
    matches.sort_unstable_by_key(|m| (m.start, m.tag, m.end));
    let mut write = 0usize;
    for read in 0..matches.len() {
        let current = matches[read];
        if write > 0
            && matches[write - 1].start == current.start
            && matches[write - 1].tag == current.tag
        {
            matches[write - 1] = current;
        } else {
            matches[write] = current;
            write += 1;
        }
    }
    matches.truncate(write);
    matches.sort_unstable_by_key(|m| (m.start, m.end, m.tag));
}
/// Validates candidate origins against an anchored regex [`CompiledDfa`],
/// extracting the full `(pattern_id, start, end)` match set that begins at each
/// origin.
///
/// Construct once per DFA (it precomputes the dead-state id for an O(1) early
/// out), then validate any number of candidate batches against different
/// haystacks.
#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct AnchoredWindowValidator<'dfa> {
    dfa: &'dfa CompiledDfa,
    /// Precomputed dead-sink state id, if the DFA has one. Reaching it ends a
    /// window walk early: it self-loops forever and never accepts, so no match
    /// can follow. Purely an optimization, correctness holds without it because
    /// the dead state never accepts.
    dead_state: Option<u32>,
}

#[cfg(test)]
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
    pub(crate) fn new(dfa: &'dfa CompiledDfa) -> Self {
        Self {
            dead_state: detect_dead_state(dfa),
            dfa,
        }
    }

    /// The longest byte window any single candidate can consume, the DFA's
    /// `max_pattern_len`. A consumer sizing a GPU per-candidate replay buffer
    /// reads this to bound the work per origin.
    #[must_use]
    pub(crate) fn window_len(&self) -> u32 {
        self.dfa.max_pattern_len
    }

    /// Replay the anchored DFA seeded at a single candidate `origin`, appending
    /// every `(pattern_id, origin, end)` it accepts to `out`.
    ///
    /// Emits one [`ByteRange`] per `(accepting state, pattern id in that state's
    /// output set)`, so a variable-length pattern that accepts at several ends,
    /// and distinct overlapping patterns that accept at one end, all surface
    /// (mirrors the whole-buffer AC dispatch's `output_records` fan-out). Does
    /// not sort or deduplicate; call [`Self::validate_candidates`] for a
    /// canonical, deduplicated batch result. Out-of-range origins are ignored.
    pub(crate) fn validate_candidate(
        &self,
        haystack: &[u8],
        origin: u32,
        out: &mut Vec<ByteRange>,
    ) {
        let origin_idx = origin as usize;
        if origin_idx >= haystack.len() {
            return;
        }
        let window = (self.dfa.max_pattern_len as usize).min(haystack.len() - origin_idx);
        let mut state = 0u32;
        for step in 0..window {
            let byte = haystack[origin_idx + step];
            let trans_idx =
                crate::builder::state_machine::TableStateMachineComposer::flat_byte_index(
                    state, byte,
                );
            state = self.dfa.transitions[trans_idx];
            if Some(state) == self.dead_state {
                // Dead sink: self-loops forever, never accepts, no match can
                // follow, so stop replaying this origin.
                break;
            }
            let end = origin + step as u32 + 1;
            let lo = self.dfa.output_offsets[state as usize] as usize;
            let hi = self.dfa.output_offsets[state as usize + 1] as usize;
            for &pattern_id in &self.dfa.output_records[lo..hi] {
                out.push(ByteRange::new(pattern_id, origin, end));
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
    pub(crate) fn validate_candidates(&self, haystack: &[u8], origins: &[u32]) -> Vec<ByteRange> {
        let mut matches = Vec::new();
        for &origin in origins {
            self.validate_candidate(haystack, origin, &mut matches);
        }
        matches.sort_unstable_by_key(|m| (m.start, m.end, m.tag));
        matches.dedup();
        matches
    }

    /// Replay the anchored DFA seeded at `origin` and append only the LONGEST
    /// match per pattern id, the leftmost-longest ("maximal munch") semantics a
    /// scanner wants (to `out`).
    ///
    /// [`Self::validate_candidate`] emits one [`ByteRange`] per accepting end (the
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
    pub(crate) fn validate_candidate_leftmost_longest(
        &self,
        haystack: &[u8],
        origin: u32,
        out: &mut Vec<ByteRange>,
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
            let trans_idx =
                crate::builder::state_machine::TableStateMachineComposer::flat_byte_index(
                    state, byte,
                );
            state = self.dfa.transitions[trans_idx];
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
            out.push(ByteRange::new(pattern_id, origin, end));
        }
    }

    /// Batch [`Self::validate_candidate_leftmost_longest`] over `origins`,
    /// returning the canonical `(start, end, pattern_id)`-ordered set with exact
    /// duplicates removed, the leftmost-longest analogue of
    /// [`Self::validate_candidates`].
    #[must_use]
    pub(crate) fn validate_candidates_leftmost_longest(
        &self,
        haystack: &[u8],
        origins: &[u32],
    ) -> Vec<ByteRange> {
        let mut matches = Vec::new();
        for &origin in origins {
            self.validate_candidate_leftmost_longest(haystack, origin, &mut matches);
        }
        canonicalize_leftmost_longest(&mut matches);
        matches
    }
}

/// Find the DFA's dead-sink state: a non-accepting state that self-loops on
/// every byte and owns no output records. Subset construction allocates at most
/// one such state (the image of the empty NFA-state set); returns its id, or
/// `None` if the automaton has no dead state (every state can still reach an
/// accept). Scans states once; O(state_count · 256) but run only at
/// construction.
#[cfg(test)]
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
/// along the way, the GPU counterpart of `AnchoredWindowValidator`, which is
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

    // Per-step of the forward walk: advance the DFA one byte through the shared
    // AC transition step, read the shared output-link span, then emit every
    // pattern id the (new) state accepts as a match ending at `step + 1`,
    // starting at the anchored `origin`.
    let mut walk_step = ac_transition_step_nodes(haystack, transitions, Expr::var("step"));
    walk_step.extend(ac_output_span_nodes(output_offsets));
    walk_step.push(output_record_loop_node(
        output_records,
        vec![append_match(
            matches,
            match_count,
            Expr::var("pattern_id"),
            Expr::var("origin"),
            Expr::add(Expr::var("step"), Expr::u32(1)),
        )],
    ));

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
        vec![wrap_anonymous_region(
            "vyre-libs::matching::regex_anchored_window",
            walk_body,
        )],
    )
}

#[cfg(all(test, feature = "pattern-regex", feature = "pattern-dfa"))]
#[path = "../../tests/internal/pattern/regex_anchored_window/mod.rs"]
mod tests;
