//! The bounded-window Aho-Corasick scan builders, and the AC walk itself.
//!
//! This module owns the transition walk for the whole crate, split across
//! three files: `walk` holds the dense
//! `state = transitions[state * 256 + byte]` step, the flat output-link span
//! and the bounded suffix replay; `emit` holds the per-region binary search
//! and everything a walk writes once it has a record span; this file holds the
//! candidate-end byte gate, the range-bound arithmetic
//! (`ac_ranges_output_records_len`), the fail-closed rejection path
//! (`ac_ranges_program_or_fail_closed`) and the scan builders themselves.
//! Every other AC builder here and under `scan/` projects from those
//! primitives and supplies only what genuinely differs: its admission
//! predicate, its emission, its prefilter shape. The gate widths and the
//! program assembly built on top of them belong to the `prefilter` submodule,
//! and the ungated scan below is one of its rows.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::pattern::builders::{append_match, append_match_subgroup, load_packed_byte_expr};

use crate::pattern::CompiledDfa;

pub(in crate::pattern) mod emit;
mod prefilter;
#[cfg(all(feature = "pattern-regex", feature = "pattern-dfa"))]
mod regex_exact;
pub(in crate::pattern) mod walk;

pub use prefilter::{
    build_ac_bounded_ranges_prefilter_program,
    build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce,
    build_ac_bounded_ranges_suffix3_prefilter_program,
    build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce,
    presence_bitmap_words, presence_by_region_words, try_build_ac_bounded_ranges_prefilter_program,
    try_build_ac_bounded_ranges_prefilter_program_with_subgroup_coalesce,
    try_build_ac_bounded_ranges_suffix3_prefilter_program,
    try_build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce,
    try_build_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program,
    try_build_ac_bounded_ranges_suffix3_presence_and_positions_by_region_program_filtered,
    try_build_ac_bounded_ranges_suffix3_presence_by_region_program,
    try_build_ac_bounded_ranges_suffix3_presence_program,
};
use prefilter::{build_ranges_scan, try_build_ranges_scan, PrefilterWidth};
#[cfg(all(feature = "pattern-regex", feature = "pattern-dfa"))]
pub(in crate::pattern) use regex_exact::regex_exact_ranges_program;

use emit::{
    match_span_start_nodes, output_record_loop_node, presence_bit_write_node,
    region_search_prologue_nodes, uniform_output_record_loop_nodes, RECORD_ACTIVE,
};
use walk::{
    bounded_walk_matched_nodes, bounded_walk_prologue_bound, bounded_walk_prologue_nodes,
    WalkBinding,
};
/// The candidate-end byte gate every prefiltered AC program opens with: bind the
/// invocation index `i`, bound it against the live `haystack_len`, unpack the
/// candidate byte, and run `accepted` only when that byte's bit is set in the
/// 8-word `candidate_end_mask`. The bound `candidate_byte` stays in scope so a
/// deeper suffix gate can reuse it instead of unpacking the same byte twice.
pub(in crate::pattern) fn candidate_end_gate_nodes(
    haystack: &str,
    haystack_len: &str,
    candidate_end_mask: &str,
    accepted: Vec<Node>,
) -> Vec<Node> {
    let i = Expr::var("i");
    vec![
        Node::let_bind("i", Expr::LogicalIndex { axis: 0 }),
        Node::if_then(
            Expr::lt(i.clone(), Expr::load(haystack_len, Expr::u32(0))),
            vec![
                Node::let_bind("candidate_byte", load_packed_byte_expr(haystack, i)),
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
                    accepted,
                ),
            ],
        ),
    ]
}

/// Bindings 0-2 of every AC program: the packed haystack, the dense transition
/// table, and the flat output-link offsets. The walk's own table ABI, so it
/// lives with the walk.
pub(in crate::pattern) fn classic_ac_dfa_buffer_decls(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    state_count: u32,
) -> Vec<BufferDecl> {
    vec![
        BufferDecl::storage(haystack, 0, BufferAccess::ReadOnly, DataType::U32),
        BufferDecl::storage(transitions, 1, BufferAccess::ReadOnly, DataType::U32)
            .with_count(state_count.saturating_mul(256)),
        BufferDecl::storage(output_offsets, 2, BufferAccess::ReadOnly, DataType::U32)
            .with_count(state_count.saturating_add(1)),
    ]
}

/// Bindings 0-5 of every bounded-RANGES AC program: the DFA tables above plus
/// the flat output records, the pattern-length table, and the live haystack
/// length. One value instead of the nine positional arguments each builder used
/// to respell, so the shared input ABI cannot drift between the plain,
/// candidate-gated, suffix3-gated and presence programs.
#[derive(Clone, Copy)]
pub(in crate::pattern) struct AcInputBindings<'a> {
    pub haystack: &'a str,
    pub transitions: &'a str,
    pub output_offsets: &'a str,
    pub output_records: &'a str,
    pub pattern_lengths: &'a str,
    pub haystack_len: &'a str,
    pub state_count: u32,
    pub output_records_len: u32,
    pub pattern_count: u32,
}

impl<'a> AcInputBindings<'a> {
    /// All six bounded-range inputs under their canonical runtime names.
    pub(in crate::pattern) const fn canonical(
        state_count: u32,
        output_records_len: u32,
        pattern_count: u32,
    ) -> Self {
        Self::from_names(
            "haystack",
            "transitions",
            "output_offsets",
            "output_records",
            "pattern_lengths",
            "haystack_len",
            state_count,
            output_records_len,
            pattern_count,
        )
    }

    /// The shared inputs under caller-selected buffer names and DFA-derived counts.
    pub(in crate::pattern) const fn from_names(
        haystack: &'a str,
        transitions: &'a str,
        output_offsets: &'a str,
        output_records: &'a str,
        pattern_lengths: &'a str,
        haystack_len: &'a str,
        state_count: u32,
        output_records_len: u32,
        pattern_count: u32,
    ) -> Self {
        Self {
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
    }

    /// The six declarations, in binding order.
    pub(in crate::pattern) fn decls(&self) -> Vec<BufferDecl> {
        let mut decls = classic_ac_dfa_buffer_decls(
            self.haystack,
            self.transitions,
            self.output_offsets,
            self.state_count,
        );
        decls.reserve(3);
        decls.extend([
            BufferDecl::storage(
                self.output_records,
                3,
                BufferAccess::ReadOnly,
                DataType::U32,
            )
            .with_count(self.output_records_len),
            BufferDecl::storage(
                self.pattern_lengths,
                4,
                BufferAccess::ReadOnly,
                DataType::U32,
            )
            .with_count(self.pattern_count),
            BufferDecl::storage(self.haystack_len, 5, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1),
        ]);
        decls
    }
}

/// The `dfa.output_records.len()` to u32 narrowing every bounded-ranges
/// `try_build_*` entrypoint performs before it can size binding 3.
///
/// `program` names the dispatch shape in the message, and it is the only thing
/// that differed across the six hand-written copies. What this deliberately does
/// NOT do is clamp, saturate, or default: an unrepresentable record count has to
/// reach the caller as an error, because a silently truncated `output_records`
/// table drops matches with no other symptom.
pub(in crate::pattern) fn ac_ranges_output_records_len(
    dfa: &CompiledDfa,
    program: &str,
) -> Result<u32, String> {
    u32::try_from(dfa.output_records.len()).map_err(|source| {
        format!(
            "AC {program} DFA output record count {} exceeds u32 GPU buffer metadata: {source}. Fix: shard the pattern set or lower the DFA budget before dispatch.",
            dfa.output_records.len()
        )
    })
}

/// Unwrap a bounded-ranges builder's `Result` for the infallible entrypoint,
/// panicking with the recovery route rather than substituting a dispatchable
/// program.
///
/// `program` names the dispatch shape and `fallible` the entrypoint a caller
/// that must recover calls instead. Those two are the only positions that
/// differed across the copies. The failure MODE is deliberately not a parameter:
/// every bounded-ranges builder loses recall the same way, because an empty
/// rejecting automaton and an all-zero candidate mask both admit nothing, so all
/// of them fail closed here instead of returning something a caller would
/// dispatch and trust.
///
/// # Panics
///
/// Panics if `built` is an `Err`, indicating the AC program build failed.
pub(in crate::pattern) fn ac_ranges_program_or_fail_closed(
    built: Result<Program, String>,
    program: &str,
    fallible: &str,
) -> Program {
    match built {
        Ok(ready) => ready,
        Err(error) => panic!(
            "AC {program} program build failed: {error}. \
             substituting an empty rejecting automaton or an all-zero candidate mask \
             would silently lose every match; \
             use {fallible} and shard oversized DFAs across multiple programs."
        ),
    }
}

/// A scan body split at the admission gate.
///
/// `gated` runs only for an admitted candidate. `uniform` runs in every lane
/// of the dispatch, because it carries a subgroup collective and a collective
/// reads across lanes: a rejected lane that never entered the gate has none of
/// the names its peers are reading. `prelude` introduces the names `gated`
/// writes and `uniform` reads, outside the gate, and is empty for an emit with
/// no collective in it.
pub(in crate::pattern) struct ScanBody {
    /// Bindings the gate assigns into and the uniform tail reads.
    pub(in crate::pattern) prelude: Vec<Node>,
    /// The walk and, for a non-collective emit, the emit itself.
    pub(in crate::pattern) gated: Vec<Node>,
    /// The collective emit, at invocation level.
    pub(in crate::pattern) uniform: Vec<Node>,
}

impl ScanBody {
    /// A body with nothing to run outside the gate.
    pub(in crate::pattern) fn gated_only(gated: Vec<Node>) -> Self {
        Self {
            prelude: Vec::new(),
            gated,
            uniform: Vec::new(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn bounded_ranges_scan_nodes(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    output_records: &str,
    pattern_lengths: &str,
    match_count: &str,
    matches: &str,
    max_pattern_len: u32,
    use_subgroup_coalesce: bool,
) -> ScanBody {
    let mut per_record = match_span_start_nodes(pattern_lengths);
    if !use_subgroup_coalesce {
        per_record.push(append_match(
            matches,
            match_count,
            Expr::var("pattern_id"),
            Expr::var("match_start"),
            Expr::var("scan_end"),
        ));
        let mut gated =
            bounded_walk_prologue_nodes(haystack, transitions, output_offsets, max_pattern_len);
        gated.push(output_record_loop_node(output_records, per_record));
        return ScanBody::gated_only(gated);
    }
    per_record.extend(append_match_subgroup(
        matches,
        match_count,
        Expr::var("pattern_id"),
        Expr::var("match_start"),
        Expr::var("scan_end"),
        Expr::var(RECORD_ACTIVE),
    ));
    // A rejected candidate leaves the span empty, so the uniform tail walks
    // zero records for that lane and emits nothing, while still reaching the
    // collective with every peer.
    ScanBody {
        prelude: vec![
            Node::let_bind("scan_end", Expr::u32(0)),
            Node::let_bind("out_begin", Expr::u32(0)),
            Node::let_bind("out_end", Expr::u32(0)),
        ],
        gated: bounded_walk_prologue_bound(
            haystack,
            transitions,
            output_offsets,
            max_pattern_len,
            WalkBinding::Assign,
        ),
        uniform: uniform_output_record_loop_nodes(output_records, None, per_record),
    }
}

/// Emit the bounded-window DFA replay for a single candidate position, writing a
/// per-pattern PRESENCE bit instead of an `(id,start,end)` match triple.
///
/// Innovation: match-DENSE literal sets (a source-code prefilter fires ~1 hit per
/// 30 bytes) make the triple-append path output-bound, every hit takes an atomic
/// counter increment + three global stores, and the host reads back tens of
/// thousands of triples. Measured on a 5090 that collapses a 676 MB/s scan kernel
/// to 4.5 MB/s. But a prefilter consumer (e.g. a downstream scanner's `collect_triggered_patterns`)
/// only needs to know WHICH patterns fired, not where. Setting a presence bit is
/// IDEMPOTENT, so concurrent lanes hitting the same pattern need no counter and no
/// per-hit serialization, just an `atomic_or` into a ~`ceil(patterns/32)`-word
/// bitmap that is the entire readback. This keeps the kernel near the scan ceiling
/// on dense inputs. `pattern_lengths` / `match_start` are unused (no positions).
fn bounded_ranges_presence_nodes(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    output_records: &str,
    presence: &str,
    max_pattern_len: u32,
) -> Vec<Node> {
    let mut nodes =
        bounded_walk_prologue_nodes(haystack, transitions, output_offsets, max_pattern_len);
    nodes.push(output_record_loop_node(
        output_records,
        vec![presence_bit_write_node(presence, None)],
    ));
    nodes
}

/// Region-attributed counterpart of [`bounded_ranges_presence_nodes`]: write the
/// presence bit into a per-REGION bitmap row instead of one global bitmap.
///
/// Innovation: a coalesced-batch consumer packs N independent
/// files into one haystack and needs to know which patterns fired *in each file*,
/// not just somewhere in the batch. The triple-append path gives exact spans the
/// consumer then reduces to a per-file trigger set on the host, paying the dense
/// per-hit atomic-counter serialization + large triple readback measured to
/// collapse a 554 MB/s scan to 4.4 MB/s. This builder keeps the idempotent
/// `atomic_or` (no counter, stays near the scan ceiling) but indexes it by region:
/// the candidate end position `i` is mapped to its region via a bounded binary
/// search over `region_starts` (ascending file start offsets in the coalesced
/// buffer; `region_starts[0]` MUST be 0), then the bit lands in
/// `presence[region * presence_words + (pattern_id >> 5)]`. The readback is the
/// `region_count × presence_words` bitmap the consumer wanted directly, no host
/// reduction, no span materialization.
///
/// `log2_max_regions` fixed binary-search iterations bound the region lookup
/// (`ceil(log2(max_regions))`); `presence_words` is the per-region row stride.
/// The kernel reads the live `region_count` from `buf_len(region_starts)`, so one
/// compiled program serves any batch with `region_count <= max_regions`. A match
/// never spans a region boundary (the consumer inserts separator bytes between
/// files), so attributing by the end position `i` equals attributing by the start.
#[allow(clippy::too_many_arguments)]
fn bounded_ranges_presence_by_region_nodes(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    output_records: &str,
    presence: &str,
    region_starts: &str,
    region_base: &str,
    max_pattern_len: u32,
    presence_words: u32,
    log2_max_regions: u32,
) -> Vec<Node> {
    // Region lookup + presence writes, gated on this candidate having matches.
    // `region = largest r with region_starts[r] <= pos` where `pos = i +
    // region_base` is the GLOBAL byte position: a sharded dispatch scans a slice
    // with local positions `i` but attributes against the whole batch's
    // `region_starts` by adding the shard's base offset (0 for the
    // single-dispatch path).
    let mut region_and_emit =
        region_search_prologue_nodes(region_starts, region_base, presence_words, log2_max_regions);
    region_and_emit.push(output_record_loop_node(
        output_records,
        vec![presence_bit_write_node(presence, Some("rs_base"))],
    ));
    bounded_walk_matched_nodes(
        haystack,
        transitions,
        output_offsets,
        max_pattern_len,
        region_and_emit,
    )
}

/// FUSED presence-AND-positions region replay: one bounded-window DFA walk that, at
/// each accepted candidate, emits BOTH the per-region presence bit (idempotent
/// `atomic_or`, exactly as [`bounded_ranges_presence_by_region_nodes`]) AND the
/// `(pattern_id, start, end)` match triple (atomic append, exactly as
/// [`bounded_ranges_scan_nodes`]).
///
/// Innovation: a coalesced-batch consumer (a GPU phase-1 scanner) needs the per-file
/// trigger SET *and* the anchor/keyword match POSITIONS. Today it pays TWO full GPU
/// scans of the same haystack: `scan_presence_by_region` (bitmap) then a second
/// `scan_into` (triples), because the presence bitmap carries no positions. Both
/// scans run the IDENTICAL suffix3 candidate gate + bounded DFA replay over the same
/// `output_records`; only the per-record EMISSION differs. Fusing them runs the
/// expensive walk ONCE and drives both outputs from the single `output_records`
/// loop, halving the consumer's phase-1 work. Recall-identical to the two separate
/// scans by construction: same candidate set, same walk, same iteration order, the
/// presence bits equal `scan_presence_by_region`'s and the triples equal
/// `scan_into`'s, just produced together.
#[allow(clippy::too_many_arguments)]
fn bounded_ranges_presence_and_positions_by_region_nodes(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    output_records: &str,
    pattern_lengths: &str,
    presence: &str,
    region_starts: &str,
    region_base: &str,
    match_count: &str,
    matches: &str,
    max_pattern_len: u32,
    presence_words: u32,
    log2_max_regions: u32,
    first_positioned_pattern_id: u32,
) -> Vec<Node> {
    // Region binary search, then ONE `output_records` loop that emits the region
    // presence bit AND the match triple per accepted pattern.
    let mut positioned = match_span_start_nodes(pattern_lengths);
    positioned.push(append_match(
        matches,
        match_count,
        Expr::var("pattern_id"),
        Expr::var("match_start"),
        Expr::var("scan_end"),
    ));
    let mut region_and_emit =
        region_search_prologue_nodes(region_starts, region_base, presence_words, log2_max_regions);
    region_and_emit.push(output_record_loop_node(
        output_records,
        vec![
            presence_bit_write_node(presence, Some("rs_base")),
            // No subgroup coalesce on the triple append: one native backend cannot
            // lower subgroup ops and the dense-hit benefit is the presence
            // bitmap's job, not this fused path's.
            Node::if_then(
                Expr::ge(
                    Expr::var("pattern_id"),
                    Expr::u32(first_positioned_pattern_id),
                ),
                positioned,
            ),
        ],
    ));
    bounded_walk_matched_nodes(
        haystack,
        transitions,
        output_offsets,
        max_pattern_len,
        region_and_emit,
    )
}

/// Build the dispatch Program for a bounded-ranges AC scan over an
/// already-compiled DFA. Pairs with
/// [`build_ac_bounded_ranges_program_with_subgroup_coalesce`]: identical buffer layout
/// and emit format, but the caller doesn't have to thread through
/// the eight derived count fields every time.
#[must_use]
pub fn build_ac_bounded_ranges_program(
    dfa: &CompiledDfa,
    pattern_count: u32,
    max_matches: u32,
) -> Program {
    build_ac_bounded_ranges_program_with_subgroup_coalesce(dfa, pattern_count, max_matches, true)
}

/// Variant of [`build_ac_bounded_ranges_program`] that exposes the
/// `use_subgroup_coalesce` selector. Pass `false` when the program
/// is going to be dispatched on a backend that cannot lower
/// `subgroup_ballot` + `subgroup_shuffle` yet.
///
/// # Panics
/// Panics when the automaton exceeds the GPU ABI limits, through the crate's
/// shared fail-closed wrapper. Callers that must recover use
/// [`try_build_ac_bounded_ranges_program_with_subgroup_coalesce`] and shard the DFA.
#[must_use]
pub fn build_ac_bounded_ranges_program_with_subgroup_coalesce(
    dfa: &CompiledDfa,
    pattern_count: u32,
    max_matches: u32,
    use_subgroup_coalesce: bool,
) -> Program {
    build_ranges_scan(
        PrefilterWidth::Unfiltered,
        dfa,
        pattern_count,
        max_matches,
        use_subgroup_coalesce,
    )
}

/// Fallible variant of [`build_ac_bounded_ranges_program`].
///
/// # Errors
///
/// Returns an actionable error when DFA metadata cannot fit the GPU program's
/// u32 buffer-count ABI.
pub fn try_build_ac_bounded_ranges_program(
    dfa: &CompiledDfa,
    pattern_count: u32,
    max_matches: u32,
) -> Result<Program, String> {
    try_build_ac_bounded_ranges_program_with_subgroup_coalesce(
        dfa,
        pattern_count,
        max_matches,
        true,
    )
}

/// Fallible variant of [`build_ac_bounded_ranges_program_with_subgroup_coalesce`].
///
/// # Errors
///
/// Returns an actionable error when DFA metadata cannot fit the GPU program's
/// u32 buffer-count ABI.
pub fn try_build_ac_bounded_ranges_program_with_subgroup_coalesce(
    dfa: &CompiledDfa,
    pattern_count: u32,
    max_matches: u32,
    use_subgroup_coalesce: bool,
) -> Result<Program, String> {
    try_build_ranges_scan(
        PrefilterWidth::Unfiltered,
        dfa,
        pattern_count,
        max_matches,
        use_subgroup_coalesce,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern::classic_ac::test_dispatch_and_decode::assert_infallible_matches_try;
    use crate::pattern::classic_ac::{classic_ac_bounded_ranges_scan, classic_ac_compile};

    /// Behavioral regression guard: the infallible builder must wire the REAL DFA
    /// metadata (delegating to the `try_` variant's Ok program), never the deleted
    /// degenerate empty-rejecting fallback (state_count=1, output_records_len=0) that
    /// silently dropped every match.
    #[test]
    fn infallible_builder_uses_real_dfa_not_empty_fallback() {
        let ac = classic_ac_compile(&[b"abc", b"de", b"abcd"]);
        let via_infallible =
            build_ac_bounded_ranges_program_with_subgroup_coalesce(&ac.dfa, 3, 128, false);
        let via_try =
            try_build_ac_bounded_ranges_program_with_subgroup_coalesce(&ac.dfa, 3, 128, false)
                .expect("valid DFA must build");
        assert_infallible_matches_try("bounded-ranges", &via_infallible, &via_try, &ac.dfa);
    }

    /// Verify try_build_ac_bounded_ranges_program_with_subgroup_coalesce returns Ok for a valid
    /// small DFA, proving the success path is intact after the panic-on-error fix.
    #[test]
    fn try_build_ac_bounded_ranges_program_ext_succeeds_for_valid_dfa() {
        let ac = classic_ac_compile(&[b"abc", b"de"]);
        let result =
            try_build_ac_bounded_ranges_program_with_subgroup_coalesce(&ac.dfa, 2, 128, false);
        assert!(
            result.is_ok(),
            "try_build must succeed for a valid small DFA: {:?}",
            result.err()
        );
        // Verify the program has the correct buffer shape for the DFA size.
        let program = result.unwrap();
        assert_eq!(
            program.workgroup_size(),
            [128, 1, 1],
            "workgroup size must be [128, 1, 1]"
        );
    }

    /// Verify the CPU reference scan panics (not silently zero-lengths) when
    /// the DFA output_records contain a pid beyond pattern_lengths.len().
    /// Before the fix, pattern_lengths.get(pid).copied().unwrap_or(0) would
    /// silently treat the OOB pid as pat_len=0, producing a zero-length match
    /// at the right position, masking the root cause of the mismatch and
    /// making parity tests impossible to detect the bug.
    #[test]
    #[should_panic]
    fn classic_ac_bounded_ranges_scan_panics_on_oob_pid() {
        use crate::pattern::CompiledDfa;

        // Craft a ClassicAcAutomaton whose output_records contains pid=5
        // but we only supply pattern_lengths of length 3.
        // state 0 -b'A'-> state 1, state 1 accepts pid=5.
        let transitions: Vec<u32> = {
            let mut t = vec![0u32; 2 * 256]; // 2 states
            t[0 * 256 + b'A' as usize] = 1; // state 0 --'A'--> state 1
                                            // state 1 loops to 0 on all other bytes (default 0)
            t
        };
        let accept = vec![0u32, 6u32]; // state 1: accept=6 (pid=5, encoded as 5+1)
        let output_offsets = vec![0u32, 0u32, 1u32]; // state 0: [], state 1: [5]
        let output_records = vec![5u32]; // pid=5

        let dfa = CompiledDfa {
            transitions,
            accept,
            state_count: 2,
            max_pattern_len: 1,
            output_offsets,
            output_records,
        };
        let ac = crate::pattern::classic_ac::ClassicAcAutomaton { dfa };
        // pattern_lengths only has 3 entries (pids 0..2) (pid=5 is OOB).
        // This must panic, not silently produce a zero-length match.
        let _result = classic_ac_bounded_ranges_scan(&ac, &[1u32, 2u32, 3u32], b"A");
    }

    /// Patterns whose output spans differ per candidate position, so some
    /// lanes of a subgroup carry records and some carry none. A pattern set
    /// with a uniform span would hide the divergence these two tests exist to
    /// catch.
    const DIVERGENT_SPANS: &[&[u8]] = &[b"he", b"she", b"his", b"hers"];
    const DIVERGENT_HAYSTACK: &[u8] = b"ushers and his hershey he she";

    fn unfiltered_ranges_scan(coalesce: bool) -> Vec<(u32, u32, u32)> {
        let ac = classic_ac_compile(DIVERGENT_SPANS);
        let lengths =
            crate::pattern::classic_ac::test_dispatch_and_decode::pattern_lengths(DIVERGENT_SPANS);
        let program = build_ac_bounded_ranges_program_with_subgroup_coalesce(
            &ac.dfa,
            DIVERGENT_SPANS.len() as u32,
            256,
            coalesce,
        );
        let mut inputs = crate::pattern::classic_ac::test_dispatch_and_decode::ac_ranges_inputs(
            &ac.dfa,
            DIVERGENT_HAYSTACK,
            &lengths,
        );
        inputs.push(crate::pattern::classic_ac::test_dispatch_and_decode::u32_input(&[0]));
        let outputs = vyre_test_support::test_parity_oracles::eval_bytes(
            "bounded_ranges_coalesce",
            &program,
            inputs,
        );
        let mut decoded =
            crate::pattern::classic_ac::test_dispatch_and_decode::decode_match_triples(&outputs);
        decoded.sort_unstable();
        decoded
    }

    /// The shipped entrypoint selects `use_subgroup_coalesce = true`, so the
    /// reference oracle has to be able to judge that program. Every other
    /// bounded-ranges test passes `false`, which left the default dispatch
    /// shape unjudged.
    #[test]
    fn the_coalesced_scan_reproduces_the_host_oracle() {
        let ac = classic_ac_compile(DIVERGENT_SPANS);
        let lengths =
            crate::pattern::classic_ac::test_dispatch_and_decode::pattern_lengths(DIVERGENT_SPANS);
        let mut expected = classic_ac_bounded_ranges_scan(&ac, &lengths, DIVERGENT_HAYSTACK);
        expected.sort_unstable();
        assert_eq!(unfiltered_ranges_scan(true), expected);
    }

    /// Subgroup coalescing changes how a hit buffer slot is reserved, never
    /// which matches are emitted.
    #[test]
    fn both_coalesce_settings_emit_the_same_matches() {
        assert_eq!(unfiltered_ranges_scan(true), unfiltered_ranges_scan(false));
    }
}
