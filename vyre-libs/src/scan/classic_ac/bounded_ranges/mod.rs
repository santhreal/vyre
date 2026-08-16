//! The bounded-window Aho-Corasick scan builders, and the AC walk itself.
//!
//! This module owns the transition walk for the whole crate: the dense
//! `state = transitions[state * 256 + byte]` step, the flat output-link span,
//! the bounded suffix replay, the candidate-end byte gate, the per-region binary
//! search, the range-bound arithmetic (`bounded_walk_prologue_nodes`,
//! `match_span_start_nodes`, `ac_ranges_output_records_len`) and the fail-closed
//! rejection path (`ac_ranges_program_or_fail_closed`). Every other AC builder
//! here and under `scan/` projects from those primitives and supplies only what
//! genuinely differs: its admission predicate, its emission, its prefilter
//! shape. The gate widths and the program assembly built on top of them belong
//! to the `prefilter` submodule, and the ungated scan below is one of its rows.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::scan::builders::{
    append_match, append_match_subgroup, load_packed_byte, load_packed_byte_expr,
};

use vyre_primitives::matching::CompiledDfa;

#[cfg(any(test, feature = "cpu-parity"))]
use super::ClassicAcAutomaton;

mod prefilter;
#[cfg(all(feature = "matching-regex", feature = "matching-dfa"))]
mod regex_exact;

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
use prefilter::{
    build_ranges_scan, ranges_scan_program, try_build_ranges_scan, PrefilterGate, PrefilterWidth,
};
#[cfg(all(feature = "matching-regex", feature = "matching-dfa"))]
pub(in crate::scan) use regex_exact::regex_exact_ranges_program;

/// Advance `state` one byte through the dense `state * 256 + byte` transition
/// row.
///
/// THE Aho-Corasick transition step. The bounded suffix replay, the anchored
/// forward walk, the per-region admission walk and the unbounded classic walk
/// are each built from this one node, so a change to the table layout reaches
/// all of them at once. `byte` is whatever the caller's haystack encoding
/// yields: a direct element load for an unpacked haystack, or the masked byte
/// [`ac_transition_step_nodes`] unpacks from a u32 word.
pub(in crate::scan) fn ac_advance_state_node(transitions: &str, byte: Expr) -> Node {
    Node::assign(
        "state",
        Expr::load(
            transitions,
            Expr::add(Expr::mul(Expr::var("state"), Expr::u32(256)), byte),
        ),
    )
}

/// One byte of the walk over a PACKED haystack: unpack the byte at `idx` from
/// its u32 word, then [`ac_advance_state_node`].
pub(in crate::scan) fn ac_transition_step_nodes(
    haystack: &str,
    transitions: &str,
    idx: Expr,
) -> Vec<Node> {
    let (load_byte, byte) = load_packed_byte(haystack, idx);
    vec![load_byte, ac_advance_state_node(transitions, byte)]
}

/// Bind `out_begin`/`out_end` to the flat output-link span of the current
/// `state`. Every walk pairs this with the transition step before emitting, so
/// an `output_offsets` layout change has one place to land.
pub(in crate::scan) fn ac_output_span_nodes(output_offsets: &str) -> Vec<Node> {
    vec![
        Node::let_bind("out_begin", Expr::load(output_offsets, Expr::var("state"))),
        Node::let_bind(
            "out_end",
            Expr::load(output_offsets, Expr::add(Expr::var("state"), Expr::u32(1))),
        ),
    ]
}

/// Bounded-window walk prologue for the scan, count and presence builders: bind
/// `state`/`scan_start`/`scan_end`, replay the suffix window
/// `haystack[max(0, i + 1 - max_pattern_len)..=i]` from state 0, and bind the
/// output-link span. Callers append their per-record emit loop.
pub(in crate::scan) fn bounded_walk_prologue_nodes(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    max_pattern_len: u32,
) -> Vec<Node> {
    let max_pattern_len = max_pattern_len.max(1);
    let i = Expr::var("i");
    let end = Expr::add(i.clone(), Expr::u32(1));
    let scan_start = Expr::select(
        Expr::lt(i, Expr::u32(max_pattern_len - 1)),
        Expr::u32(0),
        Expr::sub(end.clone(), Expr::u32(max_pattern_len)),
    );
    let mut nodes = vec![
        Node::let_bind("state", Expr::u32(0)),
        Node::let_bind("scan_start", scan_start),
        Node::let_bind("scan_end", end),
        Node::loop_for(
            "step",
            Expr::var("scan_start"),
            Expr::var("scan_end"),
            ac_transition_step_nodes(haystack, transitions, Expr::var("step")),
        ),
    ];
    nodes.extend(ac_output_span_nodes(output_offsets));
    nodes
}

/// The candidate-end byte gate every prefiltered AC program opens with: bind the
/// invocation index `i`, bound it against the live `haystack_len`, unpack the
/// candidate byte, and run `accepted` only when that byte's bit is set in the
/// 8-word `candidate_end_mask`. The bound `candidate_byte` stays in scope so a
/// deeper suffix gate can reuse it instead of unpacking the same byte twice.
pub(in crate::scan) fn candidate_end_gate_nodes(
    haystack: &str,
    haystack_len: &str,
    candidate_end_mask: &str,
    accepted: Vec<Node>,
) -> Vec<Node> {
    let i = Expr::var("i");
    vec![
        Node::let_bind("i", Expr::InvocationId { axis: 0 }),
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
pub(in crate::scan) fn classic_ac_dfa_buffer_decls(
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
pub(in crate::scan) struct AcInputBindings<'a> {
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
    /// The shared inputs under the six buffer names every bounded-ranges builder
    /// threads through, in binding order, plus the three DFA-derived counts that
    /// size bindings 1-4.
    ///
    /// Taking the names as one array is what keeps the field list from being
    /// respelled at every call site: the builders here bind the identical six
    /// names, and a struct literal spelling them out is nine lines of the same
    /// text in each.
    pub(in crate::scan) const fn new(
        names: [&'a str; 6],
        state_count: u32,
        output_records_len: u32,
        pattern_count: u32,
    ) -> Self {
        let [haystack, transitions, output_offsets, output_records, pattern_lengths, haystack_len] =
            names;
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
    pub(in crate::scan) fn decls(&self) -> Vec<BufferDecl> {
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
pub(in crate::scan) fn ac_ranges_output_records_len(
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
pub(in crate::scan) fn ac_ranges_program_or_fail_closed(
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

/// Build a Program that scans `haystack` for any AC match and emits
/// `(pattern_id, start, end)` triples through the canonical
/// [`append_match`] hit buffer. Pairs with
/// the product-side haystack packer: each invocation `i`
/// corresponds to byte position `i` of the
/// **unpacked** haystack, but loads from the packed u32 buffer via
/// [`load_packed_byte_expr`](crate::scan::builders::load_packed_byte_expr).
///
/// Buffer layout (bindings 0..7):
///
/// | binding | name | access | element shape |
/// |---|---|---|---|
/// | 0 | `haystack`        | ReadOnly  | packed u32, 4 bytes / word |
/// | 1 | `transitions`     | ReadOnly  | `state_count * 256` u32    |
/// | 2 | `output_offsets`  | ReadOnly  | `state_count + 1` u32      |
/// | 3 | `output_records`  | ReadOnly  | `output_records_len` u32   |
/// | 4 | `pattern_lengths` | ReadOnly  | `pattern_count` u32        |
/// | 5 | `haystack_len`    | ReadOnly  | 1 u32 (byte length)        |
/// | 6 | `match_count`     | ReadWrite | 1 u32 (atomic)             |
/// | 7 | `matches`         | Output    | `max_matches * 3` u32      |
///
/// Each invocation `i` replays the suffix window
/// `haystack[max(0, i+1-max_pattern_len)..=i]` from state 0, then
/// emits every `(pid, start, end)` triple that accepts at `i`. The
/// scan window cap is the only difference from the unbounded walk:
/// `max_pattern_len` must be greater than or equal to the longest
/// entry in `pattern_lengths`, or matches longer than the window are
/// invisible because the walk never sees their first byte.
#[must_use]
#[allow(clippy::too_many_arguments)]
fn classic_ac_bounded_ranges_program(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    output_records: &str,
    pattern_lengths: &str,
    haystack_len: &str,
    match_count: &str,
    matches: &str,
    state_count: u32,
    output_records_len: u32,
    pattern_count: u32,
    max_matches: u32,
    max_pattern_len: u32,
) -> Program {
    classic_ac_bounded_ranges_program_with_subgroup_coalesce(
        haystack,
        transitions,
        output_offsets,
        output_records,
        pattern_lengths,
        haystack_len,
        match_count,
        matches,
        state_count,
        output_records_len,
        pattern_count,
        max_matches,
        max_pattern_len,
        true,
    )
}

/// Variant of [`classic_ac_bounded_ranges_program`] with explicit
/// control over the match-append strategy.
///
/// Set `use_subgroup_coalesce = true` for `append_match_subgroup`
/// (Innovation I.17, one atomic per subgroup leader, the default).
/// Set `false` for the simpler `append_match` (one atomic per lane
/// per hit). Use the `false` variant on backends whose IR lowering
/// can't emit `subgroup_ballot`/`subgroup_shuffle`.
#[must_use]
#[allow(clippy::too_many_arguments)]
fn classic_ac_bounded_ranges_program_with_subgroup_coalesce(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    output_records: &str,
    pattern_lengths: &str,
    haystack_len: &str,
    match_count: &str,
    matches: &str,
    state_count: u32,
    output_records_len: u32,
    pattern_count: u32,
    max_matches: u32,
    max_pattern_len: u32,
    use_subgroup_coalesce: bool,
) -> Program {
    ranges_scan_program(
        PrefilterGate::unfiltered(),
        AcInputBindings::new(
            [
                haystack,
                transitions,
                output_offsets,
                output_records,
                pattern_lengths,
                haystack_len,
            ],
            state_count,
            output_records_len,
            pattern_count,
        ),
        match_count,
        matches,
        max_matches,
        max_pattern_len,
        use_subgroup_coalesce,
    )
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
) -> Vec<Node> {
    let mut per_record = match_span_start_nodes(pattern_lengths);
    if use_subgroup_coalesce {
        per_record.extend(append_match_subgroup(
            matches,
            match_count,
            Expr::var("pattern_id"),
            Expr::var("match_start"),
            Expr::var("scan_end"),
            Expr::bool(true),
        ));
    } else {
        per_record.push(append_match(
            matches,
            match_count,
            Expr::var("pattern_id"),
            Expr::var("match_start"),
            Expr::var("scan_end"),
        ));
    }
    let mut nodes =
        bounded_walk_prologue_nodes(haystack, transitions, output_offsets, max_pattern_len);
    nodes.push(output_record_loop_node(output_records, per_record));
    nodes
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

/// The region binary-search PROLOGUE shared by every region-attributed walk in
/// `scan/`: the presence-only and fused presence+positions bounded builders here,
/// and the anchored per-region walk in
/// [`crate::scan::regex_region_admission`]. Computes `rs_pos = i + region_base`
/// (the GLOBAL byte position so a sharded dispatch attributes against the
/// whole-batch region table), binary-searches `region_starts` for the largest
/// region whose start `<= rs_pos`, and binds `rs_base = region * presence_words`
/// (the per-region presence-row offset). The caller appends its own per-record
/// emit loop after these nodes.
///
/// The row stride is floored at one word to match every presence-word helper in
/// the crate (`presence_bitmap_words`, `presence_by_region_words`,
/// `regex_admission_presence_words`): a zero stride would alias every region
/// onto row 0 and report a batch-wide bitmap as a per-region one.
///
/// The `rs_mid - 1` arm can underflow to `u32::MAX` on the rejected `select`
/// branch; it is discarded harmlessly (`rs_mid == 0` only when
/// `rs_lo == rs_hi == 0`, where `region_starts[0] == 0 <= rs_pos` forces the
/// `cond` arm). One source of truth for the lookup keeps the builders
/// bit-identical by construction.
pub(in crate::scan) fn region_search_prologue_nodes(
    region_starts: &str,
    region_base: &str,
    presence_words: u32,
    log2_max_regions: u32,
) -> Vec<Node> {
    vec![
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
            Expr::mul(Expr::var("rs_lo"), Expr::u32(presence_words.max(1))),
        ),
    ]
}

/// Walk the flat `output_records` span bound by [`ac_output_span_nodes`],
/// binding `pattern_id` for each record before running `per_record`.
///
/// Every AC emit path iterates this one span identically and differs only in
/// what it does with `pattern_id`, so the record layout is read in one place.
pub(in crate::scan) fn output_record_loop_node(
    output_records: &str,
    per_record: Vec<Node>,
) -> Node {
    let mut body = vec![Node::let_bind(
        "pattern_id",
        Expr::load(output_records, Expr::var("out_idx")),
    )];
    body.extend(per_record);
    Node::loop_for(
        "out_idx",
        Expr::var("out_begin"),
        Expr::var("out_end"),
        body,
    )
}

/// Set this pattern's bit in a per-pattern bitset:
/// `bitset[row_base + (pattern_id >> 5)] |= 1u32 << (pattern_id & 31)`.
///
/// `row_base` names the per-region row offset bound by
/// [`region_search_prologue_nodes`]; `None` writes a single batch-wide bitmap.
/// `prev_binding` receives the previous value, discarded, so the atomic
/// read-modify-write is emitted as a side-effecting statement, the same idiom as
/// `append_match`'s `_vyre_match_slot`. Setting the bit is idempotent, which is
/// what lets concurrent lanes hitting one pattern skip the counter and the
/// per-hit serialization the triple-append path pays.
pub(in crate::scan) fn pattern_bitset_or_node(
    bitset: &str,
    row_base: Option<&str>,
    prev_binding: &str,
) -> Node {
    let word = Expr::shr(Expr::var("pattern_id"), Expr::u32(5));
    let word = match row_base {
        Some(base) => Expr::add(Expr::var(base), word),
        None => word,
    };
    Node::let_bind(
        prev_binding,
        Expr::atomic_or(
            bitset,
            word,
            Expr::shl(
                Expr::u32(1),
                Expr::bitand(Expr::var("pattern_id"), Expr::u32(31)),
            ),
        ),
    )
}

/// [`pattern_bitset_or_node`] into the presence bitmap, under the binding name
/// every presence builder in `scan/` emits.
pub(in crate::scan) fn presence_bit_write_node(presence: &str, row_base: Option<&str>) -> Node {
    pattern_bitset_or_node(presence, row_base, "_vyre_presence_prev")
}

/// Bind `pat_len` and the match start for the pattern accepted at `scan_end`.
///
/// The subtraction is floored at zero: a pattern longer than the window walked
/// so far would wrap, and the emitted span has to stay inside the haystack.
pub(in crate::scan) fn match_span_start_nodes(pattern_lengths: &str) -> Vec<Node> {
    vec![
        Node::let_bind(
            "pat_len",
            Expr::load(pattern_lengths, Expr::var("pattern_id")),
        ),
        Node::let_bind(
            "match_start",
            Expr::select(
                Expr::lt(Expr::var("scan_end"), Expr::var("pat_len")),
                Expr::u32(0),
                Expr::sub(Expr::var("scan_end"), Expr::var("pat_len")),
            ),
        ),
    ]
}

/// Bounded walk whose `matched` nodes run only for candidates that accept
/// (`out_begin < out_end`), so a miss pays the walk and nothing else.
///
/// The region-attributed builders gate on this because the region binary search
/// is pure overhead for a position with no records.
pub(in crate::scan) fn bounded_walk_matched_nodes(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    max_pattern_len: u32,
    matched: Vec<Node>,
) -> Vec<Node> {
    let mut nodes =
        bounded_walk_prologue_nodes(haystack, transitions, output_offsets, max_pattern_len);
    nodes.push(Node::if_then(
        Expr::lt(Expr::var("out_begin"), Expr::var("out_end")),
        matched,
    ));
    nodes
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
/// [`classic_ac_bounded_ranges_program`]: identical buffer layout
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

/// CPU reference for [`classic_ac_bounded_ranges_program`]. Returns
/// `(pattern_id, start, end)` triples reconstructed from
/// `output_records` plus the pattern length table.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn classic_ac_bounded_ranges_scan(
    ac: &ClassicAcAutomaton,
    pattern_lengths: &[u32],
    haystack: &[u8],
) -> Vec<(u32, u32, u32)> {
    let dfa = &ac.dfa;
    let mut state = 0u32;
    let mut out = Vec::new();
    for (pos, &b) in haystack.iter().enumerate() {
        state = dfa.transitions[(state as usize) * 256 + (b as usize)];
        let begin = dfa.output_offsets[state as usize] as usize;
        let end_off = dfa.output_offsets[state as usize + 1] as usize;
        for &pid in &dfa.output_records[begin..end_off] {
            // Index directly so an OOB pid panics here rather than silently
            // producing a zero-length hit. A mismatch between pattern_count
            // and the actual max pid in output_records is a caller bug; the
            // GPU kernel does an unchecked load that is UB on the same input,
            // so the CPU reference must fail loud-and-early instead of clamping.
            let pat_len = pattern_lengths[pid as usize];
            let end_pos = (pos as u32).saturating_add(1);
            let start = end_pos.saturating_sub(pat_len);
            out.push((pid, start, end_pos));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scan::classic_ac::classic_ac_compile;
    use crate::scan::classic_ac::test_dispatch_and_decode::assert_infallible_matches_try;

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
        use vyre_primitives::matching::CompiledDfa;

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
        let ac = crate::scan::classic_ac::ClassicAcAutomaton { dfa };
        // pattern_lengths only has 3 entries (pids 0..2) (pid=5 is OOB).
        // This must panic, not silently produce a zero-length match.
        let _result = classic_ac_bounded_ranges_scan(&ac, &[1u32, 2u32, 3u32], b"A");
    }
}
