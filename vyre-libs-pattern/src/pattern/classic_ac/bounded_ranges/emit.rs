//! What a walk does once it has an output-link span: attribute the position to
//! a region, iterate the flat record span, and write a match triple, a
//! presence bit or a pattern-bitset bit.
//!
//! The record span is read in exactly two places here, the per-lane loop and
//! its subgroup-uniform counterpart, so the `output_records` layout and the
//! clamp of `out_end` against the extent it indexes have one owner.

use vyre_foundation::composition::{bounded_index, bounded_index_when};
use vyre_foundation::ir::{Expr, Node};
use vyre_libs_builder::builder::trip_count::clamped_by_extents;

/// The region binary-search PROLOGUE shared by every region-attributed walk in
/// `scan/`: the presence-only and fused presence+positions bounded builders here,
/// and the anchored per-region walk in
/// [`crate::pattern::regex_region_admission`]. Computes `rs_pos = i + region_base`
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
pub(in crate::pattern) fn region_search_prologue_nodes(
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

/// Walk the flat `output_records` span bound by [`ac_output_span_nodes`](super::walk::ac_output_span_nodes),
/// binding `pattern_id` for each record before running `per_record`.
///
/// Every AC emit path iterates this one span identically and differs only in
/// what it does with `pattern_id`, so the record layout is read in one place.
/// That also makes this the one place `out_end`, which comes from the compiled
/// table's `output_offsets`, is clamped to the extent it indexes.
pub(in crate::pattern) fn output_record_loop_node(
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
        clamped_by_extents(Expr::var("out_end"), output_records, []),
        body,
    )
}

/// Name a lane binds to whether its current record-loop iteration addresses a
/// real record. Every emit that runs under
/// [`uniform_output_record_loop_nodes`] gates its writes on it.
pub(in crate::pattern) const RECORD_ACTIVE: &str = "out_active";

/// Subgroup-uniform counterpart of [`output_record_loop_node`], for an emit
/// that runs a subgroup collective per record.
///
/// A collective reads a value out of a peer lane, so every lane of the
/// subgroup has to reach it with that value bound. The record span is
/// per-lane, so under [`output_record_loop_node`] a lane whose span is empty
/// never enters the body, and the peer read of a name that lane never bound
/// has no result at all: the program is not merely wrong, it cannot be
/// evaluated. This walks `subgroup_max(span)` iterations in every lane of the
/// subgroup and binds [`RECORD_ACTIVE`] to `out_k < out_span`, which puts the
/// collective in uniform control flow and leaves the emit predicate to
/// `per_record`.
///
/// The record index is folded to zero on an inactive iteration, so a lane
/// walking another lane's trip count reads a slot that exists and emits
/// nothing.
///
/// `active` names a caller predicate that says whether the span itself is
/// real. A walk that runs its steps uniformly reaches this with a span left
/// over from a step its lane never took, and zeroing the span here keeps
/// [`ac_output_span_nodes`](super::walk::ac_output_span_nodes) the only writer of `out_begin` and `out_end`.
pub(in crate::pattern) fn uniform_output_record_loop_nodes(
    output_records: &str,
    active: Option<Expr>,
    per_record: Vec<Node>,
) -> Vec<Node> {
    let span_end = Expr::var("out_span_end");
    let begin = Expr::var("out_begin");
    let mut body = vec![
        Node::let_bind(
            RECORD_ACTIVE,
            Expr::lt(Expr::var("out_k"), Expr::var("out_span")),
        ),
        Node::let_bind(
            "out_idx",
            bounded_index_when(
                Expr::var(RECORD_ACTIVE),
                Expr::add(begin.clone(), Expr::var("out_k")),
            ),
        ),
        Node::let_bind(
            "pattern_id",
            Expr::load(output_records, Expr::var("out_idx")),
        ),
    ];
    body.extend(per_record);
    vec![
        Node::let_bind(
            "out_span_end",
            clamped_by_extents(Expr::var("out_end"), output_records, []),
        ),
        Node::let_bind(
            "out_span",
            match active {
                Some(active) => Expr::select(
                    Expr::and(active, Expr::lt(begin.clone(), span_end.clone())),
                    Expr::sub(span_end, begin),
                    Expr::u32(0),
                ),
                None => Expr::select(
                    Expr::lt(begin.clone(), span_end.clone()),
                    Expr::sub(span_end, begin),
                    Expr::u32(0),
                ),
            },
        ),
        Node::let_bind("out_uniform", Expr::subgroup_max(Expr::var("out_span"))),
        Node::loop_for("out_k", Expr::u32(0), Expr::var("out_uniform"), body),
    ]
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
pub(in crate::pattern) fn pattern_bitset_or_node(
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
pub(in crate::pattern) fn presence_bit_write_node(presence: &str, row_base: Option<&str>) -> Node {
    pattern_bitset_or_node(presence, row_base, "_vyre_presence_prev")
}

/// Bind `pat_len` and the match start for the pattern accepted at `scan_end`.
///
/// The subtraction is floored at zero: a pattern longer than the window walked
/// so far would wrap, and the emitted span has to stay inside the haystack.
///
/// `pattern_id` is an `output_records` entry, so the length read is folded into
/// `pattern_lengths`, which holds one length per compiled pattern.
pub(in crate::pattern) fn match_span_start_nodes(pattern_lengths: &str) -> Vec<Node> {
    vec![
        Node::let_bind(
            "pat_len",
            Expr::load(
                pattern_lengths,
                bounded_index(Expr::var("pattern_id"), Expr::buf_len(pattern_lengths)),
            ),
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
