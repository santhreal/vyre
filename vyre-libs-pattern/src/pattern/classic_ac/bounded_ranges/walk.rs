//! The Aho-Corasick transition walk: one dense table step, the flat
//! output-link span it selects, and the bounded suffix replay every
//! `bounded_ranges` builder opens with.
//!
//! Every walk in this module folds its data-derived indices against the live
//! buffer extents, because a transition-table entry is read-only input and
//! nothing constrains it to the state set.

use vyre_foundation::composition::bounded_index;
use vyre_foundation::ir::{Expr, Node};
use vyre_libs_builder::builder::state_machine::TableStateMachineComposer;

use crate::pattern::builders::load_packed_byte;

/// Advance `state` one byte through the dense `state * 256 + byte` transition
/// row, with both operands folded inside the table.
///
/// THE Aho-Corasick transition step. The bounded suffix replay, the anchored
/// forward walk, the per-region admission walk and the unbounded classic walk
/// are each built from this one node, so a change to the table layout reaches
/// all of them at once. `byte` is whatever the caller's haystack encoding
/// yields: a direct element load for an unpacked haystack, or the masked byte
/// [`ac_transition_step_nodes`] unpacks from a u32 word.
///
/// The state this reads is the previous step's table entry, and nothing
/// constrains the contents of a read-only buffer, so `state * 256 + byte`
/// indexes past the table on the step after an entry falls outside the state
/// set. The state extent is the table's own run-time length over the row
/// stride, which is identity for a table whose entries are states and covers
/// every caller without a signature change.
pub(in crate::pattern) fn ac_advance_state_node(transitions: &str, byte: Expr) -> Node {
    let composer = TableStateMachineComposer::new(transitions);
    let state_extent = Expr::div(Expr::buf_len(transitions), Expr::u32(composer.stride));
    composer.bounded_advance_node(state_extent, byte)
}

/// One byte of the walk over a PACKED haystack: unpack the byte at `idx` from
/// its u32 word, then [`ac_advance_state_node`].
pub(in crate::pattern) fn ac_transition_step_nodes(
    haystack: &str,
    transitions: &str,
    idx: Expr,
) -> Vec<Node> {
    let (load_byte, byte) = load_packed_byte(haystack, idx);
    vec![load_byte, ac_advance_state_node(transitions, byte)]
}

/// How a walk introduces the results a later emit reads.
///
/// `Let` is the ordinary shape: the emit runs where the walk ran, inside the
/// admission gate. `Assign` writes into names the caller bound outside that
/// gate, which is what an emit carrying a subgroup collective needs, because a
/// collective reads across lanes and a rejected lane never enters the gate.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::pattern) enum WalkBinding {
    /// Introduce the name here.
    Let,
    /// Write into a name the caller already bound.
    Assign,
}

impl WalkBinding {
    fn bind(self, name: &str, value: Expr) -> Node {
        match self {
            Self::Let => Node::let_bind(name, value),
            Self::Assign => Node::assign(name, value),
        }
    }
}

/// Bind `out_begin`/`out_end` to the flat output-link span of the current
/// `state`. Every walk pairs this with the transition step before emitting, so
/// an `output_offsets` layout change has one place to land.
///
/// `state` is a transition-table entry, so both offsets are read at a
/// data-derived index. Each read is folded against the offset buffer's own
/// run-time length, which is identity for a well-formed table holding one
/// offset per state plus the terminating end, and reads element zero for a
/// state the table does not describe. A `length - 1` extent would wrap for a
/// table declared with no offsets at all.
pub(in crate::pattern) fn ac_output_span_nodes(output_offsets: &str) -> Vec<Node> {
    ac_output_span_nodes_bound(output_offsets, WalkBinding::Let)
}

/// [`ac_output_span_nodes`] under an explicit [`WalkBinding`].
pub(in crate::pattern) fn ac_output_span_nodes_bound(
    output_offsets: &str,
    binding: WalkBinding,
) -> Vec<Node> {
    vec![
        binding.bind(
            "out_begin",
            Expr::load(
                output_offsets,
                bounded_index(Expr::var("state"), Expr::buf_len(output_offsets)),
            ),
        ),
        binding.bind(
            "out_end",
            Expr::load(
                output_offsets,
                bounded_index(
                    Expr::add(Expr::var("state"), Expr::u32(1)),
                    Expr::buf_len(output_offsets),
                ),
            ),
        ),
    ]
}

/// Bounded-window walk prologue for the scan, count and presence builders: bind
/// `state`/`scan_start`/`scan_end`, replay the suffix window
/// `haystack[max(0, i + 1 - max_pattern_len)..=i]` from state 0, and bind the
/// output-link span. Callers append their per-record emit loop.
pub(in crate::pattern) fn bounded_walk_prologue_nodes(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    max_pattern_len: u32,
) -> Vec<Node> {
    bounded_walk_prologue_bound(
        haystack,
        transitions,
        output_offsets,
        max_pattern_len,
        WalkBinding::Let,
    )
}

/// [`bounded_walk_prologue_nodes`] under an explicit [`WalkBinding`].
///
/// `state` and `scan_start` are always introduced here: nothing outside the
/// gate reads them. `scan_end` and the output-link span follow `binding`,
/// because those three are exactly what an emit hoisted out of the gate needs.
pub(in crate::pattern) fn bounded_walk_prologue_bound(
    haystack: &str,
    transitions: &str,
    output_offsets: &str,
    max_pattern_len: u32,
    binding: WalkBinding,
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
        // `end` derives from the live `haystack_len` load, which is not an
        // extent. The step body reads `haystack` through `load_packed_byte`,
        // four bytes per word, so the real ceiling is four times its extent.
        binding.bind(
            "scan_end",
            Expr::min(end, Expr::mul(Expr::buf_len(haystack), Expr::u32(4))),
        ),
        Node::loop_for(
            "step",
            Expr::var("scan_start"),
            Expr::var("scan_end"),
            ac_transition_step_nodes(haystack, transitions, Expr::var("step")),
        ),
    ];
    nodes.extend(ac_output_span_nodes_bound(output_offsets, binding));
    nodes
}

/// Bounded walk whose `matched` nodes run only for candidates that accept
/// (`out_begin < out_end`), so a miss pays the walk and nothing else.
///
/// The region-attributed builders gate on this because the region binary search
/// is pure overhead for a position with no records.
pub(in crate::pattern) fn bounded_walk_matched_nodes(
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
