//! The whole-buffer regex program with origin-derived exact match starts.
//!
//! Its buffer ABI matches the classic bounded-range program on purpose, so a
//! caller can swap the two without re-marshalling.

use crate::pattern::dfa::aho_corasick_programs::OP_ID as AHO_CORASICK_OP_ID;
use crate::pattern::regex_dfa::REGEX_DFA_OP_ID;
use vyre_foundation::composition::wrap_child_region;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Ident, Node, Program};

use super::{
    ac_output_span_nodes, ac_output_span_nodes_bound, ac_transition_step_nodes,
    output_record_loop_node, uniform_output_record_loop_nodes, AcInputBindings, WalkBinding,
    RECORD_ACTIVE,
};
use crate::pattern::builders::{append_match, append_match_subgroup};
use vyre_foundation::composition::bounded_index_when;

/// Build the regex whole-buffer program with exact origin-derived starts.
///
/// The buffer ABI intentionally matches the classic bounded-range program. The
/// `pattern_lengths` buffer remains present for compatibility, but regex starts
/// come from the invocation origin rather than `end - maximum_length`.
pub(in crate::pattern) fn regex_exact_ranges_program(
    inputs: AcInputBindings<'_>,
    match_count: &str,
    matches: &str,
    max_matches: u32,
    max_pattern_len: u32,
    use_subgroup_coalesce: bool,
) -> Program {
    let haystack_len = inputs.haystack_len;
    let replay_limit = max_pattern_len.max(1);
    let origin = Expr::var("origin");
    // `haystack_len` is a live load, not an extent. The step body reads
    // `haystack` through `load_packed_byte`, four bytes per word, so the real
    // ceiling is four times its extent.
    let scan_limit = Expr::min(
        Expr::load(haystack_len, Expr::u32(0)),
        Expr::mul(Expr::buf_len(inputs.haystack), Expr::u32(4)),
    );
    if use_subgroup_coalesce {
        let emit_body = append_match_subgroup(
            matches,
            match_count,
            Expr::var("pattern_id"),
            origin.clone(),
            Expr::add(Expr::var("step"), Expr::u32(1)),
            Expr::var(RECORD_ACTIVE),
        );
        return Program::wrapped(
            regex_exact_buffers(inputs, match_count, matches, max_matches),
            [128, 1, 1],
            vec![wrap_child_region(
                AHO_CORASICK_OP_ID,
                Ident::from(REGEX_DFA_OP_ID),
                uniform_regex_walk(inputs, scan_limit, replay_limit, emit_body),
            )],
        );
    }

    let emit_body: Vec<Node> = vec![append_match(
        matches,
        match_count,
        Expr::var("pattern_id"),
        origin.clone(),
        Expr::add(Expr::var("step"), Expr::u32(1)),
    )];
    let remaining = Expr::sub(Expr::var("scan_limit"), origin.clone());
    let replay_len = Expr::select(
        Expr::lt(remaining.clone(), Expr::u32(replay_limit)),
        remaining,
        Expr::u32(replay_limit),
    );
    let window_end = Expr::add(origin.clone(), replay_len);

    let mut walk_step =
        ac_transition_step_nodes(inputs.haystack, inputs.transitions, Expr::var("step"));
    walk_step.extend(ac_output_span_nodes(inputs.output_offsets));
    walk_step.push(output_record_loop_node(inputs.output_records, emit_body));

    let invocation = vec![
        Node::let_bind("origin", Expr::LogicalIndex { axis: 0 }),
        Node::let_bind("scan_limit", scan_limit),
        Node::if_then(
            Expr::lt(origin.clone(), Expr::var("scan_limit")),
            vec![
                Node::let_bind("state", Expr::u32(0)),
                Node::let_bind("window_end", window_end),
                Node::loop_for("step", origin, Expr::var("window_end"), walk_step),
            ],
        ),
    ];

    Program::wrapped(
        regex_exact_buffers(inputs, match_count, matches, max_matches),
        [128, 1, 1],
        vec![wrap_child_region(
            AHO_CORASICK_OP_ID,
            Ident::from(REGEX_DFA_OP_ID),
            invocation,
        )],
    )
}

/// The buffer list both emit strategies bind: the shared AC inputs, the match
/// counter at 6, the triple sink at 7.
fn regex_exact_buffers(
    inputs: AcInputBindings<'_>,
    match_count: &str,
    matches: &str,
    max_matches: u32,
) -> Vec<BufferDecl> {
    let mut buffers = inputs.decls();
    buffers.reserve(2);
    buffers.push(BufferDecl::read_write(match_count, 6, DataType::U32).with_count(1));
    buffers.push(
        BufferDecl::output(matches, 7, DataType::U32).with_count(max_matches.saturating_mul(3)),
    );
    buffers
}

/// The whole-buffer walk with every lane of the subgroup running the same
/// number of iterations, for the coalesced emit.
///
/// A subgroup collective reads across lanes, so it has to be reached by all of
/// them. Two things make the ordinary walk diverge: a lane whose origin is
/// past the live length skips the walk outright, and a lane near the end
/// replays a shorter window than its peers. Both become a per-lane
/// `window_len` of zero or less than the subgroup maximum, and the loop runs
/// to that maximum with `step_active` deciding whether an iteration means
/// anything. An inactive iteration reads byte zero and walks zero output
/// records, so it advances no match and still meets its peers at the
/// collective.
fn uniform_regex_walk(
    inputs: AcInputBindings<'_>,
    scan_limit: Expr,
    replay_limit: u32,
    emit_body: Vec<Node>,
) -> Vec<Node> {
    let origin = Expr::var("origin");
    let remaining = Expr::sub(Expr::var("scan_limit"), origin.clone());
    let window_len = Expr::select(
        Expr::lt(origin.clone(), Expr::var("scan_limit")),
        Expr::min(remaining, Expr::u32(replay_limit)),
        Expr::u32(0),
    );

    let mut step_body = vec![
        Node::let_bind(
            "step_active",
            Expr::lt(Expr::var("k"), Expr::var("window_len")),
        ),
        Node::let_bind(
            "step",
            bounded_index_when(
                Expr::var("step_active"),
                Expr::add(origin.clone(), Expr::var("k")),
            ),
        ),
    ];
    step_body.extend(ac_transition_step_nodes(
        inputs.haystack,
        inputs.transitions,
        Expr::var("step"),
    ));
    step_body.extend(ac_output_span_nodes_bound(
        inputs.output_offsets,
        WalkBinding::Assign,
    ));
    step_body.extend(uniform_output_record_loop_nodes(
        inputs.output_records,
        Some(Expr::var("step_active")),
        emit_body,
    ));

    vec![
        Node::let_bind("origin", Expr::LogicalIndex { axis: 0 }),
        Node::let_bind("scan_limit", scan_limit),
        Node::let_bind("window_len", window_len),
        Node::let_bind("uniform_steps", Expr::subgroup_max(Expr::var("window_len"))),
        Node::let_bind("state", Expr::u32(0)),
        Node::let_bind("out_begin", Expr::u32(0)),
        Node::let_bind("out_end", Expr::u32(0)),
        Node::loop_for("k", Expr::u32(0), Expr::var("uniform_steps"), step_body),
    ]
}
