use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

use super::{ac_output_span_nodes, ac_transition_step_nodes, AcInputBindings};
use crate::pattern::builders::{append_match, append_match_subgroup};

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
    let remaining = Expr::sub(Expr::load(haystack_len, Expr::u32(0)), origin.clone());
    let replay_len = Expr::select(
        Expr::lt(remaining.clone(), Expr::u32(replay_limit)),
        remaining,
        Expr::u32(replay_limit),
    );
    let window_end = Expr::add(origin.clone(), replay_len);

    let mut emit_body = vec![Node::let_bind(
        "pattern_id",
        Expr::load(inputs.output_records, Expr::var("out_idx")),
    )];
    if use_subgroup_coalesce {
        emit_body.extend(append_match_subgroup(
            matches,
            match_count,
            Expr::var("pattern_id"),
            origin.clone(),
            Expr::add(Expr::var("step"), Expr::u32(1)),
            Expr::bool(true),
        ));
    } else {
        emit_body.push(append_match(
            matches,
            match_count,
            Expr::var("pattern_id"),
            origin.clone(),
            Expr::add(Expr::var("step"), Expr::u32(1)),
        ));
    }

    let mut walk_step =
        ac_transition_step_nodes(inputs.haystack, inputs.transitions, Expr::var("step"));
    walk_step.extend(ac_output_span_nodes(inputs.output_offsets));
    walk_step.push(Node::loop_for(
        "out_idx",
        Expr::var("out_begin"),
        Expr::var("out_end"),
        emit_body,
    ));

    let invocation = vec![
        Node::let_bind("origin", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(origin.clone(), Expr::load(haystack_len, Expr::u32(0))),
            vec![
                Node::let_bind("state", Expr::u32(0)),
                Node::let_bind("window_end", window_end),
                Node::loop_for("step", origin, Expr::var("window_end"), walk_step),
            ],
        ),
    ];

    let mut buffers = inputs.decls();
    buffers.reserve(2);
    buffers.push(BufferDecl::read_write(match_count, 6, DataType::U32).with_count(1));
    buffers.push(
        BufferDecl::output(matches, 7, DataType::U32).with_count(max_matches.saturating_mul(3)),
    );

    Program::wrapped(
        buffers,
        [128, 1, 1],
        vec![wrap_anonymous_region(
            "vyre-libs::matching::regex_exact_ranges",
            invocation,
        )],
    )
}
