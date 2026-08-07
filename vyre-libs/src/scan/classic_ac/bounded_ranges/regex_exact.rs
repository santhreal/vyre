use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::region::wrap_anonymous;
use crate::scan::builders::{append_match, append_match_subgroup, load_packed_byte};

/// Build the regex whole-buffer program with exact origin-derived starts.
///
/// The buffer ABI intentionally matches the classic bounded-range program. The
/// `pattern_lengths` buffer remains present for compatibility, but regex starts
/// come from the invocation origin rather than `end - maximum_length`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn regex_exact_ranges_program_ext(
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
    let replay_limit = max_pattern_len.max(1);
    let origin = Expr::var("origin");
    let remaining = Expr::sub(Expr::load(haystack_len, Expr::u32(0)), origin.clone());
    let replay_len = Expr::select(
        Expr::lt(remaining.clone(), Expr::u32(replay_limit)),
        remaining,
        Expr::u32(replay_limit),
    );
    let window_end = Expr::add(origin.clone(), replay_len);
    let (load_step_byte, step_byte) = load_packed_byte(haystack, Expr::var("step"));

    let mut emit_body = vec![Node::let_bind(
        "pattern_id",
        Expr::load(output_records, Expr::var("out_idx")),
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
            emit_body,
        ),
    ];

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

    Program::wrapped(
        vec![
            BufferDecl::storage(haystack, 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(transitions, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(state_count.saturating_mul(256)),
            BufferDecl::storage(output_offsets, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(state_count.saturating_add(1)),
            BufferDecl::storage(output_records, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(output_records_len),
            BufferDecl::storage(pattern_lengths, 4, BufferAccess::ReadOnly, DataType::U32)
                .with_count(pattern_count),
            BufferDecl::storage(haystack_len, 5, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1),
            BufferDecl::read_write(match_count, 6, DataType::U32).with_count(1),
            BufferDecl::output(matches, 7, DataType::U32).with_count(max_matches.saturating_mul(3)),
        ],
        [128, 1, 1],
        vec![wrap_anonymous(
            "vyre-libs::matching::regex_exact_ranges",
            invocation,
        )],
    )
}
