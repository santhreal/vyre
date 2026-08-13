//! Token-paste branch builder for macro replacement.

use crate::parsing::c::lex::tokens::TOK_COMMA;
use crate::parsing::c::preprocess::materialization::*;
use vyre_foundation::ir::{Expr, Node};

use super::helpers::*;
use super::*;

/// Divergences between the token-only and materialized `##` branches.
///
/// Both branches validate the same two operand-presence conditions, decode the
/// right operand from the same replacement slot, synthesize the pasted token
/// type the same way, and copy the rest of a multi-token right argument through
/// the same bounded loop. They diverge in what the right operand carries (a
/// token type alone, or a token type plus a byte span), in what an empty right
/// argument means, and in the emitter used for each copied token.
pub(super) struct PasteBranchSpec<'a> {
    /// Replacement parameter-index table.
    pub(super) macro_replacement_params: &'a str,
    /// Output token-type stream.
    pub(super) out_tok_types: &'a str,
    /// Token count bounding the rest-of-argument copy loop.
    pub(super) num_tokens: Expr,
    /// Binds and assigns the right operand from the replacement slot.
    pub(super) resolve_rhs: Vec<Node>,
    /// Trap raised when no paste rule covers the operand pair.
    pub(super) synth_trap: &'static str,
    /// Appends the right operand's bytes to the pasted token. Empty for the
    /// token-only kernel, which carries no byte arena.
    pub(super) append_rhs_bytes: Vec<Node>,
    /// Selects the rows whose right operand came from a macro argument, and so
    /// may contribute tokens past the first.
    pub(super) rhs_rest_guard: Expr,
    /// Loop body copying one token past the first of the right argument.
    pub(super) rhs_rest_copy: Vec<Node>,
    /// Taken when the right operand is empty. `None` keeps the token-only
    /// kernel's policy, where an empty argument already trapped during
    /// resolution and no empty path is reachable.
    pub(super) empty_rhs: Option<Vec<Node>>,
}

/// Build the `##` branch of a function-like macro replacement.
pub(super) fn emit_function_paste_branch(spec: PasteBranchSpec<'_>) -> Vec<Node> {
    let mut paste = vec![
        Node::if_then(
            Expr::eq(Expr::var("named_out_idx"), Expr::u32(0)),
            vec![Node::trap(
                Expr::var("named_repl_i"),
                "function-like-token-paste-missing-left-token",
            )],
        ),
        Node::if_then(
            Expr::ge(
                Expr::add(Expr::var("named_repl_i"), Expr::u32(1)),
                Expr::var("named_repl_size"),
            ),
            vec![Node::trap(
                Expr::var("named_repl_i"),
                "function-like-token-paste-missing-right-token",
            )],
        ),
        Node::let_bind(
            "macro_paste_next_offset",
            Expr::add(
                Expr::var("named_macro_idx"),
                Expr::add(Expr::var("named_repl_i"), Expr::u32(1)),
            ),
        ),
        Node::let_bind(
            "macro_paste_next_param",
            Expr::load(
                spec.macro_replacement_params,
                Expr::var("macro_paste_next_offset"),
            ),
        ),
    ];
    paste.extend(spec.resolve_rhs);

    let mut nonempty_rhs = vec![
        Node::let_bind(
            "macro_paste_left_tok",
            Expr::load(
                spec.out_tok_types,
                Expr::sub(Expr::var("named_out_idx"), Expr::u32(1)),
            ),
        ),
        Node::let_bind(
            "macro_paste_synth_tok",
            synthesized_paste_token(
                Expr::var("macro_paste_left_tok"),
                Expr::var("macro_paste_right_tok"),
            ),
        ),
        Node::if_then(
            Expr::eq(
                Expr::var("macro_paste_synth_tok"),
                Expr::u32(EMPTY_MACRO_SLOT),
            ),
            vec![Node::trap(
                Expr::var("macro_paste_right_tok"),
                spec.synth_trap,
            )],
        ),
        Node::store(
            spec.out_tok_types,
            Expr::sub(Expr::var("named_out_idx"), Expr::u32(1)),
            Expr::var("macro_paste_synth_tok"),
        ),
    ];
    nonempty_rhs.extend(spec.append_rhs_bytes);
    nonempty_rhs.push(Node::if_then(
        spec.rhs_rest_guard,
        vec![Node::loop_for(
            "macro_paste_rhs_rest_rel",
            Expr::u32(1),
            spec.num_tokens,
            vec![Node::if_then(
                Expr::lt(
                    Expr::add(
                        Expr::var("macro_paste_arg_start"),
                        Expr::var("macro_paste_rhs_rest_rel"),
                    ),
                    Expr::var("macro_paste_arg_end"),
                ),
                spec.rhs_rest_copy,
            )],
        )],
    ));
    nonempty_rhs.push(Node::assign("named_skip_repl", Expr::u32(1)));

    match spec.empty_rhs {
        Some(empty_rhs) => paste.push(Node::if_then_else(
            Expr::eq(Expr::var("macro_paste_right_len"), Expr::u32(0)),
            empty_rhs,
            nonempty_rhs,
        )),
        None => paste.extend(nonempty_rhs),
    }
    paste
}

/// Buffer and bound inputs of the materialized `##` branch.
pub(super) struct MaterializedPasteBranchSpec<'a> {
    /// Input token-type stream.
    pub(super) in_tok_types: &'a str,
    /// Input token start offsets.
    pub(super) in_tok_starts: &'a str,
    /// Input token byte lengths.
    pub(super) in_tok_lens: &'a str,
    /// Input source byte arena.
    pub(super) source_words: &'a str,
    /// Element packing of `source_words`.
    pub(super) source_layout: MacroByteLayout,
    /// Replacement token-type table.
    pub(super) macro_vals: &'a str,
    /// Replacement parameter-index table.
    pub(super) macro_replacement_params: &'a str,
    /// Replacement token start offsets.
    pub(super) macro_replacement_starts: &'a str,
    /// Replacement token byte lengths.
    pub(super) macro_replacement_lens: &'a str,
    /// Replacement source byte arena.
    pub(super) macro_replacement_words: &'a str,
    /// Element packing of `macro_replacement_words`.
    pub(super) macro_replacement_layout: MacroByteLayout,
    /// Output token-type stream.
    pub(super) out_tok_types: &'a str,
    /// Output token start offsets.
    pub(super) out_tok_starts: &'a str,
    /// Output token byte lengths.
    pub(super) out_tok_lens: &'a str,
    /// Output source byte arena.
    pub(super) out_source_words: &'a str,
    /// Workgroup argument start bounds.
    pub(super) macro_arg_starts: &'a str,
    /// Workgroup argument end bounds.
    pub(super) macro_arg_ends: &'a str,
    /// Input token count.
    pub(super) num_tokens: Expr,
    /// Length of the input source arena.
    pub(super) source_len: Expr,
    /// Length of the replacement source arena.
    pub(super) macro_replacement_source_len: Expr,
    /// Output token capacity.
    pub(super) max_out_tokens: u32,
    /// Output byte capacity.
    pub(super) max_out_source_bytes: u32,
}

/// Build the materialized `##` branch, which carries byte spans alongside
/// token types and implements the GNU comma-swallow rule for an empty right
/// argument.
pub(super) fn emit_materialized_function_paste_branch(
    spec: MaterializedPasteBranchSpec<'_>,
) -> Vec<Node> {
    let mut resolve_rhs = vec![
        Node::let_bind("macro_paste_right_tok", Expr::u32(0)),
        Node::let_bind("macro_paste_right_start", Expr::u32(0)),
        Node::let_bind("macro_paste_right_len", Expr::u32(0)),
        Node::let_bind("macro_paste_right_source_limit", Expr::u32(0)),
        Node::let_bind("macro_paste_right_from_argument", Expr::u32(0)),
        Node::let_bind("macro_paste_arg_start", Expr::u32(0)),
        Node::let_bind("macro_paste_arg_end", Expr::u32(0)),
    ];
    resolve_rhs.push(Node::if_then_else(
        Expr::eq(
            Expr::var("macro_paste_next_param"),
            Expr::u32(C_MACRO_REPLACEMENT_LITERAL),
        ),
        vec![
            Node::assign(
                "macro_paste_right_tok",
                Expr::load(spec.macro_vals, Expr::var("macro_paste_next_offset")),
            ),
            Node::assign(
                "macro_paste_right_start",
                Expr::load(
                    spec.macro_replacement_starts,
                    Expr::var("macro_paste_next_offset"),
                ),
            ),
            Node::assign(
                "macro_paste_right_len",
                Expr::load(
                    spec.macro_replacement_lens,
                    Expr::var("macro_paste_next_offset"),
                ),
            ),
            Node::assign(
                "macro_paste_right_source_limit",
                spec.macro_replacement_source_len.clone(),
            ),
        ],
        {
            let arg_start =
                selected_arg_bound(spec.macro_arg_starts, Expr::var("macro_paste_next_param"));
            let arg_end =
                selected_arg_bound(spec.macro_arg_ends, Expr::var("macro_paste_next_param"));
            vec![
                Node::if_then(
                    Expr::ge(
                        Expr::var("macro_paste_next_param"),
                        Expr::var("named_param_count"),
                    ),
                    vec![Node::trap(
                        Expr::var("macro_paste_next_param"),
                        "function-like-token-paste-parameter-out-of-range",
                    )],
                ),
                Node::assign("macro_paste_arg_start", arg_start),
                Node::assign("macro_paste_arg_end", arg_end),
                Node::if_then(
                    Expr::lt(
                        Expr::var("macro_paste_arg_start"),
                        Expr::var("macro_paste_arg_end"),
                    ),
                    vec![
                        Node::assign(
                            "macro_paste_right_tok",
                            Expr::load(spec.in_tok_types, Expr::var("macro_paste_arg_start")),
                        ),
                        Node::assign(
                            "macro_paste_right_start",
                            Expr::load(spec.in_tok_starts, Expr::var("macro_paste_arg_start")),
                        ),
                        Node::assign(
                            "macro_paste_right_len",
                            Expr::load(spec.in_tok_lens, Expr::var("macro_paste_arg_start")),
                        ),
                        Node::assign("macro_paste_right_source_limit", spec.source_len.clone()),
                        Node::assign("macro_paste_right_from_argument", Expr::u32(1)),
                    ],
                ),
            ]
        },
    ));

    let append_rhs_bytes = vec![Node::if_then_else(
        Expr::eq(Expr::var("macro_paste_right_from_argument"), Expr::u32(1)),
        append_to_previous_output_token(
            "function_paste_arg_rhs",
            spec.source_words,
            spec.source_layout,
            Expr::var("macro_paste_right_start"),
            Expr::var("macro_paste_right_len"),
            spec.source_len.clone(),
            spec.out_tok_starts,
            spec.out_tok_lens,
            spec.out_source_words,
            spec.max_out_source_bytes,
            "function-like-token-paste-argument-source-span-out-of-bounds",
        ),
        append_to_previous_output_token(
            "function_paste_literal_rhs",
            spec.macro_replacement_words,
            spec.macro_replacement_layout,
            Expr::var("macro_paste_right_start"),
            Expr::var("macro_paste_right_len"),
            spec.macro_replacement_source_len.clone(),
            spec.out_tok_starts,
            spec.out_tok_lens,
            spec.out_source_words,
            spec.max_out_source_bytes,
            "function-like-token-paste-literal-source-span-out-of-bounds",
        ),
    )];

    let mut rhs_rest_copy = vec![Node::let_bind(
        "macro_paste_rhs_rest_idx",
        Expr::add(
            Expr::var("macro_paste_arg_start"),
            Expr::var("macro_paste_rhs_rest_rel"),
        ),
    )];
    rhs_rest_copy.extend(emit_materialized_output_token(
        "function_paste_rhs_rest",
        Expr::load(spec.in_tok_types, Expr::var("macro_paste_rhs_rest_idx")),
        spec.source_words,
        spec.source_layout,
        Expr::load(spec.in_tok_starts, Expr::var("macro_paste_rhs_rest_idx")),
        Expr::load(spec.in_tok_lens, Expr::var("macro_paste_rhs_rest_idx")),
        spec.source_len.clone(),
        spec.out_tok_types,
        spec.out_tok_starts,
        spec.out_tok_lens,
        spec.out_source_words,
        spec.max_out_tokens,
        spec.max_out_source_bytes,
        "function-like-token-paste-rest-source-span-out-of-bounds",
    ));

    // GNU comma swallow: `, ## __VA_ARGS__` with no variadic argument drops the
    // preceding comma instead of trapping.
    let empty_rhs = vec![
        Node::let_bind(
            "macro_paste_empty_prev_idx",
            Expr::sub(Expr::var("named_out_idx"), Expr::u32(1)),
        ),
        Node::let_bind(
            "macro_paste_empty_prev_tok",
            Expr::load(spec.out_tok_types, Expr::var("macro_paste_empty_prev_idx")),
        ),
        Node::if_then(
            Expr::eq(
                Expr::var("macro_paste_empty_prev_tok"),
                Expr::u32(TOK_COMMA),
            ),
            vec![
                Node::assign(
                    "named_source_out_idx",
                    Expr::load(spec.out_tok_starts, Expr::var("macro_paste_empty_prev_idx")),
                ),
                Node::assign("named_out_idx", Expr::var("macro_paste_empty_prev_idx")),
            ],
        ),
        Node::assign("named_skip_repl", Expr::u32(1)),
    ];

    emit_function_paste_branch(PasteBranchSpec {
        macro_replacement_params: spec.macro_replacement_params,
        out_tok_types: spec.out_tok_types,
        num_tokens: spec.num_tokens,
        resolve_rhs,
        synth_trap: "function-like-token-paste-cannot-synthesize-token-type-from-materialized-bytes",
        append_rhs_bytes,
        rhs_rest_guard: Expr::eq(Expr::var("macro_paste_right_from_argument"), Expr::u32(1)),
        rhs_rest_copy,
        empty_rhs: Some(empty_rhs),
    })
}
