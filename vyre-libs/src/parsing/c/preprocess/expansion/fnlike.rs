//! Function-like macro expansion builder.

use crate::parsing::c::preprocess::synthesis::*;
use vyre_foundation::ir::{Expr, Node};

use super::helpers::*;
use super::paste_branch::*;
use super::*;

pub(super) fn emit_function_like_replacement(
    in_tok_types: &str,
    macro_vals: &str,
    macro_replacement_params: &str,
    out_tok_types: &str,
    macro_arg_starts: &str,
    macro_arg_ends: &str,
    num_tokens: Expr,
    max_out_tokens: u32,
) -> Vec<Node> {
    emit_function_like_replacement_walk(FunctionLikeReplacementSpec {
        in_tok_types,
        macro_vals,
        macro_replacement_params,
        macro_arg_starts,
        macro_arg_ends,
        num_tokens: num_tokens.clone(),
        stringify: stringify_branch(macro_replacement_params, out_tok_types, max_out_tokens),
        paste: paste_branch(
            in_tok_types,
            macro_vals,
            macro_replacement_params,
            out_tok_types,
            macro_arg_starts,
            macro_arg_ends,
            num_tokens.clone(),
            max_out_tokens,
        ),
        regular: regular_branch(
            in_tok_types,
            out_tok_types,
            macro_arg_starts,
            macro_arg_ends,
            num_tokens,
            max_out_tokens,
        ),
    })
}

/// `#param`: emit the stringified argument as one synthetic string token.
fn stringify_branch(
    macro_replacement_params: &str,
    out_tok_types: &str,
    max_out_tokens: u32,
) -> Vec<Node> {
    vec![
        Node::let_bind(
            "macro_stringify_next_offset",
            Expr::add(
                Expr::var("named_macro_idx"),
                Expr::add(Expr::var("named_repl_i"), Expr::u32(1)),
            ),
        ),
        Node::let_bind(
            "macro_stringify_next_param",
            Expr::load(
                macro_replacement_params,
                Expr::var("macro_stringify_next_offset"),
            ),
        ),
        Node::if_then_else(
            Expr::eq(
                Expr::var("macro_stringify_next_param"),
                Expr::u32(C_MACRO_REPLACEMENT_LITERAL),
            ),
            emit_one_output_token(out_tok_types, Expr::var("named_repl_tok"), max_out_tokens),
            {
                let mut stringify = vec![Node::if_then(
                    Expr::ge(
                        Expr::var("macro_stringify_next_param"),
                        Expr::var("named_param_count"),
                    ),
                    vec![Node::trap(
                        Expr::var("macro_stringify_next_param"),
                        "function-like-stringification-parameter-out-of-range",
                    )],
                )];
                stringify.extend(emit_one_output_token(
                    out_tok_types,
                    Expr::u32(stringification_token_type()),
                    max_out_tokens,
                ));
                stringify.push(Node::assign("named_skip_repl", Expr::u32(1)));
                stringify
            },
        ),
    ]
}

/// `lhs ## rhs`, token types only. An empty argument operand has no bytes to
/// contribute and no comma to swallow, so it traps rather than expanding to
/// nothing.
fn paste_branch(
    in_tok_types: &str,
    macro_vals: &str,
    macro_replacement_params: &str,
    out_tok_types: &str,
    macro_arg_starts: &str,
    macro_arg_ends: &str,
    num_tokens: Expr,
    max_out_tokens: u32,
) -> Vec<Node> {
    let mut resolve_rhs = vec![
        Node::let_bind("macro_paste_right_tok", Expr::u32(0)),
        Node::let_bind("macro_paste_arg_start", Expr::u32(0)),
        Node::let_bind("macro_paste_arg_end", Expr::u32(0)),
    ];
    resolve_rhs.push(Node::if_then_else(
        Expr::eq(
            Expr::var("macro_paste_next_param"),
            Expr::u32(C_MACRO_REPLACEMENT_LITERAL),
        ),
        vec![Node::assign(
            "macro_paste_right_tok",
            Expr::load(macro_vals, Expr::var("macro_paste_next_offset")),
        )],
        {
            let arg_start = selected_arg_bound(macro_arg_starts, Expr::var("macro_paste_next_param"));
            let arg_end = selected_arg_bound(macro_arg_ends, Expr::var("macro_paste_next_param"));
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
                    Expr::ge(
                        Expr::var("macro_paste_arg_start"),
                        Expr::var("macro_paste_arg_end"),
                    ),
                    vec![Node::trap(
                        Expr::var("macro_paste_next_param"),
                        "function-like-token-paste-empty-argument",
                    )],
                ),
                Node::assign(
                    "macro_paste_right_tok",
                    Expr::load(in_tok_types, Expr::var("macro_paste_arg_start")),
                ),
            ]
        },
    ));

    let mut rhs_rest_copy = vec![Node::let_bind(
        "macro_paste_rhs_rest_tok",
        Expr::load(
            in_tok_types,
            Expr::add(
                Expr::var("macro_paste_arg_start"),
                Expr::var("macro_paste_rhs_rest_rel"),
            ),
        ),
    )];
    rhs_rest_copy.extend(emit_one_output_token(
        out_tok_types,
        Expr::var("macro_paste_rhs_rest_tok"),
        max_out_tokens,
    ));

    emit_function_paste_branch(PasteBranchSpec {
        macro_replacement_params,
        out_tok_types,
        num_tokens,
        resolve_rhs,
        synth_trap: "function-like-token-paste-cannot-synthesize-token-type",
        append_rhs_bytes: Vec::new(),
        rhs_rest_guard: Expr::ne(
            Expr::var("macro_paste_next_param"),
            Expr::u32(C_MACRO_REPLACEMENT_LITERAL),
        ),
        rhs_rest_copy,
        empty_rhs: None,
    })
}

/// Any other replacement token: a literal passes through, a parameter expands
/// to every token of its argument.
fn regular_branch(
    in_tok_types: &str,
    out_tok_types: &str,
    macro_arg_starts: &str,
    macro_arg_ends: &str,
    num_tokens: Expr,
    max_out_tokens: u32,
) -> Vec<Node> {
    let regular_literal =
        emit_one_output_token(out_tok_types, Expr::var("named_repl_tok"), max_out_tokens);
    let arg_start = selected_arg_bound(macro_arg_starts, Expr::var("named_repl_param"));
    let arg_end = selected_arg_bound(macro_arg_ends, Expr::var("named_repl_param"));
    vec![Node::if_then_else(
        Expr::eq(
            Expr::var("named_repl_param"),
            Expr::u32(C_MACRO_REPLACEMENT_LITERAL),
        ),
        regular_literal,
        vec![
            Node::if_then(
                Expr::ge(
                    Expr::var("named_repl_param"),
                    Expr::var("named_param_count"),
                ),
                vec![Node::trap(
                    Expr::var("named_repl_param"),
                    "function-like-macro-replacement-parameter-out-of-range",
                )],
            ),
            Node::let_bind("macro_sub_arg_start", arg_start),
            Node::let_bind("macro_sub_arg_end", arg_end),
            Node::loop_for(
                "macro_sub_arg_rel",
                Expr::u32(0),
                num_tokens,
                vec![Node::if_then(
                    Expr::lt(
                        Expr::add(
                            Expr::var("macro_sub_arg_start"),
                            Expr::var("macro_sub_arg_rel"),
                        ),
                        Expr::var("macro_sub_arg_end"),
                    ),
                    {
                        let mut copy = vec![Node::let_bind(
                            "macro_sub_arg_tok",
                            Expr::load(
                                in_tok_types,
                                Expr::add(
                                    Expr::var("macro_sub_arg_start"),
                                    Expr::var("macro_sub_arg_rel"),
                                ),
                            ),
                        )];
                        copy.extend(emit_one_output_token(
                            out_tok_types,
                            Expr::var("macro_sub_arg_tok"),
                            max_out_tokens,
                        ));
                        copy
                    },
                )],
            ),
        ],
    )]
}
