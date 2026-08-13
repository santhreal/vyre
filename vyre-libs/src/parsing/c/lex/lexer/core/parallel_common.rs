use super::helpers::{
    classify_prologue, digit_run_scan, identifier_scan, identifier_start, integer_start,
    token_start_expr, ClassifyCtx, ScanNames, TokenStartOpts,
};
use super::*;
use crate::parsing::c::lex::lexer::sections;

#[derive(Clone, Copy)]
pub(super) enum RegularParallelMode {
    Ranked,
    Sparse,
}

impl RegularParallelMode {
    fn scan_prefix(self) -> &'static str {
        match self {
            RegularParallelMode::Ranked => "ranked",
            RegularParallelMode::Sparse => "sparse",
        }
    }
}

/// Token-start rule for the expanded-haystack parallel variants: no NUL
/// padding to skip, no ellipsis in the reduced grammar, and the runtime buffer
/// length is the authoritative bound.
const REGULAR_PARALLEL_TOKEN_START: TokenStartOpts = TokenStartOpts {
    dot_pair_is_tail: false,
    nul_is_space: false,
    bound_by_declared_len: false,
};

pub(super) fn regular_parallel_token_start_expr(
    haystack: &str,
    haystack_len: u32,
    index: Expr,
) -> Expr {
    token_start_expr(
        &ClassifyCtx::expanded(haystack, haystack_len),
        index,
        &REGULAR_PARALLEL_TOKEN_START,
    )
}

pub(super) fn regular_parallel_classifier(
    haystack: &str,
    haystack_len: u32,
    t: Expr,
    mode: RegularParallelMode,
) -> Vec<Node> {
    let ctx = ClassifyCtx::expanded(haystack, haystack_len);
    let prefix = mode.scan_prefix();
    let ident_done = format!("{prefix}_ident_done");
    let ident_scan = format!("{prefix}_scan_ident");
    let number_done = format!("{prefix}_number_done");
    let number_scan = format!("{prefix}_scan_number");

    let mut classify_at_pos = classify_prologue(&ctx, &t, true);

    if matches!(mode, RegularParallelMode::Ranked) {
        classify_at_pos.push(Node::let_bind("rank", Expr::u32(0)));
        classify_at_pos.push(Node::loop_for(
            "rank_scan",
            Expr::u32(0),
            t.clone(),
            vec![Node::if_then(
                regular_parallel_token_start_expr(haystack, haystack_len, Expr::var("rank_scan")),
                vec![Node::assign(
                    "rank",
                    Expr::add(Expr::var("rank"), Expr::u32(1)),
                )],
            )],
        ));
    }

    classify_at_pos.push(identifier_start());
    classify_at_pos.push(identifier_scan(
        &ctx,
        &ScanNames {
            done: &ident_done,
            scan: &ident_scan,
        },
        false,
    ));
    classify_at_pos.push(integer_start());
    classify_at_pos.push(digit_run_scan(
        &ctx,
        &ScanNames {
            done: &number_done,
            scan: &number_scan,
        },
    ));
    classify_at_pos.extend(sections::operator_punct_pushes());
    classify_at_pos
}
