use super::helpers::{
    classify_prologue, digit_run_scan, identifier_scan, identifier_start, integer_start,
    ClassifyCtx, ScanNames, SerialLexer,
};
use super::*;
use crate::parsing::c::lex::lexer::sections;

/// Reduced serial C lexer: identifiers, decimal integer runs, and the operator
/// table. Shares the classification stages and the serial shell with
/// [`super::dense::c11_lexer`]; it differs only in which stages it composes.
pub fn c11_lexer_regular(
    haystack: &str,
    out_tok_types: &str,
    out_tok_starts: &str,
    out_tok_lens: &str,
    out_counts: &str,
    haystack_len: u32,
) -> Program {
    let ctx = ClassifyCtx::contiguous(haystack, haystack_len);
    let mut classify_at_pos = classify_prologue(&ctx, &Expr::var("pos"), false);

    classify_at_pos.push(identifier_start());
    classify_at_pos.push(identifier_scan(
        &ctx,
        &ScanNames {
            done: "ident_done",
            scan: "scan_ident",
        },
        false,
    ));
    classify_at_pos.push(integer_start());
    classify_at_pos.push(digit_run_scan(
        &ctx,
        &ScanNames {
            done: "number_done",
            scan: "scan_number",
        },
    ));

    classify_at_pos.extend(sections::operator_punct_pushes());
    classify_at_pos.extend(sections::store_token_and_advance_pushes(
        haystack,
        haystack_len,
        out_tok_types,
        out_tok_starts,
        out_tok_lens,
    ));

    SerialLexer {
        op_id: "vyre-libs::parsing::c_lexer_regular",
        haystack,
        out_tok_types,
        out_tok_starts,
        out_tok_lens,
        out_counts,
        haystack_len,
    }
    .build(classify_at_pos)
}
