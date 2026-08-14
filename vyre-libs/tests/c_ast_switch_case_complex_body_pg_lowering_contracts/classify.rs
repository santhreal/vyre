// C parser contract tests for switch/case/default bodies containing GNU
// statement expressions, compound literals, designated initializers, and
// nested labels  -  constructs likely to break VAST/PG lowering.
//
// Constructs under test:
//   - switch with a statement expression in a case body
//   - switch with a compound literal in a case body
//   - switch with designated initializers in a case body
//   - Duff's-device style interleaved switch/loop/label pattern
//   - nested switch inside a statement expression
//   - default label shared with a user label
//   - PG lowering preservation and GPU/CPU parity
//
// A missing GPU adapter is a configuration failure; tests do not skip.

use super::switch_case_complex_bodies::*;
use crate::c_frontend::rows::row_indices;
use crate::c_frontend::token_fixture::classify;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_BREAK_STMT, C_AST_KIND_CASE_STMT, C_AST_KIND_GNU_STATEMENT_EXPR,
    C_AST_KIND_SWITCH_STMT,
};

// ---------------------------------------------------------------------------
// CPU reference contracts
// ---------------------------------------------------------------------------

#[test]
pub(crate) fn cpu_switch_case_with_statement_expr_classifies() {
    let fix = fixture_switch_case_with_statement_expr();
    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_SWITCH_STMT),
        vec![7],
        "switch must classify as SWITCH_STMT"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_CASE_STMT),
        vec![12],
        "case must classify as CASE_STMT"
    );
    assert!(
        !row_indices(&typed, C_AST_KIND_GNU_STATEMENT_EXPR).is_empty(),
        "statement expression in case body must classify"
    );
    assert!(
        !row_indices(&typed, C_AST_KIND_BREAK_STMT).is_empty(),
        "break must classify as BREAK_STMT"
    );
}
