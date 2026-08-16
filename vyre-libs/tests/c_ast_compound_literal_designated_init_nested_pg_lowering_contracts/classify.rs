// C parser contract tests for compound literals with nested designated
// initializers, compound literals inside statement expressions, designated
// initializers containing builtins, and arrays of compound literals  -
// constructs likely to break VAST/PG lowering.
//
// Constructs under test:
//   - compound literal with nested designated initializers
//   - compound literal inside a statement expression
//   - designated initializer value is `__builtin_choose_expr`
//   - array of compound literals
//   - compound literal in a ternary expression
//   - PG lowering preservation and GPU/CPU parity
//
// A missing GPU adapter is a configuration failure; tests do not skip.

use super::compound_literal_designated_init::*;
use crate::c_frontend::rows::{
    assert_pg_preserves_fixture_row as assert_pg_preserves_row, row_indices,
};
use crate::c_frontend::token_fixture::classify;
use vyre_libs::parsing::c::lower::reference_ast_to_pg_nodes;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_BUILTIN_CHOOSE_EXPR, C_AST_KIND_COMPOUND_LITERAL_EXPR, C_AST_KIND_CONDITIONAL_EXPR,
    C_AST_KIND_INITIALIZER_LIST, C_AST_KIND_MEMBER_ACCESS_EXPR,
};
use vyre_libs::predicate::node_kind;

// ---------------------------------------------------------------------------
// CPU reference contracts
// ---------------------------------------------------------------------------

#[test]
pub(crate) fn cpu_compound_literal_nested_designated_classifies() {
    let fix = fixture_compound_literal_nested_designated();
    let typed = classify(&fix);
    assert!(
        !row_indices(&typed, C_AST_KIND_COMPOUND_LITERAL_EXPR).is_empty(),
        "compound literal must classify"
    );
    let lists = row_indices(&typed, C_AST_KIND_INITIALIZER_LIST);
    assert!(
        lists.len() >= 2,
        "outer and inner initializer lists must classify; got {lists:?}"
    );
    let members = row_indices(&typed, C_AST_KIND_MEMBER_ACCESS_EXPR);
    assert!(
        members.len() >= 3,
        "dot designators .a, .inner, .b, .c must classify; got {members:?}"
    );
}

#[test]
pub(crate) fn cpu_compound_literal_inside_statement_expr_classifies() {
    let fix = fixture_compound_literal_inside_statement_expr();
    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_COMPOUND_LITERAL_EXPR),
        vec![5],
        "compound literal inside statement expression must classify"
    );
    assert!(
        !row_indices(&typed, node_kind::BASIC_BLOCK).is_empty(),
        "statement expression must contain a BASIC_BLOCK"
    );
}

#[test]
pub(crate) fn cpu_designated_init_with_builtin_choose_expr_classifies() {
    let fix = fixture_designated_init_with_builtin_choose_expr();
    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_BUILTIN_CHOOSE_EXPR),
        vec![8],
        "__builtin_choose_expr as designated-init value must classify"
    );
    assert!(
        !row_indices(&typed, C_AST_KIND_INITIALIZER_LIST).is_empty(),
        "initializer list must classify"
    );
}

#[test]
pub(crate) fn cpu_array_of_compound_literals_classifies() {
    let fix = fixture_array_of_compound_literals();
    let typed = classify(&fix);
    let compounds = row_indices(&typed, C_AST_KIND_COMPOUND_LITERAL_EXPR);
    assert_eq!(
        compounds.len(),
        2,
        "both compound literals must classify; got {compounds:?}"
    );
    assert!(
        !row_indices(&typed, C_AST_KIND_INITIALIZER_LIST).is_empty(),
        "initializer list must classify"
    );
}

#[test]
pub(crate) fn cpu_compound_literal_in_ternary_classifies() {
    let fix = fixture_compound_literal_in_ternary();
    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_CONDITIONAL_EXPR),
        vec![6],
        "ternary must classify as CONDITIONAL_EXPR"
    );
    let compounds = row_indices(&typed, C_AST_KIND_COMPOUND_LITERAL_EXPR);
    assert_eq!(
        compounds.len(),
        2,
        "both compound literals in ternary arms must classify; got {compounds:?}"
    );
}

// ---------------------------------------------------------------------------
// PG lowering preservation contracts
// ---------------------------------------------------------------------------

#[test]
pub(crate) fn pg_lower_preserves_compound_literal_nested_designated() {
    let fix = fixture_compound_literal_nested_designated();
    let typed = classify(&fix);
    let pg = reference_ast_to_pg_nodes(&typed);

    for idx in row_indices(&typed, C_AST_KIND_COMPOUND_LITERAL_EXPR) {
        assert_pg_preserves_row(&typed, &pg, &fix, idx, C_AST_KIND_COMPOUND_LITERAL_EXPR);
    }
    for idx in row_indices(&typed, C_AST_KIND_INITIALIZER_LIST) {
        assert_pg_preserves_row(&typed, &pg, &fix, idx, C_AST_KIND_INITIALIZER_LIST);
    }
    for idx in row_indices(&typed, C_AST_KIND_MEMBER_ACCESS_EXPR) {
        assert_pg_preserves_row(&typed, &pg, &fix, idx, C_AST_KIND_MEMBER_ACCESS_EXPR);
    }
}

#[test]
pub(crate) fn pg_lower_preserves_compound_literal_inside_statement_expr() {
    let fix = fixture_compound_literal_inside_statement_expr();
    let typed = classify(&fix);
    let pg = reference_ast_to_pg_nodes(&typed);

    assert_pg_preserves_row(&typed, &pg, &fix, 5, C_AST_KIND_COMPOUND_LITERAL_EXPR);
}
