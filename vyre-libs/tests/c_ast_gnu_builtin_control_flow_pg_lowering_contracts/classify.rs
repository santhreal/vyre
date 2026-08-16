// C parser contract tests for GNU builtins (`__builtin_expect`,
// `__builtin_choose_expr`) in control-flow contexts that stress VAST/PG
// lowering.
//
// Constructs under test:
//   - `__builtin_expect` as an if-condition
//   - `__builtin_expect` as a switch-selector
//   - `__builtin_choose_expr` inside a statement expression
//   - `__builtin_choose_expr` inside a designated initializer value
//   - nested builtins (`__builtin_expect` around `__builtin_choose_expr`)
//   - PG lowering preservation (kind, span, parent, first_child, next_sibling)
//   - GPU/CPU parity for the full pipeline
//
// A missing GPU adapter is a configuration failure; tests do not skip.

use super::gnu_builtin_control_flow::*;
use crate::c_frontend::rows::{
    assert_pg_preserves_fixture_row as assert_pg_preserves_row, row_indices,
};
use crate::c_frontend::token_fixture::classify;
use vyre_libs::parsing::c::lower::reference_ast_to_pg_nodes;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_build_vast_nodes, reference_c11_classify_vast_node_kinds,
    C_AST_KIND_BUILTIN_CHOOSE_EXPR, C_AST_KIND_BUILTIN_EXPECT_EXPR, C_AST_KIND_IF_STMT,
    C_AST_KIND_SWITCH_STMT,
};
use vyre_libs::predicate::node_kind;

// ---------------------------------------------------------------------------
// CPU reference contracts
// ---------------------------------------------------------------------------

#[test]
pub(crate) fn cpu_builtin_expect_if_condition_classifies() {
    let fix = fixture_builtin_expect_if_condition();
    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_IF_STMT),
        vec![0],
        "if must classify as IF_STMT"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_BUILTIN_EXPECT_EXPR),
        vec![2],
        "__builtin_expect in if condition must classify as BUILTIN_EXPECT_EXPR"
    );
    assert!(
        row_indices(&typed, node_kind::CALL).is_empty(),
        "__builtin_expect must not collapse into CALL"
    );
}

#[test]
pub(crate) fn cpu_builtin_expect_switch_selector_classifies() {
    let fix = fixture_builtin_expect_switch_selector();
    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_SWITCH_STMT),
        vec![0],
        "switch must classify as SWITCH_STMT"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_BUILTIN_EXPECT_EXPR),
        vec![2],
        "__builtin_expect in switch selector must classify as BUILTIN_EXPECT_EXPR"
    );
}

#[test]
pub(crate) fn cpu_builtin_choose_expr_in_statement_expr_classifies() {
    let fix = fixture_builtin_choose_expr_in_statement_expr();
    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_BUILTIN_CHOOSE_EXPR),
        vec![5],
        "__builtin_choose_expr inside statement expression must classify"
    );
    assert!(
        !row_indices(&typed, node_kind::BASIC_BLOCK).is_empty(),
        "statement expression must contain a BASIC_BLOCK"
    );
}

#[test]
pub(crate) fn cpu_builtin_choose_expr_in_designated_init_classifies() {
    let fix = fixture_builtin_choose_expr_in_designated_init();
    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_BUILTIN_CHOOSE_EXPR),
        vec![16],
        "__builtin_choose_expr in designated initializer value must classify"
    );
}

#[test]
pub(crate) fn cpu_nested_builtin_expect_choose_expr_classifies() {
    let fix = fixture_nested_builtin_expect_choose_expr();
    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_BUILTIN_EXPECT_EXPR),
        vec![3],
        "outer __builtin_expect must classify"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_BUILTIN_CHOOSE_EXPR),
        vec![5],
        "inner __builtin_choose_expr must classify"
    );
}

#[test]
pub(crate) fn cpu_builtin_expect_in_ternary_classifies() {
    let fix = fixture_builtin_expect_in_ternary();
    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_BUILTIN_EXPECT_EXPR),
        vec![3],
        "__builtin_expect in ternary condition must classify"
    );
}

// ---------------------------------------------------------------------------
// PG lowering preservation contracts
// ---------------------------------------------------------------------------

#[test]
pub(crate) fn pg_lower_preserves_builtin_expect_if_condition() {
    let fix = fixture_builtin_expect_if_condition();
    let typed = classify(&fix);
    let pg = reference_ast_to_pg_nodes(&typed);

    assert_pg_preserves_row(&typed, &pg, &fix, 0, C_AST_KIND_IF_STMT);
    assert_pg_preserves_row(&typed, &pg, &fix, 2, C_AST_KIND_BUILTIN_EXPECT_EXPR);
}

#[test]
pub(crate) fn pg_lower_preserves_builtin_expect_switch_selector() {
    let fix = fixture_builtin_expect_switch_selector();
    let typed = classify(&fix);
    let pg = reference_ast_to_pg_nodes(&typed);

    assert_pg_preserves_row(&typed, &pg, &fix, 0, C_AST_KIND_SWITCH_STMT);
    assert_pg_preserves_row(&typed, &pg, &fix, 2, C_AST_KIND_BUILTIN_EXPECT_EXPR);
}

#[test]
pub(crate) fn pg_lower_preserves_builtin_choose_expr_in_statement_expr() {
    let fix = fixture_builtin_choose_expr_in_statement_expr();
    let typed = classify(&fix);
    let pg = reference_ast_to_pg_nodes(&typed);

    assert_pg_preserves_row(&typed, &pg, &fix, 5, C_AST_KIND_BUILTIN_CHOOSE_EXPR);
}

#[test]
pub(crate) fn pg_lower_preserves_builtin_choose_expr_in_designated_init() {
    let fix = fixture_builtin_choose_expr_in_designated_init();
    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let typed = reference_c11_classify_vast_node_kinds(&raw);
    let pg = reference_ast_to_pg_nodes(&typed);

    assert_pg_preserves_row(&typed, &pg, &fix, 16, C_AST_KIND_BUILTIN_CHOOSE_EXPR);
}

#[test]
pub(crate) fn pg_lower_preserves_nested_builtin_expect_choose_expr() {
    let fix = fixture_nested_builtin_expect_choose_expr();
    let typed = classify(&fix);
    let pg = reference_ast_to_pg_nodes(&typed);

    assert_pg_preserves_row(&typed, &pg, &fix, 3, C_AST_KIND_BUILTIN_EXPECT_EXPR);
    assert_pg_preserves_row(&typed, &pg, &fix, 5, C_AST_KIND_BUILTIN_CHOOSE_EXPR);
}
