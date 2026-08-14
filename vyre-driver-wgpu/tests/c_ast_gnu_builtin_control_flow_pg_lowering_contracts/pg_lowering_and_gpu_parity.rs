use super::gnu_builtin_control_flow::*;
use crate::c_ast_gpu_parity_support::{assert_full_pipeline_parity, run_gpu_pg_lower};
use crate::c_frontend::rows::assert_pg_preserves_fixture_row as assert_pg_preserves_row;
use vyre_libs::parsing::c::lower::reference_ast_to_pg_nodes;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_annotate_typedef_names, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds, C_AST_KIND_BUILTIN_EXPECT_EXPR,
};

#[test]
fn pg_lower_preserves_builtin_expect_in_ternary() {
    let fix = fixture_builtin_expect_in_ternary();
    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);
    let pg = reference_ast_to_pg_nodes(&typed);

    assert_pg_preserves_row(&typed, &pg, &fix, 3, C_AST_KIND_BUILTIN_EXPECT_EXPR);
}

// ---------------------------------------------------------------------------
// GPU/CPU parity contracts
// ---------------------------------------------------------------------------

#[test]
fn gpu_parity_builtin_expect_if_condition() {
    let fix = fixture_builtin_expect_if_condition();
    assert_full_pipeline_parity(&fix, "builtin_expect_if_condition");
}

#[test]
fn gpu_parity_builtin_expect_switch_selector() {
    let fix = fixture_builtin_expect_switch_selector();
    assert_full_pipeline_parity(&fix, "builtin_expect_switch_selector");
}

#[test]
fn gpu_parity_builtin_choose_expr_in_statement_expr() {
    let fix = fixture_builtin_choose_expr_in_statement_expr();
    assert_full_pipeline_parity(&fix, "builtin_choose_expr_in_statement_expr");
}

#[test]
fn gpu_parity_builtin_choose_expr_in_designated_init() {
    let fix = fixture_builtin_choose_expr_in_designated_init();
    assert_full_pipeline_parity(&fix, "builtin_choose_expr_in_designated_init");
}

#[test]
fn gpu_parity_nested_builtin_expect_choose_expr() {
    let fix = fixture_nested_builtin_expect_choose_expr();
    assert_full_pipeline_parity(&fix, "nested_builtin_expect_choose_expr");
}

#[test]
fn gpu_parity_builtin_expect_in_ternary() {
    let fix = fixture_builtin_expect_in_ternary();
    assert_full_pipeline_parity(&fix, "builtin_expect_in_ternary");
}

// ---------------------------------------------------------------------------
// GPU PG lowering parity contracts
// ---------------------------------------------------------------------------

#[test]
fn gpu_parity_pg_lower_builtin_expect_if_condition() {
    let fix = fixture_builtin_expect_if_condition();
    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);
    let expected = reference_ast_to_pg_nodes(&typed);
    let gpu = run_gpu_pg_lower(&typed);
    assert_eq!(
        gpu, expected,
        "GPU PG lowerer must match CPU for builtin_expect_if_condition"
    );
}

#[test]
fn gpu_parity_pg_lower_builtin_expect_switch_selector() {
    let fix = fixture_builtin_expect_switch_selector();
    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);
    let expected = reference_ast_to_pg_nodes(&typed);
    let gpu = run_gpu_pg_lower(&typed);
    assert_eq!(
        gpu, expected,
        "GPU PG lowerer must match CPU for builtin_expect_switch_selector"
    );
}

#[test]
fn gpu_parity_pg_lower_builtin_choose_expr_in_statement_expr() {
    let fix = fixture_builtin_choose_expr_in_statement_expr();
    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);
    let expected = reference_ast_to_pg_nodes(&typed);
    let gpu = run_gpu_pg_lower(&typed);
    assert_eq!(
        gpu, expected,
        "GPU PG lowerer must match CPU for builtin_choose_expr_in_statement_expr"
    );
}

#[test]
fn gpu_parity_pg_lower_builtin_choose_expr_in_designated_init() {
    let fix = fixture_builtin_choose_expr_in_designated_init();
    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let typed = reference_c11_classify_vast_node_kinds(&raw);
    let expected = reference_ast_to_pg_nodes(&typed);
    let gpu = run_gpu_pg_lower(&typed);
    assert_eq!(
        gpu, expected,
        "GPU PG lowerer must match CPU for builtin_choose_expr_in_designated_init"
    );
}

#[test]
fn gpu_parity_pg_lower_nested_builtin_expect_choose_expr() {
    let fix = fixture_nested_builtin_expect_choose_expr();
    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);
    let expected = reference_ast_to_pg_nodes(&typed);
    let gpu = run_gpu_pg_lower(&typed);
    assert_eq!(
        gpu, expected,
        "GPU PG lowerer must match CPU for nested_builtin_expect_choose_expr"
    );
}

#[test]
fn gpu_parity_pg_lower_builtin_expect_in_ternary() {
    let fix = fixture_builtin_expect_in_ternary();
    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);
    let expected = reference_ast_to_pg_nodes(&typed);
    let gpu = run_gpu_pg_lower(&typed);
    assert_eq!(
        gpu, expected,
        "GPU PG lowerer must match CPU for builtin_expect_in_ternary"
    );
}
