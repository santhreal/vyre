// GPU/CPU parity end-to-end tests for statement construct gaps and
// label/goto interactions that appear in Linux-grade C but lack
// dedicated coverage.
//
// Constructs under test:
//   - empty statement (`;`)
//   - for-loop with a declaration in the init clause (C99)
//   - labels inside nested loops, switch, and if bodies
//   - goto jumping across nested block boundaries
//
// A missing GPU adapter is a configuration failure.

use crate::c_ast_gpu_parity_support::{
    assert_full_pipeline_parity, classify, row_indices, word_at, Fixture, VAST_STRIDE_U32,
};
use crate::c_frontend::spelling::c_tokens;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_BREAK_STMT, C_AST_KIND_CASE_STMT, C_AST_KIND_CONTINUE_STMT, C_AST_KIND_DEFAULT_STMT,
    C_AST_KIND_DO_STMT, C_AST_KIND_FOR_STMT, C_AST_KIND_GOTO_STMT, C_AST_KIND_IF_STMT,
    C_AST_KIND_LABEL_STMT, C_AST_KIND_RETURN_STMT, C_AST_KIND_SWITCH_STMT, C_AST_KIND_WHILE_STMT,
};
use vyre_libs::predicate::node_kind;

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

/// void f() { int x; ; }
pub(crate) fn fixture_empty_statement() -> Fixture {
    c_tokens("void f ( void ) { int x ; ; }")
}

/// void g() { for (int i = 0; i < 10; i++) { } }
pub(crate) fn fixture_for_with_declaration() -> Fixture {
    c_tokens("void g ( void ) { for ( int i = 0 ; i < 10 ; i ++ ) { } }")
}

/// void h() { while (1) { label: goto label; } }
pub(crate) fn fixture_label_goto_inside_while() -> Fixture {
    c_tokens("void h ( void ) { while ( 1 ) { label : goto label ; } }")
}

/// void k(int x) { switch (x) { case 1: if (1) { goto end; } end: ; } }
pub(crate) fn fixture_goto_across_switch_case() -> Fixture {
    c_tokens("void k ( int x ) { switch ( x ) { case 1 : if ( 1 ) { goto end ; } end : ; } }")
}

/// void m(int x) { switch (x) { default: do { continue; } while (0); case 1: break; } return; }
pub(crate) fn fixture_default_do_break_continue_return() -> Fixture {
    c_tokens(
        "void m ( int x ) { switch ( x ) { default : do { continue ; } while ( 0 ) ; case 1 : \
         break ; } return ; }",
    )
}

/// void n(void) { { { return; } } }
pub(crate) fn fixture_nested_compound_return() -> Fixture {
    c_tokens("void n ( void ) { { { return ; } } }")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
pub(crate) fn empty_statement_inside_function_gpu_cpu_parity() {
    let fix = fixture_empty_statement();
    assert_full_pipeline_parity(&fix, "empty_statement");

    let typed = classify(&fix);

    assert!(
        row_indices(&typed, node_kind::VARIABLE).contains(&7),
        "x must classify as VARIABLE"
    );
    // The second semicolon is an empty statement; it should not crash and
    // should simply classify as 0 (unknown / raw token).
    assert_eq!(
        word_at(&typed, 9 * VAST_STRIDE_U32),
        0,
        "empty statement semicolon must classify as 0"
    );
}

#[test]
pub(crate) fn for_loop_with_declaration_gpu_cpu_parity() {
    let fix = fixture_for_with_declaration();
    assert_full_pipeline_parity(&fix, "for_with_declaration");

    let typed = classify(&fix);

    assert_eq!(
        row_indices(&typed, C_AST_KIND_FOR_STMT),
        vec![6],
        "for must classify as FOR_STMT"
    );
    assert!(
        row_indices(&typed, node_kind::VARIABLE).contains(&9),
        "i must classify as VARIABLE"
    );
    assert_eq!(
        word_at(&typed, 11 * VAST_STRIDE_U32),
        node_kind::LITERAL,
        "declaration initializer literal must classify as LITERAL"
    );
    assert!(
        row_indices(&typed, node_kind::BASIC_BLOCK).contains(&20),
        "for body brace must classify as BASIC_BLOCK"
    );
}

#[test]
pub(crate) fn label_goto_inside_while_loop_gpu_cpu_parity() {
    let fix = fixture_label_goto_inside_while();
    assert_full_pipeline_parity(&fix, "label_goto_inside_while");

    let typed = classify(&fix);

    assert_eq!(
        row_indices(&typed, C_AST_KIND_WHILE_STMT),
        vec![6],
        "while must classify as WHILE_STMT"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_GOTO_STMT),
        vec![13],
        "goto must classify as GOTO_STMT"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_LABEL_STMT),
        vec![11],
        "label definition must classify as LABEL_STMT"
    );
}

#[test]
pub(crate) fn goto_across_switch_case_blocks_gpu_cpu_parity() {
    let fix = fixture_goto_across_switch_case();
    assert_full_pipeline_parity(&fix, "goto_across_switch_case");

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
    assert_eq!(
        row_indices(&typed, C_AST_KIND_IF_STMT),
        vec![15],
        "if must classify as IF_STMT"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_GOTO_STMT),
        vec![20],
        "goto must classify as GOTO_STMT"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_LABEL_STMT),
        vec![24],
        "end label after the if-block must classify as LABEL_STMT"
    );
}

#[test]
pub(crate) fn default_do_break_continue_return_gpu_cpu_parity() {
    let fix = fixture_default_do_break_continue_return();
    assert_full_pipeline_parity(&fix, "default_do_break_continue_return");

    let typed = classify(&fix);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_SWITCH_STMT),
        vec![7],
        "switch must classify as SWITCH_STMT"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_DEFAULT_STMT),
        vec![12],
        "default must classify as DEFAULT_STMT"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_DO_STMT),
        vec![14],
        "do must classify as DO_STMT"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_CONTINUE_STMT),
        vec![16],
        "continue must classify as CONTINUE_STMT"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_WHILE_STMT),
        vec![19],
        "do/while trailer must classify while as WHILE_STMT"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_CASE_STMT),
        vec![24],
        "case must classify as CASE_STMT"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_BREAK_STMT),
        vec![27],
        "break must classify as BREAK_STMT"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_RETURN_STMT),
        vec![30],
        "return after switch must classify as RETURN_STMT"
    );
}
