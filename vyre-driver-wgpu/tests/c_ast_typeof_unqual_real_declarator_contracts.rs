//! Contracts for GNU/C23 `typeof_unqual` declarators without token spoofing.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;

use c_ast_gpu_parity_support::{assert_full_pipeline_parity, row_indices, Fixture};
use c_frontend::spelling::c_tokens;
use c_frontend::token_fixture::classify;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_ARRAY_DECL, C_AST_KIND_FUNCTION_DECLARATOR, C_AST_KIND_POINTER_DECL,
    C_AST_KIND_SIZEOF_EXPR,
};
use vyre_libs::predicate::node_kind;

fn real_typeof_unqual_function_pointer_table() -> Fixture {
    c_tokens("__typeof_unqual__ ( int ) * const ( * table [ 2 ] ) ( unsigned long ) ;")
}

fn typedef_over_typeof_unqual_pointer() -> Fixture {
    c_tokens("typedef typeof_unqual ( int ) * alias_t ; alias_t value ;")
}

#[test]
fn real_typeof_unqual_drives_complex_declarator_shape() {
    let fix = real_typeof_unqual_function_pointer_table();
    assert_eq!(
        fix.tok_types[0], TOK_GNU_TYPEOF_UNQUAL,
        "real __typeof_unqual__ spelling must promote through the keyword pass"
    );
    assert_full_pipeline_parity(&fix, "real_typeof_unqual_function_pointer_table");

    let typed = classify(&fix);
    assert!(
        row_indices(&typed, C_AST_KIND_SIZEOF_EXPR).contains(&0),
        "__typeof_unqual__ must classify as the typeof operator row"
    );
    assert_eq!(
        row_indices(&typed, C_AST_KIND_POINTER_DECL),
        vec![4, 7],
        "both result pointer and table element pointer must be declarator rows"
    );
    assert!(
        row_indices(&typed, C_AST_KIND_ARRAY_DECL).contains(&9),
        "table[2] must remain an array declarator"
    );
    assert!(
        row_indices(&typed, C_AST_KIND_FUNCTION_DECLARATOR).contains(&13),
        "function-pointer table suffix must classify as function declarator"
    );
    assert!(
        row_indices(&typed, node_kind::VARIABLE).contains(&8),
        "table identifier must remain the declarator variable"
    );
}

#[test]
fn typedef_over_typeof_unqual_is_visible_to_later_declarators() {
    let fix = typedef_over_typeof_unqual_pointer();
    assert_full_pipeline_parity(&fix, "typedef_over_typeof_unqual_pointer");

    let typed = classify(&fix);
    assert!(
        row_indices(&typed, C_AST_KIND_POINTER_DECL).contains(&5),
        "typedef typeof_unqual(int) *alias_t must classify pointer declarator"
    );
    assert!(
        row_indices(&typed, node_kind::VARIABLE).contains(&9),
        "alias_t value must classify value as a typedef-backed declarator"
    );
}
