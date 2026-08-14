//! GPU/CPU parity end-to-end tests for C11 _Atomic, _Generic, and typeof
//! combinations that appear in Linux-grade code but lack dedicated coverage.
//!
//! Constructs under test:
//!   - `_Atomic` as type specifier / qualifier in declarations and parameters
//!   - `_Atomic` mixed with pointer declarators
//!   - `_Generic` selection with multiple associations and default
//!   - `_Generic` nested inside call arguments
//!   - `typeof` combined with `_Atomic` in complex declarators
//!
//! A missing GPU adapter is a configuration failure.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;

use c_ast_gpu_parity_support::{assert_full_pipeline_parity, row_indices, Fixture};
use c_frontend::spelling::c_tokens;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_annotate_typedef_names, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds, C_AST_KIND_FIELD_DECL, C_AST_KIND_FUNCTION_DEFINITION,
    C_AST_KIND_GENERIC_SELECTION_EXPR, C_AST_KIND_POINTER_DECL, C_AST_KIND_STRUCT_DECL,
};
use vyre_primitives::predicate::node_kind;

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

fn fixture_atomic_variable() -> Fixture {
    c_tokens("_Atomic int counter ;")
}

fn fixture_atomic_pointer() -> Fixture {
    c_tokens("_Atomic unsigned long * p ;")
}

fn fixture_atomic_struct_member() -> Fixture {
    c_tokens("struct s { _Atomic int val ; } ;")
}

fn fixture_atomic_parameter() -> Fixture {
    c_tokens("void f ( _Atomic int * p ) { }")
}

fn fixture_generic_selection() -> Fixture {
    c_tokens("int x = _Generic ( a , int : 1 , long : 2 , default : 0 ) ;")
}

fn fixture_generic_in_call() -> Fixture {
    c_tokens("foo ( _Generic ( x , int : 1 , default : 0 ) ) ;")
}

fn fixture_typeof_atomic_combo() -> Fixture {
    c_tokens("typeof ( _Atomic int ) * q ;")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn atomic_variable_declaration_gpu_cpu_parity() {
    let fix = fixture_atomic_variable();
    assert_eq!(
        fix.tok_types[0], TOK_ATOMIC,
        "_Atomic must promote to keyword"
    );
    assert_full_pipeline_parity(&fix, "atomic_variable");

    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);
    assert!(
        row_indices(&typed, node_kind::VARIABLE).contains(&2),
        "counter must classify as VARIABLE"
    );
}

#[test]
fn atomic_pointer_declaration_gpu_cpu_parity() {
    let fix = fixture_atomic_pointer();
    assert_eq!(fix.tok_types[0], TOK_ATOMIC);
    assert_full_pipeline_parity(&fix, "atomic_pointer");

    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);
    assert!(
        row_indices(&typed, C_AST_KIND_POINTER_DECL).contains(&3),
        "pointer declarator must be present"
    );
    assert!(
        row_indices(&typed, node_kind::VARIABLE).contains(&4),
        "p must classify as VARIABLE"
    );
}

#[test]
fn atomic_struct_member_gpu_cpu_parity() {
    let fix = fixture_atomic_struct_member();
    assert_eq!(fix.tok_types[3], TOK_ATOMIC);
    assert_full_pipeline_parity(&fix, "atomic_struct_member");

    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);
    assert!(
        row_indices(&typed, C_AST_KIND_STRUCT_DECL).contains(&0),
        "struct must classify as STRUCT_DECL"
    );
    assert!(
        row_indices(&typed, C_AST_KIND_FIELD_DECL).contains(&5),
        "atomic struct member must classify as FIELD_DECL"
    );
}

#[test]
fn atomic_parameter_declaration_gpu_cpu_parity() {
    let fix = fixture_atomic_parameter();
    assert_eq!(fix.tok_types[3], TOK_ATOMIC);
    assert_full_pipeline_parity(&fix, "atomic_parameter");

    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);
    assert!(
        row_indices(&typed, C_AST_KIND_FUNCTION_DEFINITION).contains(&1)
            || row_indices(&typed, node_kind::FUNCTION_DECL).contains(&1),
        "f with a body must classify as FUNCTION_DEFINITION or FUNCTION_DECL"
    );
}

#[test]
fn generic_selection_expression_gpu_cpu_parity() {
    let fix = fixture_generic_selection();
    assert_eq!(
        fix.tok_types[3], TOK_GENERIC,
        "_Generic must promote to keyword"
    );
    assert_full_pipeline_parity(&fix, "generic_selection");

    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_GENERIC_SELECTION_EXPR),
        vec![3],
        "_Generic must classify as GENERIC_SELECTION_EXPR"
    );
}

#[test]
fn generic_in_call_argument_gpu_cpu_parity() {
    let fix = fixture_generic_in_call();
    assert_eq!(fix.tok_types[2], TOK_GENERIC);
    assert_full_pipeline_parity(&fix, "generic_in_call");

    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);
    assert_eq!(
        row_indices(&typed, C_AST_KIND_GENERIC_SELECTION_EXPR),
        vec![2],
        "_Generic inside call must classify as GENERIC_SELECTION_EXPR"
    );
}

#[test]
fn typeof_atomic_combination_gpu_cpu_parity() {
    let fix = fixture_typeof_atomic_combo();
    assert_eq!(fix.tok_types[0], TOK_GNU_TYPEOF, "typeof must promote");
    assert_eq!(
        fix.tok_types[2], TOK_ATOMIC,
        "_Atomic inside typeof must promote"
    );
    assert_full_pipeline_parity(&fix, "typeof_atomic_combo");

    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);
    assert!(
        row_indices(&typed, C_AST_KIND_POINTER_DECL).contains(&5),
        "pointer declarator must be present after typeof(_Atomic int)"
    );
}
