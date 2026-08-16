//! GPU/CPU parity end-to-end tests for typeof/typeof_unqual and complex
//! declarator constructs common in Linux headers and macro expansions.
//!
//! Constructs under test:
//!   - `typeof` in array, pointer, and function-pointer declarators
//!   - deeply parenthesised declarators with typeof type-specifiers
//!   - `typeof_unqual` treated as an identifier fallback (future C23 contract)
//!   - typeof combined with `_Atomic` and nested qualifiers
//!   - function-pointer arrays and function-returning-function pointers
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
    reference_c11_classify_vast_node_kinds, C_AST_KIND_ARRAY_DECL, C_AST_KIND_FUNCTION_DECLARATOR,
    C_AST_KIND_POINTER_DECL,
};
use vyre_libs::predicate::node_kind;

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

/// typeof(int) *(*fp[4])(void);
fn fixture_typeof_function_pointer_array() -> Fixture {
    c_tokens("typeof ( int ) * ( * fp [ 4 ] ) ( void ) ;")
}

/// typeof(int) (((*ptr)));
fn fixture_typeof_deeply_parenthesised() -> Fixture {
    c_tokens("typeof ( int ) ( ( ( * ptr ) ) ) ;")
}

/// typeof(int) * const * volatile p;
fn fixture_typeof_nested_qualifiers() -> Fixture {
    c_tokens("typeof ( int ) * const * volatile p ;")
}

/// __typeof_unqual__(int) z;
/// Simulate keyword promotion to verify the parser pipeline handles
/// typeof_unqual without panic.
fn fixture_typeof_unqual_simulated() -> Fixture {
    let mut fix = c_tokens("__typeof_unqual__ ( int ) z ;");
    // Simulate keyword promotion so the parser sees it as typeof.
    fix.tok_types[0] = TOK_GNU_TYPEOF;
    fix
}

/// _Atomic typeof(int) *q;
fn fixture_atomic_typeof_pointer() -> Fixture {
    c_tokens("_Atomic typeof ( int ) * q ;")
}

/// typeof(int) *(*f(void))(float);
fn fixture_typeof_function_returning_fnptr() -> Fixture {
    c_tokens("typeof ( int ) * ( * f ( void ) ) ( float ) ;")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn typeof_function_pointer_array_gpu_cpu_parity() {
    let fix = fixture_typeof_function_pointer_array();
    assert_eq!(fix.tok_types[0], TOK_GNU_TYPEOF, "typeof must promote");
    assert_full_pipeline_parity(&fix, "typeof_function_pointer_array");

    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);

    let ptrs = row_indices(&typed, C_AST_KIND_POINTER_DECL);
    assert_eq!(ptrs.len(), 2, "must contain two pointer declarators");
    assert!(
        row_indices(&typed, C_AST_KIND_ARRAY_DECL).contains(&8),
        "fp[4] must classify as ARRAY_DECL"
    );
    assert!(
        !row_indices(&typed, C_AST_KIND_FUNCTION_DECLARATOR).is_empty(),
        "must contain at least one function declarator"
    );
    assert!(
        row_indices(&typed, node_kind::VARIABLE).contains(&7),
        "fp must classify as VARIABLE"
    );
}

#[test]
fn typeof_deeply_parenthesised_pointer_gpu_cpu_parity() {
    let fix = fixture_typeof_deeply_parenthesised();
    assert_eq!(fix.tok_types[0], TOK_GNU_TYPEOF);
    assert_full_pipeline_parity(&fix, "typeof_deeply_parenthesised");

    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);

    assert!(
        row_indices(&typed, C_AST_KIND_POINTER_DECL).contains(&7),
        "deeply parenthesised pointer must classify as POINTER_DECL"
    );
    assert!(
        row_indices(&typed, node_kind::VARIABLE).contains(&8),
        "ptr must classify as VARIABLE"
    );
}

#[test]
fn typeof_nested_qualifiers_gpu_cpu_parity() {
    let fix = fixture_typeof_nested_qualifiers();
    assert_eq!(fix.tok_types[0], TOK_GNU_TYPEOF);
    assert_full_pipeline_parity(&fix, "typeof_nested_qualifiers");

    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);

    let ptrs = row_indices(&typed, C_AST_KIND_POINTER_DECL);
    assert_eq!(
        ptrs.len(),
        2,
        "must contain two pointer declarators for * const * volatile"
    );
    assert!(
        row_indices(&typed, node_kind::VARIABLE).contains(&8),
        "p must classify as VARIABLE"
    );
}

#[test]
fn typeof_unqual_simulated_promotion_gpu_cpu_parity() {
    let fix = fixture_typeof_unqual_simulated();
    // We manually promoted __typeof_unqual__ to TOK_GNU_TYPEOF to test
    // forward compatibility of the parser pipeline.
    assert_eq!(fix.tok_types[0], TOK_GNU_TYPEOF);
    assert_full_pipeline_parity(&fix, "typeof_unqual_simulated");

    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);
    assert!(
        row_indices(&typed, node_kind::VARIABLE).contains(&4),
        "z must classify as VARIABLE after simulated typeof_unqual"
    );
}

#[test]
fn atomic_typeof_pointer_combo_gpu_cpu_parity() {
    let fix = fixture_atomic_typeof_pointer();
    assert_eq!(fix.tok_types[0], TOK_ATOMIC, "_Atomic must promote");
    assert_eq!(fix.tok_types[1], TOK_GNU_TYPEOF, "typeof must promote");
    assert_full_pipeline_parity(&fix, "atomic_typeof_pointer");

    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);
    assert!(
        row_indices(&typed, C_AST_KIND_POINTER_DECL).contains(&5),
        "pointer declarator must be present"
    );
    assert!(
        row_indices(&typed, node_kind::VARIABLE).contains(&6),
        "q must classify as VARIABLE"
    );
}

#[test]
fn typeof_function_returning_fnptr_gpu_cpu_parity() {
    let fix = fixture_typeof_function_returning_fnptr();
    assert_eq!(fix.tok_types[0], TOK_GNU_TYPEOF);
    assert_full_pipeline_parity(&fix, "typeof_function_returning_fnptr");

    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw, fix.source.as_bytes());
    let typed = reference_c11_classify_vast_node_kinds(&annotated);

    let fn_decls = row_indices(&typed, C_AST_KIND_FUNCTION_DECLARATOR);
    assert_eq!(
        fn_decls.len(),
        2,
        "must contain two function declarators (f(void) and (float))"
    );
    let ptrs = row_indices(&typed, C_AST_KIND_POINTER_DECL);
    assert_eq!(ptrs.len(), 2, "must contain two pointer declarators");
    assert!(
        row_indices(&typed, node_kind::FUNCTION_DECL).contains(&7),
        "f must classify as FUNCTION_DECL"
    );
}
