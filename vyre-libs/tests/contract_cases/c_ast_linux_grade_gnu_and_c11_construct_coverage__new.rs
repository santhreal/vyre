// CPU-only reference tests for Linux-grade C constructs not covered by the
// existing GNU extension test suite.
//
// Constructs under test:
//   - `__auto_type` declarations
//   - `typeof` / `__typeof__` used as declaration type-specifiers
//   - `__attribute__` variants: `cleanup`, `constructor`, `destructor`, `mode`,
//     `packed`, `aligned`
//   - labels and computed-goto interactions
//   - statement expressions in initializer and declarator positions
//   - `_Static_assert`
//   - `_Alignas` / `_Alignof`
//
// These tests exercise the CPU reference pipeline (build -> annotate -> classify).

#[path = "c_ast_linux_grade_gnu_and_c11_construct_coverage__cpu_reference_attribute_constructor_parses.rs"]
mod c_ast_linux_grade_gnu_and_c11_construct_coverage_cpu_reference_attribute_constructor_parses;

use crate::c_frontend::spelling::c_tokens;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_ASSIGN_EXPR, C_AST_KIND_ATTRIBUTE_ALIGNED, C_AST_KIND_ATTRIBUTE_CLEANUP,
    C_AST_KIND_ATTRIBUTE_CONSTRUCTOR, C_AST_KIND_ATTRIBUTE_DESTRUCTOR, C_AST_KIND_ATTRIBUTE_MODE,
    C_AST_KIND_ATTRIBUTE_PACKED, C_AST_KIND_FUNCTION_DEFINITION, C_AST_KIND_GNU_ATTRIBUTE,
    C_AST_KIND_GNU_LABEL_ADDRESS_EXPR, C_AST_KIND_GOTO_STMT, C_AST_KIND_LABEL_STMT,
};
use vyre_libs::predicate::node_kind;

use crate::c_frontend::rows::{row_indices_by_stride as row_indices, word_at, VAST_STRIDE_U32};
use crate::c_frontend::token_fixture::{classify, Fixture};

fn kind_at(rows: &[u8], idx: usize) -> u32 {
    word_at(rows, idx * VAST_STRIDE_U32)
}

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

fn fixture_auto_type_decl() -> Fixture {
    c_tokens("__auto_type x = 1 ;")
}

fn fixture_typeof_in_decl() -> Fixture {
    c_tokens("__typeof__ ( int ) y ; typeof ( long ) z ;")
}

fn fixture_attribute_cleanup() -> Fixture {
    c_tokens("__attribute__ ( ( cleanup ( free ) ) ) void * p ;")
}

fn fixture_attribute_constructor() -> Fixture {
    c_tokens("__attribute__ ( ( constructor ) ) void init ( void ) { }")
}

fn fixture_attribute_destructor() -> Fixture {
    c_tokens("__attribute__ ( ( destructor ) ) void fini ( void ) { }")
}

fn fixture_attribute_mode() -> Fixture {
    c_tokens("__attribute__ ( ( mode ( __word__ ) ) ) unsigned int w ;")
}

fn fixture_attribute_packed_and_aligned() -> Fixture {
    c_tokens(
        "struct __attribute__ ( ( packed ) ) s { char c ; } ; __attribute__ ( ( aligned ( 8 ) ) ) \
         long l ;",
    )
}

fn fixture_computed_goto() -> Fixture {
    c_tokens("void f ( void * p ) { goto * p ; }")
}

fn fixture_label_and_computed_goto_interaction() -> Fixture {
    c_tokens("void g ( void ) { void * t = && lbl ; goto * t ; lbl : return ; }")
}

fn fixture_stmt_expr_initializer() -> Fixture {
    c_tokens("int a = ( { int b = 1 ; b + 2 ; } ) ;")
}

fn fixture_stmt_expr_in_declarator() -> Fixture {
    // GNU C allows statement expressions in array sizes:
    // int arr[({ int x = 2; x; })];
    c_tokens("int arr [ ( { int x = 2 ; x ; } ) ] ;")
}

fn fixture_static_assert() -> Fixture {
    c_tokens("_Static_assert ( 1 , \"ok\" ) ;")
}

fn fixture_alignas_var() -> Fixture {
    c_tokens("_Alignas ( 16 ) char buf [ 64 ] ;")
}

fn fixture_alignof_expr() -> Fixture {
    c_tokens("int a = _Alignof ( int ) ;")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn cpu_reference_auto_type_decl_parses_without_panic() {
    let fix = fixture_auto_type_decl();
    let typed = classify(&fix);

    assert!(
        !typed.is_empty(),
        "__auto_type fixture must produce a non-empty typed VAST"
    );
    assert_eq!(
        fix.tok_types[0], TOK_GNU_AUTO_TYPE,
        "__auto_type must promote to a declaration type token"
    );
}

#[test]
fn cpu_reference_typeof_in_decl_position() {
    let fix = fixture_typeof_in_decl();
    let typed = classify(&fix);

    assert!(
        !typed.is_empty(),
        "typeof declaration fixture must produce a non-empty typed VAST"
    );
    assert_eq!(
        fix.tok_types[0], TOK_GNU_TYPEOF,
        "__typeof__ must promote to TOK_GNU_TYPEOF"
    );
    assert_eq!(
        fix.tok_types[6], TOK_GNU_TYPEOF,
        "typeof must promote to TOK_GNU_TYPEOF"
    );

    // Both y and z should classify as variables (declarator identifiers)
    let vars = row_indices(&typed, VAST_STRIDE_U32, node_kind::VARIABLE);
    assert!(vars.contains(&4), "y declarator must classify as VARIABLE");
    assert!(vars.contains(&10), "z declarator must classify as VARIABLE");
}

#[test]
fn cpu_reference_attribute_cleanup_parses() {
    let fix = fixture_attribute_cleanup();
    let typed = classify(&fix);

    assert!(
        !typed.is_empty(),
        "cleanup attribute fixture must produce a non-empty typed VAST"
    );
    assert_eq!(
        row_indices(&typed, VAST_STRIDE_U32, C_AST_KIND_GNU_ATTRIBUTE),
        vec![0],
        "__attribute__ must classify as GNU_ATTRIBUTE"
    );
    assert_eq!(
        row_indices(&typed, VAST_STRIDE_U32, C_AST_KIND_ATTRIBUTE_CLEANUP),
        vec![3],
        "cleanup must classify as a specific GNU attribute kind"
    );
}
