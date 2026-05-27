//! Advanced C declaration and declarator contracts.
//!
//! Coverage gaps filled:
//!   * deeply nested struct / union / enum definitions
//!   * anonymous struct/union members
//!   * typedefs with multiple complex declarators (struct tag + pointer)
//!   * triple-star pointers with interleaved qualifiers
//!   * storage-class combinations: _Thread_local, _Atomic, register, inline
//!   * bit-fields inside nested structs
//!   * GNU attributes on struct fields and typedef declarations
//!   * pointer-to-function-pointer declarators
//!   * arrays of function pointers with qualified parameters
//!
//! Every test asserts CPU/GPU parity and meaningful AST/VAST/PG invariants.
//! A missing GPU adapter is a configuration failure, never a skip.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[allow(dead_code)]
mod c_ast_gpu_parity_support;

use c_ast_gpu_parity_support::{
    assert_full_pipeline_parity, assert_pg_preserves_row, build_fixture, kind_at, row_indices,
    run_gpu_pg_lower_with_count as run_gpu_pg_lower, word_at, Fixture, FixtureToken,
    VAST_STRIDE_U32,
};
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::lower::reference_ast_to_pg_nodes;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_annotate_typedef_names, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds, C_AST_KIND_ARRAY_DECL, C_AST_KIND_BIT_FIELD_DECL,
    C_AST_KIND_ENUMERATOR_DECL, C_AST_KIND_ENUM_DECL, C_AST_KIND_FIELD_DECL,
    C_AST_KIND_FUNCTION_DECLARATOR, C_AST_KIND_GNU_ATTRIBUTE, C_AST_KIND_POINTER_DECL,
    C_AST_KIND_STRUCT_DECL, C_AST_KIND_TYPEDEF_DECL, C_AST_KIND_UNION_DECL,
};
use vyre_primitives::predicate::node_kind;

const TYPEDEF_FLAGS_FIELD: usize = 7;
const TYPEDEF_FLAG_DECL: u32 = 1 << 1;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn flags_at(rows: &[u8], idx: usize) -> u32 {
    word_at(rows, idx * VAST_STRIDE_U32 + TYPEDEF_FLAGS_FIELD)
}

fn node_count_from_vast(vast: &[u8]) -> u32 {
    (vast.len() / (VAST_STRIDE_U32 * 4)) as u32
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// ```c
/// struct outer {
///     union {
///         struct { int x; } s;
///         int y;
///     } u;
///     enum { A = 1, B = 2 } e;
/// };
/// ```
fn fixture_nested_struct_union_enum() -> Fixture {
    build_fixture(&[
        FixtureToken::new("struct", TOK_IDENTIFIER),
        FixtureToken::new("outer", TOK_IDENTIFIER),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("union", TOK_IDENTIFIER),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("struct", TOK_IDENTIFIER),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("x", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new("s", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("y", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new("u", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("enum", TOK_IDENTIFIER),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("A", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("1", TOK_INTEGER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("B", TOK_IDENTIFIER),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("2", TOK_INTEGER),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new("e", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// struct {
///     union {
///         int i;
///         float f;
///     };
///     int tag;
/// };
/// ```
fn fixture_anonymous_struct_union() -> Fixture {
    build_fixture(&[
        FixtureToken::new("struct", TOK_IDENTIFIER),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("union", TOK_IDENTIFIER),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("i", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("float", TOK_IDENTIFIER),
        FixtureToken::new("f", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("tag", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// typedef struct Node { int v; } Node, *NodePtr;
/// ```
fn fixture_typedef_multiple_declarators() -> Fixture {
    build_fixture(&[
        FixtureToken::new("typedef", TOK_IDENTIFIER),
        FixtureToken::new("struct", TOK_IDENTIFIER),
        FixtureToken::new("Node", TOK_IDENTIFIER),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("v", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new("Node", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("NodePtr", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// const int * const * volatile * restrict p;
/// ```
fn fixture_deeply_nested_pointer() -> Fixture {
    build_fixture(&[
        FixtureToken::new("const", TOK_IDENTIFIER),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("const", TOK_IDENTIFIER),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("volatile", TOK_IDENTIFIER),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("restrict", TOK_IDENTIFIER),
        FixtureToken::new("p", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// static inline int f(void);
/// extern register int x;
/// _Thread_local _Atomic int y;
/// ```
fn fixture_storage_class_combinations() -> Fixture {
    build_fixture(&[
        FixtureToken::new("static", TOK_IDENTIFIER),
        FixtureToken::new("inline", TOK_IDENTIFIER),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("f", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("void", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("extern", TOK_IDENTIFIER),
        FixtureToken::new("register", TOK_IDENTIFIER),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("x", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("_Thread_local", TOK_IDENTIFIER),
        FixtureToken::new("_Atomic", TOK_IDENTIFIER),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("y", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// struct {
///     unsigned int a : 4;
///     struct {
///         int b : 8;
///         unsigned int : 0;
///     } inner;
/// };
/// ```
fn fixture_bitfield_nested_struct() -> Fixture {
    build_fixture(&[
        FixtureToken::new("struct", TOK_IDENTIFIER),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("unsigned", TOK_IDENTIFIER),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("a", TOK_IDENTIFIER),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("4", TOK_INTEGER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("struct", TOK_IDENTIFIER),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("b", TOK_IDENTIFIER),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("8", TOK_INTEGER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("unsigned", TOK_IDENTIFIER),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("0", TOK_INTEGER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new("inner", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// struct {
///     __attribute__((aligned(8))) int x;
/// };
/// typedef int __attribute__((packed)) packed_int;
/// ```
fn fixture_gnu_attribute_field_and_typedef() -> Fixture {
    build_fixture(&[
        FixtureToken::new("struct", TOK_IDENTIFIER),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("__attribute__", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("aligned", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("8", TOK_INTEGER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("x", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("typedef", TOK_IDENTIFIER),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("__attribute__", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("packed", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("packed_int", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// int (**fp)(void);
/// ```
fn fixture_function_pointer_to_pointer() -> Fixture {
    build_fixture(&[
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("fp", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("void", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

/// ```c
/// int (*handlers[4])(int, const char * restrict);
/// ```
fn fixture_array_of_function_pointers_qualified() -> Fixture {
    build_fixture(&[
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("handlers", TOK_IDENTIFIER),
        FixtureToken::new("[", TOK_LBRACKET),
        FixtureToken::new("4", TOK_INTEGER),
        FixtureToken::new("]", TOK_RBRACKET),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("const", TOK_IDENTIFIER),
        FixtureToken::new("char", TOK_IDENTIFIER),
        FixtureToken::new("*", TOK_STAR),
        FixtureToken::new("restrict", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

// ---------------------------------------------------------------------------
// CPU reference contracts  -  nested struct/union/enum
// ---------------------------------------------------------------------------

mod c_ast_declaration_advanced_contracts_part1 {

    include!("__split/c_ast_declaration_advanced_contracts_part1.rs");
}
mod c_ast_declaration_advanced_contracts_part2 {
    include!("__split/c_ast_declaration_advanced_contracts_part2.rs");
}
