//! Deep GPU/CPU parity contracts for extended GNU asm forms and GNU attributes.
//!
//! Covers:
//!   - asm volatile, asm goto, asm volatile goto with outputs/inputs/clobbers/labels
//!   - named asm operands (`[name] "constraint" (expr)`)
//!   - asm aliases on declarations (`int foo asm("bar") = 1;`)
//!   - GNU attributes: section, weak, alias, aligned, packed, cleanup,
//!     constructor, destructor, mode, visibility, naked, used, unused
//!
//! A missing GPU adapter is a configuration failure; tests panic loudly.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;

use c_ast_gpu_parity_support::{
    assert_full_pipeline_parity, build_fixture, classify, row_indices, word_at, Fixture,
    FixtureToken, VAST_STRIDE_U32,
};
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_ASM_CLOBBERS_LIST, C_AST_KIND_ASM_GOTO_LABELS, C_AST_KIND_ASM_INPUT_OPERAND,
    C_AST_KIND_ASM_OUTPUT_OPERAND, C_AST_KIND_ASM_QUALIFIER, C_AST_KIND_ASM_TEMPLATE,
    C_AST_KIND_ATTRIBUTE_ALIAS, C_AST_KIND_ATTRIBUTE_ALIGNED, C_AST_KIND_ATTRIBUTE_CLEANUP,
    C_AST_KIND_ATTRIBUTE_CONSTRUCTOR, C_AST_KIND_ATTRIBUTE_DESTRUCTOR, C_AST_KIND_ATTRIBUTE_MODE,
    C_AST_KIND_ATTRIBUTE_NAKED, C_AST_KIND_ATTRIBUTE_PACKED, C_AST_KIND_ATTRIBUTE_SECTION,
    C_AST_KIND_ATTRIBUTE_UNUSED, C_AST_KIND_ATTRIBUTE_USED, C_AST_KIND_ATTRIBUTE_VISIBILITY,
    C_AST_KIND_ATTRIBUTE_WEAK, C_AST_KIND_FIELD_DECL, C_AST_KIND_FUNCTION_DEFINITION,
    C_AST_KIND_GNU_ATTRIBUTE, C_AST_KIND_GOTO_STMT, C_AST_KIND_INLINE_ASM,
};
use vyre_primitives::predicate::node_kind;

// ---------------------------------------------------------------------------
// Fixture builders  -  extended GNU asm
// ---------------------------------------------------------------------------

fn fixture_asm_goto_with_outputs_inputs_clobbers_labels() -> Fixture {
    build_fixture(&[
        FixtureToken::new("asm", TOK_IDENTIFIER),
        FixtureToken::new("goto", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("\"jmp %l0\"", TOK_STRING),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("\"=r\"", TOK_STRING),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("out", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("\"r\"", TOK_STRING),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("in", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("\"memory\"", TOK_STRING),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("\"cc\"", TOK_STRING),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("fail", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("ok", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

fn fixture_asm_volatile_goto_with_outputs_inputs_clobbers_labels() -> Fixture {
    build_fixture(&[
        FixtureToken::new("asm", TOK_IDENTIFIER),
        FixtureToken::new("volatile", TOK_IDENTIFIER),
        FixtureToken::new("goto", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("\"jmp %l0\"", TOK_STRING),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("\"=r\"", TOK_STRING),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("out", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("\"r\"", TOK_STRING),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("in", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("\"memory\"", TOK_STRING),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("fail", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

fn fixture_asm_volatile_with_outputs_inputs_clobbers() -> Fixture {
    build_fixture(&[
        FixtureToken::new("asm", TOK_IDENTIFIER),
        FixtureToken::new("volatile", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("\"mov %1, %0\"", TOK_STRING),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("\"=r\"", TOK_STRING),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("out", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("\"r\"", TOK_STRING),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("in", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("\"memory\"", TOK_STRING),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("\"cc\"", TOK_STRING),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

fn fixture_asm_named_operands() -> Fixture {
    build_fixture(&[
        FixtureToken::new("asm", TOK_IDENTIFIER),
        FixtureToken::new("volatile", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("\"mov %[src], %[dst]\"", TOK_STRING),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("[", TOK_LBRACKET),
        FixtureToken::new("dst", TOK_IDENTIFIER),
        FixtureToken::new("]", TOK_RBRACKET),
        FixtureToken::new("\"=r\"", TOK_STRING),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("out", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("[", TOK_LBRACKET),
        FixtureToken::new("src", TOK_IDENTIFIER),
        FixtureToken::new("]", TOK_RBRACKET),
        FixtureToken::new("\"r\"", TOK_STRING),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("in", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

fn fixture_asm_alias_declaration() -> Fixture {
    build_fixture(&[
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("foo", TOK_IDENTIFIER),
        FixtureToken::new("asm", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("\"bar\"", TOK_STRING),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("=", TOK_ASSIGN),
        FixtureToken::new("1", TOK_INTEGER),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

// ---------------------------------------------------------------------------
// Fixture builders  -  GNU attribute-specific kinds (deep coverage)
// ---------------------------------------------------------------------------

fn fixture_attribute_cleanup() -> Fixture {
    build_fixture(&[
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("x", TOK_IDENTIFIER),
        FixtureToken::new("__attribute__", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("cleanup", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("clean", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

fn fixture_attribute_constructor() -> Fixture {
    build_fixture(&[
        FixtureToken::new("__attribute__", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("constructor", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("void", TOK_IDENTIFIER),
        FixtureToken::new("init", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("void", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("}", TOK_RBRACE),
    ])
}

fn fixture_attribute_destructor() -> Fixture {
    build_fixture(&[
        FixtureToken::new("__attribute__", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("destructor", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("void", TOK_IDENTIFIER),
        FixtureToken::new("fini", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("void", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("}", TOK_RBRACE),
    ])
}

fn fixture_attribute_mode() -> Fixture {
    build_fixture(&[
        FixtureToken::new("__attribute__", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("mode", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("SI", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("x", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

fn fixture_attribute_packed_struct() -> Fixture {
    build_fixture(&[
        FixtureToken::new("struct", TOK_IDENTIFIER),
        FixtureToken::new("__attribute__", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("packed", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("S", TOK_IDENTIFIER),
        FixtureToken::new("{", TOK_LBRACE),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("x", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
        FixtureToken::new("}", TOK_RBRACE),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

fn fixture_attribute_combined_variable() -> Fixture {
    build_fixture(&[
        FixtureToken::new("__attribute__", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("section", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("\".text\"", TOK_STRING),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("weak", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("alias", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("\"real\"", TOK_STRING),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("aligned", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("8", TOK_INTEGER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("used", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("unused", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("naked", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("visibility", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("\"hidden\"", TOK_STRING),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("combo", TOK_IDENTIFIER),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

fn fixture_non_attribute_identifiers() -> Fixture {
    build_fixture(&[
        FixtureToken::new("void", TOK_IDENTIFIER),
        FixtureToken::new("cleanup", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("int", TOK_IDENTIFIER),
        FixtureToken::new("section", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

// ---------------------------------------------------------------------------
// Tests  -  extended GNU asm forms
// ---------------------------------------------------------------------------

mod c_ast_gnu_asm_attribute_deep_contracts_part1 {

    include!("__split/c_ast_gnu_asm_attribute_deep_contracts_part1.rs");
}
mod c_ast_gnu_asm_attribute_deep_contracts_part2 {
    include!("__split/c_ast_gnu_asm_attribute_deep_contracts_part2.rs");
}
