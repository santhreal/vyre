// CPU-only reference tests for extended GNU asm decomposition and
// GNU attribute-specific AST kinds.
//
// These tests exercise `reference_c11_classify_vast_node_kinds` through
// the full three-stage CPU pipeline (build → annotate → classify) to
// verify that:
//   - asm templates, output operands, input operands, clobbers, and goto
//     labels each receive a distinct `C_AST_KIND_ASM_*` kind.
//   - the eight supported GNU attribute names (`section`, `weak`, `alias`,
//     `aligned`, `used`, `unused`, `naked`, `visibility`) each receive a
//     distinct `C_AST_KIND_ATTRIBUTE_*` kind.
//   - identifiers outside attribute contexts are never mis-classified as
//     attribute-specific kinds.

#[path = "c_ast_gnu_asm_decomposition_and_attribute_kinds__cpu_reference_classifies_attribute_naked.rs"]
mod c_ast_gnu_asm_decomposition_and_attribute_kinds_cpu_reference_classifies_attribute_naked;

use crate::c_frontend::spelling::c_tokens;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_ASM_CLOBBERS_LIST, C_AST_KIND_ASM_GOTO_LABELS, C_AST_KIND_ASM_INPUT_OPERAND,
    C_AST_KIND_ASM_OUTPUT_OPERAND, C_AST_KIND_ASM_TEMPLATE, C_AST_KIND_ATTRIBUTE_ALIAS,
    C_AST_KIND_ATTRIBUTE_ALIGNED, C_AST_KIND_ATTRIBUTE_NAKED, C_AST_KIND_ATTRIBUTE_SECTION,
    C_AST_KIND_ATTRIBUTE_UNUSED, C_AST_KIND_ATTRIBUTE_USED, C_AST_KIND_ATTRIBUTE_VISIBILITY,
    C_AST_KIND_ATTRIBUTE_WEAK, C_AST_KIND_GNU_ATTRIBUTE, C_AST_KIND_GOTO_STMT,
    C_AST_KIND_INLINE_ASM,
};
use vyre_primitives::predicate::node_kind;

use crate::c_frontend::rows::{row_indices_by_stride as row_indices, word_at, VAST_STRIDE_U32};
use crate::c_frontend::token_fixture::{build_fixture, classify, Fixture, FixtureToken};

// ---------------------------------------------------------------------------
// Fixture builders  -  GNU attribute-specific kinds
// ---------------------------------------------------------------------------

fn fixture_attribute_section() -> Fixture {
    c_tokens("__attribute__ ( ( section ( \".text.foo\" ) ) ) void foo ( ) { }")
}

fn fixture_attribute_weak() -> Fixture {
    c_tokens("__attribute__ ( ( weak ) ) int x ;")
}

fn fixture_attribute_alias() -> Fixture {
    c_tokens("__attribute__ ( ( alias ( \"bar\" ) ) ) void foo ( ) ;")
}

fn fixture_attribute_aligned() -> Fixture {
    c_tokens("__attribute__ ( ( aligned ( 16 ) ) ) char buf [ 64 ] ;")
}

fn fixture_attribute_used() -> Fixture {
    c_tokens("__attribute__ ( ( used ) ) static int x ;")
}

fn fixture_attribute_unused() -> Fixture {
    c_tokens("__attribute__ ( ( unused ) ) int x ;")
}

fn fixture_attribute_naked() -> Fixture {
    c_tokens("__attribute__ ( ( naked ) ) void entry ( ) { }")
}

fn fixture_attribute_visibility() -> Fixture {
    c_tokens("__attribute__ ( ( visibility ( \"hidden\" ) ) ) void foo ( ) ;")
}

// ---------------------------------------------------------------------------
// Fixture builders  -  extended GNU asm decomposition
// ---------------------------------------------------------------------------

fn fixture_asm_multiple_outputs() -> Fixture {
    build_fixture(&[
        FixtureToken::new("asm", TOK_IDENTIFIER),
        FixtureToken::new("volatile", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("\"mov %1, %0\"", TOK_STRING),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("\"=r\"", TOK_STRING),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("a", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("\"=r\"", TOK_STRING),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("b", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("\"r\"", TOK_STRING),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("c", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("\"memory\"", TOK_STRING),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("\"cc\"", TOK_STRING),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

fn fixture_asm_multiple_inputs() -> Fixture {
    build_fixture(&[
        FixtureToken::new("asm", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("\"add %0, %1, %2\"", TOK_STRING),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("\"=r\"", TOK_STRING),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("dst", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("\"r\"", TOK_STRING),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("src1", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("\"r\"", TOK_STRING),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("src2", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

fn fixture_asm_goto_multiple_labels() -> Fixture {
    build_fixture(&[
        FixtureToken::new("asm", TOK_IDENTIFIER),
        FixtureToken::new("goto", TOK_IDENTIFIER),
        FixtureToken::new("(", TOK_LPAREN),
        FixtureToken::new("\"jmp %l0\"", TOK_STRING),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new(":", TOK_COLON),
        FixtureToken::new("error", TOK_IDENTIFIER),
        FixtureToken::new(",", TOK_COMMA),
        FixtureToken::new("done", TOK_IDENTIFIER),
        FixtureToken::new(")", TOK_RPAREN),
        FixtureToken::new(";", TOK_SEMICOLON),
    ])
}

fn fixture_basic_asm() -> Fixture {
    c_tokens("asm ( \"nop\" ) ;")
}

fn fixture_non_attribute_identifier() -> Fixture {
    c_tokens("void foo ( section ) ;")
}

// ---------------------------------------------------------------------------
// Tests  -  GNU attribute-specific kinds
// ---------------------------------------------------------------------------

#[test]
fn cpu_reference_classifies_attribute_section() {
    let fix = fixture_attribute_section();
    let typed = classify(&fix);

    assert_eq!(
        row_indices(&typed, VAST_STRIDE_U32, C_AST_KIND_ATTRIBUTE_SECTION),
        vec![3],
        "section attribute name must classify as ATTRIBUTE_SECTION"
    );
    assert_eq!(
        row_indices(&typed, VAST_STRIDE_U32, C_AST_KIND_GNU_ATTRIBUTE),
        vec![0],
        "__attribute__ keyword must classify as GNU_ATTRIBUTE"
    );
}

#[test]
fn cpu_reference_classifies_attribute_weak() {
    let fix = fixture_attribute_weak();
    let typed = classify(&fix);

    assert_eq!(
        row_indices(&typed, VAST_STRIDE_U32, C_AST_KIND_ATTRIBUTE_WEAK),
        vec![3],
        "weak attribute name must classify as ATTRIBUTE_WEAK"
    );
}

#[test]
fn cpu_reference_classifies_attribute_alias() {
    let fix = fixture_attribute_alias();
    let typed = classify(&fix);

    assert_eq!(
        row_indices(&typed, VAST_STRIDE_U32, C_AST_KIND_ATTRIBUTE_ALIAS),
        vec![3],
        "alias attribute name must classify as ATTRIBUTE_ALIAS"
    );
}

#[test]
fn cpu_reference_classifies_attribute_aligned() {
    let fix = fixture_attribute_aligned();
    let typed = classify(&fix);

    assert_eq!(
        row_indices(&typed, VAST_STRIDE_U32, C_AST_KIND_ATTRIBUTE_ALIGNED),
        vec![3],
        "aligned attribute name must classify as ATTRIBUTE_ALIGNED"
    );
}

#[test]
fn cpu_reference_classifies_attribute_used() {
    let fix = fixture_attribute_used();
    let typed = classify(&fix);

    assert_eq!(
        row_indices(&typed, VAST_STRIDE_U32, C_AST_KIND_ATTRIBUTE_USED),
        vec![3],
        "used attribute name must classify as ATTRIBUTE_USED"
    );
}

#[test]
fn cpu_reference_classifies_attribute_unused() {
    let fix = fixture_attribute_unused();
    let typed = classify(&fix);

    assert_eq!(
        row_indices(&typed, VAST_STRIDE_U32, C_AST_KIND_ATTRIBUTE_UNUSED),
        vec![3],
        "unused attribute name must classify as ATTRIBUTE_UNUSED"
    );
}
