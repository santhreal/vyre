//! End-to-end C VAST typedef annotation regression for parameter-scope restore.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "c_ast_gpu_parity_support/mod.rs"]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;

use c_ast_gpu_parity_support::{
    kind_at, run_gpu_classifier_with_count, run_gpu_full_typedef_annotation, word_at,
    VAST_STRIDE_U32,
};
use c_frontend::scope_fixture::{
    annotate_cpu, classify_cpu_annotated, fixture, ident, raw_vast, tok, ScopeFixture,
};
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_classify_vast_node_kinds, C_AST_KIND_POINTER_DECL,
};
use vyre_primitives::predicate::node_kind;

const TYPEDEF_FLAGS_FIELD: usize = 7;
const TYPEDEF_FLAG_VISIBLE: u32 = 1;
const TYPEDEF_FLAG_DECL: u32 = 1 << 1;
const ORDINARY_FLAG_DECL: u32 = 1 << 2;

fn typedef_restore_fixture() -> ScopeFixture {
    fixture(
        "typedef_parameter_scope_restore",
        &[
            tok(TOK_TYPEDEF),
            tok(TOK_INT),
            ident("T"),
            tok(TOK_SEMICOLON),
            tok(TOK_VOID),
            ident("f"),
            tok(TOK_LPAREN),
            tok(TOK_INT),
            ident("T"),
            tok(TOK_RPAREN),
            tok(TOK_LBRACE),
            ident("T"),
            tok(TOK_STAR),
            ident("y"),
            tok(TOK_SEMICOLON),
            tok(TOK_RBRACE),
            tok(TOK_VOID),
            ident("g"),
            tok(TOK_LPAREN),
            ident("T"),
            tok(TOK_STAR),
            ident("p"),
            tok(TOK_RPAREN),
            tok(TOK_LBRACE),
            tok(TOK_RBRACE),
        ],
    )
}

fn assert_words_eq(actual: &[u8], expected: &[u8]) {
    if actual != expected {
        let actual_words = actual.len() / 4;
        let expected_words = expected.len() / 4;
        let limit = actual_words.min(expected_words);
        for word in 0..limit {
            let actual_word = word_at(actual, word);
            let expected_word = word_at(expected, word);
            if actual_word != expected_word {
                panic!(
                    "word {word} differs: actual={actual_word}, expected={expected_word}, row={}, field={}",
                    word / VAST_STRIDE_U32,
                    word % VAST_STRIDE_U32
                );
            }
        }
        panic!(
            "byte lengths differ: actual={}, expected={}",
            actual.len(),
            expected.len()
        );
    }
}

fn flags_at(rows: &[u8], idx: usize) -> u32 {
    word_at(rows, idx * VAST_STRIDE_U32 + TYPEDEF_FLAGS_FIELD)
}

fn annotate_gpu(fix: &ScopeFixture) -> Vec<u8> {
    let raw = raw_vast(fix);
    run_gpu_full_typedef_annotation(&fix.haystack, &raw)
}

fn classify_gpu(annotated: &[u8], node_count: usize) -> Vec<u8> {
    run_gpu_classifier_with_count(annotated, node_count as u32)
}

fn assert_parameter_shadow_restores_typedef(annotated: &[u8], typed: &[u8]) {
    assert_ne!(
        flags_at(annotated, 2) & TYPEDEF_FLAG_DECL,
        0,
        "global typedef declaration `T` must be marked as a typedef declaration"
    );
    assert_ne!(
        flags_at(annotated, 8) & ORDINARY_FLAG_DECL,
        0,
        "function parameter `T` must be marked as an ordinary declaration"
    );
    assert_eq!(
        flags_at(annotated, 11) & TYPEDEF_FLAG_VISIBLE,
        0,
        "parameter `T` must shadow the typedef inside `f`"
    );
    assert_eq!(
        kind_at(typed, 12),
        node_kind::BINARY,
        "`T * y` inside `f` must classify `*` as multiplication while the typedef is shadowed"
    );
    assert_ne!(
        flags_at(annotated, 19) & TYPEDEF_FLAG_VISIBLE,
        0,
        "typedef `T` must be visible again in the later function parameter list"
    );
    assert_eq!(
        kind_at(typed, 20),
        C_AST_KIND_POINTER_DECL,
        "`T * p` in later function `g` must classify `*` as a pointer declarator"
    );
}

#[test]
fn cpu_parameter_scope_restores_typedef_for_later_function() {
    let fix = typedef_restore_fixture();
    let annotated = annotate_cpu(&fix);
    let typed = classify_cpu_annotated(&fix);

    assert_parameter_shadow_restores_typedef(&annotated, &typed);
}

#[test]
fn gpu_parameter_scope_restores_typedef_for_later_function() {
    let fix = typedef_restore_fixture();
    let expected_annotations = annotate_cpu(&fix);
    let gpu_annotations = annotate_gpu(&fix);
    assert_words_eq(&gpu_annotations, &expected_annotations);

    let expected_typed = reference_c11_classify_vast_node_kinds(&expected_annotations);
    let gpu_typed = classify_gpu(&gpu_annotations, fix.tok_types.len());
    assert_words_eq(&gpu_typed, &expected_typed);
    assert_parameter_shadow_restores_typedef(&gpu_annotations, &gpu_typed);
}
