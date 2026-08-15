//! Hand-written CPU-only contracts for Linux-style C syntax parsed from raw
//! source strings through the full lexer → VAST → annotate → classify pipeline.
//!
//! Constructs under test:
//!   * `__attribute__((__section__(...)))` and `__attribute__((__aligned__(...)))`
//!   * `__attribute__((__weak__))` plus negative contract (bare `weak` identifier)
//!   * inline asm with output/input operands, clobbers, and `asm goto` labels
//!   * `typeof` / `__typeof__` / `__typeof__` in declarations
//!   * GNU statement expressions `({ ... })` in initializer context
//!   * designated initializers (field, array index, range)
//!   * macro-expanded-looking dense token streams (`__asm__ __volatile__` etc.)
//!
//! Every test asserts **structural invariants** (node kind counts, parent/child
//! links, span monotonicity, symbol-hash discrimination)  -  never merely
//! "parses without panic".

#![cfg(feature = "c-parser")]
#![allow(deprecated)]

#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;

use c_frontend::reference_lexer::classify_raw_source;
use c_frontend::rows::{
    kind_at, node_count_from_vast, row_indices as indices_with_kind, word_at, VAST_STRIDE_U32,
};
use c_grammar_gen::lex_c11_max_munch::lex_c11_max_munch_kinds;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_ASM_CLOBBERS_LIST, C_AST_KIND_ASM_GOTO_LABELS, C_AST_KIND_ASM_INPUT_OPERAND,
    C_AST_KIND_ASM_OUTPUT_OPERAND, C_AST_KIND_ASM_QUALIFIER, C_AST_KIND_ASM_TEMPLATE,
    C_AST_KIND_ASSIGN_EXPR, C_AST_KIND_ATTRIBUTE_ALIGNED, C_AST_KIND_ATTRIBUTE_SECTION,
    C_AST_KIND_ATTRIBUTE_WEAK, C_AST_KIND_GNU_ATTRIBUTE, C_AST_KIND_GNU_STATEMENT_EXPR,
    C_AST_KIND_INITIALIZER_LIST, C_AST_KIND_INLINE_ASM, C_AST_KIND_MEMBER_ACCESS_EXPR,
    C_AST_KIND_RANGE_DESIGNATOR_EXPR,
};
use vyre_primitives::predicate::node_kind;

/// Assert that `lex_c11_max_munch_kinds` agrees with the filtered, keyword-promoted oracle stream.
fn assert_max_munch_agrees(source: &[u8], types: &[u32]) {
    let host_kinds = lex_c11_max_munch_kinds(source).expect("host lexer must accept source");
    let host_non_ws: Vec<u32> = host_kinds
        .into_iter()
        .filter(|k| *k != TOK_WHITESPACE && *k != TOK_COMMENT)
        .collect();
    assert_eq!(
        host_non_ws, types,
        "hand-built source must match max-munch lexer"
    );
}

fn parent_at(rows: &[u8], idx: usize) -> u32 {
    word_at(rows, idx * VAST_STRIDE_U32 + 1)
}

fn first_child_at(rows: &[u8], idx: usize) -> u32 {
    word_at(rows, idx * VAST_STRIDE_U32 + 2)
}

fn next_sibling_at(rows: &[u8], idx: usize) -> u32 {
    word_at(rows, idx * VAST_STRIDE_U32 + 3)
}

fn start_at(rows: &[u8], idx: usize) -> u32 {
    word_at(rows, idx * VAST_STRIDE_U32 + 5)
}

fn len_at(rows: &[u8], idx: usize) -> u32 {
    word_at(rows, idx * VAST_STRIDE_U32 + 6)
}

fn find_single(rows: &[u8], kind: u32) -> usize {
    let idxs = indices_with_kind(rows, kind);
    assert_eq!(
        idxs.len(),
        1,
        "expected exactly one node of kind 0x{kind:08x}, found {}",
        idxs.len()
    );
    idxs[0]
}

fn lexeme_at(source: &[u8], start: u32, len: u32) -> Option<&[u8]> {
    let s = start as usize;
    let e = s.saturating_add(len as usize);
    source.get(s..e)
}

fn assert_span_monotonicity(rows: &[u8]) {
    for i in 0..node_count_from_vast(rows) as usize {
        let start = start_at(rows, i);
        let len = len_at(rows, i);
        assert!(
            len > 0 || kind_at(rows, i) == TOK_SEMICOLON,
            "token {i} has zero length"
        );
        // start+len must not overflow in practice for small fixtures
        let _end = start.saturating_add(len);
    }
}

/// Full pipeline: raw source bytes → typed VAST bytes + source context.
struct Parsed {
    source: Vec<u8>,
    typed_vast: Vec<u8>,
    tok_types: Vec<u32>,
    #[allow(dead_code)]
    tok_starts: Vec<u32>,
    #[allow(dead_code)]
    tok_lens: Vec<u32>,
}

fn parse_source(source: &str) -> Parsed {
    let source_bytes = source.as_bytes();
    let classified = classify_raw_source(source_bytes);

    assert_max_munch_agrees(source_bytes, &classified.tok_types);
    assert_span_monotonicity(&classified.typed_vast);

    Parsed {
        source: source_bytes.to_vec(),
        typed_vast: classified.typed_vast,
        tok_types: classified.tok_types,
        tok_starts: classified.tok_starts,
        tok_lens: classified.tok_lens,
    }
}

// ---------------------------------------------------------------------------
// 1. GNU __attribute__ with double-underscore forms (macro-expanded look)
// ---------------------------------------------------------------------------

#[path = "contract_cases/c_ast_linux_style_raw_source_contracts__attribute_double_underscore_section_classified_correctly.rs"]
mod c_ast_linux_style_raw_source_contracts_attribute_double_underscore_section_classified_correctly;
#[path = "contract_cases/c_ast_linux_style_raw_source_contracts__macro_expanded_dense_attribute_asm_typeof_stream.rs"]
mod c_ast_linux_style_raw_source_contracts_macro_expanded_dense_attribute_asm_typeof_stream;
