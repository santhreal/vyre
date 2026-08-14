//! The GPU-lexer half of the C token harness.
//!
//! Fixture assembly, row accessors, the CPU reference pipeline and the row
//! assertions live in `tests/support/c_frontend`, which `vyre-libs` shares, so
//! what stays here is the lexer op run through the reference oracle and the
//! assertions that read a lowered row against a fixture's own spans.

#![allow(deprecated)]

use vyre_libs::parsing::c::lex::tokens::{TOK_COMMENT, TOK_WHITESPACE};
use vyre_libs::parsing::c::parse::vast::{
    C_EXPR_ASSOC_NONE, C_EXPR_SHAPE_NONE, C_EXPR_SHAPE_STRIDE_U32,
};

// Each consumer includes this module and uses a subset of what it re-exports,
// so an unused re-export here is a fact about one test binary, not dead code.
#[allow(unused_imports)]
pub(crate) use crate::c_frontend::expression_pipeline::{
    run_pipeline_for_fixture as run_cpu_pipeline, PipelineRows,
};

#[allow(unused_imports)]
pub(crate) use crate::c_frontend::rows::{
    node_count_from_vast, word_at, PG_STRIDE_U32, VAST_STRIDE_U32,
};
#[allow(unused_imports)]
pub(crate) use crate::c_frontend::token_fixture::Fixture;

/// Find the first token row whose source span equals `needle`.
pub(crate) fn find_row_for_lexeme(assembled: &Fixture, needle: &str) -> usize {
    assembled
        .tok_starts
        .iter()
        .zip(&assembled.tok_lens)
        .position(|(start, len)| {
            let start = *start as usize;
            let end = start.saturating_add(*len as usize);
            assembled.source.as_bytes().get(start..end) == Some(needle.as_bytes())
        })
        .unwrap_or_else(|| panic!("lexeme {needle:?} not found in fixture"))
}

/// Read the kind field from one typed VAST row.
pub(crate) fn row_typed_kind(typed: &[u8], row: usize) -> u32 {
    word_at(typed, row * VAST_STRIDE_U32)
}

/// Assert a typed VAST row and its lowered Program Graph source span.
pub(crate) fn assert_pg_row(assembled: &Fixture, pg: &[u8], typed: &[u8], idx: usize, kind: u32) {
    assert_eq!(
        word_at(typed, idx * VAST_STRIDE_U32),
        kind,
        "typed kind at {idx}"
    );
    assert_eq!(word_at(pg, idx * PG_STRIDE_U32), kind, "PG kind at {idx}");
    assert_eq!(
        word_at(pg, idx * PG_STRIDE_U32 + 1),
        assembled.tok_starts[idx],
        "PG span_start at {idx}"
    );
    assert_eq!(
        word_at(pg, idx * PG_STRIDE_U32 + 2),
        assembled.tok_starts[idx] + assembled.tok_lens[idx],
        "PG span_end at {idx}"
    );
}

/// Assert the fixture's own lexemes re-lex to the kinds it declares, so a
/// hand-built stream cannot assert a tokenization the lexer does not produce.
pub(crate) fn assert_lex_matches_non_ws(assembled: &Fixture) {
    let kinds = c_grammar_gen::lex_c11_max_munch_kinds(assembled.source.as_bytes())
        .expect("lex fixture source");
    let filtered: Vec<u32> = kinds
        .into_iter()
        .filter(|k| *k != TOK_WHITESPACE && *k != TOK_COMMENT)
        .collect();
    assert_eq!(
        filtered, assembled.raw_kinds,
        "hand-built fixture must match max-munch lexer (no fake tokenization)"
    );
}

/// A structural or preprocessing row carries no expression shape.
pub(crate) fn assert_shape_none(expr_shape: &[u8], idx: usize) {
    let base = idx * C_EXPR_SHAPE_STRIDE_U32 as usize;
    assert_eq!(
        word_at(expr_shape, base),
        C_EXPR_SHAPE_NONE,
        "preproc/structural rows stay shape-none"
    );
    assert_eq!(word_at(expr_shape, base + 4), C_EXPR_ASSOC_NONE);
}

#[allow(unused_imports)]
pub(crate) use crate::c_frontend::reference_lexer::run_c11_lexer;
