//! Deep contract tests for C preprocessor macro behavior and directive
//! preservation across host lexer, GPU lexer, dynamic macro expansion,
//! conditional masking, and VAST→PG lowering.
//!
//! Covers:
//! - directive preservation (host & GPU lexer, VAST, PG)
//! - line continuations (backslash-newline splicing)
//! - function-like macro shapes (expansion + argument token preservation)
//! - nested macro calls (single-pass non-recursive replacement, VAST CALL survival)
//! - token pasting (## lexing and expansion-level passthrough)
//! - stringification (# lexing and expansion-level passthrough)
//! - variadic trailing comma behavior (definition lexing + call classification)
//! - conditional directives as token streams (raw/typed VAST, PG, GPU parity)
//! - malformed directives fail-loud behavior (zero-length expansion, malformed rows)
//! - span preservation (lexer → VAST → PG)

#![cfg(feature = "c-parser")]
#![allow(clippy::erasing_op)]
#![allow(deprecated)]

#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
mod c_token_support;
mod common;
use c_frontend::macro_expansion::{run_dynamic_macro_expansion, MacroFixture};
use c_frontend::token_fixture::c_fixture;
use c_token_support::{
    assert_pg_row, assert_shape_none, find_row_for_lexeme, node_count_from_vast, row_typed_kind,
    run_c11_lexer, run_cpu_pipeline, word_at, PG_STRIDE_U32,
};
use common::{decode_u32_words, u32_bytes};
use std::sync::OnceLock;

use c_grammar_gen::lex_c11_max_munch_kinds;
use vyre::ir::Expr;
use vyre_driver::VyreBackend;
use vyre_driver_wgpu::WgpuBackend;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::lower::c_lower_ast_to_pg_nodes;
use vyre_libs::parsing::c::parse::vast::{
    c11_build_expression_shape_nodes, c11_classify_vast_node_kinds,
};
use vyre_libs::parsing::c::preprocess::expansion::opt_conditional_mask_with_directives;
use vyre_libs::parsing::c::preprocess::{
    c_translation_phase_line_splice, reference_c_preprocessor_directive_metadata,
};
use vyre_primitives::predicate::node_kind;
use vyre_reference::value::Value;

// ---------------------------------------------------------------------------
// Byte / word helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GPU lexer helper
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------

fn run_conditional_mask_with_directives(
    tok_types: &[u32],
    directive_kinds: &[u32],
    directive_values: &[u32],
) -> Result<Vec<Value>, vyre_reference::ReferenceError> {
    let program = opt_conditional_mask_with_directives(
        "tok_types",
        "directive_kinds",
        "directive_values",
        "out_mask",
        Expr::u32(tok_types.len() as u32),
    );
    let values = [
        Value::from(u32_bytes(tok_types)),
        Value::from(u32_bytes(directive_kinds)),
        Value::from(u32_bytes(directive_values)),
        Value::from(vec![0u8; tok_types.len() * 4]),
    ];
    vyre_reference::reference_eval(&program, &values)
}

// ---------------------------------------------------------------------------
// VAST / PG pipeline helpers
// ---------------------------------------------------------------------------

const VAST_STRIDE_U32: usize = 10;

// ---------------------------------------------------------------------------
// GPU backend helpers
// ---------------------------------------------------------------------------

fn gpu_backend() -> &'static WgpuBackend {
    static BACKEND: OnceLock<WgpuBackend> = OnceLock::new();
    BACKEND.get_or_init(|| {
        WgpuBackend::acquire().expect(
            "WgpuBackend::acquire failed on a machine that must have a GPU. \
             This is a configuration bug, not a graceful skip.",
        )
    })
}

fn run_gpu_classify(vast: &[u8]) -> Vec<u8> {
    let n = node_count_from_vast(vast);
    let program = c11_classify_vast_node_kinds("vast_nodes", Expr::u32(n), "typed_vast_nodes");
    let inputs: Vec<&[u8]> = vec![vast];
    gpu_backend()
        .dispatch_borrowed(&program, &inputs, &Default::default())
        .expect("GPU classify dispatch must succeed")
        .into_iter()
        .next()
        .expect("one typed VAST output")
}

fn run_gpu_expr_shape(raw_vast: &[u8], typed_vast: &[u8]) -> Vec<u8> {
    let program = c11_build_expression_shape_nodes(
        "raw_vast_nodes",
        "typed_vast_nodes",
        Expr::u32(node_count_from_vast(raw_vast)),
        "expr_shape_nodes",
    );
    let inputs: Vec<&[u8]> = vec![raw_vast, typed_vast];
    gpu_backend()
        .dispatch_borrowed(&program, &inputs, &Default::default())
        .expect("GPU expression-shape dispatch must succeed")
        .into_iter()
        .next()
        .expect("one expr-shape output")
}

fn run_gpu_pg_lower(typed_vast: &[u8]) -> Vec<u8> {
    let program = c_lower_ast_to_pg_nodes(
        "vast_nodes",
        Expr::u32(node_count_from_vast(typed_vast)),
        "pg_nodes",
    );
    let inputs: Vec<&[u8]> = vec![typed_vast];
    gpu_backend()
        .dispatch_borrowed(&program, &inputs, &Default::default())
        .expect("GPU PG lowerer dispatch must succeed")
        .into_iter()
        .next()
        .expect("one PG output")
}

// ---------------------------------------------------------------------------
// 1. Directive preservation
// ---------------------------------------------------------------------------

#[path = "c_preprocess_macro_deep_contracts/conditional_directives_and_span_preservation.rs"]
mod conditional_directives_and_span_preservation;
#[path = "c_preprocess_macro_deep_contracts/directive_rows_and_macro_operators.rs"]
mod directive_rows_and_macro_operators;
