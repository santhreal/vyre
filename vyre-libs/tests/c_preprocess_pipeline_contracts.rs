//! Contract tests for the C preprocessor pipeline.
//!
//! Covers: object-like macros, nested function-like macro shapes, token paste,
//! stringize, escaped newlines, directive-position hash versus operator hash,
//! include guards, and overflow/determinism contracts.
//!
//! GPU and host lexing must agree on directive-position `#`: only `#` at the
//! start of a logical line after whitespace starts a preprocessor row; mid-line
//! `#` remains a normal hash token.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]

#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
mod common;

use c_frontend::macro_expansion::{run_dynamic_macro_expansion, MacroFixture};
use c_frontend::reference_lexer::run_c11_lexer;
use common::{decode_u32_words, u32_bytes};
use std::panic::{catch_unwind, AssertUnwindSafe};

use c_grammar_gen::lex_c11_max_munch::lex_c11_max_munch_kinds;
use vyre::ir::Expr;
use vyre_libs::parsing::c::lex::lexer::c11_lexer_regular_sparse_packed_haystack_with_flags;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::preprocess::expansion::opt_conditional_mask;
use vyre_libs::parsing::c::preprocess::{
    c_translation_phase_line_splice, reference_c_preprocessor_directive_metadata,
};
use vyre_reference::value::Value;

fn run_sparse_c11_lexer_positions(source: &[u8]) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let haystack_len = source.len() as u32;
    let program = c11_lexer_regular_sparse_packed_haystack_with_flags(
        "haystack",
        "sparse_types",
        "sparse_starts",
        "sparse_lens",
        "sparse_flags",
        haystack_len,
    );
    let padded_len = source.len().div_ceil(4).max(1) * 4;
    let mut haystack = Vec::with_capacity(padded_len);
    haystack.extend_from_slice(source);
    haystack.resize(padded_len, 0);
    let zero_buf = vec![0u8; source.len() * 4];
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(haystack),
            Value::from(zero_buf.clone()),
            Value::from(zero_buf.clone()),
            Value::from(zero_buf.clone()),
            Value::from(zero_buf),
        ],
    )
    .expect("sparse c11 lexer must execute under the reference oracle");
    assert_eq!(
        outputs.len(),
        4,
        "expected [sparse_types, sparse_starts, sparse_lens, sparse_flags]"
    );
    (
        decode_u32_words(&outputs[0].to_bytes()),
        decode_u32_words(&outputs[1].to_bytes()),
        decode_u32_words(&outputs[2].to_bytes()),
        decode_u32_words(&outputs[3].to_bytes()),
    )
}

// ---------------------------------------------------------------------------

fn run_conditional_mask(tok_types: &[u32]) -> Result<Vec<Value>, vyre_reference::ReferenceError> {
    let program = opt_conditional_mask("tok_types", "out_mask", Expr::u32(tok_types.len() as u32));
    let input_bytes = if tok_types.is_empty() {
        vec![0u8; 4]
    } else {
        u32_bytes(tok_types)
    };
    let out_bytes = vec![0u8; tok_types.len().max(1) * 4];
    let values = [Value::from(input_bytes), Value::from(out_bytes)];
    vyre_reference::reference_eval(&program, &values)
}

// ---------------------------------------------------------------------------
// 1. Object-like macros
// ---------------------------------------------------------------------------

#[path = "contract_cases/c_preprocess_pipeline_contracts__leading_hash_becomes_preproc_row_gpu_lexer.rs"]
mod c_preprocess_pipeline_contracts_leading_hash_becomes_preproc_row_gpu_lexer;
#[path = "contract_cases/c_preprocess_pipeline_contracts__object_like_macro_replaces_identifier_with_token_sequence.rs"]
mod c_preprocess_pipeline_contracts_object_like_macro_replaces_identifier_with_token_sequence;
