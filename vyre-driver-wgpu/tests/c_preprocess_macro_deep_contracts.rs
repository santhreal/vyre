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

mod c_token_support;
mod common;
use c_token_support::{assemble, assert_pg_row, find_row_for_lexeme, row_typed_kind, Assembled};
use common::{decode_u32_words, u32_bytes};
use std::sync::OnceLock;

use c_grammar_gen::lex_c11_max_munch_kinds;
use vyre::ir::Expr;
use vyre_driver::VyreBackend;
use vyre_driver_wgpu::WgpuBackend;
use vyre_libs::parsing::c::lex::lexer::c11_lexer;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::lower::{c_lower_ast_to_pg_nodes, reference_ast_to_pg_nodes};
use vyre_libs::parsing::c::parse::vast::{
    c11_build_expression_shape_nodes, c11_classify_vast_node_kinds,
    reference_c11_build_expression_shape_nodes, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds, C_EXPR_ASSOC_NONE, C_EXPR_SHAPE_NONE,
    C_EXPR_SHAPE_STRIDE_U32,
};
use vyre_libs::parsing::c::preprocess::expansion::{
    opt_conditional_mask_with_directives, opt_dynamic_macro_expansion,
};
use vyre_libs::parsing::c::preprocess::{
    c_translation_phase_line_splice, reference_c_preprocessor_directive_metadata,
};
use vyre_primitives::predicate::node_kind;
use vyre_reference::value::Value;

// ---------------------------------------------------------------------------
// Byte / word helpers
// ---------------------------------------------------------------------------

fn haystack_words(source: &[u8]) -> Vec<u32> {
    source.iter().map(|b| u32::from(*b)).collect()
}

// ---------------------------------------------------------------------------
// GPU lexer helper
// ---------------------------------------------------------------------------

fn run_c11_lexer(source: &[u8], haystack_len: u32) -> (Vec<u32>, Vec<u32>, Vec<u32>, u32) {
    let program = c11_lexer(
        "haystack",
        "out_tok_types",
        "out_tok_starts",
        "out_tok_lens",
        "out_counts",
        haystack_len,
    );
    let haystack_buf = u32_bytes(&haystack_words(source));
    let zero_buf = vec![0u8; haystack_len as usize * 4];
    let count_zero = vec![0u8; 4];
    let inputs = [
        Value::from(haystack_buf),
        Value::from(zero_buf.clone()),
        Value::from(zero_buf.clone()),
        Value::from(zero_buf),
        Value::from(count_zero),
    ];
    let outputs = vyre_reference::reference_eval(&program, &inputs)
        .expect("c11_lexer must execute under the reference oracle");
    assert_eq!(
        outputs.len(),
        4,
        "expected [tok_types, tok_starts, tok_lens, counts]"
    );
    let tok_types = decode_u32_words(&outputs[0].to_bytes());
    let tok_starts = decode_u32_words(&outputs[1].to_bytes());
    let tok_lens = decode_u32_words(&outputs[2].to_bytes());
    let counts = decode_u32_words(&outputs[3].to_bytes());
    let tok_count = counts.first().copied().unwrap_or(0);
    (
        tok_types[..tok_count as usize].to_vec(),
        tok_starts[..tok_count as usize].to_vec(),
        tok_lens[..tok_count as usize].to_vec(),
        tok_count,
    )
}

// ---------------------------------------------------------------------------
// Dynamic macro-expansion helpers
// ---------------------------------------------------------------------------

const EMPTY_SLOT: u32 = u32::MAX;
const TABLE_SLOTS: usize = 4096;

fn hash_token(tok: u32) -> usize {
    (tok.wrapping_mul(2_654_435_769) & (TABLE_SLOTS as u32 - 1)) as usize
}

struct MacroFixture {
    keys: Vec<u32>,
    vals: Vec<u32>,
    sizes: Vec<u32>,
}

impl MacroFixture {
    fn empty() -> Self {
        Self {
            keys: vec![EMPTY_SLOT; TABLE_SLOTS],
            vals: vec![0; TABLE_SLOTS],
            sizes: vec![0; TABLE_SLOTS],
        }
    }

    fn insert(&mut self, token: u32, replacement_offset: usize, replacement: &[u32]) {
        let slot = hash_token(token);
        self.keys[slot] = token;
        self.vals[slot] = replacement_offset as u32;
        self.sizes[replacement_offset] = replacement.len() as u32;
        for (idx, value) in replacement.iter().enumerate() {
            self.vals[replacement_offset + idx] = *value;
        }
    }
}

fn run_dynamic_macro_expansion(
    input: &[u32],
    fixture: &MacroFixture,
    max_out_tokens: u32,
) -> Result<Vec<Value>, vyre_reference::ReferenceError> {
    let program = opt_dynamic_macro_expansion(
        "in_tok_types",
        "macro_keys",
        "macro_vals",
        "macro_sizes",
        "out_tok_types",
        "out_tok_counts",
        Expr::u32(input.len() as u32),
        max_out_tokens,
    );
    let input_bytes = if input.is_empty() {
        vec![0u8; 4]
    } else {
        u32_bytes(input)
    };
    let out_len = max_out_tokens.max(1) as usize * 4;
    let values = [
        Value::from(input_bytes),
        Value::from(u32_bytes(&fixture.keys)),
        Value::from(u32_bytes(&fixture.vals)),
        Value::from(u32_bytes(&fixture.sizes)),
        Value::from(vec![0u8; out_len]),
        Value::from(vec![0u8; 4]),
    ];
    vyre_reference::reference_eval(&program, &values)
}

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
const PG_STRIDE_U32: usize = 6;

fn word_at(buf: &[u8], word: usize) -> u32 {
    let offset = word * 4;
    u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
}

fn node_count_from_vast(vast: &[u8]) -> u32 {
    u32::try_from(vast.len() / (VAST_STRIDE_U32 * 4)).unwrap_or_default()
}

fn run_reference_pg_lower(typed_vast: &[u8]) -> Vec<u8> {
    let num_nodes = node_count_from_vast(typed_vast);
    let program = c_lower_ast_to_pg_nodes("vast_nodes", Expr::u32(num_nodes), "pg_nodes");
    let output_len = num_nodes.saturating_mul(PG_STRIDE_U32 as u32).max(1) as usize * 4;
    let values = [
        Value::from(typed_vast.to_vec()),
        Value::from(vec![0; output_len]),
    ];
    let outputs = vyre_reference::reference_eval(&program, &values)
        .unwrap_or_else(|e| panic!("C AST PG lowerer must execute on CPU: {e}"));
    assert_eq!(outputs.len(), 1);
    outputs[0].to_bytes()
}

struct PipelineOut {
    raw_vast: Vec<u8>,
    typed_vast: Vec<u8>,
    expr_shape: Vec<u8>,
    pg: Vec<u8>,
}

fn run_cpu_pipeline(assembled: &Assembled) -> PipelineOut {
    let raw_vast = reference_c11_build_vast_nodes(
        &assembled.tok_types,
        &assembled.tok_starts,
        &assembled.tok_lens,
    );
    let typed_vast = reference_c11_classify_vast_node_kinds(&raw_vast);
    let expr_shape = reference_c11_build_expression_shape_nodes(&raw_vast, &typed_vast);
    let pg = run_reference_pg_lower(&typed_vast);
    assert_eq!(
        pg,
        reference_ast_to_pg_nodes(&typed_vast),
        "executable PG lowerer must match byte oracle"
    );
    PipelineOut {
        raw_vast,
        typed_vast,
        expr_shape,
        pg,
    }
}

fn assert_shape_none(expr_shape: &[u8], idx: usize) {
    let base = idx * C_EXPR_SHAPE_STRIDE_U32 as usize;
    assert_eq!(
        word_at(expr_shape, base),
        C_EXPR_SHAPE_NONE,
        "preproc/structural rows stay shape-none"
    );
    assert_eq!(word_at(expr_shape, base + 4), C_EXPR_ASSOC_NONE);
}

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

mod c_preprocess_macro_deep_contracts_host_lexer_preserves_complex_directive_block_as_preproc_rows {

    include!("contract_cases/c_preprocess_macro_deep_contracts__host_lexer_preserves_complex_directive_block_as_preproc_rows.rs");
}
mod c_preprocess_macro_deep_contracts_macro_call_with_trailing_comma_survives_as_call_in_vast {
    include!("contract_cases/c_preprocess_macro_deep_contracts__macro_call_with_trailing_comma_survives_as_call_in_vast.rs");
}
