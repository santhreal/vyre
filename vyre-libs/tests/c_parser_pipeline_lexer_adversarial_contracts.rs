//! Adversarial contract tests for the C11 GPU lexer.
//!
//! Covers string literals, character literals, comments, line continuations,
//! preprocessor directives, and source-span integrity under hostile inputs.
//! Every test asserts either exact token-kind sequences, exact byte spans,
//! or host-vs-GPU parity  -  never silent acceptance of empty or default output.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]

#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;

use c_frontend::reference_lexer::{c11_lexer_outputs, run_c11_lexer_promoted as run_gpu_lexer};
use c_grammar_gen::lex_c11_max_munch_kinds;
use vyre_libs::parsing::c::lex::diagnostics::{first_c11_lexer_diagnostic, C11LexerDiagnosticKind};
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::pipeline::stages::C11_AST_MAX_TOK_SCAN;

/// Assert that the GPU lexer and the host max-munch lexer agree on the
/// non-whitespace, non-comment token sequence for `source`.
fn assert_host_gpu_agree(source: &[u8]) {
    let host_kinds = lex_c11_max_munch_kinds(source).expect("host lexer must accept source");
    let host_non_ws: Vec<u32> = host_kinds
        .into_iter()
        .filter(|k| *k != TOK_WHITESPACE && *k != TOK_COMMENT)
        .collect();
    let (gpu_types, _, _, gpu_count) = run_gpu_lexer(source, source.len() as u32);
    assert_eq!(
        gpu_count as usize,
        gpu_types.len(),
        "GPU count must match trimmed length"
    );
    assert_eq!(
        gpu_types,
        host_non_ws,
        "GPU lexer disagrees with host lexer for source: {:?}",
        std::str::from_utf8(source).unwrap_or("<binary>")
    );
}

fn assert_first_diagnostic(
    source: &[u8],
    expected_kind: C11LexerDiagnosticKind,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, u32) {
    let (types, starts, lens, count) = run_gpu_lexer(source, source.len() as u32);
    let diag = first_c11_lexer_diagnostic(&types, &starts, &lens)
        .expect("malformed fixture must emit a lexer diagnostic token");
    assert_eq!(diag.kind, expected_kind);
    assert!(
        diag.byte_start + diag.byte_len <= source.len() as u32,
        "diagnostic span must stay inside the source"
    );
    assert!(
        is_c_lexer_error_token(types[diag.token_index as usize]),
        "diagnostic token must be encoded as a lexer error token"
    );
    (types, starts, lens, count)
}

// ---------------------------------------------------------------------------
// 1. String literal adversarial contracts
// ---------------------------------------------------------------------------

#[path = "contract_cases/c_parser_pipeline_lexer_adversarial_contracts__empty_string_literal_emits_one_string_token_with_len_two.rs"]
mod c_parser_pipeline_lexer_adversarial_contracts_empty_string_literal_emits_one_string_token_with_len_two;
#[path = "contract_cases/c_parser_pipeline_lexer_adversarial_contracts__mid_line_hash_is_operator_not_directive.rs"]
mod c_parser_pipeline_lexer_adversarial_contracts_mid_line_hash_is_operator_not_directive;
