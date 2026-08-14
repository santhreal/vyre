//! Contract tests for dynamic C preprocessor macro expansion bounds.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]

#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
mod common;
use common::decode_u32_words;
use std::panic::{catch_unwind, AssertUnwindSafe};

use c_frontend::macro_expansion::{run_dynamic_macro_expansion, MacroFixture};
use vyre_libs::parsing::c::lex::tokens::{TOK_IDENTIFIER, TOK_INTEGER, TOK_PLUS, TOK_STAR};

#[test]
fn dynamic_macro_expansion_emits_replacement_tokens_and_count() {
    let mut fixture = MacroFixture::empty();
    fixture.insert(TOK_IDENTIFIER, 512, &[TOK_INTEGER, TOK_PLUS, TOK_INTEGER]);

    let outputs = run_dynamic_macro_expansion(&[TOK_IDENTIFIER, TOK_STAR], &fixture, 8)
        .expect("bounded macro expansion must succeed");
    assert_eq!(outputs.len(), 2);

    let out_tokens = decode_u32_words(&outputs[0].to_bytes());
    let out_count = decode_u32_words(&outputs[1].to_bytes());
    assert_eq!(
        &out_tokens[..4],
        &[TOK_INTEGER, TOK_PLUS, TOK_INTEGER, TOK_STAR]
    );
    assert_eq!(out_count, vec![4]);
}

#[test]
fn dynamic_macro_expansion_passthrough_counts_unmapped_tokens() {
    let fixture = MacroFixture::empty();
    let input = [TOK_IDENTIFIER, TOK_PLUS, TOK_INTEGER];

    let outputs = run_dynamic_macro_expansion(&input, &fixture, 8)
        .expect("unmapped macro tokens must pass through");
    let out_tokens = decode_u32_words(&outputs[0].to_bytes());
    let out_count = decode_u32_words(&outputs[1].to_bytes());
    assert_eq!(&out_tokens[..input.len()], &input);
    assert_eq!(out_count, vec![input.len() as u32]);
}

#[test]
fn dynamic_macro_expansion_rejects_output_capacity_overflow_without_panic() {
    let mut fixture = MacroFixture::empty();
    fixture.insert(TOK_IDENTIFIER, 512, &[TOK_INTEGER, TOK_PLUS, TOK_INTEGER]);

    let result = catch_unwind(AssertUnwindSafe(|| {
        run_dynamic_macro_expansion(&[TOK_IDENTIFIER, TOK_IDENTIFIER], &fixture, 5)
    }));
    let eval_result = result.expect("output-capacity overflow must return an error, not panic");
    let err = eval_result.expect_err("two 3-token expansions into five output slots must reject");
    let msg = err.to_string();
    assert!(
        msg.contains("capacity") || msg.contains("overflow") || msg.contains("Fix:"),
        "capacity overflow error: {msg}"
    );
}
