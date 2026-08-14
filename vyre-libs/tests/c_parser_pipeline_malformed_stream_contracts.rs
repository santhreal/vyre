//! Adversarial contract tests for malformed streams and parser stage boundaries.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]

#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
use c_frontend::expression_pipeline::run_reference_pg_lower;

mod common;
use common::u32_bytes;

#[path = "contract_cases/c_parser_pipeline_malformed_stream_contracts__word_at.rs"]
mod c_parser_pipeline_malformed_stream_contracts_word_at;
