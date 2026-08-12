//! Adversarial contract tests for malformed streams and parser stage boundaries.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]

mod common;
use common::u32_bytes;

include!("contract_cases/c_parser_pipeline_malformed_stream_contracts__word_at.rs");
include!("contract_cases/c_parser_pipeline_malformed_stream_contracts__classifier_does_not_emit_all_zeros_for_nonempty_vast.rs");
