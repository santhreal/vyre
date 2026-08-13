//! Contract tests for c ast statement construct gaps e2e.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "c_ast_statement_construct_gaps_e2e/classify.rs"]
mod classify;
#[path = "c_ast_statement_construct_gaps_e2e/nested_compound_statements_preserve_blocks_and_return.rs"]
mod nested_compound_statements_preserve_blocks_and_return;
