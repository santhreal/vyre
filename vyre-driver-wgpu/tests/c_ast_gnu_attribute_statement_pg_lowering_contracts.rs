//! Contract tests for c ast gnu attribute statement pg lowering contracts.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "c_ast_gnu_attribute_statement_pg_lowering_contracts/classify.rs"]
mod classify;
#[path = "c_ast_gnu_attribute_statement_pg_lowering_contracts/pg_lower_preserves_attribute_aligned_on_label.rs"]
mod pg_lower_preserves_attribute_aligned_on_label;
