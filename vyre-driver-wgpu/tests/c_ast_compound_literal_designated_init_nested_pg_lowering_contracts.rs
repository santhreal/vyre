//! Contract tests for c ast compound literal designated init nested pg lowering contracts.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "c_ast_compound_literal_designated_init_nested_pg_lowering_contracts/classify.rs"]
mod classify;
#[path = "c_ast_compound_literal_designated_init_nested_pg_lowering_contracts/pg_lower_preserves_designated_init_with_builtin_choose_expr.rs"]
mod pg_lower_preserves_designated_init_with_builtin_choose_expr;
