//! Contract tests for c ast gnu builtin control flow pg lowering contracts.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "c_ast_gnu_builtin_control_flow_pg_lowering_contracts/classify.rs"]
mod classify;
#[path = "c_ast_gnu_builtin_control_flow_pg_lowering_contracts/pg_lower_preserves_builtin_expect_in_ternary.rs"]
mod pg_lower_preserves_builtin_expect_in_ternary;
