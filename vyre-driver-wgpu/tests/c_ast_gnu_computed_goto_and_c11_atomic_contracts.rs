//! Contract tests for c ast gnu computed goto and c11 atomic contracts.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "c_ast_gnu_computed_goto_and_c11_atomic_contracts/classify.rs"]
mod classify;
#[path = "c_ast_gnu_computed_goto_and_c11_atomic_contracts/pg_lower_preserves_computed_goto_rows.rs"]
mod pg_lower_preserves_computed_goto_rows;
