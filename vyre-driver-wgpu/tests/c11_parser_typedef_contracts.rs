//! Contract tests for c11 parser typedef contracts.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "c11_parser_typedef_contracts/assert_kind.rs"]
mod assert_kind;
mod c_ast_gpu_parity_support;
#[path = "c11_parser_typedef_contracts/pg_lower_preserves_typedef_cast_vs_expr_kinds.rs"]
mod pg_lower_preserves_typedef_cast_vs_expr_kinds;
