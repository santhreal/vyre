//! Property-graph lowering of GNU attributes attached to labels, compound statements, if arms, and
//! switch cases.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "c_ast_gnu_attribute_statement_pg_lowering_contracts/classify.rs"]
mod classify;
#[path = "c_ast_gnu_attribute_statement_pg_lowering_contracts/pg_lowering_and_gpu_parity.rs"]
mod pg_lowering_and_gpu_parity;
