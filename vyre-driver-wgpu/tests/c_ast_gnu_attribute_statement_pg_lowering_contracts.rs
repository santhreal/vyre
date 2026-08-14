//! Property-graph lowering of GNU attributes attached to labels, compound statements, if arms, and
//! switch cases.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/gnu_attribute_statements.rs"]
mod gnu_attribute_statements;
#[path = "c_ast_gnu_attribute_statement_pg_lowering_contracts/pg_lowering_and_gpu_parity.rs"]
mod pg_lowering_and_gpu_parity;
