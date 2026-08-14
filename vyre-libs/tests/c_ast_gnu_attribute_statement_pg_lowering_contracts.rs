//! Property-graph lowering of GNU attributes attached to labels, compound statements, if arms, and
//! switch cases.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "c_ast_gnu_attribute_statement_pg_lowering_contracts/classify.rs"]
mod classify;
#[path = "../../tests/support/c_frontend/fixtures/gnu_attribute_statements.rs"]
mod gnu_attribute_statements;
