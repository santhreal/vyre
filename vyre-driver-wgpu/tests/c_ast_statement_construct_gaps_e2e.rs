//! Statement constructs end to end: nested compound statements, control-flow rows, and label and
//! goto rows.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
mod c_ast_gpu_parity_support;
#[path = "c_ast_statement_construct_gaps_e2e/classify.rs"]
mod classify;
#[path = "c_ast_statement_construct_gaps_e2e/compound_statements_and_control_flow.rs"]
mod compound_statements_and_control_flow;
