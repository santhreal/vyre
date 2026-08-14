//! Property-graph lowering of __builtin_expect and __builtin_choose_expr in control-flow and
//! initializer positions.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
mod c_ast_gpu_parity_support;
#[path = "c_ast_gnu_builtin_control_flow_pg_lowering_contracts/classify.rs"]
mod classify;
#[path = "c_ast_gnu_builtin_control_flow_pg_lowering_contracts/pg_lowering_and_gpu_parity.rs"]
mod pg_lowering_and_gpu_parity;
