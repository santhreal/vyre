//! Switch bodies that mix compound literals, designated initializers, statement expressions, nested
//! switches, and Duff's device.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
mod c_ast_gpu_parity_support;
#[path = "c_ast_switch_case_complex_body_pg_lowering_contracts/classify.rs"]
mod classify;
#[path = "c_ast_switch_case_complex_body_pg_lowering_contracts/cpu_pg_and_gpu_parity.rs"]
mod cpu_pg_and_gpu_parity;
