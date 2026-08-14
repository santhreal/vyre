//! Switch bodies that mix compound literals, designated initializers, statement expressions, nested
//! switches, and Duff's device.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "c_ast_switch_case_complex_body_pg_lowering_contracts/classify.rs"]
mod classify;
#[path = "../../tests/support/c_frontend/fixtures/switch_case_complex_bodies.rs"]
mod switch_case_complex_bodies;
