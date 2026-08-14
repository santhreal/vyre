//! Property-graph lowering of compound literals and nested designated initializers, including their
//! use inside ternaries and statement expressions.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/fixtures/compound_literal_designated_init.rs"]
mod compound_literal_designated_init;
#[path = "c_ast_compound_literal_designated_init_nested_pg_lowering_contracts/pg_lowering_and_gpu_parity.rs"]
mod pg_lowering_and_gpu_parity;
