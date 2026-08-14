//! GPU/CPU parity tests for difficult C declarator edge cases.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "c_ast_declarator_edge_cases/cpu_reference_and_gpu_parity.rs"]
mod cpu_reference_and_gpu_parity;
#[path = "../../tests/support/c_frontend/fixtures/gemini_named_fixtures.rs"]
mod gemini_named_fixtures;
#[path = "c_ast_declarator_edge_cases/gpu_parity.rs"]
mod gpu_parity;
#[path = "c_ast_declarator_edge_cases/struct_fixtures.rs"]
mod struct_fixtures;
#[path = "c_ast_declarator_edge_cases/support.rs"]
mod support;
