//! Tag separation, GNU attributes, and compound literals in the C AST, with GPU parity against the
//! CPU reference.
#![cfg(feature = "c-parser")]
#![allow(clippy::erasing_op)]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "gemini_c_ast_contracts/cpu_reference_and_gpu_parity.rs"]
mod cpu_reference_and_gpu_parity;
#[path = "../../tests/support/c_frontend/fixtures/gemini_named_fixtures.rs"]
mod gemini_named_fixtures;
