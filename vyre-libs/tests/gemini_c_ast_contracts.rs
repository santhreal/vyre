//! Tag separation, GNU attributes, and compound literals in the C AST, with GPU parity against the
//! CPU reference.
#![cfg(feature = "c-parser")]
#![allow(clippy::erasing_op)]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/gemini_named_fixtures.rs"]
mod gemini_named_fixtures;
#[path = "gemini_c_ast_contracts/cpu_named_fixture_classification.rs"]
mod cpu_named_fixture_classification;
