//! Member and pointer member access rows, and the cast versus parenthesized expression ambiguities
//! that surround them.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "c_ast_expression_member_ptr_access_and_ambiguity_contracts/classify.rs"]
mod classify;
#[path = "c_ast_expression_member_ptr_access_and_ambiguity_contracts/member_access_and_cast_ambiguity.rs"]
mod member_access_and_cast_ambiguity;
