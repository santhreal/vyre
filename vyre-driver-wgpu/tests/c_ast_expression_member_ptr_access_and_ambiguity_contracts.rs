//! Member and pointer member access rows, and the cast versus parenthesized expression ambiguities
//! that surround them.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "c_ast_expression_member_ptr_access_and_ambiguity_contracts/classify.rs"]
mod classify;
#[path = "c_ast_expression_member_ptr_access_and_ambiguity_contracts/paren_expr_then_mul_is_binary_not_cast.rs"]
mod paren_expr_then_mul_is_binary_not_cast;
