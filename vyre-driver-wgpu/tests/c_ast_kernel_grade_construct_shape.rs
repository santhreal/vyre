//! AST node shape and CPU/GPU parity for the constructs kernel sources rely on: nested declarators,
//! asm with attributes, control flow, typedef shadowing, and statement expressions.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
mod c_ast_gpu_parity_support;
#[path = "c_ast_kernel_grade_construct_shape/classify.rs"]
mod classify;
#[path = "c_ast_kernel_grade_construct_shape/kernel_construct_parity_and_shape.rs"]
mod kernel_construct_parity_and_shape;
