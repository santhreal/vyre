//! AST node shape and CPU/GPU parity for the constructs kernel sources rely on: nested declarators,
//! asm with attributes, control flow, typedef shadowing, and statement expressions.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "c_ast_kernel_grade_construct_shape/classify.rs"]
mod classify;
#[path = "c_ast_kernel_grade_construct_shape/nested_declarator_parity_and_shape.rs"]
mod nested_declarator_parity_and_shape;
