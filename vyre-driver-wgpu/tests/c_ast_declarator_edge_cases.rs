//! GPU/CPU parity tests for difficult C declarator edge cases.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "c_ast_declarator_edge_cases/cpu_array_of_function_pointers_kinds.rs"]
mod cpu_array_of_function_pointers_kinds;
#[path = "c_ast_declarator_edge_cases/gpu_parity_classifier_abstract_declarator_cast.rs"]
mod gpu_parity_classifier_abstract_declarator_cast;
#[path = "c_ast_declarator_edge_cases/struct_fixtures.rs"]
mod struct_fixtures;
#[path = "c_ast_declarator_edge_cases/support.rs"]
mod support;
