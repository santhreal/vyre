//! Contract tests for C AST declaration container node classification.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "c_ast_declaration_container_nodes/cpu.rs"]
mod cpu;
#[path = "c_ast_declaration_container_nodes/fixtures.rs"]
mod fixtures;
#[path = "c_ast_declaration_container_nodes/gpu.rs"]
mod gpu;
#[path = "c_ast_declaration_container_nodes/support.rs"]
mod support;
