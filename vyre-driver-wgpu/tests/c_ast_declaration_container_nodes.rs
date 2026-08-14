//! Contract tests for C AST declaration container node classification.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/declaration_container_nodes.rs"]
mod declaration_container_nodes;
#[path = "c_ast_declaration_container_nodes/gpu.rs"]
mod gpu;
#[path = "c_ast_declaration_container_nodes/gpu_classify_runners.rs"]
mod gpu_classify_runners;
