//! Contract tests for C AST declaration container node classification.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "c_ast_declaration_container_nodes/cpu_declaration_container_classification.rs"]
mod cpu_declaration_container_classification;
#[path = "../../tests/support/c_frontend/fixtures/declaration_container_nodes.rs"]
mod declaration_container_nodes;
