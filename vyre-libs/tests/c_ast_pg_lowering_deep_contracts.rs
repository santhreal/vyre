//! Semantic categories, roles, and edges the deep property-graph lowering must assign, checked
//! against a GPU oracle.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "c_ast_pg_lowering_deep_contracts/cpu_semantic_node_and_edge_shapes.rs"]
mod cpu_semantic_node_and_edge_shapes;
#[path = "../../tests/support/c_frontend/fixtures/pg_lowering_deep_constructs.rs"]
mod pg_lowering_deep_constructs;
