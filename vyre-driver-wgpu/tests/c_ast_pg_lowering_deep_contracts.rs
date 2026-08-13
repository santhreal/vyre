//! Semantic categories, roles, and edges the deep property-graph lowering must assign, checked
//! against a GPU oracle.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "c_ast_pg_lowering_deep_contracts/classify.rs"]
mod classify;
#[path = "c_ast_pg_lowering_deep_contracts/semantic_categories_roles_and_edges.rs"]
mod semantic_categories_roles_and_edges;
