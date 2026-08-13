//! Expression shape from C source through property-graph lowering, covering designators, nested
//! conditionals, and labelled switch bodies.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "c_ast_pg_expression_shape_e2e/bytes.rs"]
mod bytes;
mod c_ast_gpu_parity_support;
#[path = "c_ast_pg_expression_shape_e2e/expression_and_statement_rows.rs"]
mod expression_and_statement_rows;
