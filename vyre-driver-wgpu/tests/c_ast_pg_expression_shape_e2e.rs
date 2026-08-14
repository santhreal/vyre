//! Expression shape from C source through property-graph lowering, covering designators, nested
//! conditionals, and labelled switch bodies.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/fixtures/expression_shape_pg.rs"]
mod expression_shape_pg;
#[path = "c_ast_pg_expression_shape_e2e/expression_and_statement_rows.rs"]
mod expression_and_statement_rows;
