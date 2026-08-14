//! Expression shape from C source through property-graph lowering, covering designators, nested
//! conditionals, and labelled switch bodies.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/expression_shape_pg.rs"]
mod expression_shape_pg;
#[path = "c_ast_pg_expression_shape_e2e/cpu_expression_shape_pg_lowering.rs"]
mod cpu_expression_shape_pg_lowering;
