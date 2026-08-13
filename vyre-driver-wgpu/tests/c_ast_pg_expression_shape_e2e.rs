//! Contract tests for c ast pg expression shape e2e.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "c_ast_pg_expression_shape_e2e/bytes.rs"]
mod bytes;
mod c_ast_gpu_parity_support;
#[path = "c_ast_pg_expression_shape_e2e/compound_literal_designators_and_nested_conditional_lower_to_pg.rs"]
mod compound_literal_designators_and_nested_conditional_lower_to_pg;
