//! End-to-end C parser coverage for expression-shape gaps.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "c_ast_expression_shape_gaps_e2e/cpu_kind_and_shape_rows.rs"]
mod cpu_kind_and_shape_rows;
#[path = "../../tests/support/c_frontend/fixtures/expression_shape_gap_constructs.rs"]
mod expression_shape_gap_constructs;
