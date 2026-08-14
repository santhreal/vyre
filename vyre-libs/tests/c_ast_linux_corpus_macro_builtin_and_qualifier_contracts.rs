//! Linux/kernel-grade C AST contracts for macro shapes, builtins, and qualifiers.
//!
//! Constructs under test:
//!   * container_of macro definition as preprocessor token stream + call-shaped usage
//!   * list_entry / list_for_each macro patterns
//!   * __builtin_expect (likely/unlikely) direct usage and macro wrapper preservation
//!   * static inline __attribute__((always_inline)) function definitions
//!   * volatile / _Atomic qualifier promotions in declarations and parameters
//!   * Linux error-label cleanup patterns (goto err; ... err: return -errno;)
//!
//! Every fixture asserts full GPU/CPU parity for build, annotate, and classify.
//! PG preservation is asserted for rows that carry semantic payload.
//! A missing GPU adapter is a configuration failure  -  tests panic loudly.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/linux_macro_builtin_qualifier.rs"]
mod linux_macro_builtin_qualifier;

use crate::c_frontend::rows::{assert_pg_preserves_row, row_indices};
use linux_macro_builtin_qualifier::*;
use vyre_libs::parsing::c::lower::reference_ast_to_pg_nodes;
use vyre_libs::parsing::c::parse::vast::{
    C_AST_KIND_GOTO_STMT, C_AST_KIND_IF_STMT, C_AST_KIND_RETURN_STMT,
};

#[path = "c_ast_linux_corpus_macro_builtin_and_qualifier_contracts/error_label_control_flow_rows.rs"]
mod error_label_control_flow_rows;
