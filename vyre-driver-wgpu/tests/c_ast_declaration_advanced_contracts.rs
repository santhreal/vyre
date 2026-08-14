//! Advanced C declaration and declarator contracts.
//!
//! Coverage gaps filled:
//!   * deeply nested struct / union / enum definitions
//!   * anonymous struct/union members
//!   * typedefs with multiple complex declarators (struct tag + pointer)
//!   * triple-star pointers with interleaved qualifiers
//!   * storage-class combinations: _Thread_local, _Atomic, register, inline
//!   * bit-fields inside nested structs
//!   * GNU attributes on struct fields and typedef declarations
//!   * pointer-to-function-pointer declarators
//!   * arrays of function pointers with qualified parameters
//!
//! Every test asserts CPU/GPU parity and meaningful AST/VAST/PG invariants.
//! A missing GPU adapter is a configuration failure, never a skip.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/fixtures/declaration_advanced_constructs.rs"]
mod declaration_advanced_constructs;

use c_ast_gpu_parity_support::{
    assert_full_pipeline_parity, assert_pg_preserves_row, kind_at, node_count_from_vast,
    run_gpu_pg_lower_with_count as run_gpu_pg_lower,
};
use declaration_advanced_constructs::*;
use vyre_libs::parsing::c::lower::reference_ast_to_pg_nodes;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_annotate_typedef_names, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds,
};

#[path = "c_ast_declaration_advanced_contracts/pg_lowering_and_gpu_parity.rs"]
mod pg_lowering_and_gpu_parity;
