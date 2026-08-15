//! Backend parity arm for the Linux-grade C declarator matrix.
//!
//! The construct list, the four-stage scaffold and the property-graph row
//! contract are owned by `tests/support/c_frontend/parity_matrix` and
//! `tests/support/c_frontend/fixtures/declarator_matrix_constructs`; the CPU arm
//! in `vyre-libs/tests/c_ast_declarator_matrix_contracts` iterates the same
//! `CASES`. What this root adds is the GPU arm those cases run on.
//!
//! Covered constructs:
//!   * pointer-to-array declarators (`int (*p)[4];`)
//!   * storage-class specifiers threaded through multi-declarator lists
//!   * parameter array declarators with `static` / `restrict` (C99)
//!   * nested typedef names inside declarators (function-pointer typedef reuse)
//!   * struct / union / enum tag definitions followed by mixed declarators
//!   * abstract declarators with qualifiers in cast contexts
//!   * GNU `__restrict` normalized to the C restrict qualifier
//!
//! A missing GPU adapter is a configuration failure, never a silent skip.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;
#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
#[path = "../../tests/support/c_frontend/fixtures/declarator_matrix_constructs.rs"]
mod declarator_matrix_constructs;

use c_ast_gpu_parity_support::{
    assert_family_parity, assert_words_eq, run_gpu_classifier, run_gpu_vast_builder_from_parts,
    GpuArm,
};
use declarator_matrix_constructs::fixture_abstract_declarator_with_qualifiers;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_build_vast_nodes, reference_c11_classify_vast_node_kinds,
};

#[path = "c_ast_declarator_matrix_contracts/pg_lowering_and_gpu_parity.rs"]
mod pg_lowering_and_gpu_parity;
