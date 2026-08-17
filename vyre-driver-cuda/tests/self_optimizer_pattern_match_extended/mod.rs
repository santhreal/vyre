//! Coverage for the arithmetic, bitwise, min/max and boolean identity rules in
//! the GPU pattern-match pass. Each case runs a Program through the full
//! persistent-resident pipeline and asserts the post-pipeline IR has the
//! expected collapsed form.
//!
//! The pipeline runner and the program shapes live in
//! `harness::self_optimizer`; this file only groups the per-rule submodules and
//! puts the harness in scope for their `use super::*`.

#![cfg(test)]

#[path = "../harness/mod.rs"]
mod harness;

pub(crate) use harness::self_optimizer::{
    assert_branch_folded_to, assert_cond_not_headed_by, assert_lit_bool, assert_lit_u32,
    assert_var, b_load_branch_program, binop, folded_x_store_value, folded_xy_store_value,
    run_pipeline, unop,
};
pub(crate) use vyre::ir::UnOp;
pub(crate) use vyre::ir::{BinOp, Expr};

mod arithmetic_cse_contracts;
mod arithmetic_identity_contracts;
mod bitwise_shift_contracts;
mod bitxor_chain_contracts;
mod boolean_comparison_contracts;
mod minmax_contracts;
mod self_cse_contracts;
