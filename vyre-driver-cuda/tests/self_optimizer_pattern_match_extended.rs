//! Coverage for the arithmetic, bitwise, min/max and boolean identity rules in
//! the GPU pattern-match pass. Each case runs a Program through the full
//! persistent-resident pipeline and asserts the post-pipeline IR has the
//! expected collapsed form.
//!
//! The pipeline runner and the program shapes live in
//! `harness::self_optimizer`; this file only groups the per-rule submodules and
//! puts the harness in scope for their `use super::*`.

#![cfg(test)]

mod harness;

pub(crate) use harness::self_optimizer::{
    assert_branch_folded_to, assert_cond_not_headed_by, assert_lit_bool, assert_lit_u32,
    assert_var, b_load_branch_program, binop, folded_x_store_value, folded_xy_store_value,
    run_pipeline, unop,
};
pub(crate) use vyre::ir::UnOp;
pub(crate) use vyre::ir::{BinOp, Expr};

#[path = "self_optimizer_pattern_match_extended/arithmetic_cse_contracts.rs"]
mod arithmetic_cse_contracts;
#[path = "self_optimizer_pattern_match_extended/arithmetic_identity_contracts.rs"]
mod arithmetic_identity_contracts;
#[path = "self_optimizer_pattern_match_extended/bitwise_shift_contracts.rs"]
mod bitwise_shift_contracts;
#[path = "self_optimizer_pattern_match_extended/bitxor_chain_contracts.rs"]
mod bitxor_chain_contracts;
#[path = "self_optimizer_pattern_match_extended/boolean_comparison_contracts.rs"]
mod boolean_comparison_contracts;
#[path = "self_optimizer_pattern_match_extended/minmax_contracts.rs"]
mod minmax_contracts;
#[path = "self_optimizer_pattern_match_extended/self_cse_contracts.rs"]
mod self_cse_contracts;
