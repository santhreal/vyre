//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `tools/xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/adversarial_emit_program_matrix.rs`.
#[path = "adversarial_emit_program_matrix.rs"]
mod adversarial_emit_program_matrix;

/// Integration tests from `tests/barrier_scope_parity.rs`.
#[path = "barrier_scope_parity.rs"]
mod barrier_scope_parity;


/// Integration tests from `tests/divergent_trap_and_grid_barrier.rs`.
#[path = "divergent_trap_and_grid_barrier.rs"]
mod divergent_trap_and_grid_barrier;

/// Integration tests from `tests/emit_contracts/mod.rs`.
#[path = "emit_contracts/mod.rs"]
mod emit_contracts;

/// Integration tests from `tests/global_store_bounds.rs`.
#[path = "global_store_bounds.rs"]
mod global_store_bounds;

/// Integration tests from `tests/grid_sync_loop_refusal.rs`.
#[path = "grid_sync_loop_refusal.rs"]
mod grid_sync_loop_refusal;

/// Integration tests from `tests/nested_return_branch.rs`.
#[path = "nested_return_branch.rs"]
mod nested_return_branch;

/// Integration tests from `tests/nvrtc_compile_gate/mod.rs`.
#[path = "nvrtc_compile_gate/mod.rs"]
mod nvrtc_compile_gate;

/// Integration tests from `tests/pattern_analysis_contracts/mod.rs`.
#[path = "pattern_analysis_contracts/mod.rs"]
mod pattern_analysis_contracts;

/// Integration tests from `tests/ping_pong_schedule_contracts.rs`.
#[path = "ping_pong_schedule_contracts.rs"]
mod ping_pong_schedule_contracts;

/// Integration tests from `tests/regression_emit_fixes.rs`.
#[path = "regression_emit_fixes.rs"]
mod regression_emit_fixes;

/// Integration tests from `tests/shared_branch_walk_equality.rs`.
#[path = "shared_branch_walk_equality.rs"]
mod shared_branch_walk_equality;

/// Integration tests from `tests/ulp_budget_is_not_an_admission_gate.rs`.
#[path = "ulp_budget_is_not_an_admission_gate.rs"]
mod ulp_budget_is_not_an_admission_gate;

/// Integration tests from `tests/physical_ir_variant_exhaustiveness.rs`.
#[path = "physical_ir_variant_exhaustiveness.rs"]
mod physical_ir_variant_exhaustiveness;

/// Integration tests from `tests/proptest_shared_extent_padding.rs`.
#[path = "proptest_shared_extent_padding.rs"]
mod proptest_shared_extent_padding;

/// Integration tests from `tests/workgroup_zero_init.rs`.
#[path = "workgroup_zero_init.rs"]
mod workgroup_zero_init;
