//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/adversarial_emit_program_matrix.rs`.
#[path = "adversarial_emit_program_matrix.rs"]
pub mod adversarial_emit_program_matrix;

/// Integration tests from `tests/barrier_scope_parity.rs`.
#[path = "barrier_scope_parity.rs"]
pub mod barrier_scope_parity;

/// Integration tests from `tests/cross_emitter_parity.rs`.
#[path = "cross_emitter_parity.rs"]
pub mod cross_emitter_parity;

/// Integration tests from `tests/divergent_trap_and_grid_barrier.rs`.
#[path = "divergent_trap_and_grid_barrier.rs"]
pub mod divergent_trap_and_grid_barrier;

/// Integration tests from `tests/emit_contracts/mod.rs`.
#[path = "emit_contracts/mod.rs"]
pub mod emit_contracts;

/// Integration tests from `tests/emitted_artifact_byte_stability.rs`.
#[path = "emitted_artifact_byte_stability.rs"]
pub mod emitted_artifact_byte_stability;

/// Integration tests from `tests/global_store_bounds.rs`.
#[path = "global_store_bounds.rs"]
pub mod global_store_bounds;

/// Integration tests from `tests/grid_sync_loop_refusal.rs`.
#[path = "grid_sync_loop_refusal.rs"]
pub mod grid_sync_loop_refusal;

/// Integration tests from `tests/lowering_digest.rs`.
#[path = "lowering_digest.rs"]
pub mod lowering_digest;

/// Integration tests from `tests/nested_return_branch.rs`.
#[path = "nested_return_branch.rs"]
pub mod nested_return_branch;

/// Integration tests from `tests/nvrtc_compile_gate/mod.rs`.
#[path = "nvrtc_compile_gate/mod.rs"]
pub mod nvrtc_compile_gate;

/// Integration tests from `tests/pattern_analysis_contracts/mod.rs`.
#[path = "pattern_analysis_contracts/mod.rs"]
pub mod pattern_analysis_contracts;

/// Integration tests from `tests/ping_pong_schedule_contracts.rs`.
#[path = "ping_pong_schedule_contracts.rs"]
pub mod ping_pong_schedule_contracts;

/// Integration tests from `tests/regression_emit_fixes.rs`.
#[path = "regression_emit_fixes.rs"]
pub mod regression_emit_fixes;

/// Integration tests from `tests/shared_branch_walk_equality.rs`.
#[path = "shared_branch_walk_equality.rs"]
pub mod shared_branch_walk_equality;

/// Integration tests from `tests/ulp_budget_is_not_an_admission_gate.rs`.
#[path = "ulp_budget_is_not_an_admission_gate.rs"]
pub mod ulp_budget_is_not_an_admission_gate;
