//! One binary for every fixpoint, math-kernels integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/loop_unroll_trip1_idempotence.rs`.
#[cfg(all(feature = "math-kernels", feature = "fixpoint"))]
#[path = "loop_unroll_trip1_idempotence.rs"]
pub mod loop_unroll_trip1_idempotence;

/// Integration tests from `tests/scallop_join_grid_contract.rs`.
#[cfg(all(feature = "math-kernels", feature = "fixpoint"))]
#[path = "scallop_join_grid_contract.rs"]
pub mod scallop_join_grid_contract;
