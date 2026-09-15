//! One binary for every math, math-linalg, nn-activation, nn-attention, nn-linear, nn-norm integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/operation_registry.rs`.
#[path = "operation_registry.rs"]
pub mod operation_registry;
