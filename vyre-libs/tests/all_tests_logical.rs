//! One binary for every logical integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/sweep_logical_reference_matrix.rs`.
#[cfg(feature = "logical")]
#[allow(deprecated)]
#[path = "sweep_logical_reference_matrix.rs"]
pub mod sweep_logical_reference_matrix;
