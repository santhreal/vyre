//! One binary for every math-scan integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/scan_prefix_sum_size_contract.rs`.
#[cfg(feature = "math-scan")]
#[path = "scan_prefix_sum_size_contract.rs"]
pub mod scan_prefix_sum_size_contract;
