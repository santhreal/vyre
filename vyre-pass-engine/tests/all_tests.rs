//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/consumer_boundary.rs`.
#[path = "consumer_boundary.rs"]
pub mod consumer_boundary;

/// Integration tests from `tests/encoded_csr_layout_contract.rs`.
#[path = "encoded_csr_layout_contract.rs"]
pub mod encoded_csr_layout_contract;

/// Integration tests from `tests/feature_boundaries.rs`.
#[path = "feature_boundaries.rs"]
pub mod feature_boundaries;
