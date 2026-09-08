//! One binary for every nn-norm integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/wire_words/mod.rs`.
#[path = "wire_words/mod.rs"]
pub mod wire_words;

/// Integration tests from `tests/gated_rms_norm_contract.rs`.
#[path = "gated_rms_norm_contract.rs"]
pub mod gated_rms_norm_contract;

/// Integration tests from `tests/last_dim_l2_norm_contract.rs`.
#[path = "last_dim_l2_norm_contract.rs"]
pub mod last_dim_l2_norm_contract;
