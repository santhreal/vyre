//! One binary for every math integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/quantized_packing_contracts.rs`.
#[cfg(feature = "math")]
#[path = "quantized_packing_contracts.rs"]
pub mod quantized_packing_contracts;
