//! One binary for every nn-linear integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/wire_words/mod.rs`.
#[path = "wire_words/mod.rs"]
pub mod wire_words;

/// Integration tests from `tests/linear_rows_contract.rs`.
#[path = "linear_rows_contract.rs"]
pub mod linear_rows_contract;

/// Integration tests from `tests/quantized_linear_affine_fma.rs`.
#[cfg(feature = "nn-linear-4bit")]
#[path = "quantized_linear_affine_fma.rs"]
pub mod quantized_linear_affine_fma;
