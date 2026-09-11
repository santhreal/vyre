//! One binary for every spirv-val integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/target_artifacts/mod.rs`.
#[path = "target_artifacts/mod.rs"]
pub mod target_artifacts;

/// Integration tests from `tests/spirv_parity.rs`.
#[path = "spirv_parity.rs"]
pub mod spirv_parity;
