//! One binary for every pattern-kernels integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/wire_words/mod.rs`.
#[cfg(feature = "pattern")]
#[path = "wire_words/mod.rs"]
pub mod wire_words;

/// Integration tests from `tests/region_adversarial.rs`.
#[cfg(feature = "pattern")]
#[path = "region_adversarial.rs"]
pub mod region_adversarial;
