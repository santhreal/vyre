//! One binary for every hash integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/wire_words/mod.rs`.
#[cfg(feature = "hash")]
#[allow(deprecated)]
#[path = "wire_words/mod.rs"]
pub mod wire_words;

/// Integration tests from `tests/hash_registration_witnesses.rs`.
#[cfg(feature = "hash")]
#[path = "hash_registration_witnesses.rs"]
pub mod hash_registration_witnesses;
