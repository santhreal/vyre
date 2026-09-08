//! One binary for every pattern-regex integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/scan_ac_transition_walk_single_owner.rs`.
#[cfg(all(feature = "pattern-regex", feature = "pattern-dfa"))]
#[path = "scan_ac_transition_walk_single_owner.rs"]
pub mod scan_ac_transition_walk_single_owner;
