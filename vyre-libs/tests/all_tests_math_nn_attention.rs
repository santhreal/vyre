//! One binary for every math, nn-attention integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/name_collision.rs`.
#[cfg(all(feature = "math-linalg", feature = "nn-attention", feature = "nn-norm",))]
#[path = "name_collision.rs"]
pub mod name_collision;
