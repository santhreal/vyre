//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/float_lowering/mod.rs`.
#[path = "float_lowering/mod.rs"]
pub mod float_lowering;

/// Integration tests from `tests/level_stage_closure.rs`.
#[path = "level_stage_closure.rs"]
pub mod level_stage_closure;

/// Integration tests from `tests/float_lowering_decisions.rs`.
#[path = "float_lowering_decisions.rs"]
pub mod float_lowering_decisions;
