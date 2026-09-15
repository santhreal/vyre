//! One binary for every fixpoint, graph, math-kernels, parsing-kernels integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/loop_back_edge_audit.rs`.
#[path = "loop_back_edge_audit.rs"]
pub mod loop_back_edge_audit;
