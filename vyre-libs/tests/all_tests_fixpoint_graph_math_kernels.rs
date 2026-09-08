//! One binary for every fixpoint, graph, math-kernels integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/wire_words/mod.rs`.
#[path = "wire_words/mod.rs"]
pub mod wire_words;

/// Integration tests from `tests/adversarial_graph_reachability_fixpoint/mod.rs`.
#[path = "adversarial_graph_reachability_fixpoint/mod.rs"]
pub mod adversarial_graph_reachability_fixpoint;
