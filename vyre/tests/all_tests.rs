//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/program_fixtures/mod.rs`.
#[path = "program_fixtures/mod.rs"]
pub mod program_fixtures;

/// Integration tests from `tests/artifact_workflow.rs`.
#[path = "artifact_workflow.rs"]
pub mod artifact_workflow;

/// Integration tests from `tests/connected_graph_execution.rs`.
#[path = "connected_graph_execution.rs"]
pub mod connected_graph_execution;

/// Integration tests from `tests/ir_surface.rs`.
#[path = "ir_surface.rs"]
pub mod ir_surface;

/// Integration tests from `tests/wire_malformed_adversarial.rs`.
#[path = "wire_malformed_adversarial.rs"]
pub mod wire_malformed_adversarial;

/// Integration tests from `tests/wire_v1_round_trip.rs`.
#[path = "wire_v1_round_trip.rs"]
pub mod wire_v1_round_trip;
