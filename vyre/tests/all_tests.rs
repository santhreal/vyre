//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

#[path = "collected_errors.rs"]
pub mod collected_errors;

#[path = "consumer_manifest_boundary.rs"]
mod consumer_manifest_boundary;

#[path = "program_fixtures/mod.rs"]
pub mod program_fixtures;

#[path = "artifact_workflow.rs"]
mod artifact_workflow;

#[path = "connected_graph_execution.rs"]
mod connected_graph_execution;

#[path = "ir_surface.rs"]
mod ir_surface;

#[path = "wire_malformed_adversarial.rs"]
mod wire_malformed_adversarial;

#[path = "wire_v1_round_trip.rs"]
mod wire_v1_round_trip;

#[path = "downstream_workflow_fixture.rs"]
mod downstream_workflow_fixture;

#[path = "publication_classes.rs"]
mod publication_classes;

#[path = "facade_boundary.rs"]
mod facade_boundary;
