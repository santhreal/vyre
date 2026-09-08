//! One binary for every llm integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/harness/mod.rs`.
#[path = "harness/mod.rs"]
pub mod harness;

/// Integration tests from `tests/attention_layout_launch_domain.rs`.
#[path = "attention_layout_launch_domain.rs"]
pub mod attention_layout_launch_domain;
