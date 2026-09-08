//! One binary for every nn-inference integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/wire_words/mod.rs`.
#[path = "wire_words/mod.rs"]
pub mod wire_words;

/// Integration tests from `tests/causal_conv_state_transition_contract.rs`.
#[path = "causal_conv_state_transition_contract.rs"]
pub mod causal_conv_state_transition_contract;

/// Integration tests from `tests/dense_gated_mlp_graph_contract.rs`.
#[path = "dense_gated_mlp_graph_contract.rs"]
pub mod dense_gated_mlp_graph_contract;

/// Integration tests from `tests/depthwise_causal_conv1d_contract.rs`.
#[path = "depthwise_causal_conv1d_contract.rs"]
pub mod depthwise_causal_conv1d_contract;
