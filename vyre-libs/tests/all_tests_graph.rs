//! One binary for every graph integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/csr_sweep/mod.rs`.
#[path = "csr_sweep/mod.rs"]
pub mod csr_sweep;

/// Shared fixture module from `tests/gate_fixtures/mod.rs`.
#[cfg(feature = "graph")]
#[macro_use]
#[allow(clippy::needless_range_loop, deprecated)]
#[path = "gate_fixtures/mod.rs"]
pub mod gate_fixtures;

/// Shared fixture module from `tests/graph_sweep_fixtures/mod.rs`.
#[cfg(feature = "graph")]
#[allow(clippy::needless_range_loop, deprecated)]
#[path = "graph_sweep_fixtures/mod.rs"]
pub mod graph_sweep_fixtures;

/// Shared fixture module from `tests/wire_words/mod.rs`.
#[allow(clippy::needless_range_loop, deprecated)]
#[path = "wire_words/mod.rs"]
pub mod wire_words;

/// Integration tests from `tests/adaptive_four_russians_dense_generated.rs`.
#[cfg(feature = "graph")]
#[path = "adaptive_four_russians_dense_generated.rs"]
pub mod adaptive_four_russians_dense_generated;

/// Integration tests from `tests/adversarial_graph_csr_validation_contracts/mod.rs`.
#[cfg(feature = "graph")]
#[allow(clippy::needless_range_loop)]
#[path = "adversarial_graph_csr_validation_contracts/mod.rs"]
pub mod adversarial_graph_csr_validation_contracts;

/// Integration tests from `tests/adversarial_graph_ops/mod.rs`.
#[cfg(feature = "graph")]
#[path = "adversarial_graph_ops/mod.rs"]
pub mod adversarial_graph_ops;

/// Integration tests from `tests/csr_sweep_arm_coverage.rs`.
#[path = "csr_sweep_arm_coverage.rs"]
pub mod csr_sweep_arm_coverage;

/// Integration tests from `tests/dominator_tree_pristine.rs`.
#[cfg(feature = "graph")]
#[path = "dominator_tree_pristine.rs"]
pub mod dominator_tree_pristine;

/// Integration tests from `tests/dominator_tree_proptest.rs`.
#[cfg(feature = "graph")]
#[path = "dominator_tree_proptest.rs"]
pub mod dominator_tree_proptest;

/// Integration tests from `tests/dominator_tree_scale_gate.rs`.
#[cfg(feature = "graph")]
#[path = "dominator_tree_scale_gate.rs"]
pub mod dominator_tree_scale_gate;

/// Integration tests from `tests/graph_motif_contracts.rs`.
#[path = "graph_motif_contracts.rs"]
pub mod graph_motif_contracts;

/// Integration tests from `tests/graph_primitive_binding_contracts.rs`.
#[cfg(feature = "graph")]
#[path = "graph_primitive_binding_contracts.rs"]
pub mod graph_primitive_binding_contracts;

/// Integration tests from `tests/graph_toposort_contracts.rs`.
#[path = "graph_toposort_contracts.rs"]
pub mod graph_toposort_contracts;

/// Integration tests from `tests/parser_graph_navigation_contracts.rs`.
#[cfg(feature = "graph")]
#[allow(deprecated)]
#[path = "parser_graph_navigation_contracts.rs"]
pub mod parser_graph_navigation_contracts;

/// Integration tests from `tests/proptest_csr_frontier_queue.rs`.
#[cfg(feature = "graph")]
#[path = "proptest_csr_frontier_queue.rs"]
pub mod proptest_csr_frontier_queue;

/// Integration tests from `tests/proptest_csr_frontier_shard.rs`.
#[cfg(feature = "graph")]
#[path = "proptest_csr_frontier_shard.rs"]
pub mod proptest_csr_frontier_shard;

/// Integration tests from `tests/proptest_csr_queue_split.rs`.
#[cfg(feature = "graph")]
#[path = "proptest_csr_queue_split.rs"]
pub mod proptest_csr_queue_split;

/// Integration tests from `tests/proptest_csr_queue_strided.rs`.
#[cfg(feature = "graph")]
#[path = "proptest_csr_queue_strided.rs"]
pub mod proptest_csr_queue_strided;

/// Integration tests from `tests/sweep_graph_csr_backward_traverse_volume_oracle_matrix.rs`.
#[cfg(feature = "graph")]
#[path = "sweep_graph_csr_backward_traverse_volume_oracle_matrix.rs"]
pub mod sweep_graph_csr_backward_traverse_volume_oracle_matrix;

/// Integration tests from `tests/sweep_graph_csr_bidirectional_oracle_matrix.rs`.
#[path = "sweep_graph_csr_bidirectional_oracle_matrix.rs"]
pub mod sweep_graph_csr_bidirectional_oracle_matrix;

/// Integration tests from `tests/sweep_graph_csr_forward_or_changed_volume_oracle_matrix.rs`.
#[cfg(feature = "graph")]
#[path = "sweep_graph_csr_forward_or_changed_volume_oracle_matrix.rs"]
pub mod sweep_graph_csr_forward_or_changed_volume_oracle_matrix;

/// Integration tests from `tests/sweep_graph_csr_forward_traverse_volume_oracle_matrix.rs`.
#[cfg(feature = "graph")]
#[path = "sweep_graph_csr_forward_traverse_volume_oracle_matrix.rs"]
pub mod sweep_graph_csr_forward_traverse_volume_oracle_matrix;

/// Integration tests from `tests/sweep_graph_motif_oracle_matrix.rs`.
#[path = "sweep_graph_motif_oracle_matrix.rs"]
pub mod sweep_graph_motif_oracle_matrix;

/// Integration tests from `tests/sweep_graph_path_reconstruct_oracle_matrix.rs`.
#[path = "sweep_graph_path_reconstruct_oracle_matrix.rs"]
pub mod sweep_graph_path_reconstruct_oracle_matrix;

/// Integration tests from `tests/sweep_graph_reachable_volume_oracle_matrix.rs`.
#[cfg(feature = "graph")]
#[path = "sweep_graph_reachable_volume_oracle_matrix.rs"]
pub mod sweep_graph_reachable_volume_oracle_matrix;

/// Integration tests from `tests/sweep_graph_scc_decompose_volume_oracle_matrix.rs`.
#[cfg(feature = "graph")]
#[path = "sweep_graph_scc_decompose_volume_oracle_matrix.rs"]
pub mod sweep_graph_scc_decompose_volume_oracle_matrix;

/// Integration tests from `tests/sweep_toposort_oracle_matrix.rs`.
#[path = "sweep_toposort_oracle_matrix.rs"]
pub mod sweep_toposort_oracle_matrix;
