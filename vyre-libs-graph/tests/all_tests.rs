//! One binary for every integration test in this crate.

#[path = "adaptive_four_russians_dense_generated.rs"]
pub mod adaptive_four_russians_dense_generated;

#[path = "adversarial_graph_csr_validation_contracts/mod.rs"]
pub mod adversarial_graph_csr_validation_contracts;

#[path = "adversarial_graph_ops/mod.rs"]
pub mod adversarial_graph_ops;

#[path = "bounded_compile_policy.rs"]
pub mod bounded_compile_policy;

#[path = "csr_sweep/mod.rs"]
pub mod csr_sweep;

#[path = "csr_sweep_arm_coverage.rs"]
pub mod csr_sweep_arm_coverage;

#[path = "dominator_tree_pristine.rs"]
pub mod dominator_tree_pristine;

#[path = "dominator_tree_proptest.rs"]
pub mod dominator_tree_proptest;

#[path = "dominator_tree_scale_gate.rs"]
pub mod dominator_tree_scale_gate;

#[path = "gate_fixtures/mod.rs"]
pub mod gate_fixtures;

#[path = "graph_motif_contracts.rs"]
pub mod graph_motif_contracts;

#[path = "graph_primitive_binding_contracts.rs"]
pub mod graph_primitive_binding_contracts;

#[path = "graph_single_source_contracts.rs"]
pub mod graph_single_source_contracts;

#[path = "graph_sweep_fixtures/mod.rs"]
pub mod graph_sweep_fixtures;

#[path = "graph_toposort_contracts.rs"]
pub mod graph_toposort_contracts;

#[path = "match_motif_via_reference_parity.rs"]
pub mod match_motif_via_reference_parity;

#[path = "parser_graph_navigation_contracts.rs"]
pub mod parser_graph_navigation_contracts;

#[path = "proptest_csr_frontier_queue.rs"]
pub mod proptest_csr_frontier_queue;

#[path = "proptest_csr_frontier_shard.rs"]
pub mod proptest_csr_frontier_shard;

#[path = "proptest_csr_queue_split.rs"]
pub mod proptest_csr_queue_split;

#[path = "proptest_csr_queue_strided.rs"]
pub mod proptest_csr_queue_strided;

#[path = "reconstruct_path_via_reference_parity.rs"]
pub mod reconstruct_path_via_reference_parity;

#[path = "sweep_graph_cpu_oracle_matrix.rs"]
pub mod sweep_graph_cpu_oracle_matrix;

#[path = "sweep_graph_csr_backward_traverse_volume_oracle_matrix.rs"]
pub mod sweep_graph_csr_backward_traverse_volume_oracle_matrix;

#[path = "sweep_graph_csr_bidirectional_oracle_matrix.rs"]
pub mod sweep_graph_csr_bidirectional_oracle_matrix;

#[path = "sweep_graph_csr_forward_or_changed_volume_oracle_matrix.rs"]
pub mod sweep_graph_csr_forward_or_changed_volume_oracle_matrix;

#[path = "sweep_graph_csr_forward_traverse_volume_oracle_matrix.rs"]
pub mod sweep_graph_csr_forward_traverse_volume_oracle_matrix;

#[path = "sweep_graph_motif_oracle_matrix.rs"]
pub mod sweep_graph_motif_oracle_matrix;

#[path = "sweep_graph_path_reconstruct_oracle_matrix.rs"]
pub mod sweep_graph_path_reconstruct_oracle_matrix;

#[path = "sweep_graph_persistent_bfs_volume_oracle_matrix.rs"]
pub mod sweep_graph_persistent_bfs_volume_oracle_matrix;

#[path = "sweep_graph_reachable_volume_oracle_matrix.rs"]
pub mod sweep_graph_reachable_volume_oracle_matrix;

#[path = "sweep_graph_scc_decompose_volume_oracle_matrix.rs"]
pub mod sweep_graph_scc_decompose_volume_oracle_matrix;

#[path = "sweep_toposort_oracle_matrix.rs"]
pub mod sweep_toposort_oracle_matrix;

#[path = "union_find_alias_via_reference_parity.rs"]
pub mod union_find_alias_via_reference_parity;

#[path = "wire_words/mod.rs"]
pub mod wire_words;
