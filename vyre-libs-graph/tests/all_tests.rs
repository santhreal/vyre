//! One binary for every integration test in this crate.

#[path = "adversarial_graph_csr_validation_contracts/mod.rs"]
pub mod adversarial_graph_csr_validation_contracts;

#[path = "adversarial_graph_ops/mod.rs"]
pub mod adversarial_graph_ops;

#[path = "csr_sweep/mod.rs"]
pub mod csr_sweep;

#[path = "gate_fixtures/mod.rs"]
pub mod gate_fixtures;

#[path = "graph_sweep_fixtures/mod.rs"]
pub mod graph_sweep_fixtures;

#[path = "wire_words/mod.rs"]
pub mod wire_words;

#[path = "adaptive_four_russians_dense_generated.rs"]
pub mod adaptive_four_russians_dense_generated;

#[path = "csr_backward_or_changed_ir_fixpoint.rs"]
pub mod csr_backward_or_changed_ir_fixpoint;

#[path = "csr_sweep_arm_coverage.rs"]
pub mod csr_sweep_arm_coverage;

#[path = "csr_traversal_clone_family_equality.rs"]
pub mod csr_traversal_clone_family_equality;

#[path = "csr_traverse_ir_parity_proptest.rs"]
pub mod csr_traverse_ir_parity_proptest;

#[path = "do_calculus_rule2_value_parity.rs"]
pub mod do_calculus_rule2_value_parity;

#[path = "dominator_tree_pristine.rs"]
pub mod dominator_tree_pristine;

#[path = "dominator_tree_proptest.rs"]
pub mod dominator_tree_proptest;

#[path = "dominator_tree_scale_gate.rs"]
pub mod dominator_tree_scale_gate;

#[path = "frontier_to_queue_multi_workgroup_span.rs"]
pub mod frontier_to_queue_multi_workgroup_span;

#[path = "functor_apply_ir_parity_proptest.rs"]
pub mod functor_apply_ir_parity_proptest;

#[path = "graph_builders_emit_valid_ir.rs"]
pub mod graph_builders_emit_valid_ir;

#[path = "graph_motif_contracts.rs"]
pub mod graph_motif_contracts;

#[path = "graph_primitive_binding_contracts.rs"]
pub mod graph_primitive_binding_contracts;

#[path = "graph_single_source_contracts.rs"]
pub mod graph_single_source_contracts;

#[path = "graph_toposort_contracts.rs"]
pub mod graph_toposort_contracts;

#[path = "match_motif_via_reference_parity.rs"]
pub mod match_motif_via_reference_parity;

#[path = "motif_ir_parity_proptest.rs"]
pub mod motif_ir_parity_proptest;

#[path = "parser_graph_navigation_contracts.rs"]
pub mod parser_graph_navigation_contracts;

#[path = "proptest_csr_frontier_queue.rs"]
pub mod proptest_csr_frontier_queue;

#[path = "proptest_csr_frontier_queue_clear_out.rs"]
pub mod proptest_csr_frontier_queue_clear_out;

#[path = "proptest_csr_frontier_shard.rs"]
pub mod proptest_csr_frontier_shard;

#[path = "proptest_csr_queue_split.rs"]
pub mod proptest_csr_queue_split;

#[path = "proptest_csr_queue_strided.rs"]
pub mod proptest_csr_queue_strided;

#[path = "proptest_graph_reachable.rs"]
pub mod proptest_graph_reachable;

#[path = "proptest_toposort_dag.rs"]
pub mod proptest_toposort_dag;

#[path = "reconstruct_path_via_reference_parity.rs"]
pub mod reconstruct_path_via_reference_parity;

#[path = "sheaf_diffusion_step_signed_parity.rs"]
pub mod sheaf_diffusion_step_signed_parity;

#[path = "simplicial_triangle_message_fixed_point_parity.rs"]
pub mod simplicial_triangle_message_fixed_point_parity;

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

#[path = "toposort_program_value_parity.rs"]
pub mod toposort_program_value_parity;

#[path = "union_find_alias_via_reference_parity.rs"]
pub mod union_find_alias_via_reference_parity;

#[path = "union_find_connectivity_parity.rs"]
pub mod union_find_connectivity_parity;
