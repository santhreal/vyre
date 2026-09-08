//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/harness/mod.rs`.
#[cfg(feature = "device-tests")]
#[path = "harness/mod.rs"]
pub mod harness;

/// Integration tests from `tests/aot_launcher_contracts.rs`.
#[cfg(feature = "device-tests")]
#[path = "aot_launcher_contracts.rs"]
pub mod aot_launcher_contracts;

/// Integration tests from `tests/argmax_of_marginals_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "argmax_of_marginals_gpu_parity.rs"]
pub mod argmax_of_marginals_gpu_parity;

/// Integration tests from `tests/async_transfer_byte_span_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "async_transfer_byte_span_parity.rs"]
pub mod async_transfer_byte_span_parity;

/// Integration tests from `tests/autodiff_cuda_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "autodiff_cuda_parity.rs"]
pub mod autodiff_cuda_parity;

/// Integration tests from `tests/bellman_tn_order_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "bellman_tn_order_gpu_parity.rs"]
pub mod bellman_tn_order_gpu_parity;

/// Integration tests from `tests/bigint_add_carry_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "bigint_add_carry_gpu_parity.rs"]
pub mod bigint_add_carry_gpu_parity;

/// Integration tests from `tests/binding_plan_contracts.rs`.
#[cfg(feature = "device-tests")]
#[path = "binding_plan_contracts.rs"]
pub mod binding_plan_contracts;

/// Integration tests from `tests/bitset_pairwise_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "bitset_pairwise_gpu_parity.rs"]
pub mod bitset_pairwise_gpu_parity;

/// Integration tests from `tests/bitset_popcount_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "bitset_popcount_gpu_parity.rs"]
pub mod bitset_popcount_gpu_parity;

/// Integration tests from `tests/bitset_popcount_primitive_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "bitset_popcount_primitive_gpu_parity.rs"]
pub mod bitset_popcount_primitive_gpu_parity;

/// Integration tests from `tests/bitset_primitives_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "bitset_primitives_gpu_parity.rs"]
pub mod bitset_primitives_gpu_parity;

/// Integration tests from `tests/buffer_argument_op_lowers.rs`.
#[cfg(feature = "device-tests")]
#[path = "buffer_argument_op_lowers.rs"]
pub mod buffer_argument_op_lowers;

/// Integration tests from `tests/byte_histogram_utf8_shape_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "byte_histogram_utf8_shape_gpu_parity.rs"]
pub mod byte_histogram_utf8_shape_gpu_parity;

/// Integration tests from `tests/capability_contracts.rs`.
#[cfg(feature = "device-tests")]
#[path = "capability_contracts.rs"]
pub mod capability_contracts;

/// Integration tests from `tests/causal_graph_primitives_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "causal_graph_primitives_gpu_parity.rs"]
pub mod causal_graph_primitives_gpu_parity;

/// Integration tests from `tests/char_class_bracket_match_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "char_class_bracket_match_gpu_parity.rs"]
pub mod char_class_bracket_match_gpu_parity;

/// Integration tests from `tests/closure_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "closure_gpu_parity.rs"]
pub mod closure_gpu_parity;

/// Integration tests from `tests/cooperative_launch_contracts.rs`.
#[cfg(feature = "device-tests")]
#[path = "cooperative_launch_contracts.rs"]
pub mod cooperative_launch_contracts;

/// Integration tests from `tests/csr_backward_or_changed_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "csr_backward_or_changed_gpu_parity.rs"]
pub mod csr_backward_or_changed_gpu_parity;

/// Integration tests from `tests/csr_backward_traverse_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "csr_backward_traverse_gpu_parity.rs"]
pub mod csr_backward_traverse_gpu_parity;

/// Integration tests from `tests/csr_bidirectional_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "csr_bidirectional_gpu_parity.rs"]
pub mod csr_bidirectional_gpu_parity;

/// Integration tests from `tests/csr_forward_or_changed_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "csr_forward_or_changed_gpu_parity.rs"]
pub mod csr_forward_or_changed_gpu_parity;

/// Integration tests from `tests/csr_frontier_degree_sum_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "csr_frontier_degree_sum_gpu_parity.rs"]
pub mod csr_frontier_degree_sum_gpu_parity;

/// Integration tests from `tests/cuda_external_probe_contract.rs`.
#[cfg(feature = "device-tests")]
#[path = "cuda_external_probe_contract.rs"]
pub mod cuda_external_probe_contract;

/// Integration tests from `tests/cuda_ffi_template_contracts.rs`.
#[cfg(feature = "device-tests")]
#[path = "cuda_ffi_template_contracts.rs"]
pub mod cuda_ffi_template_contracts;

/// Integration tests from `tests/cuda_graph_dispatch_parity/mod.rs`.
#[cfg(feature = "device-tests")]
#[path = "cuda_graph_dispatch_parity/mod.rs"]
pub mod cuda_graph_dispatch_parity;

/// Integration tests from `tests/cuda_graph_update_evidence.rs`.
#[cfg(feature = "device-tests")]
#[path = "cuda_graph_update_evidence.rs"]
pub mod cuda_graph_update_evidence;

/// Integration tests from `tests/cuda_scan_memory_pool_registry.rs`.
#[cfg(feature = "device-tests")]
#[path = "cuda_scan_memory_pool_registry.rs"]
pub mod cuda_scan_memory_pool_registry;

/// Integration tests from `tests/cuda_stream_ordered_pool_planner.rs`.
#[cfg(feature = "device-tests")]
#[path = "cuda_stream_ordered_pool_planner.rs"]
pub mod cuda_stream_ordered_pool_planner;

/// Integration tests from `tests/cuda_warp_nfa_plan_registry.rs`.
#[cfg(feature = "device-tests")]
#[path = "cuda_warp_nfa_plan_registry.rs"]
pub mod cuda_warp_nfa_plan_registry;

/// Integration tests from `tests/decode_hex_gpu_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "decode_hex_gpu_parity.rs"]
pub mod decode_hex_gpu_parity;

/// Integration tests from `tests/device_pool_hit_rate_evidence.rs`.
#[cfg(feature = "device-tests")]
#[path = "device_pool_hit_rate_evidence.rs"]
pub mod device_pool_hit_rate_evidence;

/// Integration tests from `tests/dispatch_overhead_breakdown.rs`.
#[cfg(feature = "device-tests")]
#[path = "dispatch_overhead_breakdown.rs"]
pub mod dispatch_overhead_breakdown;

/// Integration tests from `tests/dispatch_overhead_profile.rs`.
#[cfg(feature = "device-tests")]
#[path = "dispatch_overhead_profile.rs"]
pub mod dispatch_overhead_profile;

/// Integration tests from `tests/div_zero_shift_mask_cuda_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "div_zero_shift_mask_cuda_parity.rs"]
pub mod div_zero_shift_mask_cuda_parity;

/// Integration tests from `tests/dominator_frontier_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "dominator_frontier_gpu_parity.rs"]
pub mod dominator_frontier_gpu_parity;

/// Integration tests from `tests/egraph_device_image_upload/mod.rs`.
#[cfg(feature = "device-tests")]
#[path = "egraph_device_image_upload/mod.rs"]
pub mod egraph_device_image_upload;

/// Integration tests from `tests/emitted_ptx_byte_stability.rs`.
#[cfg(feature = "device-tests")]
#[path = "emitted_ptx_byte_stability.rs"]
pub mod emitted_ptx_byte_stability;

/// Integration tests from `tests/emitted_resources_contract.rs`.
#[cfg(feature = "device-tests")]
#[path = "emitted_resources_contract.rs"]
pub mod emitted_resources_contract;

/// Integration tests from `tests/encoding_classify_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "encoding_classify_gpu_parity.rs"]
pub mod encoding_classify_gpu_parity;

/// Integration tests from `tests/execution_contracts.rs`.
#[cfg(feature = "device-tests")]
#[path = "execution_contracts.rs"]
pub mod execution_contracts;

/// Integration tests from `tests/exploded_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "exploded_gpu_parity.rs"]
pub mod exploded_gpu_parity;

/// Integration tests from `tests/fixpoint_visual_region_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "fixpoint_visual_region_gpu_parity.rs"]
pub mod fixpoint_visual_region_gpu_parity;

/// Integration tests from `tests/four_russians_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "four_russians_gpu_parity.rs"]
pub mod four_russians_gpu_parity;

/// Integration tests from `tests/functor_matroid_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "functor_matroid_gpu_parity.rs"]
pub mod functor_matroid_gpu_parity;

/// Integration tests from `tests/generated_atomic_cuda_reference_matrix.rs`.
#[cfg(feature = "device-tests")]
#[path = "generated_atomic_cuda_reference_matrix.rs"]
pub mod generated_atomic_cuda_reference_matrix;

/// Integration tests from `tests/generated_cast_fma_cuda_reference_matrix.rs`.
#[cfg(feature = "device-tests")]
#[path = "generated_cast_fma_cuda_reference_matrix.rs"]
pub mod generated_cast_fma_cuda_reference_matrix;

/// Integration tests from `tests/generated_control_cuda_reference_matrix.rs`.
#[cfg(feature = "device-tests")]
#[path = "generated_control_cuda_reference_matrix.rs"]
pub mod generated_control_cuda_reference_matrix;

/// Integration tests from `tests/generated_f32_cuda_reference_matrix.rs`.
#[cfg(feature = "device-tests")]
#[path = "generated_f32_cuda_reference_matrix.rs"]
pub mod generated_f32_cuda_reference_matrix;

/// Integration tests from `tests/generated_i32_cuda_reference_matrix.rs`.
#[cfg(feature = "device-tests")]
#[path = "generated_i32_cuda_reference_matrix.rs"]
pub mod generated_i32_cuda_reference_matrix;

/// Integration tests from `tests/generated_memory_cuda_reference_matrix.rs`.
#[cfg(feature = "device-tests")]
#[path = "generated_memory_cuda_reference_matrix.rs"]
pub mod generated_memory_cuda_reference_matrix;

/// Integration tests from `tests/generated_resident_cuda_reference_matrix/mod.rs`.
#[cfg(feature = "device-tests")]
#[path = "generated_resident_cuda_reference_matrix/mod.rs"]
pub mod generated_resident_cuda_reference_matrix;

/// Integration tests from `tests/generated_resident_sequence_cuda_reference_matrix/mod.rs`.
#[cfg(feature = "device-tests")]
#[path = "generated_resident_sequence_cuda_reference_matrix/mod.rs"]
pub mod generated_resident_sequence_cuda_reference_matrix;

/// Integration tests from `tests/generated_scalar_cuda_reference_matrix.rs`.
#[cfg(feature = "device-tests")]
#[path = "generated_scalar_cuda_reference_matrix.rs"]
pub mod generated_scalar_cuda_reference_matrix;

/// Integration tests from `tests/gpu_automata_load_balance_registry.rs`.
#[cfg(feature = "device-tests")]
#[path = "gpu_automata_load_balance_registry.rs"]
pub mod gpu_automata_load_balance_registry;

/// Integration tests from `tests/gpu_elementwise_conformance.rs`.
#[cfg(feature = "device-tests")]
#[path = "gpu_elementwise_conformance.rs"]
pub mod gpu_elementwise_conformance;

/// Integration tests from `tests/gqa_attention_primitive_composition_cuda.rs`.
#[cfg(feature = "device-tests")]
#[path = "gqa_attention_primitive_composition_cuda.rs"]
pub mod gqa_attention_primitive_composition_cuda;

/// Integration tests from `tests/graph_toposort_reachable_level_wave_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "graph_toposort_reachable_level_wave_gpu_parity.rs"]
pub mod graph_toposort_reachable_level_wave_gpu_parity;

/// Integration tests from `tests/grid_barrier_arrival_audit.rs`.
#[cfg(feature = "device-tests")]
#[path = "grid_barrier_arrival_audit.rs"]
pub mod grid_barrier_arrival_audit;

/// Integration tests from `tests/grid_sync_capability_probe_audit.rs`.
#[cfg(feature = "device-tests")]
#[path = "grid_sync_capability_probe_audit.rs"]
pub mod grid_sync_capability_probe_audit;

/// Integration tests from `tests/grid_sync_dispatch_contracts.rs`.
#[cfg(feature = "device-tests")]
#[path = "grid_sync_dispatch_contracts.rs"]
pub mod grid_sync_dispatch_contracts;

/// Integration tests from `tests/grid_sync_split_policy_contracts.rs`.
#[cfg(feature = "device-tests")]
#[path = "grid_sync_split_policy_contracts.rs"]
pub mod grid_sync_split_policy_contracts;

/// Integration tests from `tests/hash_parsing_primitives_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "hash_parsing_primitives_gpu_parity.rs"]
pub mod hash_parsing_primitives_gpu_parity;

/// Integration tests from `tests/int4_quantized_gpu_parity/mod.rs`.
#[cfg(feature = "device-tests")]
#[path = "int4_quantized_gpu_parity/mod.rs"]
pub mod int4_quantized_gpu_parity;

/// Integration tests from `tests/interval_merge_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "interval_merge_gpu_parity.rs"]
pub mod interval_merge_gpu_parity;

/// Integration tests from `tests/kfac_autotune_step_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "kfac_autotune_step_gpu_parity.rs"]
pub mod kfac_autotune_step_gpu_parity;

/// Integration tests from `tests/label_predicate_primitives_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "label_predicate_primitives_gpu_parity.rs"]
pub mod label_predicate_primitives_gpu_parity;

/// Integration tests from `tests/launch_geometry_contracts.rs`.
#[cfg(feature = "device-tests")]
#[path = "launch_geometry_contracts.rs"]
pub mod launch_geometry_contracts;

/// Integration tests from `tests/line_splice_classify_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "line_splice_classify_gpu_parity.rs"]
pub mod line_splice_classify_gpu_parity;

/// Integration tests from `tests/math_primitives_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "math_primitives_gpu_parity.rs"]
pub mod math_primitives_gpu_parity;

/// Integration tests from `tests/math_scan_prefix_sum_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "math_scan_prefix_sum_gpu_parity.rs"]
pub mod math_scan_prefix_sum_gpu_parity;

/// Integration tests from `tests/mla_decode_shared_memory_scaling.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "mla_decode_shared_memory_scaling.rs"]
pub mod mla_decode_shared_memory_scaling;

/// Integration tests from `tests/module_cache_accounting.rs`.
#[cfg(feature = "device-tests")]
#[path = "module_cache_accounting.rs"]
pub mod module_cache_accounting;

/// Integration tests from `tests/module_cache_contracts.rs`.
#[cfg(feature = "device-tests")]
#[path = "module_cache_contracts.rs"]
pub mod module_cache_contracts;

/// Integration tests from `tests/motif_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "motif_gpu_parity.rs"]
pub mod motif_gpu_parity;

/// Integration tests from `tests/multi_block_prefix_scan_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "multi_block_prefix_scan_gpu_parity.rs"]
pub mod multi_block_prefix_scan_gpu_parity;

/// Integration tests from `tests/multi_block_scan_device_answer.rs`.
#[cfg(feature = "device-tests")]
#[path = "multi_block_scan_device_answer.rs"]
pub mod multi_block_scan_device_answer;

/// Integration tests from `tests/narrowing_cast_cuda_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "narrowing_cast_cuda_parity.rs"]
pub mod narrowing_cast_cuda_parity;

/// Integration tests from `tests/occupancy_contracts.rs`.
#[path = "occupancy_contracts.rs"]
pub mod occupancy_contracts;

/// Integration tests from `tests/occupancy_evidence.rs`.
#[cfg(feature = "device-tests")]
#[path = "occupancy_evidence.rs"]
pub mod occupancy_evidence;

/// Integration tests from `tests/path_reconstruct_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "path_reconstruct_gpu_parity.rs"]
pub mod path_reconstruct_gpu_parity;

/// Integration tests from `tests/persistent_bfs_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "persistent_bfs_gpu_parity.rs"]
pub mod persistent_bfs_gpu_parity;

/// Integration tests from `tests/persistent_bfs_primitive_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "persistent_bfs_primitive_gpu_parity.rs"]
pub mod persistent_bfs_primitive_gpu_parity;

/// Integration tests from `tests/persistent_fixpoint_partial_exit_cuda.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "persistent_fixpoint_partial_exit_cuda.rs"]
pub mod persistent_fixpoint_partial_exit_cuda;

/// Integration tests from `tests/planar_rewrite_schedule_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "planar_rewrite_schedule_gpu_parity.rs"]
pub mod planar_rewrite_schedule_gpu_parity;

/// Integration tests from `tests/predicate_call_traversal_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "predicate_call_traversal_gpu_parity.rs"]
pub mod predicate_call_traversal_gpu_parity;

/// Integration tests from `tests/predicate_edge_kind_mask_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "predicate_edge_kind_mask_gpu_parity.rs"]
pub mod predicate_edge_kind_mask_gpu_parity;

/// Integration tests from `tests/predicate_node_kind_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "predicate_node_kind_gpu_parity.rs"]
pub mod predicate_node_kind_gpu_parity;

/// Integration tests from `tests/predicate_size_arg_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "predicate_size_arg_gpu_parity.rs"]
pub mod predicate_size_arg_gpu_parity;

/// Integration tests from `tests/preferred_dispatch_backend.rs`.
#[cfg(feature = "device-tests")]
#[path = "preferred_dispatch_backend.rs"]
pub mod preferred_dispatch_backend;

/// Integration tests from `tests/ptx_codegen_smoke.rs`.
#[cfg(feature = "device-tests")]
#[path = "ptx_codegen_smoke.rs"]
pub mod ptx_codegen_smoke;

/// Integration tests from `tests/ptx_key_digest_memo_lifetime.rs`.
#[cfg(feature = "device-tests")]
#[path = "ptx_key_digest_memo_lifetime.rs"]
pub mod ptx_key_digest_memo_lifetime;

/// Integration tests from `tests/reduce_array_primitives_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "reduce_array_primitives_gpu_parity.rs"]
pub mod reduce_array_primitives_gpu_parity;

/// Integration tests from `tests/reduce_scalar_primitives_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "reduce_scalar_primitives_gpu_parity.rs"]
pub mod reduce_scalar_primitives_gpu_parity;

/// Integration tests from `tests/regex_bitstream_program_registry.rs`.
#[cfg(feature = "device-tests")]
#[path = "regex_bitstream_program_registry.rs"]
pub mod regex_bitstream_program_registry;

/// Integration tests from `tests/resident_buffer_contracts/mod.rs`.
#[cfg(feature = "device-tests")]
#[path = "resident_buffer_contracts/mod.rs"]
pub mod resident_buffer_contracts;

/// Integration tests from `tests/resident_dispatch_contracts/mod.rs`.
#[cfg(feature = "device-tests")]
#[path = "resident_dispatch_contracts/mod.rs"]
pub mod resident_dispatch_contracts;

/// Integration tests from `tests/resident_handle_ownership_contracts.rs`.
#[cfg(feature = "device-tests")]
#[path = "resident_handle_ownership_contracts.rs"]
pub mod resident_handle_ownership_contracts;

/// Integration tests from `tests/resident_work_queue_scale_scheduler_contracts.rs`.
#[cfg(feature = "device-tests")]
#[path = "resident_work_queue_scale_scheduler_contracts.rs"]
pub mod resident_work_queue_scale_scheduler_contracts;

/// Integration tests from `tests/resident_work_queue_wave_policy_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "resident_work_queue_wave_policy_parity.rs"]
pub mod resident_work_queue_wave_policy_parity;

/// Integration tests from `tests/rle_segment_lengths_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "rle_segment_lengths_gpu_parity.rs"]
pub mod rle_segment_lengths_gpu_parity;

/// Integration tests from `tests/scallop_join_ddnnf_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "scallop_join_ddnnf_gpu_parity.rs"]
pub mod scallop_join_ddnnf_gpu_parity;

/// Integration tests from `tests/scallop_provenance_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "scallop_provenance_gpu_parity.rs"]
pub mod scallop_provenance_gpu_parity;

/// Integration tests from `tests/scc_decompose_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "scc_decompose_gpu_parity.rs"]
pub mod scc_decompose_gpu_parity;

/// Integration tests from `tests/self_optimizer_const_fold_extended.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "self_optimizer_const_fold_extended.rs"]
pub mod self_optimizer_const_fold_extended;

/// Integration tests from `tests/self_optimizer_const_prop_e2e.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "self_optimizer_const_prop_e2e.rs"]
pub mod self_optimizer_const_prop_e2e;

/// Integration tests from `tests/self_optimizer_cross_scope_cse_e2e.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "self_optimizer_cross_scope_cse_e2e.rs"]
pub mod self_optimizer_cross_scope_cse_e2e;

/// Integration tests from `tests/self_optimizer_cse_e2e.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "self_optimizer_cse_e2e.rs"]
pub mod self_optimizer_cse_e2e;

/// Integration tests from `tests/self_optimizer_cse_let_dedupe_e2e.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "self_optimizer_cse_let_dedupe_e2e.rs"]
pub mod self_optimizer_cse_let_dedupe_e2e;

/// Integration tests from `tests/self_optimizer_dead_branch_e2e.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "self_optimizer_dead_branch_e2e.rs"]
pub mod self_optimizer_dead_branch_e2e;

/// Integration tests from `tests/self_optimizer_differential.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "self_optimizer_differential.rs"]
pub mod self_optimizer_differential;

/// Integration tests from `tests/self_optimizer_e2e.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "self_optimizer_e2e.rs"]
pub mod self_optimizer_e2e;

/// Integration tests from `tests/self_optimizer_licm_e2e.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "self_optimizer_licm_e2e.rs"]
pub mod self_optimizer_licm_e2e;

/// Integration tests from `tests/self_optimizer_pattern_match_e2e.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "self_optimizer_pattern_match_e2e.rs"]
pub mod self_optimizer_pattern_match_e2e;

/// Integration tests from `tests/self_optimizer_pattern_match_extended/mod.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "self_optimizer_pattern_match_extended/mod.rs"]
pub mod self_optimizer_pattern_match_extended;

/// Integration tests from `tests/self_optimizer_scaling_bench.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "self_optimizer_scaling_bench.rs"]
pub mod self_optimizer_scaling_bench;

/// Integration tests from `tests/self_optimizer_validate_e2e.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "self_optimizer_validate_e2e.rs"]
pub mod self_optimizer_validate_e2e;

/// Integration tests from `tests/semantic_execution.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "semantic_execution.rs"]
pub mod semantic_execution;

/// Integration tests from `tests/semiring_gemm_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "semiring_gemm_gpu_parity.rs"]
pub mod semiring_gemm_gpu_parity;

/// Integration tests from `tests/sketch_sparse_fft_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "sketch_sparse_fft_gpu_parity.rs"]
pub mod sketch_sparse_fft_gpu_parity;

/// Integration tests from `tests/sparse_binding_param_cuda.rs`.
#[cfg(feature = "device-tests")]
#[path = "sparse_binding_param_cuda.rs"]
pub mod sparse_binding_param_cuda;

/// Integration tests from `tests/split_op_lowers_through_registry.rs`.
#[cfg(feature = "device-tests")]
#[path = "split_op_lowers_through_registry.rs"]
pub mod split_op_lowers_through_registry;

/// Integration tests from `tests/subgroup_reduce_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "subgroup_reduce_gpu_parity.rs"]
pub mod subgroup_reduce_gpu_parity;

/// Integration tests from `tests/synthetic_binop_cuda_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "synthetic_binop_cuda_parity.rs"]
pub mod synthetic_binop_cuda_parity;

/// Integration tests from `tests/synthetic_device_caps_contracts.rs`.
#[cfg(feature = "device-tests")]
#[path = "synthetic_device_caps_contracts.rs"]
pub mod synthetic_device_caps_contracts;

/// Integration tests from `tests/target_compiler.rs`.
#[cfg(feature = "device-tests")]
#[path = "target_compiler.rs"]
pub mod target_compiler;

/// Integration tests from `tests/telemetry_contracts.rs`.
#[cfg(feature = "device-tests")]
#[path = "telemetry_contracts.rs"]
pub mod telemetry_contracts;

/// Integration tests from `tests/tensor_flow_forward_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "tensor_flow_forward_gpu_parity.rs"]
pub mod tensor_flow_forward_gpu_parity;

/// Integration tests from `tests/tensor_scc_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "tensor_scc_gpu_parity.rs"]
pub mod tensor_scc_gpu_parity;

/// Integration tests from `tests/text_primitives_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "text_primitives_gpu_parity.rs"]
pub mod text_primitives_gpu_parity;

/// Integration tests from `tests/trap_propagation_capability.rs`.
#[cfg(feature = "device-tests")]
#[path = "trap_propagation_capability.rs"]
pub mod trap_propagation_capability;

/// Integration tests from `tests/trap_readback_launch_coverage.rs`.
#[cfg(feature = "device-tests")]
#[path = "trap_readback_launch_coverage.rs"]
pub mod trap_readback_launch_coverage;

/// Integration tests from `tests/union_find_gpu_parity.rs`.
#[cfg(all(test, feature = "device-tests"))]
#[path = "union_find_gpu_parity.rs"]
pub mod union_find_gpu_parity;

/// Integration tests from `tests/unsupported_ir_errors.rs`.
#[cfg(feature = "device-tests")]
#[path = "unsupported_ir_errors.rs"]
pub mod unsupported_ir_errors;

/// Integration tests from `tests/vectorized_memory_live_cuda.rs`.
#[cfg(feature = "device-tests")]
#[path = "vectorized_memory_live_cuda.rs"]
pub mod vectorized_memory_live_cuda;

/// Integration tests from `tests/widening_cast_64_cuda_parity.rs`.
#[cfg(feature = "device-tests")]
#[path = "widening_cast_64_cuda_parity.rs"]
pub mod widening_cast_64_cuda_parity;
