# Testing `vyre-driver-cuda`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda
```

Own pure PTX target compilation, native device acquisition, materialization, dispatch, graphs, and release-path evidence.

The crate lives at `vyre-driver-cuda`. The `cuda-driver` owner maintains its
`concrete-backend` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda
```

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --all-features
```

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda -- --ignored --nocapture
```

## Feature sets

- Default feature members: None
- Available manifest features: `cuda`, `default`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `cuda_release_surface` | `vyre-driver-cuda/examples/cuda_release_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --example cuda_release_surface` |
| `lib` | `vyre_driver_cuda` | `vyre-driver-cuda/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda` |
| `test` | `adaptive_sparse_queue_generated_gpu_parity` | `vyre-driver-cuda/tests/adaptive_sparse_queue_generated_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test adaptive_sparse_queue_generated_gpu_parity` |
| `test` | `adaptive_traverse_vast_walk_gpu_parity` | `vyre-driver-cuda/tests/adaptive_traverse_vast_walk_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test adaptive_traverse_vast_walk_gpu_parity` |
| `test` | `aot_launcher_contracts` | `vyre-driver-cuda/tests/aot_launcher_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test aot_launcher_contracts` |
| `test` | `argmax_of_marginals_gpu_parity` | `vyre-driver-cuda/tests/argmax_of_marginals_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test argmax_of_marginals_gpu_parity` |
| `test` | `autodiff_cuda_parity` | `vyre-driver-cuda/tests/autodiff_cuda_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test autodiff_cuda_parity` |
| `test` | `bellman_tn_order_gpu_parity` | `vyre-driver-cuda/tests/bellman_tn_order_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test bellman_tn_order_gpu_parity` |
| `test` | `bigint_add_carry_gpu_parity` | `vyre-driver-cuda/tests/bigint_add_carry_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test bigint_add_carry_gpu_parity` |
| `test` | `binding_plan_contracts` | `vyre-driver-cuda/tests/binding_plan_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test binding_plan_contracts` |
| `test` | `bitset_pairwise_gpu_parity` | `vyre-driver-cuda/tests/bitset_pairwise_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test bitset_pairwise_gpu_parity` |
| `test` | `bitset_popcount_gpu_parity` | `vyre-driver-cuda/tests/bitset_popcount_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test bitset_popcount_gpu_parity` |
| `test` | `bitset_popcount_primitive_gpu_parity` | `vyre-driver-cuda/tests/bitset_popcount_primitive_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test bitset_popcount_primitive_gpu_parity` |
| `test` | `bitset_primitives_gpu_parity` | `vyre-driver-cuda/tests/bitset_primitives_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test bitset_primitives_gpu_parity` |
| `test` | `borrowck_reachability_cuda` | `vyre-driver-cuda/tests/borrowck_reachability_cuda.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test borrowck_reachability_cuda` |
| `test` | `buffer_argument_op_lowers` | `vyre-driver-cuda/tests/buffer_argument_op_lowers.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test buffer_argument_op_lowers` |
| `test` | `byte_histogram_utf8_shape_gpu_parity` | `vyre-driver-cuda/tests/byte_histogram_utf8_shape_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test byte_histogram_utf8_shape_gpu_parity` |
| `test` | `c_preprocess_filter_cuda` | `vyre-driver-cuda/tests/c_preprocess_filter_cuda.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test c_preprocess_filter_cuda` |
| `test` | `c_preprocess_macro_expansion_cuda` | `vyre-driver-cuda/tests/c_preprocess_macro_expansion_cuda.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test c_preprocess_macro_expansion_cuda` |
| `test` | `c_preprocess_payloads_cuda` | `vyre-driver-cuda/tests/c_preprocess_payloads_cuda.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test c_preprocess_payloads_cuda` |
| `test` | `c_preprocess_tokenize_cuda` | `vyre-driver-cuda/tests/c_preprocess_tokenize_cuda.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test c_preprocess_tokenize_cuda` |
| `test` | `capability_contracts` | `vyre-driver-cuda/tests/capability_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test capability_contracts` |
| `test` | `causal_graph_primitives_gpu_parity` | `vyre-driver-cuda/tests/causal_graph_primitives_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test causal_graph_primitives_gpu_parity` |
| `test` | `char_class_bracket_match_gpu_parity` | `vyre-driver-cuda/tests/char_class_bracket_match_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test char_class_bracket_match_gpu_parity` |
| `test` | `closure_gpu_parity` | `vyre-driver-cuda/tests/closure_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test closure_gpu_parity` |
| `test` | `cooperative_launch_contracts` | `vyre-driver-cuda/tests/cooperative_launch_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test cooperative_launch_contracts` |
| `test` | `csr_backward_or_changed_gpu_parity` | `vyre-driver-cuda/tests/csr_backward_or_changed_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test csr_backward_or_changed_gpu_parity` |
| `test` | `csr_backward_traverse_gpu_parity` | `vyre-driver-cuda/tests/csr_backward_traverse_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test csr_backward_traverse_gpu_parity` |
| `test` | `csr_bidirectional_gpu_parity` | `vyre-driver-cuda/tests/csr_bidirectional_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test csr_bidirectional_gpu_parity` |
| `test` | `csr_forward_or_changed_gpu_parity` | `vyre-driver-cuda/tests/csr_forward_or_changed_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test csr_forward_or_changed_gpu_parity` |
| `test` | `csr_frontier_degree_sum_gpu_parity` | `vyre-driver-cuda/tests/csr_frontier_degree_sum_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test csr_frontier_degree_sum_gpu_parity` |
| `test` | `csr_frontier_queue_gpu_parity` | `vyre-driver-cuda/tests/csr_frontier_queue_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test csr_frontier_queue_gpu_parity` |
| `test` | `csr_frontier_queue_word_prefix_multiblock_gpu_parity` | `vyre-driver-cuda/tests/csr_frontier_queue_word_prefix_multiblock_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test csr_frontier_queue_word_prefix_multiblock_gpu_parity` |
| `test` | `csr_queue_split_gpu_parity` | `vyre-driver-cuda/tests/csr_queue_split_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test csr_queue_split_gpu_parity` |
| `test` | `csr_queue_strided_gpu_parity` | `vyre-driver-cuda/tests/csr_queue_strided_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test csr_queue_strided_gpu_parity` |
| `test` | `cuda_external_probe_contract` | `vyre-driver-cuda/tests/cuda_external_probe_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test cuda_external_probe_contract` |
| `test` | `cuda_ffi_template_contracts` | `vyre-driver-cuda/tests/cuda_ffi_template_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test cuda_ffi_template_contracts` |
| `test` | `cuda_graph_dispatch_parity` | `vyre-driver-cuda/tests/cuda_graph_dispatch_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test cuda_graph_dispatch_parity` |
| `test` | `cuda_graph_update_evidence` | `vyre-driver-cuda/tests/cuda_graph_update_evidence.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test cuda_graph_update_evidence` |
| `test` | `cuda_scan_memory_pool_registry` | `vyre-driver-cuda/tests/cuda_scan_memory_pool_registry.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test cuda_scan_memory_pool_registry` |
| `test` | `cuda_stream_ordered_pool_planner` | `vyre-driver-cuda/tests/cuda_stream_ordered_pool_planner.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test cuda_stream_ordered_pool_planner` |
| `test` | `cuda_warp_nfa_plan_registry` | `vyre-driver-cuda/tests/cuda_warp_nfa_plan_registry.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test cuda_warp_nfa_plan_registry` |
| `test` | `dce_workgroup_redundancy_and_cost` | `vyre-driver-cuda/tests/dce_workgroup_redundancy_and_cost.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test dce_workgroup_redundancy_and_cost` |
| `test` | `decode_hex_gpu_parity` | `vyre-driver-cuda/tests/decode_hex_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test decode_hex_gpu_parity` |
| `test` | `device_pool_hit_rate_evidence` | `vyre-driver-cuda/tests/device_pool_hit_rate_evidence.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test device_pool_hit_rate_evidence` |
| `test` | `dispatch_overhead_breakdown` | `vyre-driver-cuda/tests/dispatch_overhead_breakdown.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test dispatch_overhead_breakdown` |
| `test` | `dispatch_overhead_profile` | `vyre-driver-cuda/tests/dispatch_overhead_profile.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test dispatch_overhead_profile` |
| `test` | `div_zero_shift_mask_cuda_parity` | `vyre-driver-cuda/tests/div_zero_shift_mask_cuda_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test div_zero_shift_mask_cuda_parity` |
| `test` | `dominator_frontier_gpu_parity` | `vyre-driver-cuda/tests/dominator_frontier_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test dominator_frontier_gpu_parity` |
| `test` | `egraph_device_image_upload` | `vyre-driver-cuda/tests/egraph_device_image_upload.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test egraph_device_image_upload` |
| `test` | `emitted_ptx_byte_stability` | `vyre-driver-cuda/tests/emitted_ptx_byte_stability.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test emitted_ptx_byte_stability` |
| `test` | `encoding_classify_gpu_parity` | `vyre-driver-cuda/tests/encoding_classify_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test encoding_classify_gpu_parity` |
| `test` | `execution_contracts` | `vyre-driver-cuda/tests/execution_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test execution_contracts` |
| `test` | `exploded_gpu_parity` | `vyre-driver-cuda/tests/exploded_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test exploded_gpu_parity` |
| `test` | `fixpoint_visual_region_gpu_parity` | `vyre-driver-cuda/tests/fixpoint_visual_region_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test fixpoint_visual_region_gpu_parity` |
| `test` | `four_russians_gpu_parity` | `vyre-driver-cuda/tests/four_russians_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test four_russians_gpu_parity` |
| `test` | `functor_matroid_gpu_parity` | `vyre-driver-cuda/tests/functor_matroid_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test functor_matroid_gpu_parity` |
| `test` | `generated_atomic_cuda_reference_matrix` | `vyre-driver-cuda/tests/generated_atomic_cuda_reference_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test generated_atomic_cuda_reference_matrix` |
| `test` | `generated_cast_fma_cuda_reference_matrix` | `vyre-driver-cuda/tests/generated_cast_fma_cuda_reference_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test generated_cast_fma_cuda_reference_matrix` |
| `test` | `generated_control_cuda_reference_matrix` | `vyre-driver-cuda/tests/generated_control_cuda_reference_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test generated_control_cuda_reference_matrix` |
| `test` | `generated_f32_cuda_reference_matrix` | `vyre-driver-cuda/tests/generated_f32_cuda_reference_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test generated_f32_cuda_reference_matrix` |
| `test` | `generated_i32_cuda_reference_matrix` | `vyre-driver-cuda/tests/generated_i32_cuda_reference_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test generated_i32_cuda_reference_matrix` |
| `test` | `generated_memory_cuda_reference_matrix` | `vyre-driver-cuda/tests/generated_memory_cuda_reference_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test generated_memory_cuda_reference_matrix` |
| `test` | `generated_resident_cuda_reference_matrix` | `vyre-driver-cuda/tests/generated_resident_cuda_reference_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test generated_resident_cuda_reference_matrix` |
| `test` | `generated_resident_sequence_cuda_reference_matrix` | `vyre-driver-cuda/tests/generated_resident_sequence_cuda_reference_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test generated_resident_sequence_cuda_reference_matrix` |
| `test` | `generated_scalar_cuda_reference_matrix` | `vyre-driver-cuda/tests/generated_scalar_cuda_reference_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test generated_scalar_cuda_reference_matrix` |
| `test` | `gpu_automata_load_balance_registry` | `vyre-driver-cuda/tests/gpu_automata_load_balance_registry.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test gpu_automata_load_balance_registry` |
| `test` | `gpu_elementwise_conformance` | `vyre-driver-cuda/tests/gpu_elementwise_conformance.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test gpu_elementwise_conformance` |
| `test` | `gqa_attention_primitive_composition_cuda` | `vyre-driver-cuda/tests/gqa_attention_primitive_composition_cuda.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test gqa_attention_primitive_composition_cuda` |
| `test` | `graph_toposort_reachable_level_wave_gpu_parity` | `vyre-driver-cuda/tests/graph_toposort_reachable_level_wave_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test graph_toposort_reachable_level_wave_gpu_parity` |
| `test` | `grid_barrier_arrival_audit` | `vyre-driver-cuda/tests/grid_barrier_arrival_audit.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test grid_barrier_arrival_audit` |
| `test` | `grid_sync_capability_probe_audit` | `vyre-driver-cuda/tests/grid_sync_capability_probe_audit.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test grid_sync_capability_probe_audit` |
| `test` | `grid_sync_dispatch_contracts` | `vyre-driver-cuda/tests/grid_sync_dispatch_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test grid_sync_dispatch_contracts` |
| `test` | `grid_sync_split_policy_contracts` | `vyre-driver-cuda/tests/grid_sync_split_policy_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test grid_sync_split_policy_contracts` |
| `test` | `hash_parsing_primitives_gpu_parity` | `vyre-driver-cuda/tests/hash_parsing_primitives_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test hash_parsing_primitives_gpu_parity` |
| `test` | `int4_quantized_gpu_parity` | `vyre-driver-cuda/tests/int4_quantized_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test int4_quantized_gpu_parity` |
| `test` | `interval_merge_gpu_parity` | `vyre-driver-cuda/tests/interval_merge_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test interval_merge_gpu_parity` |
| `test` | `kfac_autotune_step_gpu_parity` | `vyre-driver-cuda/tests/kfac_autotune_step_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test kfac_autotune_step_gpu_parity` |
| `test` | `label_predicate_primitives_gpu_parity` | `vyre-driver-cuda/tests/label_predicate_primitives_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test label_predicate_primitives_gpu_parity` |
| `test` | `launch_geometry_contracts` | `vyre-driver-cuda/tests/launch_geometry_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test launch_geometry_contracts` |
| `test` | `line_splice_classify_gpu_parity` | `vyre-driver-cuda/tests/line_splice_classify_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test line_splice_classify_gpu_parity` |
| `test` | `math_primitives_gpu_parity` | `vyre-driver-cuda/tests/math_primitives_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test math_primitives_gpu_parity` |
| `test` | `math_scan_prefix_sum_gpu_parity` | `vyre-driver-cuda/tests/math_scan_prefix_sum_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test math_scan_prefix_sum_gpu_parity` |
| `test` | `megakernel_scale_scheduler_contracts` | `vyre-driver-cuda/tests/megakernel_scale_scheduler_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test megakernel_scale_scheduler_contracts` |
| `test` | `megakernel_wave_policy_parity` | `vyre-driver-cuda/tests/megakernel_wave_policy_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test megakernel_wave_policy_parity` |
| `test` | `mla_decode_shared_memory_scaling` | `vyre-driver-cuda/tests/mla_decode_shared_memory_scaling.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test mla_decode_shared_memory_scaling` |
| `test` | `module_cache_contracts` | `vyre-driver-cuda/tests/module_cache_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test module_cache_contracts` |
| `test` | `motif_gpu_parity` | `vyre-driver-cuda/tests/motif_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test motif_gpu_parity` |
| `test` | `multi_block_prefix_scan_gpu_parity` | `vyre-driver-cuda/tests/multi_block_prefix_scan_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test multi_block_prefix_scan_gpu_parity` |
| `test` | `narrowing_cast_cuda_parity` | `vyre-driver-cuda/tests/narrowing_cast_cuda_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test narrowing_cast_cuda_parity` |
| `test` | `occupancy_choice_autotune_persistence` | `vyre-driver-cuda/tests/occupancy_choice_autotune_persistence.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test occupancy_choice_autotune_persistence` |
| `test` | `occupancy_evidence` | `vyre-driver-cuda/tests/occupancy_evidence.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test occupancy_evidence` |
| `test` | `path_reconstruct_gpu_parity` | `vyre-driver-cuda/tests/path_reconstruct_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test path_reconstruct_gpu_parity` |
| `test` | `persistent_bfs_gpu_parity` | `vyre-driver-cuda/tests/persistent_bfs_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test persistent_bfs_gpu_parity` |
| `test` | `persistent_bfs_primitive_gpu_parity` | `vyre-driver-cuda/tests/persistent_bfs_primitive_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test persistent_bfs_primitive_gpu_parity` |
| `test` | `persistent_fixpoint_partial_exit_cuda` | `vyre-driver-cuda/tests/persistent_fixpoint_partial_exit_cuda.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test persistent_fixpoint_partial_exit_cuda` |
| `test` | `planar_rewrite_schedule_gpu_parity` | `vyre-driver-cuda/tests/planar_rewrite_schedule_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test planar_rewrite_schedule_gpu_parity` |
| `test` | `predicate_call_traversal_gpu_parity` | `vyre-driver-cuda/tests/predicate_call_traversal_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test predicate_call_traversal_gpu_parity` |
| `test` | `predicate_edge_kind_mask_gpu_parity` | `vyre-driver-cuda/tests/predicate_edge_kind_mask_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test predicate_edge_kind_mask_gpu_parity` |
| `test` | `predicate_node_kind_gpu_parity` | `vyre-driver-cuda/tests/predicate_node_kind_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test predicate_node_kind_gpu_parity` |
| `test` | `predicate_size_arg_gpu_parity` | `vyre-driver-cuda/tests/predicate_size_arg_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test predicate_size_arg_gpu_parity` |
| `test` | `preferred_dispatch_backend` | `vyre-driver-cuda/tests/preferred_dispatch_backend.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test preferred_dispatch_backend` |
| `test` | `ptx_codegen_smoke` | `vyre-driver-cuda/tests/ptx_codegen_smoke.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test ptx_codegen_smoke` |
| `test` | `ptx_key_digest_memo_lifetime` | `vyre-driver-cuda/tests/ptx_key_digest_memo_lifetime.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test ptx_key_digest_memo_lifetime` |
| `test` | `reduce_array_primitives_gpu_parity` | `vyre-driver-cuda/tests/reduce_array_primitives_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test reduce_array_primitives_gpu_parity` |
| `test` | `reduce_scalar_primitives_gpu_parity` | `vyre-driver-cuda/tests/reduce_scalar_primitives_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test reduce_scalar_primitives_gpu_parity` |
| `test` | `regex_bitstream_program_registry` | `vyre-driver-cuda/tests/regex_bitstream_program_registry.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test regex_bitstream_program_registry` |
| `test` | `resident_buffer_contracts` | `vyre-driver-cuda/tests/resident_buffer_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test resident_buffer_contracts` |
| `test` | `resident_dispatch_contracts` | `vyre-driver-cuda/tests/resident_dispatch_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test resident_dispatch_contracts` |
| `test` | `resident_handle_ownership_contracts` | `vyre-driver-cuda/tests/resident_handle_ownership_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test resident_handle_ownership_contracts` |
| `test` | `rle_segment_lengths_gpu_parity` | `vyre-driver-cuda/tests/rle_segment_lengths_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test rle_segment_lengths_gpu_parity` |
| `test` | `scallop_join_ddnnf_gpu_parity` | `vyre-driver-cuda/tests/scallop_join_ddnnf_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test scallop_join_ddnnf_gpu_parity` |
| `test` | `scallop_provenance_gpu_parity` | `vyre-driver-cuda/tests/scallop_provenance_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test scallop_provenance_gpu_parity` |
| `test` | `scc_decompose_gpu_parity` | `vyre-driver-cuda/tests/scc_decompose_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test scc_decompose_gpu_parity` |
| `test` | `self_optimizer_const_fold_extended` | `vyre-driver-cuda/tests/self_optimizer_const_fold_extended.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test self_optimizer_const_fold_extended` |
| `test` | `self_optimizer_const_prop_e2e` | `vyre-driver-cuda/tests/self_optimizer_const_prop_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test self_optimizer_const_prop_e2e` |
| `test` | `self_optimizer_cross_scope_cse_e2e` | `vyre-driver-cuda/tests/self_optimizer_cross_scope_cse_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test self_optimizer_cross_scope_cse_e2e` |
| `test` | `self_optimizer_cse_e2e` | `vyre-driver-cuda/tests/self_optimizer_cse_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test self_optimizer_cse_e2e` |
| `test` | `self_optimizer_cse_let_dedupe_e2e` | `vyre-driver-cuda/tests/self_optimizer_cse_let_dedupe_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test self_optimizer_cse_let_dedupe_e2e` |
| `test` | `self_optimizer_dead_branch_e2e` | `vyre-driver-cuda/tests/self_optimizer_dead_branch_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test self_optimizer_dead_branch_e2e` |
| `test` | `self_optimizer_differential` | `vyre-driver-cuda/tests/self_optimizer_differential.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test self_optimizer_differential` |
| `test` | `self_optimizer_e2e` | `vyre-driver-cuda/tests/self_optimizer_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test self_optimizer_e2e` |
| `test` | `self_optimizer_licm_e2e` | `vyre-driver-cuda/tests/self_optimizer_licm_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test self_optimizer_licm_e2e` |
| `test` | `self_optimizer_pattern_match_e2e` | `vyre-driver-cuda/tests/self_optimizer_pattern_match_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test self_optimizer_pattern_match_e2e` |
| `test` | `self_optimizer_pattern_match_extended` | `vyre-driver-cuda/tests/self_optimizer_pattern_match_extended.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test self_optimizer_pattern_match_extended` |
| `test` | `self_optimizer_pipeline_resident_e2e` | `vyre-driver-cuda/tests/self_optimizer_pipeline_resident_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test self_optimizer_pipeline_resident_e2e` |
| `test` | `self_optimizer_scaling_bench` | `vyre-driver-cuda/tests/self_optimizer_scaling_bench.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test self_optimizer_scaling_bench` |
| `test` | `self_optimizer_validate_e2e` | `vyre-driver-cuda/tests/self_optimizer_validate_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test self_optimizer_validate_e2e` |
| `test` | `semiring_gemm_gpu_parity` | `vyre-driver-cuda/tests/semiring_gemm_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test semiring_gemm_gpu_parity` |
| `test` | `sketch_sparse_fft_gpu_parity` | `vyre-driver-cuda/tests/sketch_sparse_fft_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test sketch_sparse_fft_gpu_parity` |
| `test` | `sparse_binding_param_cuda` | `vyre-driver-cuda/tests/sparse_binding_param_cuda.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test sparse_binding_param_cuda` |
| `test` | `split_op_lowers_through_registry` | `vyre-driver-cuda/tests/split_op_lowers_through_registry.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test split_op_lowers_through_registry` |
| `test` | `subgroup_reduce_gpu_parity` | `vyre-driver-cuda/tests/subgroup_reduce_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test subgroup_reduce_gpu_parity` |
| `test` | `synthetic_binop_cuda_parity` | `vyre-driver-cuda/tests/synthetic_binop_cuda_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test synthetic_binop_cuda_parity` |
| `test` | `target_compiler` | `vyre-driver-cuda/tests/target_compiler.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test target_compiler` |
| `test` | `telemetry_contracts` | `vyre-driver-cuda/tests/telemetry_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test telemetry_contracts` |
| `test` | `tensor_flow_forward_gpu_parity` | `vyre-driver-cuda/tests/tensor_flow_forward_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test tensor_flow_forward_gpu_parity` |
| `test` | `tensor_scc_gpu_parity` | `vyre-driver-cuda/tests/tensor_scc_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test tensor_scc_gpu_parity` |
| `test` | `text_primitives_gpu_parity` | `vyre-driver-cuda/tests/text_primitives_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test text_primitives_gpu_parity` |
| `test` | `union_find_gpu_parity` | `vyre-driver-cuda/tests/union_find_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test union_find_gpu_parity` |
| `test` | `unsupported_ir_errors` | `vyre-driver-cuda/tests/unsupported_ir_errors.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test unsupported_ir_errors` |
| `test` | `vectorized_memory_live_cuda` | `vyre-driver-cuda/tests/vectorized_memory_live_cuda.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test vectorized_memory_live_cuda` |
| `test` | `widening_cast_64_cuda_parity` | `vyre-driver-cuda/tests/widening_cast_64_cuda_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-cuda --test widening_cast_64_cuda_parity` |

## Test classes

- Device and capability contracts
- Lowering and artifact semantics
- Dispatch, graph, memory, and backend parity tests

## Hardware requirements

You need an NVIDIA GPU and a working CUDA driver for device, dispatch, graph, and ignored physical-adapter tests. Probe failure is a configuration failure, not a skip.

## Evidence outputs

- `release/evidence/conformance/release-all-backends-certificate.json`
- `release/evidence/benchmarks/cuda-release-suite.json`
- Command status and exact backend parity assertions

## Skips and failures

The default command omits only tests marked `#[ignore]`. Run the ignored-test command on an NVIDIA host. An ignored hardware test must fail if CUDA was requested but cannot be acquired.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
