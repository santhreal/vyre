//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/bitset_law_properties/mod.rs`.
#[cfg(feature = "bitset")]
#[macro_use]
#[allow(deprecated)]
#[path = "bitset_law_properties/mod.rs"]
pub mod bitset_law_properties;

/// Shared fixture module from `tests/csr_sweep/mod.rs`.
#[cfg(feature = "graph")]
#[path = "csr_sweep/mod.rs"]
pub mod csr_sweep;

/// Shared fixture module from `tests/graph_sweep_fixtures/mod.rs`.
#[cfg(feature = "graph")]
#[allow(deprecated)]
#[path = "graph_sweep_fixtures/mod.rs"]
pub mod graph_sweep_fixtures;

/// Shared fixture module from `tests/harness/mod.rs`.
#[allow(deprecated)]
#[path = "harness/mod.rs"]
pub mod harness;

/// Shared fixture module from `tests/ir_shape/mod.rs`.
#[cfg(feature = "reduce")]
#[allow(deprecated)]
#[path = "ir_shape/mod.rs"]
pub mod ir_shape;

/// Shared fixture module from `tests/presence_oracle/mod.rs`.
#[allow(deprecated)]
#[path = "presence_oracle/mod.rs"]
pub mod presence_oracle;

/// Shared fixture module from `tests/succinct_words/mod.rs`.
#[cfg(feature = "math-succinct")]
#[allow(deprecated)]
#[path = "succinct_words/mod.rs"]
pub mod succinct_words;

/// Shared fixture module from `tests/text_char_class_runner/mod.rs`.
#[cfg(feature = "text")]
#[allow(deprecated)]
#[path = "text_char_class_runner/mod.rs"]
pub mod text_char_class_runner;

/// Shared fixture module from `tests/wire_words/mod.rs`.
#[allow(deprecated)]
#[path = "wire_words/mod.rs"]
pub mod wire_words;

/// Integration tests from `tests/ac_count_suffix3_naga_validation.rs`.
#[path = "ac_count_suffix3_naga_validation.rs"]
pub mod ac_count_suffix3_naga_validation;

/// Integration tests from `tests/adversarial.rs`.
#[path = "adversarial.rs"]
pub mod adversarial;

/// Integration tests from `tests/adversarial_decode.rs`.
#[cfg(feature = "decode")]
#[path = "adversarial_decode.rs"]
pub mod adversarial_decode;

/// Integration tests from `tests/adversarial_fixpoint.rs`.
#[cfg(feature = "fixpoint")]
#[path = "adversarial_fixpoint.rs"]
pub mod adversarial_fixpoint;

/// Integration tests from `tests/adversarial_frontier_queue_clear.rs`.
#[cfg(feature = "graph")]
#[path = "adversarial_frontier_queue_clear.rs"]
pub mod adversarial_frontier_queue_clear;

/// Integration tests from `tests/adversarial_graph.rs`.
#[cfg(feature = "graph")]
#[path = "adversarial_graph.rs"]
pub mod adversarial_graph;

/// Integration tests from `tests/adversarial_hash.rs`.
#[cfg(feature = "hash")]
#[path = "adversarial_hash.rs"]
pub mod adversarial_hash;

/// Integration tests from `tests/adversarial_label.rs`.
#[cfg(feature = "label")]
#[path = "adversarial_label.rs"]
pub mod adversarial_label;

/// Integration tests from `tests/adversarial_matching.rs`.
#[cfg(feature = "pattern")]
#[path = "adversarial_matching.rs"]
pub mod adversarial_matching;

/// Integration tests from `tests/adversarial_nfa.rs`.
#[cfg(feature = "nfa")]
#[path = "adversarial_nfa.rs"]
pub mod adversarial_nfa;

/// Integration tests from `tests/aho_corasick_kat.rs`.
#[cfg(feature = "pattern-dfa")]
#[allow(deprecated)]
#[path = "aho_corasick_kat.rs"]
pub mod aho_corasick_kat;

/// Integration tests from `tests/algebra_lattice_semiring_contracts.rs`.
#[cfg(feature = "math-algebra")]
#[allow(deprecated)]
#[path = "algebra_lattice_semiring_contracts.rs"]
pub mod algebra_lattice_semiring_contracts;

/// Integration tests from `tests/amg_v_cycle_ir_parity.rs`.
#[cfg(feature = "math")]
#[path = "amg_v_cycle_ir_parity.rs"]
pub mod amg_v_cycle_ir_parity;

/// Integration tests from `tests/argmax_of_marginals_ir_parity_proptest.rs`.
#[cfg(feature = "math")]
#[path = "argmax_of_marginals_ir_parity_proptest.rs"]
pub mod argmax_of_marginals_ir_parity_proptest;

/// Integration tests from `tests/bellman_oob_edge_parity.rs`.
#[cfg(feature = "math")]
#[path = "bellman_oob_edge_parity.rs"]
pub mod bellman_oob_edge_parity;

/// Integration tests from `tests/betti_persistence_parity.rs`.
#[path = "betti_persistence_parity.rs"]
pub mod betti_persistence_parity;

/// Integration tests from `tests/bigint_add_carry_ir_parity_proptest.rs`.
#[cfg(feature = "math")]
#[path = "bigint_add_carry_ir_parity_proptest.rs"]
pub mod bigint_add_carry_ir_parity_proptest;

/// Integration tests from `tests/bitset_fixpoint_warm_start_parity.rs`.
#[cfg(feature = "fixpoint")]
#[path = "bitset_fixpoint_warm_start_parity.rs"]
pub mod bitset_fixpoint_warm_start_parity;

/// Integration tests from `tests/bitset_scalar_ir_parity_proptest.rs`.
#[cfg(feature = "bitset")]
#[path = "bitset_scalar_ir_parity_proptest.rs"]
pub mod bitset_scalar_ir_parity_proptest;

/// Integration tests from `tests/bitset_words_sizing_contracts.rs`.
#[cfg(feature = "bitset")]
#[path = "bitset_words_sizing_contracts.rs"]
pub mod bitset_words_sizing_contracts;

/// Integration tests from `tests/blake3_compress_optimizer_idempotence_contract.rs`.
#[path = "blake3_compress_optimizer_idempotence_contract.rs"]
pub mod blake3_compress_optimizer_idempotence_contract;

/// Integration tests from `tests/blake3_kat.rs`.
#[cfg(feature = "crypto-blake3")]
#[allow(deprecated)]
#[path = "blake3_kat.rs"]
pub mod blake3_kat;

/// Integration tests from `tests/blake3_program.rs`.
#[path = "blake3_program.rs"]
pub mod blake3_program;

/// Integration tests from `tests/blake3_wrong_size.rs`.
#[cfg(feature = "crypto-blake3")]
#[allow(deprecated)]
#[path = "blake3_wrong_size.rs"]
pub mod blake3_wrong_size;

/// Integration tests from `tests/bounded_compile_policy.rs`.
#[path = "bounded_compile_policy.rs"]
pub mod bounded_compile_policy;

/// Integration tests from `tests/bracket_match_proptest.rs`.
#[cfg(feature = "pattern")]
#[path = "bracket_match_proptest.rs"]
pub mod bracket_match_proptest;

/// Integration tests from `tests/buffer_name_cross_family.rs`.
#[path = "buffer_name_cross_family.rs"]
pub mod buffer_name_cross_family;

/// Integration tests from `tests/cat_a_conform.rs`.
#[allow(deprecated)]
#[path = "cat_a_conform.rs"]
pub mod cat_a_conform;

/// Integration tests from `tests/clifford_geometric_product_program_parity.rs`.
#[cfg(feature = "geom")]
#[path = "clifford_geometric_product_program_parity.rs"]
pub mod clifford_geometric_product_program_parity;

/// Integration tests from `tests/composed_regions_resolve_in_the_catalog.rs`.
#[path = "composed_regions_resolve_in_the_catalog.rs"]
pub mod composed_regions_resolve_in_the_catalog;

/// Integration tests from `tests/consumer_boundary.rs`.
#[path = "consumer_boundary.rs"]
pub mod consumer_boundary;

/// Integration tests from `tests/contraction_tiling_epilogue_closure_contracts.rs`.
#[path = "contraction_tiling_epilogue_closure_contracts.rs"]
pub mod contraction_tiling_epilogue_closure_contracts;

/// Integration tests from `tests/corpus_privacy_retention_controls.rs`.
#[path = "corpus_privacy_retention_controls.rs"]
pub mod corpus_privacy_retention_controls;

/// Integration tests from `tests/cpu_witnesses.rs`.
#[allow(deprecated)]
#[path = "cpu_witnesses.rs"]
pub mod cpu_witnesses;

/// Integration tests from `tests/crc32_map_reduce_generated.rs`.
#[path = "crc32_map_reduce_generated.rs"]
pub mod crc32_map_reduce_generated;

/// Integration tests from `tests/csr_backward_or_changed_ir_fixpoint.rs`.
#[cfg(all(feature = "graph", feature = "bitset"))]
#[path = "csr_backward_or_changed_ir_fixpoint.rs"]
pub mod csr_backward_or_changed_ir_fixpoint;

/// Integration tests from `tests/csr_backward_traverse_ir_parity_proptest.rs`.
#[cfg(feature = "graph")]
#[path = "csr_backward_traverse_ir_parity_proptest.rs"]
pub mod csr_backward_traverse_ir_parity_proptest;

/// Integration tests from `tests/csr_certificates.rs`.
#[path = "csr_certificates.rs"]
pub mod csr_certificates;

/// Integration tests from `tests/csr_closure_argument_bundle_gate.rs`.
#[path = "csr_closure_argument_bundle_gate.rs"]
pub mod csr_closure_argument_bundle_gate;

/// Integration tests from `tests/csr_forward_traverse_ir_parity_proptest.rs`.
#[cfg(feature = "graph")]
#[path = "csr_forward_traverse_ir_parity_proptest.rs"]
pub mod csr_forward_traverse_ir_parity_proptest;

/// Integration tests from `tests/csr_frontier_degree_sum_ir_parity_proptest.rs`.
#[cfg(feature = "graph")]
#[path = "csr_frontier_degree_sum_ir_parity_proptest.rs"]
pub mod csr_frontier_degree_sum_ir_parity_proptest;

/// Integration tests from `tests/csr_queue_strided_ir_parity_proptest.rs`.
#[cfg(feature = "graph")]
#[path = "csr_queue_strided_ir_parity_proptest.rs"]
pub mod csr_queue_strided_ir_parity_proptest;

/// Integration tests from `tests/csr_traversal_clone_family_equality.rs`.
#[cfg(feature = "graph")]
#[path = "csr_traversal_clone_family_equality.rs"]
pub mod csr_traversal_clone_family_equality;

/// Integration tests from `tests/decode_primitive_composition_contracts.rs`.
#[path = "decode_primitive_composition_contracts.rs"]
pub mod decode_primitive_composition_contracts;

/// Integration tests from `tests/dedup_conv_ast_walk_family_guard.rs`.
#[cfg(feature = "graph")]
#[path = "dedup_conv_ast_walk_family_guard.rs"]
pub mod dedup_conv_ast_walk_family_guard;

/// Integration tests from `tests/delegating_builder_equivalence.rs`.
#[path = "delegating_builder_equivalence.rs"]
pub mod delegating_builder_equivalence;

/// Integration tests from `tests/delta_flow_arrangements.rs`.
#[path = "delta_flow_arrangements.rs"]
pub mod delta_flow_arrangements;

/// Integration tests from `tests/device_resident_token_fact_graph_ownership.rs`.
#[path = "device_resident_token_fact_graph_ownership.rs"]
pub mod device_resident_token_fact_graph_ownership;

/// Integration tests from `tests/dfa_wire_contracts.rs`.
#[cfg(feature = "pattern")]
#[path = "dfa_wire_contracts.rs"]
pub mod dfa_wire_contracts;

/// Integration tests from `tests/do_calculus_rule2_value_parity.rs`.
#[cfg(feature = "graph")]
#[path = "do_calculus_rule2_value_parity.rs"]
pub mod do_calculus_rule2_value_parity;

/// Integration tests from `tests/dominator_tree_composition.rs`.
#[cfg(feature = "graph")]
#[path = "dominator_tree_composition.rs"]
pub mod dominator_tree_composition;

/// Integration tests from `tests/dp_clip_signed_newton_parity.rs`.
#[cfg(feature = "math")]
#[path = "dp_clip_signed_newton_parity.rs"]
pub mod dp_clip_signed_newton_parity;

/// Integration tests from `tests/elementwise_u32_builder_parity.rs`.
#[cfg(feature = "builder")]
#[path = "elementwise_u32_builder_parity.rs"]
pub mod elementwise_u32_builder_parity;

/// Integration tests from `tests/f32_adversarial.rs`.
#[cfg(all(feature = "nn-attention", feature = "nn-norm"))]
#[allow(deprecated)]
#[path = "f32_adversarial.rs"]
pub mod f32_adversarial;

/// Integration tests from `tests/f32_states_its_contractions.rs`.
#[path = "f32_states_its_contractions.rs"]
pub mod f32_states_its_contractions;

/// Integration tests from `tests/family_duplication_budget.rs`.
#[path = "family_duplication_budget.rs"]
pub mod family_duplication_budget;

/// Integration tests from `tests/filesystem_path_archive_policies.rs`.
#[path = "filesystem_path_archive_policies.rs"]
pub mod filesystem_path_archive_policies;

/// Integration tests from `tests/flow_precision_planner.rs`.
#[path = "flow_precision_planner.rs"]
pub mod flow_precision_planner;

/// Integration tests from `tests/fmm_program_parity.rs`.
#[cfg(feature = "math")]
#[path = "fmm_program_parity.rs"]
pub mod fmm_program_parity;

/// Integration tests from `tests/fnv1a_dyn_parity.rs`.
#[cfg(feature = "hash")]
#[path = "fnv1a_dyn_parity.rs"]
pub mod fnv1a_dyn_parity;

/// Integration tests from `tests/fractional_kernel_parity.rs`.
#[path = "fractional_kernel_parity.rs"]
pub mod fractional_kernel_parity;

/// Integration tests from `tests/frontend_dialect_contracts.rs`.
#[path = "frontend_dialect_contracts.rs"]
pub mod frontend_dialect_contracts;

/// Integration tests from `tests/frontier_absorb_parity.rs`.
#[cfg(feature = "bitset")]
#[path = "frontier_absorb_parity.rs"]
pub mod frontier_absorb_parity;

/// Integration tests from `tests/frontier_load_balancing_policies.rs`.
#[path = "frontier_load_balancing_policies.rs"]
pub mod frontier_load_balancing_policies;

/// Integration tests from `tests/frontier_to_queue_multi_workgroup_span.rs`.
#[cfg(feature = "graph")]
#[path = "frontier_to_queue_multi_workgroup_span.rs"]
pub mod frontier_to_queue_multi_workgroup_span;

/// Integration tests from `tests/functor_apply_ir_parity_proptest.rs`.
#[cfg(feature = "graph")]
#[path = "functor_apply_ir_parity_proptest.rs"]
pub mod functor_apply_ir_parity_proptest;

/// Integration tests from `tests/fuse_decode_scan_error.rs`.
#[cfg(feature = "decode")]
#[path = "fuse_decode_scan_error.rs"]
pub mod fuse_decode_scan_error;

/// Integration tests from `tests/fuzz_target_inventory.rs`.
#[path = "fuzz_target_inventory.rs"]
pub mod fuzz_target_inventory;

/// Integration tests from `tests/go_channel_creation_parity.rs`.
#[cfg(feature = "go-parser")]
#[path = "go_channel_creation_parity.rs"]
pub mod go_channel_creation_parity;

/// Integration tests from `tests/go_frontend_corpus.rs`.
#[cfg(feature = "go-parser")]
#[allow(deprecated)]
#[path = "go_frontend_corpus.rs"]
pub mod go_frontend_corpus;

/// Integration tests from `tests/go_tokenizer_semantics.rs`.
#[cfg(feature = "go-parser")]
#[allow(deprecated)]
#[path = "go_tokenizer_semantics.rs"]
pub mod go_tokenizer_semantics;

/// Integration tests from `tests/gpu_columnar_string_ingress.rs`.
#[path = "gpu_columnar_string_ingress.rs"]
pub mod gpu_columnar_string_ingress;

/// Integration tests from `tests/graph_builders_emit_valid_ir.rs`.
#[cfg(feature = "graph")]
#[path = "graph_builders_emit_valid_ir.rs"]
pub mod graph_builders_emit_valid_ir;

/// Integration tests from `tests/graph_fixpoint_adversarial_generated.rs`.
#[cfg(all(feature = "graph", feature = "bitset"))]
#[path = "graph_fixpoint_adversarial_generated.rs"]
pub mod graph_fixpoint_adversarial_generated;

/// Integration tests from `tests/grid_invariant_catalog.rs`.
#[path = "grid_invariant_catalog.rs"]
pub mod grid_invariant_catalog;

/// Integration tests from `tests/grid_stride_tree_buffer_contract.rs`.
#[path = "grid_stride_tree_buffer_contract.rs"]
pub mod grid_stride_tree_buffer_contract;

/// Integration tests from `tests/grid_stride_tree_sum_covers_every_element.rs`.
#[cfg(feature = "reduce")]
#[path = "grid_stride_tree_sum_covers_every_element.rs"]
pub mod grid_stride_tree_sum_covers_every_element;

/// Integration tests from `tests/hash_crc32_ir_parity_proptest.rs`.
#[cfg(feature = "hash")]
#[path = "hash_crc32_ir_parity_proptest.rs"]
pub mod hash_crc32_ir_parity_proptest;

/// Integration tests from `tests/hash_incremental_adversarial_generated.rs`.
#[path = "hash_incremental_adversarial_generated.rs"]
pub mod hash_incremental_adversarial_generated;

/// Integration tests from `tests/hash_stream_ir_parity_proptest.rs`.
#[cfg(feature = "hash")]
#[path = "hash_stream_ir_parity_proptest.rs"]
pub mod hash_stream_ir_parity_proptest;

/// Integration tests from `tests/hex_decode_scan_fused.rs`.
#[cfg(feature = "pattern-dfa")]
#[allow(deprecated)]
#[path = "hex_decode_scan_fused.rs"]
pub mod hex_decode_scan_fused;

/// Integration tests from `tests/histogram_atomic_scatter_parity.rs`.
#[cfg(feature = "reduce")]
#[path = "histogram_atomic_scatter_parity.rs"]
pub mod histogram_atomic_scatter_parity;

/// Integration tests from `tests/homotopy_euler_signed_parity.rs`.
#[cfg(feature = "opt")]
#[path = "homotopy_euler_signed_parity.rs"]
pub mod homotopy_euler_signed_parity;

/// Integration tests from `tests/host_dispatch_is_parity_only.rs`.
#[path = "host_dispatch_is_parity_only.rs"]
pub mod host_dispatch_is_parity_only;

/// Integration tests from `tests/hypervector_ir_parity_proptest.rs`.
#[cfg(feature = "hash")]
#[path = "hypervector_ir_parity_proptest.rs"]
pub mod hypervector_ir_parity_proptest;

/// Integration tests from `tests/iht_threshold_ir_parity_proptest.rs`.
#[cfg(feature = "math")]
#[path = "iht_threshold_ir_parity_proptest.rs"]
pub mod iht_threshold_ir_parity_proptest;

/// Integration tests from `tests/indexed_move_gather_oob_parity.rs`.
#[cfg(feature = "reduce")]
#[path = "indexed_move_gather_oob_parity.rs"]
pub mod indexed_move_gather_oob_parity;

/// Integration tests from `tests/inflate_stored_ir_parity_proptest.rs`.
#[cfg(feature = "decode")]
#[path = "inflate_stored_ir_parity_proptest.rs"]
pub mod inflate_stored_ir_parity_proptest;

/// Integration tests from `tests/jacobi_serial_body_matches_per_lane.rs`.
#[cfg(feature = "math-kernels")]
#[path = "jacobi_serial_body_matches_per_lane.rs"]
pub mod jacobi_serial_body_matches_per_lane;

/// Integration tests from `tests/jacobi_workgroup_cooperative_contracts.rs`.
#[cfg(feature = "math")]
#[path = "jacobi_workgroup_cooperative_contracts.rs"]
pub mod jacobi_workgroup_cooperative_contracts;

/// Integration tests from `tests/kfac_block_inverse_proptest.rs`.
#[cfg(feature = "math")]
#[path = "kfac_block_inverse_proptest.rs"]
pub mod kfac_block_inverse_proptest;

/// Integration tests from `tests/launch_seam_registry_closure.rs`.
#[path = "launch_seam_registry_closure.rs"]
pub mod launch_seam_registry_closure;

/// Integration tests from `tests/library_operation_provenance.rs`.
#[path = "library_operation_provenance.rs"]
pub mod library_operation_provenance;

/// Integration tests from `tests/literal_set_presence_by_region_ground_truth.rs`.
#[path = "literal_set_presence_by_region_ground_truth.rs"]
pub mod literal_set_presence_by_region_ground_truth;

/// Integration tests from `tests/llm_fused_sampler_matches_the_unfused_pipeline.rs`.
#[cfg(feature = "llm")]
#[path = "llm_fused_sampler_matches_the_unfused_pipeline.rs"]
pub mod llm_fused_sampler_matches_the_unfused_pipeline;

/// Integration tests from `tests/llm_sampler_rejects_degenerate_shapes.rs`.
#[cfg(feature = "llm")]
#[path = "llm_sampler_rejects_degenerate_shapes.rs"]
pub mod llm_sampler_rejects_degenerate_shapes;

/// Integration tests from `tests/logical_proptest.rs`.
#[cfg(feature = "logical")]
#[allow(deprecated)]
#[path = "logical_proptest.rs"]
pub mod logical_proptest;

/// Integration tests from `tests/logical_should_panic.rs`.
#[cfg(feature = "logical")]
#[allow(deprecated)]
#[path = "logical_should_panic.rs"]
pub mod logical_should_panic;

/// Integration tests from `tests/loop_trip_count_bounds.rs`.
#[path = "loop_trip_count_bounds.rs"]
pub mod loop_trip_count_bounds;

/// Integration tests from `tests/lowered_input_abi_closure.rs`.
#[path = "lowered_input_abi_closure.rs"]
pub mod lowered_input_abi_closure;

/// Integration tests from `tests/matching_nfa_scan_program_contracts.rs`.
#[cfg(feature = "pattern-nfa")]
#[allow(deprecated)]
#[path = "matching_nfa_scan_program_contracts.rs"]
pub mod matching_nfa_scan_program_contracts;

/// Integration tests from `tests/math_algebra_branchless_contracts.rs`.
#[cfg(feature = "math-algebra")]
#[path = "math_algebra_branchless_contracts.rs"]
pub mod math_algebra_branchless_contracts;

/// Integration tests from `tests/matroid_intersection_full_proptest.rs`.
#[cfg(feature = "math")]
#[path = "matroid_intersection_full_proptest.rs"]
pub mod matroid_intersection_full_proptest;

/// Integration tests from `tests/matroid_intersection_full_value_parity.rs`.
#[cfg(all(feature = "math-kernels", feature = "graph"))]
#[path = "matroid_intersection_full_value_parity.rs"]
pub mod matroid_intersection_full_value_parity;

/// Integration tests from `tests/motif_ir_parity_proptest.rs`.
#[cfg(feature = "graph")]
#[path = "motif_ir_parity_proptest.rs"]
pub mod motif_ir_parity_proptest;

/// Integration tests from `tests/multi_block_prefix_scan_carry_parity.rs`.
#[cfg(feature = "reduce")]
#[path = "multi_block_prefix_scan_carry_parity.rs"]
pub mod multi_block_prefix_scan_carry_parity;

/// Integration tests from `tests/operator_reporting_interchange.rs`.
#[path = "operator_reporting_interchange.rs"]
pub mod operator_reporting_interchange;

/// Integration tests from `tests/output_encoding_unicode_policies.rs`.
#[path = "output_encoding_unicode_policies.rs"]
pub mod output_encoding_unicode_policies;

/// Integration tests from `tests/padic_hensel_signed_parity.rs`.
#[cfg(feature = "math")]
#[path = "padic_hensel_signed_parity.rs"]
pub mod padic_hensel_signed_parity;

/// Integration tests from `tests/paged_attention_eval.rs`.
#[cfg(feature = "nn-attention")]
#[path = "paged_attention_eval.rs"]
pub mod paged_attention_eval;

/// Integration tests from `tests/parser_edit_delta_contracts.rs`.
#[path = "parser_edit_delta_contracts.rs"]
pub mod parser_edit_delta_contracts;

/// Integration tests from `tests/parser_recovery_corpus_registry.rs`.
#[path = "parser_recovery_corpus_registry.rs"]
pub mod parser_recovery_corpus_registry;

/// Integration tests from `tests/pass_research_trace_artifacts.rs`.
#[path = "pass_research_trace_artifacts.rs"]
pub mod pass_research_trace_artifacts;

/// Integration tests from `tests/persistent_fixpoint_loop_contracts.rs`.
#[cfg(feature = "fixpoint")]
#[path = "persistent_fixpoint_loop_contracts.rs"]
pub mod persistent_fixpoint_loop_contracts;

/// Integration tests from `tests/planar_rewrite_ir_parity_proptest.rs`.
#[cfg(feature = "parsing")]
#[path = "planar_rewrite_ir_parity_proptest.rs"]
pub mod planar_rewrite_ir_parity_proptest;

/// Integration tests from `tests/primitive_surface_contracts.rs`.
#[path = "primitive_surface_contracts.rs"]
pub mod primitive_surface_contracts;

/// Integration tests from `tests/production_ir_parity.rs`.
#[path = "production_ir_parity.rs"]
pub mod production_ir_parity;

/// Integration tests from `tests/property.rs`.
#[path = "property.rs"]
pub mod property;

/// Integration tests from `tests/property_differential_oracles.rs`.
#[path = "property_differential_oracles.rs"]
pub mod property_differential_oracles;

/// Integration tests from `tests/proptest_base64_decode.rs`.
#[cfg(feature = "decode")]
#[path = "proptest_base64_decode.rs"]
pub mod proptest_base64_decode;

/// Integration tests from `tests/proptest_bitset_and_laws.rs`.
#[cfg(feature = "bitset")]
#[path = "proptest_bitset_and_laws.rs"]
pub mod proptest_bitset_and_laws;

/// Integration tests from `tests/proptest_bitset_any.rs`.
#[cfg(feature = "bitset")]
#[path = "proptest_bitset_any.rs"]
pub mod proptest_bitset_any;

/// Integration tests from `tests/proptest_bitset_boolean_algebra.rs`.
#[cfg(feature = "bitset")]
#[path = "proptest_bitset_boolean_algebra.rs"]
pub mod proptest_bitset_boolean_algebra;

/// Integration tests from `tests/proptest_bitset_contains.rs`.
#[cfg(feature = "bitset")]
#[path = "proptest_bitset_contains.rs"]
pub mod proptest_bitset_contains;

/// Integration tests from `tests/proptest_bitset_copy.rs`.
#[cfg(feature = "bitset")]
#[path = "proptest_bitset_copy.rs"]
pub mod proptest_bitset_copy;

/// Integration tests from `tests/proptest_bitset_equal.rs`.
#[cfg(feature = "bitset")]
#[path = "proptest_bitset_equal.rs"]
pub mod proptest_bitset_equal;

/// Integration tests from `tests/proptest_bitset_not_involution.rs`.
#[cfg(feature = "bitset")]
#[path = "proptest_bitset_not_involution.rs"]
pub mod proptest_bitset_not_involution;

/// Integration tests from `tests/proptest_bitset_not_laws.rs`.
#[cfg(feature = "bitset")]
#[path = "proptest_bitset_not_laws.rs"]
pub mod proptest_bitset_not_laws;

/// Integration tests from `tests/proptest_bitset_or_laws.rs`.
#[cfg(feature = "bitset")]
#[path = "proptest_bitset_or_laws.rs"]
pub mod proptest_bitset_or_laws;

/// Integration tests from `tests/proptest_bitset_popcount.rs`.
#[cfg(feature = "bitset")]
#[path = "proptest_bitset_popcount.rs"]
pub mod proptest_bitset_popcount;

/// Integration tests from `tests/proptest_bitset_popcount_laws.rs`.
#[cfg(feature = "bitset")]
#[path = "proptest_bitset_popcount_laws.rs"]
pub mod proptest_bitset_popcount_laws;

/// Integration tests from `tests/proptest_bitset_subset_of.rs`.
#[cfg(feature = "bitset")]
#[path = "proptest_bitset_subset_of.rs"]
pub mod proptest_bitset_subset_of;

/// Integration tests from `tests/proptest_bitset_words.rs`.
#[cfg(feature = "bitset")]
#[path = "proptest_bitset_words.rs"]
pub mod proptest_bitset_words;

/// Integration tests from `tests/proptest_bitset_xor_laws.rs`.
#[cfg(feature = "bitset")]
#[path = "proptest_bitset_xor_laws.rs"]
pub mod proptest_bitset_xor_laws;

/// Integration tests from `tests/proptest_csr_forward_traverse.rs`.
#[cfg(feature = "graph")]
#[path = "proptest_csr_forward_traverse.rs"]
pub mod proptest_csr_forward_traverse;

/// Integration tests from `tests/proptest_dispatch_pack_roundtrip.rs`.
#[cfg(feature = "parsing")]
#[path = "proptest_dispatch_pack_roundtrip.rs"]
pub mod proptest_dispatch_pack_roundtrip;

/// Integration tests from `tests/proptest_dominator_frontier.rs`.
#[cfg(feature = "graph")]
#[path = "proptest_dominator_frontier.rs"]
pub mod proptest_dominator_frontier;

/// Integration tests from `tests/proptest_graph_reachable.rs`.
#[cfg(feature = "graph")]
#[path = "proptest_graph_reachable.rs"]
pub mod proptest_graph_reachable;

/// Integration tests from `tests/proptest_hash_crc32.rs`.
#[cfg(feature = "hash")]
#[path = "proptest_hash_crc32.rs"]
pub mod proptest_hash_crc32;

/// Integration tests from `tests/proptest_hash_fnv1a.rs`.
#[cfg(feature = "hash")]
#[path = "proptest_hash_fnv1a.rs"]
pub mod proptest_hash_fnv1a;

/// Integration tests from `tests/proptest_hex_decode.rs`.
#[cfg(feature = "decode")]
#[path = "proptest_hex_decode.rs"]
pub mod proptest_hex_decode;

/// Integration tests from `tests/proptest_multi_block_prefix_scan.rs`.
#[cfg(feature = "reduce")]
#[path = "proptest_multi_block_prefix_scan.rs"]
pub mod proptest_multi_block_prefix_scan;

/// Integration tests from `tests/proptest_reduce_all.rs`.
#[cfg(feature = "reduce")]
#[path = "proptest_reduce_all.rs"]
pub mod proptest_reduce_all;

/// Integration tests from `tests/proptest_reduce_any.rs`.
#[cfg(feature = "reduce")]
#[path = "proptest_reduce_any.rs"]
pub mod proptest_reduce_any;

/// Integration tests from `tests/proptest_reduce_any_all.rs`.
#[cfg(feature = "reduce")]
#[path = "proptest_reduce_any_all.rs"]
pub mod proptest_reduce_any_all;

/// Integration tests from `tests/proptest_reduce_count_laws.rs`.
#[cfg(feature = "reduce")]
#[path = "proptest_reduce_count_laws.rs"]
pub mod proptest_reduce_count_laws;

/// Integration tests from `tests/proptest_reduce_count_non_zero.rs`.
#[cfg(feature = "reduce")]
#[path = "proptest_reduce_count_non_zero.rs"]
pub mod proptest_reduce_count_non_zero;

/// Integration tests from `tests/proptest_reduce_min_max_laws.rs`.
#[cfg(feature = "reduce")]
#[path = "proptest_reduce_min_max_laws.rs"]
pub mod proptest_reduce_min_max_laws;

/// Integration tests from `tests/proptest_reduce_sum_laws.rs`.
#[cfg(feature = "reduce")]
#[path = "proptest_reduce_sum_laws.rs"]
pub mod proptest_reduce_sum_laws;

/// Integration tests from `tests/proptest_text_byte_histogram.rs`.
#[cfg(feature = "text")]
#[path = "proptest_text_byte_histogram.rs"]
pub mod proptest_text_byte_histogram;

/// Integration tests from `tests/proptest_text_char_class.rs`.
#[cfg(feature = "text")]
#[path = "proptest_text_char_class.rs"]
pub mod proptest_text_char_class;

/// Integration tests from `tests/proptest_toposort_dag.rs`.
#[cfg(feature = "graph")]
#[path = "proptest_toposort_dag.rs"]
pub mod proptest_toposort_dag;

/// Integration tests from `tests/proptest_ziftsieve.rs`.
#[cfg(feature = "decode")]
#[path = "proptest_ziftsieve.rs"]
pub mod proptest_ziftsieve;

/// Integration tests from `tests/public_program_builders_behavioral_coverage.rs`.
#[path = "public_program_builders_behavioral_coverage.rs"]
pub mod public_program_builders_behavioral_coverage;

/// Integration tests from `tests/randomized_svd_signed_parity.rs`.
#[cfg(feature = "math")]
#[path = "randomized_svd_signed_parity.rs"]
pub mod randomized_svd_signed_parity;

/// Integration tests from `tests/range_counts_ir_parity_proptest.rs`.
#[cfg(feature = "reduce")]
#[path = "range_counts_ir_parity_proptest.rs"]
pub mod range_counts_ir_parity_proptest;

/// Integration tests from `tests/reduce_atomic_ir_parity_proptest.rs`.
#[cfg(feature = "reduce")]
#[path = "reduce_atomic_ir_parity_proptest.rs"]
pub mod reduce_atomic_ir_parity_proptest;

/// Integration tests from `tests/reduction_route_parity.rs`.
#[cfg(feature = "reduce")]
#[path = "reduction_route_parity.rs"]
pub mod reduction_route_parity;

/// Integration tests from `tests/reference_step_ceiling_corpus.rs`.
#[allow(deprecated)]
#[path = "reference_step_ceiling_corpus.rs"]
pub mod reference_step_ceiling_corpus;

/// Integration tests from `tests/regex_adversarial_class_catalog.rs`.
#[path = "regex_adversarial_class_catalog.rs"]
pub mod regex_adversarial_class_catalog;

/// Integration tests from `tests/regex_capture_mode_contracts.rs`.
#[path = "regex_capture_mode_contracts.rs"]
pub mod regex_capture_mode_contracts;

/// Integration tests from `tests/regex_columnar_output_contracts.rs`.
#[path = "regex_columnar_output_contracts.rs"]
pub mod regex_columnar_output_contracts;

/// Integration tests from `tests/regex_compile_adversarial.rs`.
#[cfg(feature = "pattern-regex")]
#[path = "regex_compile_adversarial.rs"]
pub mod regex_compile_adversarial;

/// Integration tests from `tests/regex_compile_ascii_class_contracts.rs`.
#[cfg(feature = "pattern-regex")]
#[path = "regex_compile_ascii_class_contracts.rs"]
pub mod regex_compile_ascii_class_contracts;

/// Integration tests from `tests/regex_compile_property.rs`.
#[cfg(feature = "pattern-regex")]
#[path = "regex_compile_property.rs"]
pub mod regex_compile_property;

/// Integration tests from `tests/regex_dfa_anchoring_differential.rs`.
#[cfg(feature = "pattern-regex")]
#[path = "regex_dfa_anchoring_differential.rs"]
pub mod regex_dfa_anchoring_differential;

/// Integration tests from `tests/regex_dfa_char_class_exhaustive.rs`.
#[cfg(feature = "pattern-regex")]
#[path = "regex_dfa_char_class_exhaustive.rs"]
pub mod regex_dfa_char_class_exhaustive;

/// Integration tests from `tests/regex_dfa_leftmost_longest_differential.rs`.
#[cfg(feature = "pattern-regex")]
#[path = "regex_dfa_leftmost_longest_differential.rs"]
pub mod regex_dfa_leftmost_longest_differential;

/// Integration tests from `tests/regex_dialect_lattice.rs`.
#[path = "regex_dialect_lattice.rs"]
pub mod regex_dialect_lattice;

/// Integration tests from `tests/regex_logical_pattern_planner.rs`.
#[path = "regex_logical_pattern_planner.rs"]
pub mod regex_logical_pattern_planner;

/// Integration tests from `tests/regex_match_policy_contracts.rs`.
#[path = "regex_match_policy_contracts.rs"]
pub mod regex_match_policy_contracts;

/// Integration tests from `tests/regex_prefilter_planner_registry.rs`.
#[path = "regex_prefilter_planner_registry.rs"]
pub mod regex_prefilter_planner_registry;

/// Integration tests from `tests/regex_streaming_state_ledger.rs`.
#[path = "regex_streaming_state_ledger.rs"]
pub mod regex_streaming_state_ledger;

/// Integration tests from `tests/regex_unicode_profiles.rs`.
#[path = "regex_unicode_profiles.rs"]
pub mod regex_unicode_profiles;

/// Integration tests from `tests/regex_unsupported_diagnostic_registry.rs`.
#[path = "regex_unsupported_diagnostic_registry.rs"]
pub mod regex_unsupported_diagnostic_registry;

/// Integration tests from `tests/region_chain_discipline.rs`.
#[path = "region_chain_discipline.rs"]
pub mod region_chain_discipline;

/// Integration tests from `tests/region_chain_invariant.rs`.
#[path = "region_chain_invariant.rs"]
pub mod region_chain_invariant;

/// Integration tests from `tests/region_dedup_property.rs`.
#[cfg(feature = "pattern")]
#[path = "region_dedup_property.rs"]
pub mod region_dedup_property;

/// Integration tests from `tests/region_gpu_flag_contracts.rs`.
#[cfg(feature = "pattern")]
#[path = "region_gpu_flag_contracts.rs"]
pub mod region_gpu_flag_contracts;

/// Integration tests from `tests/region_inline_let_scope.rs`.
#[path = "region_inline_let_scope.rs"]
pub mod region_inline_let_scope;

/// Integration tests from `tests/registered_operation_validation.rs`.
#[path = "registered_operation_validation.rs"]
pub mod registered_operation_validation;

/// Integration tests from `tests/registration_drift.rs`.
#[path = "registration_drift.rs"]
pub mod registration_drift;

/// Integration tests from `tests/registry_closure.rs`.
#[path = "registry_closure.rs"]
pub mod registry_closure;

/// Integration tests from `tests/registry_oob_clean.rs`.
#[path = "registry_oob_clean.rs"]
pub mod registry_oob_clean;

/// Integration tests from `tests/resource_budget_complexity_policies.rs`.
#[path = "resource_budget_complexity_policies.rs"]
pub mod resource_budget_complexity_policies;

/// Integration tests from `tests/rle_segment_lengths_contracts.rs`.
#[cfg(feature = "decode")]
#[path = "rle_segment_lengths_contracts.rs"]
pub mod rle_segment_lengths_contracts;

/// Integration tests from `tests/rle_segment_lengths_ir_parity_proptest.rs`.
#[cfg(feature = "decode")]
#[path = "rle_segment_lengths_ir_parity_proptest.rs"]
pub mod rle_segment_lengths_ir_parity_proptest;

/// Integration tests from `tests/scan_cpu_api_boundary.rs`.
#[path = "scan_cpu_api_boundary.rs"]
pub mod scan_cpu_api_boundary;

/// Integration tests from `tests/scan_prefilter_width_closure.rs`.
#[cfg(feature = "pattern-dfa")]
#[path = "scan_prefilter_width_closure.rs"]
pub mod scan_prefilter_width_closure;

/// Integration tests from `tests/score_denoise_signed_parity.rs`.
#[cfg(feature = "math")]
#[path = "score_denoise_signed_parity.rs"]
pub mod score_denoise_signed_parity;

/// Integration tests from `tests/secret_crypto_policies.rs`.
#[path = "secret_crypto_policies.rs"]
pub mod secret_crypto_policies;

/// Integration tests from `tests/security_flows_to_alias_only_parity.rs`.
#[cfg(feature = "security")]
#[path = "security_flows_to_alias_only_parity.rs"]
pub mod security_flows_to_alias_only_parity;

/// Integration tests from `tests/security_privacy_path_corpus_guards.rs`.
#[path = "security_privacy_path_corpus_guards.rs"]
pub mod security_privacy_path_corpus_guards;

/// Integration tests from `tests/security_terminal_output_closure.rs`.
#[cfg(feature = "security")]
#[path = "security_terminal_output_closure.rs"]
pub mod security_terminal_output_closure;

/// Integration tests from `tests/segment_reduce_ir_parity_proptest.rs`.
#[cfg(feature = "reduce")]
#[path = "segment_reduce_ir_parity_proptest.rs"]
pub mod segment_reduce_ir_parity_proptest;

/// Integration tests from `tests/select_arm_range_catalog.rs`.
#[path = "select_arm_range_catalog.rs"]
pub mod select_arm_range_catalog;

/// Integration tests from `tests/semiring_gemm_wide_parity.rs`.
#[cfg(feature = "math")]
#[path = "semiring_gemm_wide_parity.rs"]
pub mod semiring_gemm_wide_parity;

/// Integration tests from `tests/semiring_registry.rs`.
#[path = "semiring_registry.rs"]
pub mod semiring_registry;

/// Integration tests from `tests/set_domain_selector.rs`.
#[path = "set_domain_selector.rs"]
pub mod set_domain_selector;

/// Integration tests from `tests/shared_emitter_artifact_schema.rs`.
#[path = "shared_emitter_artifact_schema.rs"]
pub mod shared_emitter_artifact_schema;

/// Integration tests from `tests/shared_owner_closure.rs`.
#[path = "shared_owner_closure.rs"]
pub mod shared_owner_closure;

/// Integration tests from `tests/sheaf_diffusion_step_signed_parity.rs`.
#[cfg(feature = "graph")]
#[path = "sheaf_diffusion_step_signed_parity.rs"]
pub mod sheaf_diffusion_step_signed_parity;

/// Integration tests from `tests/sheaf_laplacian_eigenvalue_dispatch_parity.rs`.
#[cfg(feature = "math-kernels")]
#[path = "sheaf_laplacian_eigenvalue_dispatch_parity.rs"]
pub mod sheaf_laplacian_eigenvalue_dispatch_parity;

/// Integration tests from `tests/simplicial_triangle_message_fixed_point_parity.rs`.
#[cfg(feature = "topology")]
#[path = "simplicial_triangle_message_fixed_point_parity.rs"]
pub mod simplicial_triangle_message_fixed_point_parity;

/// Integration tests from `tests/single_invocation_guard_contracts.rs`.
#[path = "single_invocation_guard_contracts.rs"]
pub mod single_invocation_guard_contracts;

/// Integration tests from `tests/sinkhorn_iterate_ir_parity.rs`.
#[cfg(feature = "math")]
#[path = "sinkhorn_iterate_ir_parity.rs"]
pub mod sinkhorn_iterate_ir_parity;

/// Integration tests from `tests/sinkhorn_scale_ir_parity_proptest.rs`.
#[cfg(feature = "math")]
#[path = "sinkhorn_scale_ir_parity_proptest.rs"]
pub mod sinkhorn_scale_ir_parity_proptest;

/// Integration tests from `tests/sos_gram_construct_proptest.rs`.
#[cfg(feature = "math")]
#[path = "sos_gram_construct_proptest.rs"]
pub mod sos_gram_construct_proptest;

/// Integration tests from `tests/sos_gram_oob_parity.rs`.
#[cfg(feature = "math")]
#[path = "sos_gram_oob_parity.rs"]
pub mod sos_gram_oob_parity;

/// Integration tests from `tests/source_span_witness_records.rs`.
#[path = "source_span_witness_records.rs"]
pub mod source_span_witness_records;

/// Integration tests from `tests/ssa_dominance_phi_overflow_parity.rs`.
#[cfg(feature = "parsing")]
#[path = "ssa_dominance_phi_overflow_parity.rs"]
pub mod ssa_dominance_phi_overflow_parity;

/// Integration tests from `tests/stream_compact_proptest.rs`.
#[cfg(feature = "math")]
#[path = "stream_compact_proptest.rs"]
pub mod stream_compact_proptest;

/// Integration tests from `tests/subgroup_nfa_ir_parity_proptest.rs`.
#[cfg(feature = "nfa")]
#[path = "subgroup_nfa_ir_parity_proptest.rs"]
pub mod subgroup_nfa_ir_parity_proptest;

/// Integration tests from `tests/succinct_rank_contracts.rs`.
#[cfg(feature = "math-succinct")]
#[allow(deprecated)]
#[path = "succinct_rank_contracts.rs"]
pub mod succinct_rank_contracts;

/// Integration tests from `tests/succinct_rank_select_adversarial_contracts.rs`.
#[cfg(feature = "math-succinct")]
#[allow(deprecated)]
#[path = "succinct_rank_select_adversarial_contracts.rs"]
pub mod succinct_rank_select_adversarial_contracts;

/// Integration tests from `tests/sum_product_depth_order_is_a_contract.rs`.
#[cfg(feature = "graph")]
#[path = "sum_product_depth_order_is_a_contract.rs"]
pub mod sum_product_depth_order_is_a_contract;

/// Integration tests from `tests/sum_product_signed_parity.rs`.
#[cfg(feature = "graph")]
#[path = "sum_product_signed_parity.rs"]
pub mod sum_product_signed_parity;

/// Integration tests from `tests/symmetric_eigen_jacobi_parity.rs`.
#[cfg(feature = "math")]
#[path = "symmetric_eigen_jacobi_parity.rs"]
pub mod symmetric_eigen_jacobi_parity;

/// Integration tests from `tests/symmetric_eigen_jacobi_registration.rs`.
#[cfg(feature = "math")]
#[path = "symmetric_eigen_jacobi_registration.rs"]
pub mod symmetric_eigen_jacobi_registration;

/// Integration tests from `tests/syntax_motif_frontier_compiler.rs`.
#[path = "syntax_motif_frontier_compiler.rs"]
pub mod syntax_motif_frontier_compiler;

/// Integration tests from `tests/taint_pollution_grid_sync_planner_cut.rs`.
#[cfg(feature = "security")]
#[path = "taint_pollution_grid_sync_planner_cut.rs"]
pub mod taint_pollution_grid_sync_planner_cut;

/// Integration tests from `tests/target_instruction_capabilities.rs`.
#[path = "target_instruction_capabilities.rs"]
pub mod target_instruction_capabilities;

/// Integration tests from `tests/tensor_scc_value_parity.rs`.
#[cfg(feature = "math-kernels")]
#[path = "tensor_scc_value_parity.rs"]
pub mod tensor_scc_value_parity;

/// Integration tests from `tests/tensor_train_contract_signed_parity.rs`.
#[cfg(feature = "math")]
#[path = "tensor_train_contract_signed_parity.rs"]
pub mod tensor_train_contract_signed_parity;

/// Integration tests from `tests/tensor_train_decompose_eigen_contract.rs`.
#[cfg(feature = "math")]
#[path = "tensor_train_decompose_eigen_contract.rs"]
pub mod tensor_train_decompose_eigen_contract;

/// Integration tests from `tests/tensor_train_decompose_step_parity.rs`.
#[cfg(feature = "math")]
#[path = "tensor_train_decompose_step_parity.rs"]
pub mod tensor_train_decompose_step_parity;

/// Integration tests from `tests/tfn_scalar_mix_signed_parity.rs`.
#[cfg(feature = "geom")]
#[path = "tfn_scalar_mix_signed_parity.rs"]
pub mod tfn_scalar_mix_signed_parity;

/// Integration tests from `tests/the_shipped_library_build_compiles_no_cpu_oracle.rs`.
#[path = "the_shipped_library_build_compiles_no_cpu_oracle.rs"]
pub mod the_shipped_library_build_compiles_no_cpu_oracle;

/// Integration tests from `tests/toposort_program_value_parity.rs`.
#[cfg(feature = "graph")]
#[path = "toposort_program_value_parity.rs"]
pub mod toposort_program_value_parity;

/// Integration tests from `tests/union_find_connectivity_parity.rs`.
#[cfg(feature = "graph")]
#[path = "union_find_connectivity_parity.rs"]
pub mod union_find_connectivity_parity;

/// Integration tests from `tests/universal_harness.rs`.
#[allow(deprecated)]
#[path = "universal_harness.rs"]
pub mod universal_harness;

/// Integration tests from `tests/unsafe_ffi_policies.rs`.
#[path = "unsafe_ffi_policies.rs"]
pub mod unsafe_ffi_policies;

/// Integration tests from `tests/url_network_security_policies.rs`.
#[path = "url_network_security_policies.rs"]
pub mod url_network_security_policies;

/// Integration tests from `tests/utf8_shape_counts_ir_parity_proptest.rs`.
#[cfg(feature = "text")]
#[path = "utf8_shape_counts_ir_parity_proptest.rs"]
pub mod utf8_shape_counts_ir_parity_proptest;

/// Integration tests from `tests/vast_tree_walk_ir_parity_proptest.rs`.
#[cfg(feature = "graph")]
#[path = "vast_tree_walk_ir_parity_proptest.rs"]
pub mod vast_tree_walk_ir_parity_proptest;

/// Integration tests from `tests/vector_neighbor_graph.rs`.
#[path = "vector_neighbor_graph.rs"]
pub mod vector_neighbor_graph;

/// Integration tests from `tests/visual_compositions.rs`.
#[allow(deprecated)]
#[path = "visual_compositions.rs"]
pub mod visual_compositions;

/// Integration tests from `tests/wire_cross_crate_compat.rs`.
#[path = "wire_cross_crate_compat.rs"]
pub mod wire_cross_crate_compat;

/// Integration tests from `tests/witness_input_abi_closure.rs`.
#[allow(deprecated)]
#[path = "witness_input_abi_closure.rs"]
pub mod witness_input_abi_closure;

/// Integration tests from `tests/workgroup_any_ir_parity_proptest.rs`.
#[cfg(feature = "reduce")]
#[path = "workgroup_any_ir_parity_proptest.rs"]
pub mod workgroup_any_ir_parity_proptest;
