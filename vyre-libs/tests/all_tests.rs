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

/// Shared fixture module from `tests/harness/mod.rs`.
#[allow(deprecated)]
#[path = "harness/mod.rs"]
pub mod harness;

/// Shared fixture module from `tests/ir_shape/mod.rs`.
#[cfg(feature = "reduce")]
#[allow(deprecated)]
#[path = "ir_shape/mod.rs"]
pub mod ir_shape;

/// Shared fixture module from `tests/succinct_words/mod.rs`.
#[cfg(feature = "math-succinct")]
#[allow(deprecated)]
#[path = "succinct_words/mod.rs"]
pub mod succinct_words;

/// Shared fixture module from `tests/wire_words/mod.rs`.
#[allow(deprecated)]
#[path = "wire_words/mod.rs"]
pub mod wire_words;

/// Integration tests from `tests/ac_count_suffix3_naga_validation.rs`.
#[path = "ac_count_suffix3_naga_validation.rs"]
pub mod ac_count_suffix3_naga_validation;

/// Integration tests from `tests/adversarial_fixpoint.rs`.
#[cfg(feature = "fixpoint")]
#[path = "adversarial_fixpoint.rs"]
pub mod adversarial_fixpoint;

/// Integration tests from `tests/adversarial_label.rs`.
#[cfg(feature = "label")]
#[path = "adversarial_label.rs"]
pub mod adversarial_label;

/// Integration tests from `tests/amg_v_cycle_ir_parity.rs`.
#[cfg(feature = "math")]
#[path = "amg_v_cycle_ir_parity.rs"]
pub mod amg_v_cycle_ir_parity;

/// Integration tests from `tests/betti_persistence_parity.rs`.
#[path = "betti_persistence_parity.rs"]
pub mod betti_persistence_parity;

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

/// Integration tests from `tests/buffer_name_cross_family.rs`.
#[path = "buffer_name_cross_family.rs"]
pub mod buffer_name_cross_family;

/// Integration tests from `tests/cat_a_conform.rs`.
#[allow(deprecated)]
#[path = "cat_a_conform.rs"]
pub mod cat_a_conform;

/// Integration tests from `tests/composed_regions_resolve_in_the_catalog.rs`.
#[path = "composed_regions_resolve_in_the_catalog.rs"]
pub mod composed_regions_resolve_in_the_catalog;

/// Integration tests from `tests/contraction_candidate_parity_contracts.rs`.
#[cfg(all(feature = "math-linalg", feature = "nn-linear"))]
#[path = "contraction_candidate_parity_contracts.rs"]
pub mod contraction_candidate_parity_contracts;

/// Integration tests from `tests/consumer_boundary.rs`.
#[path = "consumer_boundary.rs"]
pub mod consumer_boundary;

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

/// Integration tests from `tests/csr_certificates.rs`.
#[path = "csr_certificates.rs"]
pub mod csr_certificates;

/// Integration tests from `tests/csr_closure_argument_bundle_gate.rs`.
#[path = "csr_closure_argument_bundle_gate.rs"]
pub mod csr_closure_argument_bundle_gate;

/// Integration tests from `tests/csr_frontier_degree_sum_ir_parity_proptest.rs`.
#[cfg(feature = "graph")]
#[path = "csr_frontier_degree_sum_ir_parity_proptest.rs"]
pub mod csr_frontier_degree_sum_ir_parity_proptest;

/// Integration tests from `tests/csr_queue_strided_ir_parity_proptest.rs`.
#[cfg(feature = "graph")]
#[path = "csr_queue_strided_ir_parity_proptest.rs"]
pub mod csr_queue_strided_ir_parity_proptest;

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

/// Integration tests from `tests/dominator_tree_composition.rs`.
#[cfg(feature = "graph")]
#[path = "dominator_tree_composition.rs"]
pub mod dominator_tree_composition;

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

/// Integration tests from `tests/flow_precision_planner.rs`.
#[path = "flow_precision_planner.rs"]
pub mod flow_precision_planner;

/// Integration tests from `tests/fractional_kernel_parity.rs`.
#[path = "fractional_kernel_parity.rs"]
pub mod fractional_kernel_parity;

/// Integration tests from `tests/frontend_dialect_contracts.rs`.
#[path = "frontend_dialect_contracts.rs"]
pub mod frontend_dialect_contracts;

/// Integration tests from `tests/frontier_load_balancing_policies.rs`.
#[path = "frontier_load_balancing_policies.rs"]
pub mod frontier_load_balancing_policies;

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

/// Integration tests from `tests/graph_fixpoint_adversarial_generated.rs`.
#[cfg(all(feature = "graph", feature = "bitset"))]
#[path = "graph_fixpoint_adversarial_generated.rs"]
pub mod graph_fixpoint_adversarial_generated;

/// Integration tests from `tests/grid_invariant_catalog.rs`.
#[path = "grid_invariant_catalog.rs"]
pub mod grid_invariant_catalog;

/// Integration tests from `tests/hash_incremental_adversarial_generated.rs`.
#[path = "hash_incremental_adversarial_generated.rs"]
pub mod hash_incremental_adversarial_generated;

/// Integration tests from `tests/hex_decode_scan_fused.rs`.
#[cfg(feature = "pattern-dfa")]
#[allow(deprecated)]
#[path = "hex_decode_scan_fused.rs"]
pub mod hex_decode_scan_fused;

/// Integration tests from `tests/host_dispatch_is_parity_only.rs`.
#[path = "host_dispatch_is_parity_only.rs"]
pub mod host_dispatch_is_parity_only;

/// Integration tests from `tests/host_oracle_elimination_parity.rs`.
#[path = "host_oracle_elimination_parity.rs"]
pub mod host_oracle_elimination_parity;

/// Integration tests from `tests/library_operation_provenance.rs`.
#[path = "library_operation_provenance.rs"]
pub mod library_operation_provenance;

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

/// Integration tests from `tests/lowered_input_abi_closure.rs`.
#[path = "lowered_input_abi_closure.rs"]
pub mod lowered_input_abi_closure;

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

/// Integration tests from `tests/operation_tier_readers.rs`.
#[path = "operation_tier_readers.rs"]
pub mod operation_tier_readers;

/// Integration tests from `tests/operator_reporting_interchange.rs`.
#[path = "operator_reporting_interchange.rs"]
pub mod operator_reporting_interchange;

/// Integration tests from `tests/output_encoding_unicode_policies.rs`.
#[path = "output_encoding_unicode_policies.rs"]
pub mod output_encoding_unicode_policies;

/// Integration tests from `tests/parser_edit_delta_contracts.rs`.
#[path = "parser_edit_delta_contracts.rs"]
pub mod parser_edit_delta_contracts;

/// Integration tests from `tests/parser_recovery_corpus_registry.rs`.
#[path = "parser_recovery_corpus_registry.rs"]
pub mod parser_recovery_corpus_registry;

/// Integration tests from `tests/pass_research_trace_artifacts.rs`.
#[path = "pass_research_trace_artifacts.rs"]
pub mod pass_research_trace_artifacts;

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

/// Integration tests from `tests/proptest_bitset_xor_laws.rs`.
#[cfg(feature = "bitset")]
#[path = "proptest_bitset_xor_laws.rs"]
pub mod proptest_bitset_xor_laws;

/// Integration tests from `tests/proptest_csr_forward_traverse.rs`.
#[cfg(feature = "graph")]
#[path = "proptest_csr_forward_traverse.rs"]
pub mod proptest_csr_forward_traverse;

/// Integration tests from `tests/proptest_dominator_frontier.rs`.
#[cfg(feature = "graph")]
#[path = "proptest_dominator_frontier.rs"]
pub mod proptest_dominator_frontier;

/// Integration tests from `tests/proptest_hash_crc32.rs`.
#[cfg(feature = "hash")]
#[path = "proptest_hash_crc32.rs"]
pub mod proptest_hash_crc32;

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

/// Integration tests from `tests/proptest_ziftsieve.rs`.
#[cfg(feature = "decode")]
#[path = "proptest_ziftsieve.rs"]
pub mod proptest_ziftsieve;

/// Integration tests from `tests/public_program_builders_behavioral_coverage.rs`.
#[path = "public_program_builders_behavioral_coverage.rs"]
pub mod public_program_builders_behavioral_coverage;

/// Integration tests from `tests/reference_step_ceiling_corpus.rs`.
#[allow(deprecated)]
#[path = "reference_step_ceiling_corpus.rs"]
pub mod reference_step_ceiling_corpus;

/// Integration tests from `tests/regex_adversarial_class_catalog.rs`.
#[path = "regex_adversarial_class_catalog.rs"]
pub mod regex_adversarial_class_catalog;

/// Integration tests from `tests/regex_columnar_output_contracts.rs`.
#[path = "regex_columnar_output_contracts.rs"]
pub mod regex_columnar_output_contracts;

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

/// Integration tests from `tests/region_chain_discipline.rs`.
#[path = "region_chain_discipline.rs"]
pub mod region_chain_discipline;

/// Integration tests from `tests/region_chain_invariant.rs`.
#[path = "region_chain_invariant.rs"]
pub mod region_chain_invariant;

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

/// Integration tests from `tests/scan_cpu_api_boundary.rs`.
#[path = "scan_cpu_api_boundary.rs"]
pub mod scan_cpu_api_boundary;

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

/// Integration tests from `tests/select_arm_range_catalog.rs`.
#[path = "select_arm_range_catalog.rs"]
pub mod select_arm_range_catalog;

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

/// Integration tests from `tests/single_invocation_guard_contracts.rs`.
#[path = "single_invocation_guard_contracts.rs"]
pub mod single_invocation_guard_contracts;

/// Integration tests from `tests/sinkhorn_iterate_ir_parity.rs`.
#[cfg(feature = "math")]
#[path = "sinkhorn_iterate_ir_parity.rs"]
pub mod sinkhorn_iterate_ir_parity;

/// Integration tests from `tests/source_span_witness_records.rs`.
#[path = "source_span_witness_records.rs"]
pub mod source_span_witness_records;

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

/// Integration tests from `tests/the_shipped_library_build_compiles_no_cpu_oracle.rs`.
#[path = "the_shipped_library_build_compiles_no_cpu_oracle.rs"]
pub mod the_shipped_library_build_compiles_no_cpu_oracle;

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

/// Integration tests from `tests/vast_tree_walk_ir_parity_proptest.rs`.
#[cfg(feature = "graph")]
#[path = "vast_tree_walk_ir_parity_proptest.rs"]
pub mod vast_tree_walk_ir_parity_proptest;

/// Integration tests from `tests/vector_neighbor_graph.rs`.
#[path = "vector_neighbor_graph.rs"]
pub mod vector_neighbor_graph;

/// Integration tests from `tests/witness_input_abi_closure.rs`.
#[allow(deprecated)]
#[path = "witness_input_abi_closure.rs"]
pub mod witness_input_abi_closure;

/// Integration tests from `tests/multi_domain_production_graph_compilation.rs`.
#[cfg(feature = "graph")]
#[path = "multi_domain_production_graph_compilation.rs"]
pub mod multi_domain_production_graph_compilation;

/// Integration tests from `tests/facade_feature_closure.rs`.
#[path = "facade_feature_closure.rs"]
pub mod facade_feature_closure;

/// The facade re-export surface, against the partition crates that own it.
#[path = "facade_reexport_closure.rs"]
pub mod facade_reexport_closure;
