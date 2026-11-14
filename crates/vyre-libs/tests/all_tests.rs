//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `tools/xtask/test-harness-isolation.toml`.

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

/// Shared fixture module from `vyre_test_support::ir_shape`.
#[cfg(feature = "reduce")]
pub use vyre_test_support::ir_shape;

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
#[cfg(feature = "bitset")]
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

/// Integration tests from `tests/cpu_witnesses.rs`.
#[allow(deprecated)]
#[path = "cpu_witnesses.rs"]
pub mod cpu_witnesses;

/// Integration tests from `tests/crc32_map_reduce_generated.rs`.
#[path = "crc32_map_reduce_generated.rs"]
pub mod crc32_map_reduce_generated;

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

/// Integration tests from `tests/fractional_kernel_parity.rs`.
#[path = "fractional_kernel_parity.rs"]
pub mod fractional_kernel_parity;

/// Integration tests from `tests/frontend_dialect_contracts.rs`.
#[path = "frontend_dialect_contracts.rs"]
pub mod frontend_dialect_contracts;

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

/// Integration tests from `tests/primitive_surface_contracts.rs`.
#[path = "primitive_surface_contracts.rs"]
pub mod primitive_surface_contracts;

/// Integration tests from `tests/production_ir_parity.rs`.
#[path = "production_ir_parity.rs"]
pub mod production_ir_parity;

/// Integration tests from `tests/property.rs`.
#[path = "property.rs"]
pub mod property;

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

/// Integration tests from `tests/scan_cpu_api_boundary.rs`.
#[path = "scan_cpu_api_boundary.rs"]
pub mod scan_cpu_api_boundary;

/// Integration tests from `tests/select_arm_range_catalog.rs`.
#[path = "select_arm_range_catalog.rs"]
pub mod select_arm_range_catalog;

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

/// Integration tests from `tests/taint_pollution_grid_sync_planner_cut.rs`.
#[cfg(feature = "reduce")]
#[path = "taint_pollution_grid_sync_planner_cut.rs"]
pub mod taint_pollution_grid_sync_planner_cut;

/// Integration tests from `tests/the_shipped_library_build_compiles_no_cpu_oracle.rs`.
#[path = "the_shipped_library_build_compiles_no_cpu_oracle.rs"]
pub mod the_shipped_library_build_compiles_no_cpu_oracle;

/// Integration tests from `tests/universal_harness.rs`.
#[allow(deprecated)]
#[path = "universal_harness.rs"]
pub mod universal_harness;

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

/// Domain integration test suites absorbed from per-domain library crates.
#[path = "domains/mod.rs"]
pub mod domains;

/// Integration tests from `tests/adversarial_graph_reachability_fixpoint/mod.rs`.
#[cfg(all(feature = "fixpoint", feature = "graph", feature = "math-kernels"))]
#[path = "adversarial_graph_reachability_fixpoint/mod.rs"]
pub mod adversarial_graph_reachability_fixpoint;

/// Integration tests from `tests/loop_back_edge_audit.rs`.
#[cfg(all(
    feature = "fixpoint",
    feature = "graph",
    feature = "math-kernels",
    feature = "parsing-kernels"
))]
#[path = "loop_back_edge_audit.rs"]
pub mod loop_back_edge_audit;

/// Integration tests from `tests/loop_unroll_trip1_idempotence.rs`.
#[cfg(all(feature = "math-kernels", feature = "fixpoint"))]
#[path = "loop_unroll_trip1_idempotence.rs"]
pub mod loop_unroll_trip1_idempotence;

/// Integration tests from `tests/scallop_join_grid_contract.rs`.
#[cfg(all(feature = "math-kernels", feature = "fixpoint"))]
#[path = "scallop_join_grid_contract.rs"]
pub mod scallop_join_grid_contract;

/// Integration tests from `tests/hash_registration_witnesses.rs`.
#[cfg(feature = "hash")]
#[path = "hash_registration_witnesses.rs"]
pub mod hash_registration_witnesses;

/// Integration tests from `tests/integration.rs`.
#[cfg(all(
    feature = "hash",
    feature = "math",
    feature = "nn-activation",
    feature = "nn-linear",
    feature = "pattern"
))]
#[path = "integration.rs"]
pub mod integration;

/// Integration tests from `tests/fingerprint_lock.rs`.
#[cfg(all(
    feature = "hash",
    feature = "nn-activation",
    feature = "nn-attention",
    feature = "nn-linear",
    feature = "nn-norm"
))]
#[path = "fingerprint_lock.rs"]
pub mod fingerprint_lock;

/// Integration tests from `tests/surface_contracts.rs`.
#[cfg(all(feature = "hash", feature = "nn-attention", feature = "nn-norm"))]
#[allow(deprecated)]
#[path = "surface_contracts.rs"]
pub mod surface_contracts;

/// Integration tests from `tests/operation_registry.rs`.
#[cfg(all(
    feature = "math",
    feature = "math-linalg",
    feature = "nn-activation",
    feature = "nn-attention",
    feature = "nn-linear",
    feature = "nn-norm"
))]
#[path = "operation_registry.rs"]
pub mod operation_registry;

/// Integration tests from `tests/name_collision.rs`.
#[cfg(all(
    feature = "math",
    feature = "math-linalg",
    feature = "nn-attention",
    feature = "nn-norm"
))]
#[path = "name_collision.rs"]
pub mod name_collision;
