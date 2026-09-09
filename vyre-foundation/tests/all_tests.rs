//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/contract_cases/optimizer_program_corpus.rs`.
#[allow(missing_docs)]
#[path = "contract_cases/optimizer_program_corpus.rs"]
pub mod corpus;

/// Shared fixture module from `tests/support/opaque_echo_extension.rs`.
///
/// Hoisted here rather than included by each consumer: the file registers its
/// resolver pair with `inventory`, `OpaqueExprResolver` rejects a second
/// registration of one kind, and a rejected table answers every later
/// `Program::from_wire` in the binary with that error.
#[allow(missing_docs)]
#[path = "support/opaque_echo_extension.rs"]
pub mod opaque_echo_extension;

/// Integration tests from `tests/adversarial_graph_canonical_laws.rs`.
#[path = "adversarial_graph_canonical_laws.rs"]
pub mod adversarial_graph_canonical_laws;

/// Integration tests from `tests/adversarial_loop_induction_rebind.rs`.
#[path = "adversarial_loop_induction_rebind.rs"]
pub mod adversarial_loop_induction_rebind;

/// Integration tests from `tests/adversarial_loop_peel_first_iter.rs`.
#[path = "adversarial_loop_peel_first_iter.rs"]
pub mod adversarial_loop_peel_first_iter;

/// Integration tests from `tests/adversarial_program_canonical_laws.rs`.
#[path = "adversarial_program_canonical_laws.rs"]
pub mod adversarial_program_canonical_laws;

/// Integration tests from `tests/algebraic_reordering.rs`.
#[path = "algebraic_reordering.rs"]
pub mod algebraic_reordering;

/// Integration tests from `tests/async_destination_writability.rs`.
#[path = "async_destination_writability.rs"]
pub mod async_destination_writability;

/// Integration tests from `tests/async_tag_discipline.rs`.
#[path = "async_tag_discipline.rs"]
pub mod async_tag_discipline;

/// Integration tests from `tests/async_uniformity.rs`.
#[path = "async_uniformity.rs"]
pub mod async_uniformity;

/// Integration tests from `tests/atomic_linear_type_gap.rs`.
#[path = "atomic_linear_type_gap.rs"]
pub mod atomic_linear_type_gap;

/// Integration tests from `tests/atomic_minimize_operand_positions.rs`.
#[path = "atomic_minimize_operand_positions.rs"]
pub mod atomic_minimize_operand_positions;

/// Integration tests from `tests/autodiff_forward_local_fail_closed.rs`.
#[path = "autodiff_forward_local_fail_closed.rs"]
pub mod autodiff_forward_local_fail_closed;

/// Integration tests from `tests/autodiff_transform_contracts.rs`.
#[path = "autodiff_transform_contracts.rs"]
pub mod autodiff_transform_contracts;

/// Integration tests from `tests/bench_corpus_duplication.rs`.
#[path = "bench_corpus_duplication.rs"]
pub mod bench_corpus_duplication;

/// Integration tests from `tests/binop_operand_swap_law.rs`.
#[path = "binop_operand_swap_law.rs"]
pub mod binop_operand_swap_law;

/// Integration tests from `tests/branch_value_hoist_scope.rs`.
#[path = "branch_value_hoist_scope.rs"]
pub mod branch_value_hoist_scope;

/// Integration tests from `tests/buffer_decl_boundary.rs`.
#[path = "buffer_decl_boundary.rs"]
pub mod buffer_decl_boundary;

/// Integration tests from `tests/buffer_decl_with_count.rs`.
#[path = "buffer_decl_with_count.rs"]
pub mod buffer_decl_with_count;

/// Integration tests from `tests/buffer_layout_proptest.rs`.
#[path = "buffer_layout_proptest.rs"]
pub mod buffer_layout_proptest;

/// Integration tests from `tests/canonical_determinism.rs`.
#[path = "canonical_determinism.rs"]
pub mod canonical_determinism;

/// Integration tests from `tests/canonicalization_corpus_contracts.rs`.
#[path = "canonicalization_corpus_contracts.rs"]
pub mod canonicalization_corpus_contracts;

/// Integration tests from `tests/capability_contracts.rs`.
#[path = "capability_contracts.rs"]
pub mod capability_contracts;

/// Integration tests from `tests/collective_ir_contracts.rs`.
#[path = "collective_ir_contracts.rs"]
pub mod collective_ir_contracts;

/// Integration tests from `tests/composition_tagging_contracts.rs`.
#[path = "composition_tagging_contracts.rs"]
pub mod composition_tagging_contracts;

/// Integration tests from `tests/const_fold_shift_fusion_amount_overflow.rs`.
#[path = "const_fold_shift_fusion_amount_overflow.rs"]
pub mod const_fold_shift_fusion_amount_overflow;

/// Integration tests from `tests/construct_law_closure.rs`.
#[path = "construct_law_closure.rs"]
pub mod construct_law_closure;

/// Integration tests from `tests/consumer_boundary.rs`.
#[path = "consumer_boundary.rs"]
pub mod consumer_boundary;

/// Integration tests from `tests/cse_effect_in_if_cond_invalidates_loads.rs`.
#[path = "cse_effect_in_if_cond_invalidates_loads.rs"]
pub mod cse_effect_in_if_cond_invalidates_loads;

/// Integration tests from `tests/dataflow_fixpoint_merge_contracts.rs`.
#[path = "dataflow_fixpoint_merge_contracts.rs"]
pub mod dataflow_fixpoint_merge_contracts;

/// Integration tests from `tests/dce_subgroup_operand_liveness.rs`.
#[path = "dce_subgroup_operand_liveness.rs"]
pub mod dce_subgroup_operand_liveness;

/// Integration tests from `tests/dead_buffer_dangling_ref.rs`.
#[path = "dead_buffer_dangling_ref.rs"]
pub mod dead_buffer_dangling_ref;

/// Integration tests from `tests/dead_store_elim_overwriter_reads.rs`.
#[path = "dead_store_elim_overwriter_reads.rs"]
pub mod dead_store_elim_overwriter_reads;

/// Integration tests from `tests/demos_orphan_risk.rs`.
#[path = "demos_orphan_risk.rs"]
pub mod demos_orphan_risk;

/// Integration tests from `tests/diagnostic_protocol.rs`.
#[path = "diagnostic_protocol.rs"]
pub mod diagnostic_protocol;

/// Integration tests from `tests/dialect_contracts.rs`.
#[path = "dialect_contracts.rs"]
pub mod dialect_contracts;

/// Integration tests from `tests/dialect_schema_translation_closure_contracts.rs`.
#[path = "dialect_schema_translation_closure_contracts.rs"]
pub mod dialect_schema_translation_closure_contracts;

/// Integration tests from `tests/eqsat_gpu_mirror.rs`.
#[path = "eqsat_gpu_mirror.rs"]
pub mod eqsat_gpu_mirror;

/// Integration tests from `tests/eqsat_unproved_rule_refusal.rs`.
#[path = "eqsat_unproved_rule_refusal.rs"]
pub mod eqsat_unproved_rule_refusal;

/// Integration tests from `tests/execution_plan.rs`.
#[path = "execution_plan.rs"]
pub mod execution_plan;

/// Integration tests from `tests/expr_builder_surface.rs`.
#[path = "expr_builder_surface.rs"]
pub mod expr_builder_surface;

/// Integration tests from `tests/expr_type_single_owner.rs`.
#[path = "expr_type_single_owner.rs"]
pub mod expr_type_single_owner;

/// Integration tests from `tests/expr_variant_traversal_closure.rs`.
#[path = "expr_variant_traversal_closure.rs"]
pub mod expr_variant_traversal_closure;

/// Integration tests from `tests/extension_adversarial.rs`.
#[path = "extension_adversarial.rs"]
pub mod extension_adversarial;

/// Integration tests from `tests/fingerprint_perf_contracts.rs`.
#[path = "fingerprint_perf_contracts.rs"]
pub mod fingerprint_perf_contracts;

/// Integration tests from `tests/foundation_validate_contract.rs`.
#[path = "foundation_validate_contract.rs"]
pub mod foundation_validate_contract;

/// Integration tests from `tests/fusion_atomic_aliasing.rs`.
#[path = "fusion_atomic_aliasing.rs"]
pub mod fusion_atomic_aliasing;

/// Integration tests from `tests/fusion_composability_metadata.rs`.
#[path = "fusion_composability_metadata.rs"]
pub mod fusion_composability_metadata;

/// Integration tests from `tests/fusion_stress.rs`.
#[path = "fusion_stress.rs"]
pub mod fusion_stress;

/// Integration tests from `tests/fusion_substitute_into_subgroup_operand.rs`.
#[path = "fusion_substitute_into_subgroup_operand.rs"]
pub mod fusion_substitute_into_subgroup_operand;

/// Integration tests from `tests/fusion_workgroup_geometry.rs`.
#[path = "fusion_workgroup_geometry.rs"]
pub mod fusion_workgroup_geometry;

/// Integration tests from `tests/geometry_foundation_contracts.rs`.
#[path = "geometry_foundation_contracts.rs"]
pub mod geometry_foundation_contracts;

/// Integration tests from `tests/graph_delta_contract.rs`.
#[path = "graph_delta_contract.rs"]
pub mod graph_delta_contract;
/// Integration tests from `tests/graph_invariants.rs`.
#[path = "graph_invariants.rs"]
pub mod graph_invariants;

/// Integration tests from `tests/inline_buffer_reference_arguments.rs`.
#[path = "inline_buffer_reference_arguments.rs"]
pub mod inline_buffer_reference_arguments;

/// Integration tests from `tests/inline_callee_local_rename_in_trap_and_async.rs`.
#[path = "inline_callee_local_rename_in_trap_and_async.rs"]
pub mod inline_callee_local_rename_in_trap_and_async;

/// Integration tests from `tests/inline_expands_a_call_in_every_operand_slot.rs`.
#[path = "inline_expands_a_call_in_every_operand_slot.rs"]
pub mod inline_expands_a_call_in_every_operand_slot;

/// Integration tests from `tests/inline_nested_call_argument_rebinding.rs`.
#[path = "inline_nested_call_argument_rebinding.rs"]
pub mod inline_nested_call_argument_rebinding;

/// Integration tests from `tests/inline_per_invocation_builtin_rejection.rs`.
#[path = "inline_per_invocation_builtin_rejection.rs"]
pub mod inline_per_invocation_builtin_rejection;

/// Integration tests from `tests/ir_literal_identity.rs`.
#[path = "ir_literal_identity.rs"]
pub mod ir_literal_identity;

/// Integration tests from `tests/ir_variant_shape_owner_closure.rs`.
#[path = "ir_variant_shape_owner_closure.rs"]
pub mod ir_variant_shape_owner_closure;

/// Integration tests from `tests/law_derived_region_alternatives.rs`.
#[path = "law_derived_region_alternatives.rs"]
pub mod law_derived_region_alternatives;

/// Integration tests from `tests/level_pipeline_partition.rs`.
#[path = "level_pipeline_partition.rs"]
pub mod level_pipeline_partition;

/// Integration tests from `tests/level_stage_verdicts.rs`.
#[path = "level_stage_verdicts.rs"]
pub mod level_stage_verdicts;

/// Integration tests from `tests/licm_hoist_scope_safety.rs`.
#[path = "licm_hoist_scope_safety.rs"]
pub mod licm_hoist_scope_safety;

/// Integration tests from `tests/licm_speculation_bounds.rs`.
#[path = "licm_speculation_bounds.rs"]
pub mod licm_speculation_bounds;

/// Integration tests from `tests/linear_type_validation.rs`.
#[path = "linear_type_validation.rs"]
pub mod linear_type_validation;

/// Integration tests from `tests/logical_partition_facts.rs`.
#[path = "logical_partition_facts.rs"]
pub mod logical_partition_facts;

/// Integration tests from `tests/logical_span_contracts.rs`.
#[path = "logical_span_contracts.rs"]
pub mod logical_span_contracts;

/// Integration tests from `tests/logical_stage_identity.rs`.
#[path = "logical_stage_identity.rs"]
pub mod logical_stage_identity;

/// Integration tests from `tests/logical_region_closure_contracts.rs`.
#[path = "logical_region_closure_contracts.rs"]
pub mod logical_region_closure_contracts;

/// Integration tests from `tests/loop_fusion_atomic_expected_scalar_dependency.rs`.
#[path = "loop_fusion_atomic_expected_scalar_dependency.rs"]
pub mod loop_fusion_atomic_expected_scalar_dependency;

/// Integration tests from `tests/loop_fusion_binding_collision.rs`.
#[path = "loop_fusion_binding_collision.rs"]
pub mod loop_fusion_binding_collision;

/// Integration tests from `tests/loop_fusion_pair_after_a_refusal.rs`.
#[path = "loop_fusion_pair_after_a_refusal.rs"]
pub mod loop_fusion_pair_after_a_refusal;

/// Integration tests from `tests/loop_fusion_rename_reaches_every_operand.rs`.
#[path = "loop_fusion_rename_reaches_every_operand.rs"]
pub mod loop_fusion_rename_reaches_every_operand;

/// Integration tests from `tests/loop_fusion_scalar_dependency.rs`.
#[path = "loop_fusion_scalar_dependency.rs"]
pub mod loop_fusion_scalar_dependency;

/// Integration tests from `tests/loop_induction_var_guards.rs`.
#[path = "loop_induction_var_guards.rs"]
pub mod loop_induction_var_guards;

/// Integration tests from `tests/loop_legality_collector_closure.rs`.
#[path = "loop_legality_collector_closure.rs"]
pub mod loop_legality_collector_closure;

/// Integration tests from `tests/loop_licm_scope_extension.rs`.
#[path = "loop_licm_scope_extension.rs"]
pub mod loop_licm_scope_extension;

/// Integration tests from `tests/loop_software_pipeline_loopvar_in_value.rs`.
#[path = "loop_software_pipeline_loopvar_in_value.rs"]
pub mod loop_software_pipeline_loopvar_in_value;

/// Integration tests from `tests/loop_strip_mine_fresh_ident_scope.rs`.
#[path = "loop_strip_mine_fresh_ident_scope.rs"]
pub mod loop_strip_mine_fresh_ident_scope;

/// Integration tests from `tests/loop_unroll_local_scope.rs`.
#[path = "loop_unroll_local_scope.rs"]
pub mod loop_unroll_local_scope;

/// Integration tests from `tests/loop_var_range_fold_scope.rs`.
#[path = "loop_var_range_fold_scope.rs"]
pub mod loop_var_range_fold_scope;

/// Integration tests from `tests/memo_key_completeness/mod.rs`.
#[path = "memo_key_completeness/mod.rs"]
pub mod memo_key_completeness;

/// Integration tests from `tests/memory_ordering.rs`.
#[path = "memory_ordering.rs"]
pub mod memory_ordering;

/// Integration tests from `tests/memory_ordering_adversarial.rs`.
#[path = "memory_ordering_adversarial.rs"]
pub mod memory_ordering_adversarial;

/// Integration tests from `tests/memory_model_contracts.rs`.
#[path = "memory_model_contracts.rs"]
pub mod memory_model_contracts;
/// Integration tests from `tests/memory_model_closed_types_contracts.rs`.
#[path = "memory_model_closed_types_contracts.rs"]
pub mod memory_model_closed_types_contracts;

/// Integration tests from `tests/type_system_composition_contracts.rs`.
#[path = "type_system_composition_contracts.rs"]
pub mod type_system_composition_contracts;

/// Integration tests from `tests/symbolic_shape_interner.rs`.
#[path = "symbolic_shape_interner.rs"]
pub mod symbolic_shape_interner;

/// Integration tests from `tests/declarative_verifier.rs`.
#[path = "declarative_verifier.rs"]
pub mod declarative_verifier;

/// Integration tests from `tests/memory_pass_alias_owner.rs`.
#[path = "memory_pass_alias_owner.rs"]
pub mod memory_pass_alias_owner;

/// Integration tests from `tests/node_rewrite_walk_closure.rs`.
#[path = "node_rewrite_walk_closure.rs"]
pub mod node_rewrite_walk_closure;

/// Integration tests from `tests/node_variant_traversal_closure.rs`.
#[path = "node_variant_traversal_closure.rs"]
pub mod node_variant_traversal_closure;

/// Integration tests from `tests/numeric_contract.rs`.
#[path = "numeric_contract.rs"]
pub mod numeric_contract;

/// Integration tests from `tests/numeric_range_proof.rs`.
#[path = "numeric_range_proof.rs"]
pub mod numeric_range_proof;

/// Integration tests from `tests/numeric_region_budget.rs`.
#[path = "numeric_region_budget.rs"]
pub mod numeric_region_budget;

/// Integration tests from `tests/numeric_scalar_format.rs`.
#[path = "numeric_scalar_format.rs"]
pub mod numeric_scalar_format;

/// Integration tests from `tests/opaque_payload_endian.rs`.
#[path = "opaque_payload_endian.rs"]
pub mod opaque_payload_endian;

/// Integration tests from `tests/opaque_wire_round_trip.rs`.
#[path = "opaque_wire_round_trip.rs"]
pub mod opaque_wire_round_trip;

/// Integration tests from `tests/operation_call_graph_closure.rs`.
#[path = "operation_call_graph_closure.rs"]
pub mod operation_call_graph_closure;

/// Integration tests from `tests/operation_namespace.rs`.
#[path = "operation_namespace.rs"]
pub mod operation_namespace;

/// Integration tests from `tests/operation_registry.rs`.
#[path = "operation_registry.rs"]
pub mod operation_registry;

/// Integration tests from `tests/optimizer_algebraic_rules_contracts.rs`.
#[path = "optimizer_algebraic_rules_contracts.rs"]
pub mod optimizer_algebraic_rules_contracts;

/// Integration tests from `tests/optimizer_dataflow_value_differential.rs`.
#[path = "optimizer_dataflow_value_differential.rs"]
pub mod optimizer_dataflow_value_differential;

/// Integration tests from `tests/optimizer_idempotence_proptest.rs`.
#[path = "optimizer_idempotence_proptest.rs"]
pub mod optimizer_idempotence_proptest;

/// Integration tests from `tests/optimizer_loop_value_differential.rs`.
#[path = "optimizer_loop_value_differential.rs"]
pub mod optimizer_loop_value_differential;

/// Integration tests from `tests/optimizer_pass_borrow_preservation.rs`.
#[path = "optimizer_pass_borrow_preservation.rs"]
pub mod optimizer_pass_borrow_preservation;

/// Integration tests from `tests/optimizer_perf_regression.rs`.
#[allow(clippy::match_like_matches_macro)]
#[path = "optimizer_perf_regression.rs"]
pub mod optimizer_perf_regression;

/// Integration tests from `tests/optimizer_proof_contracts.rs`.
#[path = "optimizer_proof_contracts.rs"]
pub mod optimizer_proof_contracts;

/// Integration tests from `tests/optimizer_reference_parity_smoke.rs`.
#[path = "optimizer_reference_parity_smoke.rs"]
pub mod optimizer_reference_parity_smoke;

/// Integration tests from `tests/optimizer_rewrite_proof_contracts.rs`.
#[path = "optimizer_rewrite_proof_contracts.rs"]
pub mod optimizer_rewrite_proof_contracts;

/// Integration tests from `tests/optimizer_rewrite_proof_registry_contracts.rs`.
#[path = "optimizer_rewrite_proof_registry_contracts.rs"]
pub mod optimizer_rewrite_proof_registry_contracts;

/// Integration tests from `tests/optimizer_value_dependent_reference_parity.rs`.
#[path = "optimizer_value_dependent_reference_parity.rs"]
pub mod optimizer_value_dependent_reference_parity;

/// Integration tests from `tests/output_set_roundtrip.rs`.
#[path = "output_set_roundtrip.rs"]
pub mod output_set_roundtrip;

/// Integration tests from `tests/parallelism_subgroup_operand_reads.rs`.
#[path = "parallelism_subgroup_operand_reads.rs"]
pub mod parallelism_subgroup_operand_reads;

/// Integration tests from `tests/pipeline_compiles_for_the_adapter_it_was_given.rs`.
#[path = "pipeline_compiles_for_the_adapter_it_was_given.rs"]
pub mod pipeline_compiles_for_the_adapter_it_was_given;

/// Integration tests from `tests/program_builder_invariants.rs`.
#[path = "program_builder_invariants.rs"]
pub mod program_builder_invariants;

/// Integration tests from `tests/program_canonical_commutative.rs`.
#[path = "program_canonical_commutative.rs"]
pub mod program_canonical_commutative;

/// Integration tests from `tests/program_graph_analysis_contract.rs`.
#[path = "program_graph_analysis_contract.rs"]
pub mod program_graph_analysis_contract;

/// Integration tests from `tests/program_graph_contract.rs`.
#[path = "program_graph_contract.rs"]
pub mod program_graph_contract;

/// Integration tests from `tests/program_graph_from_program.rs`.
#[path = "program_graph_from_program.rs"]
pub mod program_graph_from_program;

/// Integration tests from `tests/program_graph_identity_contract.rs`.
#[path = "program_graph_identity_contract.rs"]
pub mod program_graph_identity_contract;

/// Integration tests from `tests/program_graph_multi_domain_contracts.rs`.
#[path = "program_graph_multi_domain_contracts.rs"]
pub mod program_graph_multi_domain_contracts;

/// Integration tests from `tests/program_meta_surface.rs`.
#[path = "program_meta_surface.rs"]
pub mod program_meta_surface;

/// Integration tests from `tests/program_rebuild_preserves_metadata.rs`.
#[path = "program_rebuild_preserves_metadata.rs"]
pub mod program_rebuild_preserves_metadata;

/// Integration tests from `tests/program_soa_facts.rs`.
#[path = "program_soa_facts.rs"]
pub mod program_soa_facts;

/// Integration tests from `tests/program_stats_proptest.rs`.
#[allow(dead_code)]
#[path = "program_stats_proptest.rs"]
pub mod program_stats_proptest;

/// Integration tests from `tests/program_wire_property_contracts.rs`.
#[path = "program_wire_property_contracts.rs"]
pub mod program_wire_property_contracts;

/// Integration tests from `tests/quantized_contract.rs`.
#[path = "quantized_contract.rs"]
pub mod quantized_contract;

/// Integration tests from `tests/quantized_datatype_wire.rs`.
#[path = "quantized_datatype_wire.rs"]
pub mod quantized_datatype_wire;

/// Integration tests from `tests/read_only_load_hoist_scope.rs`.
#[path = "read_only_load_hoist_scope.rs"]
pub mod read_only_load_hoist_scope;

/// Integration tests from `tests/region_chain_adversarial.rs`.
#[path = "region_chain_adversarial.rs"]
pub mod region_chain_adversarial;

/// Integration tests from `tests/region_inline_invalidates.rs`.
#[path = "region_inline_invalidates.rs"]
pub mod region_inline_invalidates;

/// Integration tests from `tests/region_inline_scope.rs`.
#[path = "region_inline_scope.rs"]
pub mod region_inline_scope;

/// Integration tests from `tests/region_law_derivation.rs`.
#[path = "region_law_derivation.rs"]
pub mod region_law_derivation;

/// Integration tests from `tests/region_ssa_contracts.rs`.
#[path = "region_ssa_contracts.rs"]
pub mod region_ssa_contracts;

/// Integration tests from `tests/registry_closure.rs`.
#[path = "registry_closure.rs"]
pub mod registry_closure;

/// Integration tests from `tests/resource_exhaustion_adversarial.rs`.
#[path = "resource_exhaustion_adversarial.rs"]
pub mod resource_exhaustion_adversarial;

/// Integration tests from `tests/rewrite_contract_closure.rs`.
#[path = "rewrite_contract_closure.rs"]
pub mod rewrite_contract_closure;

/// Integration tests from `tests/rewrite_driver_descends_into_async_offset.rs`.
#[path = "rewrite_driver_descends_into_async_offset.rs"]
pub mod rewrite_driver_descends_into_async_offset;

/// Integration tests from `tests/scalar_operator_agreement.rs`.
#[path = "scalar_operator_agreement.rs"]
pub mod scalar_operator_agreement;

/// Integration tests from `tests/scan_database_wire_contract.rs`.
#[path = "scan_database_wire_contract.rs"]
pub mod scan_database_wire_contract;

/// Integration tests from `tests/schedule_calculus_contracts.rs`.
#[path = "schedule_calculus_contracts.rs"]
pub mod schedule_calculus_contracts;

/// Integration tests from `tests/schedule_ir_contracts.rs`.
#[path = "schedule_ir_contracts.rs"]
pub mod schedule_ir_contracts;

/// Integration tests from `tests/schedule_lowering_contracts.rs`.
#[path = "schedule_lowering_contracts.rs"]
pub mod schedule_lowering_contracts;

/// Integration tests from `tests/scope_cow.rs`.
#[path = "scope_cow.rs"]
pub mod scope_cow;

/// Integration tests from `tests/scope_rewrite_owner_contract.rs`.
#[path = "scope_rewrite_owner_contract.rs"]
pub mod scope_rewrite_owner_contract;

/// Integration tests from `tests/section_190_compiler_bounds_determinism_autodiff_concurrency.rs`.
#[path = "section_190_compiler_bounds_determinism_autodiff_concurrency.rs"]
pub mod section_190_compiler_bounds_determinism_autodiff_concurrency;

/// Integration tests from `tests/serial_envelope.rs`.
#[path = "serial_envelope.rs"]
pub mod serial_envelope;

/// Integration tests from `tests/serial_envelope_boundary.rs`.
#[path = "serial_envelope_boundary.rs"]
pub mod serial_envelope_boundary;

/// Integration tests from `tests/serial_envelope_corruption.rs`.
#[path = "serial_envelope_corruption.rs"]
pub mod serial_envelope_corruption;

/// Integration tests from `tests/shape_predicate_evaluation.rs`.
#[path = "shape_predicate_evaluation.rs"]
pub mod shape_predicate_evaluation;

/// Integration tests from `tests/source_tree_digest.rs`.
#[path = "source_tree_digest.rs"]
pub mod source_tree_digest;

/// Integration tests from `tests/store_to_load_forward_value_invalidation.rs`.
#[path = "store_to_load_forward_value_invalidation.rs"]
pub mod store_to_load_forward_value_invalidation;

/// Integration tests from `tests/structural_sharing_and_bounded_compilation_contracts.rs`.
#[path = "structural_sharing_and_bounded_compilation_contracts.rs"]
pub mod structural_sharing_and_bounded_compilation_contracts;

/// Integration tests from `tests/strength_reduce_shift_fusion_overflow.rs`.
#[path = "strength_reduce_shift_fusion_overflow.rs"]
pub mod strength_reduce_shift_fusion_overflow;

/// Integration tests from `tests/strict_expansion_convergence.rs`.
#[path = "strict_expansion_convergence.rs"]
pub mod strict_expansion_convergence;

/// Integration tests from `tests/strict_transcendental_expansion.rs`.
#[path = "strict_transcendental_expansion.rs"]
pub mod strict_transcendental_expansion;

/// Integration tests from `tests/subst_preserves_subgroup_reduce_op.rs`.
#[path = "subst_preserves_subgroup_reduce_op.rs"]
pub mod subst_preserves_subgroup_reduce_op;

/// Integration tests from `tests/sweep_validation_rejection_oracle_matrix.rs`.
#[path = "sweep_validation_rejection_oracle_matrix.rs"]
pub mod sweep_validation_rejection_oracle_matrix;

/// Integration tests from `tests/sweep_validation_rejection_volume_oracle_matrix.rs`.
#[path = "sweep_validation_rejection_volume_oracle_matrix.rs"]
pub mod sweep_validation_rejection_volume_oracle_matrix;

/// Integration tests from `tests/tail_duplication_scope.rs`.
#[path = "tail_duplication_scope.rs"]
pub mod tail_duplication_scope;

/// Integration tests from `tests/terminal_wire_round_trip.rs`.
#[path = "terminal_wire_round_trip.rs"]
pub mod terminal_wire_round_trip;

/// Integration tests from `tests/text_format_boundary.rs`.
#[path = "text_format_boundary.rs"]
pub mod text_format_boundary;

/// Integration tests from `tests/tile_nodes_contracts.rs`.
#[path = "tile_nodes_contracts.rs"]
pub mod tile_nodes_contracts;

/// Integration tests from `tests/transform_contract_classification.rs`.
#[path = "transform_contract_classification.rs"]
pub mod transform_contract_classification;

/// Integration tests from `tests/transform_rewrites_still_fire.rs`.
#[path = "transform_rewrites_still_fire.rs"]
pub mod transform_rewrites_still_fire;

/// Integration tests from `tests/type_boundary_adversarial.rs`.
#[path = "type_boundary_adversarial.rs"]
pub mod type_boundary_adversarial;

/// Integration tests from `tests/v055_uniform_exit.rs`.
#[path = "v055_uniform_exit.rs"]
pub mod v055_uniform_exit;

/// Integration tests from `tests/validation_contract_gaps.rs`.
#[path = "validation_contract_gaps.rs"]
pub mod validation_contract_gaps;

/// Integration tests from `tests/validation_depth_limits.rs`.
#[path = "validation_depth_limits.rs"]
pub mod validation_depth_limits;

/// Integration tests from `tests/validation_edge_cases.rs`.
#[path = "validation_edge_cases.rs"]
pub mod validation_edge_cases;

/// Integration tests from `tests/validation_output_markers.rs`.
#[path = "validation_output_markers.rs"]
pub mod validation_output_markers;

/// Integration tests from `tests/validation_rejection_contract.rs`.
#[path = "validation_rejection_contract.rs"]
pub mod validation_rejection_contract;

/// Integration tests from `tests/validator_error_docs.rs`.
#[path = "validator_error_docs.rs"]
pub mod validator_error_docs;

/// Integration tests from `tests/validator_limits_and_options.rs`.
#[path = "validator_limits_and_options.rs"]
pub mod validator_limits_and_options;

/// Integration tests from `tests/validator_uniformity.rs`.
#[path = "validator_uniformity.rs"]
pub mod validator_uniformity;

/// Integration tests from `tests/vast_invariants.rs`.
#[path = "vast_invariants.rs"]
pub mod vast_invariants;

/// Integration tests from `tests/vast_layout_overflow_contracts.rs`.
#[path = "vast_layout_overflow_contracts.rs"]
pub mod vast_layout_overflow_contracts;

/// Integration tests from `tests/vast_proptest.rs`.
#[path = "vast_proptest.rs"]
pub mod vast_proptest;

/// Integration tests from `tests/visitor_walk.rs`.
#[path = "visitor_walk.rs"]
pub mod visitor_walk;

/// Integration tests from `tests/wire_adversarial.rs`.
#[path = "wire_adversarial.rs"]
pub mod wire_adversarial;

/// Integration tests from `tests/wire_buffer_ref_round_trip.rs`.
#[path = "wire_buffer_ref_round_trip.rs"]
pub mod wire_buffer_ref_round_trip;

/// Integration tests from `tests/wire_decode_corruption.rs`.
#[path = "wire_decode_corruption.rs"]
pub mod wire_decode_corruption;

/// Integration tests from `tests/wire_decode_oom_guard.rs`.
#[path = "wire_decode_oom_guard.rs"]
pub mod wire_decode_oom_guard;

/// Integration tests from `tests/wire_format_corpus.rs`.
#[path = "wire_format_corpus.rs"]
pub mod wire_format_corpus;

/// Integration tests from `tests/wire_fuzz_infra_contracts.rs`.
#[path = "wire_fuzz_infra_contracts.rs"]
pub mod wire_fuzz_infra_contracts;

/// Integration tests from `tests/wire_generated_hostile_inputs.rs`.
#[path = "wire_generated_hostile_inputs.rs"]
pub mod wire_generated_hostile_inputs;

/// Integration tests from `tests/wire_roundtrip_exhaustive.rs`.
#[path = "wire_roundtrip_exhaustive.rs"]
pub mod wire_roundtrip_exhaustive;

/// Integration tests from `tests/wire_roundtrip_non_composable.rs`.
#[path = "wire_roundtrip_non_composable.rs"]
pub mod wire_roundtrip_non_composable;

/// Integration tests from `tests/wire_roundtrip_proptest.rs`.
#[allow(dead_code)]
#[path = "wire_roundtrip_proptest.rs"]
pub mod wire_roundtrip_proptest;

/// Integration tests from `tests/wire_version_mismatch.rs`.
#[path = "wire_version_mismatch.rs"]
pub mod wire_version_mismatch;

/// Integration tests from `tests/workspace_naming_footguns.rs`.
#[path = "workspace_naming_footguns.rs"]
pub mod workspace_naming_footguns;

/// Integration tests from `tests/causal_contracts.rs`.
#[path = "causal_contracts.rs"]
pub mod causal_contracts;

/// Integration tests from `tests/platform_support_matrix.rs`.
#[path = "platform_support_matrix.rs"]
pub mod platform_support_matrix;

/// Integration tests from `tests/security_contracts.rs`.
#[path = "security_contracts.rs"]
pub mod security_contracts;

/// Integration tests from `tests/declarative_schema_registry_contracts.rs`.
#[path = "declarative_schema_registry_contracts.rs"]
pub mod declarative_schema_registry_contracts;

/// Integration tests from `tests/typed_configuration_schema_contracts.rs`.
#[path = "typed_configuration_schema_contracts.rs"]
pub mod typed_configuration_schema_contracts;

/// Integration tests from `tests/resource_abi_contract.rs`.
#[path = "resource_abi_contract.rs"]
pub mod resource_abi_contract;

/// Integration tests from `tests/schema_authority_contract.rs`.
#[path = "schema_authority_contract.rs"]
pub mod schema_authority_contract;
