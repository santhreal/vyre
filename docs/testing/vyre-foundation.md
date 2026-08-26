# Testing `vyre-foundation`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-foundation
```

Own validated ProgramGraph, versioned schedule-free LogicalProgramGraph domains, versioned backend-neutral schedule IR and transform legality, semantic identity, neutral schedule-constraint composition, diagnostics, serialization, semantic operation registration, and backend-neutral optimization.

The crate lives at `vyre-foundation`. The `foundation-ir` owner maintains its
`foundation` testing contract.

## Commands

```console
./cargo_full test -p vyre-foundation
```

```console
./cargo_full test -p vyre-foundation --all-features
```

## Feature sets

- Default feature members: None
- Available manifest features: `default`, `serde`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bench` | `optimizer_pipeline` | `vyre-foundation/benches/optimizer_pipeline.rs` | None | `./cargo_full test -p vyre-foundation --bench optimizer_pipeline` |
| `example` | `vyre_foundation_release_surface` | `vyre-foundation/examples/vyre_foundation_release_surface.rs` | None | `./cargo_full test -p vyre-foundation --example vyre_foundation_release_surface` |
| `lib` | `vyre_foundation` | `vyre-foundation/src/lib.rs` | None | `./cargo_full test -p vyre-foundation` |
| `test` | `adversarial_graph_canonical_laws` | `vyre-foundation/tests/adversarial_graph_canonical_laws.rs` | None | `./cargo_full test -p vyre-foundation --test adversarial_graph_canonical_laws` |
| `test` | `adversarial_loop_induction_rebind` | `vyre-foundation/tests/adversarial_loop_induction_rebind.rs` | None | `./cargo_full test -p vyre-foundation --test adversarial_loop_induction_rebind` |
| `test` | `adversarial_loop_peel_first_iter` | `vyre-foundation/tests/adversarial_loop_peel_first_iter.rs` | None | `./cargo_full test -p vyre-foundation --test adversarial_loop_peel_first_iter` |
| `test` | `adversarial_program_canonical_laws` | `vyre-foundation/tests/adversarial_program_canonical_laws.rs` | None | `./cargo_full test -p vyre-foundation --test adversarial_program_canonical_laws` |
| `test` | `algebraic_reordering` | `vyre-foundation/tests/algebraic_reordering.rs` | None | `./cargo_full test -p vyre-foundation --test algebraic_reordering` |
| `test` | `async_destination_writability` | `vyre-foundation/tests/async_destination_writability.rs` | None | `./cargo_full test -p vyre-foundation --test async_destination_writability` |
| `test` | `async_tag_discipline` | `vyre-foundation/tests/async_tag_discipline.rs` | None | `./cargo_full test -p vyre-foundation --test async_tag_discipline` |
| `test` | `async_uniformity` | `vyre-foundation/tests/async_uniformity.rs` | None | `./cargo_full test -p vyre-foundation --test async_uniformity` |
| `test` | `atomic_linear_type_gap` | `vyre-foundation/tests/atomic_linear_type_gap.rs` | None | `./cargo_full test -p vyre-foundation --test atomic_linear_type_gap` |
| `test` | `atomic_minimize_operand_positions` | `vyre-foundation/tests/atomic_minimize_operand_positions.rs` | None | `./cargo_full test -p vyre-foundation --test atomic_minimize_operand_positions` |
| `test` | `autodiff_forward_local_fail_closed` | `vyre-foundation/tests/autodiff_forward_local_fail_closed.rs` | None | `./cargo_full test -p vyre-foundation --test autodiff_forward_local_fail_closed` |
| `test` | `autodiff_transform_contracts` | `vyre-foundation/tests/autodiff_transform_contracts.rs` | None | `./cargo_full test -p vyre-foundation --test autodiff_transform_contracts` |
| `test` | `bench_corpus_duplication` | `vyre-foundation/tests/bench_corpus_duplication.rs` | None | `./cargo_full test -p vyre-foundation --test bench_corpus_duplication` |
| `test` | `branch_value_hoist_scope` | `vyre-foundation/tests/branch_value_hoist_scope.rs` | None | `./cargo_full test -p vyre-foundation --test branch_value_hoist_scope` |
| `test` | `buffer_decl_boundary` | `vyre-foundation/tests/buffer_decl_boundary.rs` | None | `./cargo_full test -p vyre-foundation --test buffer_decl_boundary` |
| `test` | `buffer_decl_with_count` | `vyre-foundation/tests/buffer_decl_with_count.rs` | None | `./cargo_full test -p vyre-foundation --test buffer_decl_with_count` |
| `test` | `buffer_layout_proptest` | `vyre-foundation/tests/buffer_layout_proptest.rs` | None | `./cargo_full test -p vyre-foundation --test buffer_layout_proptest` |
| `test` | `canonical_determinism` | `vyre-foundation/tests/canonical_determinism.rs` | None | `./cargo_full test -p vyre-foundation --test canonical_determinism` |
| `test` | `canonicalization_corpus_contracts` | `vyre-foundation/tests/canonicalization_corpus_contracts.rs` | None | `./cargo_full test -p vyre-foundation --test canonicalization_corpus_contracts` |
| `test` | `capability_contracts` | `vyre-foundation/tests/capability_contracts.rs` | None | `./cargo_full test -p vyre-foundation --test capability_contracts` |
| `test` | `collective_ir_contracts` | `vyre-foundation/tests/collective_ir_contracts.rs` | None | `./cargo_full test -p vyre-foundation --test collective_ir_contracts` |
| `test` | `composition_tagging_contracts` | `vyre-foundation/tests/composition_tagging_contracts.rs` | None | `./cargo_full test -p vyre-foundation --test composition_tagging_contracts` |
| `test` | `const_fold_shift_fusion_amount_overflow` | `vyre-foundation/tests/const_fold_shift_fusion_amount_overflow.rs` | None | `./cargo_full test -p vyre-foundation --test const_fold_shift_fusion_amount_overflow` |
| `test` | `consumer_boundary` | `vyre-foundation/tests/consumer_boundary.rs` | None | `./cargo_full test -p vyre-foundation --test consumer_boundary` |
| `test` | `cse_effect_in_if_cond_invalidates_loads` | `vyre-foundation/tests/cse_effect_in_if_cond_invalidates_loads.rs` | None | `./cargo_full test -p vyre-foundation --test cse_effect_in_if_cond_invalidates_loads` |
| `test` | `dataflow_fixpoint_merge_contracts` | `vyre-foundation/tests/dataflow_fixpoint_merge_contracts.rs` | None | `./cargo_full test -p vyre-foundation --test dataflow_fixpoint_merge_contracts` |
| `test` | `dce_subgroup_operand_liveness` | `vyre-foundation/tests/dce_subgroup_operand_liveness.rs` | None | `./cargo_full test -p vyre-foundation --test dce_subgroup_operand_liveness` |
| `test` | `dead_buffer_dangling_ref` | `vyre-foundation/tests/dead_buffer_dangling_ref.rs` | None | `./cargo_full test -p vyre-foundation --test dead_buffer_dangling_ref` |
| `test` | `dead_store_elim_overwriter_reads` | `vyre-foundation/tests/dead_store_elim_overwriter_reads.rs` | None | `./cargo_full test -p vyre-foundation --test dead_store_elim_overwriter_reads` |
| `test` | `demos_orphan_risk` | `vyre-foundation/tests/demos_orphan_risk.rs` | None | `./cargo_full test -p vyre-foundation --test demos_orphan_risk` |
| `test` | `diagnostic_protocol` | `vyre-foundation/tests/diagnostic_protocol.rs` | None | `./cargo_full test -p vyre-foundation --test diagnostic_protocol` |
| `test` | `eqsat_gpu_mirror` | `vyre-foundation/tests/eqsat_gpu_mirror.rs` | None | `./cargo_full test -p vyre-foundation --test eqsat_gpu_mirror` |
| `test` | `execution_plan` | `vyre-foundation/tests/execution_plan.rs` | None | `./cargo_full test -p vyre-foundation --test execution_plan` |
| `test` | `expr_builder_surface` | `vyre-foundation/tests/expr_builder_surface.rs` | None | `./cargo_full test -p vyre-foundation --test expr_builder_surface` |
| `test` | `expr_type_single_owner` | `vyre-foundation/tests/expr_type_single_owner.rs` | None | `./cargo_full test -p vyre-foundation --test expr_type_single_owner` |
| `test` | `extension_adversarial` | `vyre-foundation/tests/extension_adversarial.rs` | None | `./cargo_full test -p vyre-foundation --test extension_adversarial` |
| `test` | `fingerprint_perf_contracts` | `vyre-foundation/tests/fingerprint_perf_contracts.rs` | None | `./cargo_full test -p vyre-foundation --test fingerprint_perf_contracts` |
| `test` | `foundation_validate_contract` | `vyre-foundation/tests/foundation_validate_contract.rs` | None | `./cargo_full test -p vyre-foundation --test foundation_validate_contract` |
| `test` | `fusion_atomic_aliasing` | `vyre-foundation/tests/fusion_atomic_aliasing.rs` | None | `./cargo_full test -p vyre-foundation --test fusion_atomic_aliasing` |
| `test` | `fusion_composability_metadata` | `vyre-foundation/tests/fusion_composability_metadata.rs` | None | `./cargo_full test -p vyre-foundation --test fusion_composability_metadata` |
| `test` | `fusion_stress` | `vyre-foundation/tests/fusion_stress.rs` | None | `./cargo_full test -p vyre-foundation --test fusion_stress` |
| `test` | `fusion_substitute_into_subgroup_operand` | `vyre-foundation/tests/fusion_substitute_into_subgroup_operand.rs` | None | `./cargo_full test -p vyre-foundation --test fusion_substitute_into_subgroup_operand` |
| `test` | `fusion_workgroup_geometry` | `vyre-foundation/tests/fusion_workgroup_geometry.rs` | None | `./cargo_full test -p vyre-foundation --test fusion_workgroup_geometry` |
| `test` | `geometry_foundation_contracts` | `vyre-foundation/tests/geometry_foundation_contracts.rs` | None | `./cargo_full test -p vyre-foundation --test geometry_foundation_contracts` |
| `test` | `graph_invariants` | `vyre-foundation/tests/graph_invariants.rs` | None | `./cargo_full test -p vyre-foundation --test graph_invariants` |
| `test` | `inline_buffer_reference_arguments` | `vyre-foundation/tests/inline_buffer_reference_arguments.rs` | None | `./cargo_full test -p vyre-foundation --test inline_buffer_reference_arguments` |
| `test` | `inline_callee_local_rename_in_trap_and_async` | `vyre-foundation/tests/inline_callee_local_rename_in_trap_and_async.rs` | None | `./cargo_full test -p vyre-foundation --test inline_callee_local_rename_in_trap_and_async` |
| `test` | `inline_expands_a_call_in_every_operand_slot` | `vyre-foundation/tests/inline_expands_a_call_in_every_operand_slot.rs` | None | `./cargo_full test -p vyre-foundation --test inline_expands_a_call_in_every_operand_slot` |
| `test` | `inline_nested_call_argument_rebinding` | `vyre-foundation/tests/inline_nested_call_argument_rebinding.rs` | None | `./cargo_full test -p vyre-foundation --test inline_nested_call_argument_rebinding` |
| `test` | `inline_per_invocation_builtin_rejection` | `vyre-foundation/tests/inline_per_invocation_builtin_rejection.rs` | None | `./cargo_full test -p vyre-foundation --test inline_per_invocation_builtin_rejection` |
| `test` | `ir_variant_shape_owner_closure` | `vyre-foundation/tests/ir_variant_shape_owner_closure.rs` | None | `./cargo_full test -p vyre-foundation --test ir_variant_shape_owner_closure` |
| `test` | `licm_hoist_scope_safety` | `vyre-foundation/tests/licm_hoist_scope_safety.rs` | None | `./cargo_full test -p vyre-foundation --test licm_hoist_scope_safety` |
| `test` | `licm_speculation_bounds` | `vyre-foundation/tests/licm_speculation_bounds.rs` | None | `./cargo_full test -p vyre-foundation --test licm_speculation_bounds` |
| `test` | `linear_type_validation` | `vyre-foundation/tests/linear_type_validation.rs` | None | `./cargo_full test -p vyre-foundation --test linear_type_validation` |
| `test` | `logical_span_contracts` | `vyre-foundation/tests/logical_span_contracts.rs` | None | `./cargo_full test -p vyre-foundation --test logical_span_contracts` |
| `test` | `logical_stage_identity` | `vyre-foundation/tests/logical_stage_identity.rs` | None | `./cargo_full test -p vyre-foundation --test logical_stage_identity` |
| `test` | `loop_fusion_atomic_expected_scalar_dependency` | `vyre-foundation/tests/loop_fusion_atomic_expected_scalar_dependency.rs` | None | `./cargo_full test -p vyre-foundation --test loop_fusion_atomic_expected_scalar_dependency` |
| `test` | `loop_fusion_binding_collision` | `vyre-foundation/tests/loop_fusion_binding_collision.rs` | None | `./cargo_full test -p vyre-foundation --test loop_fusion_binding_collision` |
| `test` | `loop_fusion_pair_after_a_refusal` | `vyre-foundation/tests/loop_fusion_pair_after_a_refusal.rs` | None | `./cargo_full test -p vyre-foundation --test loop_fusion_pair_after_a_refusal` |
| `test` | `loop_fusion_rename_reaches_every_operand` | `vyre-foundation/tests/loop_fusion_rename_reaches_every_operand.rs` | None | `./cargo_full test -p vyre-foundation --test loop_fusion_rename_reaches_every_operand` |
| `test` | `loop_fusion_scalar_dependency` | `vyre-foundation/tests/loop_fusion_scalar_dependency.rs` | None | `./cargo_full test -p vyre-foundation --test loop_fusion_scalar_dependency` |
| `test` | `loop_induction_var_guards` | `vyre-foundation/tests/loop_induction_var_guards.rs` | None | `./cargo_full test -p vyre-foundation --test loop_induction_var_guards` |
| `test` | `loop_legality_collector_closure` | `vyre-foundation/tests/loop_legality_collector_closure.rs` | None | `./cargo_full test -p vyre-foundation --test loop_legality_collector_closure` |
| `test` | `loop_licm_scope_extension` | `vyre-foundation/tests/loop_licm_scope_extension.rs` | None | `./cargo_full test -p vyre-foundation --test loop_licm_scope_extension` |
| `test` | `loop_software_pipeline_loopvar_in_value` | `vyre-foundation/tests/loop_software_pipeline_loopvar_in_value.rs` | None | `./cargo_full test -p vyre-foundation --test loop_software_pipeline_loopvar_in_value` |
| `test` | `loop_strip_mine_fresh_ident_scope` | `vyre-foundation/tests/loop_strip_mine_fresh_ident_scope.rs` | None | `./cargo_full test -p vyre-foundation --test loop_strip_mine_fresh_ident_scope` |
| `test` | `loop_unroll_local_scope` | `vyre-foundation/tests/loop_unroll_local_scope.rs` | None | `./cargo_full test -p vyre-foundation --test loop_unroll_local_scope` |
| `test` | `loop_var_range_fold_scope` | `vyre-foundation/tests/loop_var_range_fold_scope.rs` | None | `./cargo_full test -p vyre-foundation --test loop_var_range_fold_scope` |
| `test` | `memo_key_completeness` | `vyre-foundation/tests/memo_key_completeness/mod.rs` | None | `./cargo_full test -p vyre-foundation --test memo_key_completeness` |
| `test` | `memory_ordering` | `vyre-foundation/tests/memory_ordering.rs` | None | `./cargo_full test -p vyre-foundation --test memory_ordering` |
| `test` | `memory_ordering_adversarial` | `vyre-foundation/tests/memory_ordering_adversarial.rs` | None | `./cargo_full test -p vyre-foundation --test memory_ordering_adversarial` |
| `test` | `memory_pass_alias_owner` | `vyre-foundation/tests/memory_pass_alias_owner.rs` | None | `./cargo_full test -p vyre-foundation --test memory_pass_alias_owner` |
| `test` | `node_rewrite_walk_closure` | `vyre-foundation/tests/node_rewrite_walk_closure.rs` | None | `./cargo_full test -p vyre-foundation --test node_rewrite_walk_closure` |
| `test` | `node_variant_traversal_closure` | `vyre-foundation/tests/node_variant_traversal_closure.rs` | None | `./cargo_full test -p vyre-foundation --test node_variant_traversal_closure` |
| `test` | `opaque_payload_endian` | `vyre-foundation/tests/opaque_payload_endian.rs` | None | `./cargo_full test -p vyre-foundation --test opaque_payload_endian` |
| `test` | `opaque_wire_round_trip` | `vyre-foundation/tests/opaque_wire_round_trip.rs` | None | `./cargo_full test -p vyre-foundation --test opaque_wire_round_trip` |
| `test` | `operation_call_graph_closure` | `vyre-foundation/tests/operation_call_graph_closure.rs` | None | `./cargo_full test -p vyre-foundation --test operation_call_graph_closure` |
| `test` | `operation_namespace` | `vyre-foundation/tests/operation_namespace.rs` | None | `./cargo_full test -p vyre-foundation --test operation_namespace` |
| `test` | `operation_registry` | `vyre-foundation/tests/operation_registry.rs` | None | `./cargo_full test -p vyre-foundation --test operation_registry` |
| `test` | `optimizer_algebraic_rules_contracts` | `vyre-foundation/tests/optimizer_algebraic_rules_contracts.rs` | None | `./cargo_full test -p vyre-foundation --test optimizer_algebraic_rules_contracts` |
| `test` | `optimizer_dataflow_value_differential` | `vyre-foundation/tests/optimizer_dataflow_value_differential.rs` | None | `./cargo_full test -p vyre-foundation --test optimizer_dataflow_value_differential` |
| `test` | `optimizer_idempotence_proptest` | `vyre-foundation/tests/optimizer_idempotence_proptest.rs` | None | `./cargo_full test -p vyre-foundation --test optimizer_idempotence_proptest` |
| `test` | `optimizer_loop_value_differential` | `vyre-foundation/tests/optimizer_loop_value_differential.rs` | None | `./cargo_full test -p vyre-foundation --test optimizer_loop_value_differential` |
| `test` | `optimizer_pass_borrow_preservation` | `vyre-foundation/tests/optimizer_pass_borrow_preservation.rs` | None | `./cargo_full test -p vyre-foundation --test optimizer_pass_borrow_preservation` |
| `test` | `optimizer_perf_regression` | `vyre-foundation/tests/optimizer_perf_regression.rs` | None | `./cargo_full test -p vyre-foundation --test optimizer_perf_regression` |
| `test` | `optimizer_reference_parity_smoke` | `vyre-foundation/tests/optimizer_reference_parity_smoke.rs` | None | `./cargo_full test -p vyre-foundation --test optimizer_reference_parity_smoke` |
| `test` | `optimizer_rewrite_proof_contracts` | `vyre-foundation/tests/optimizer_rewrite_proof_contracts.rs` | None | `./cargo_full test -p vyre-foundation --test optimizer_rewrite_proof_contracts` |
| `test` | `optimizer_rewrite_proof_registry_contracts` | `vyre-foundation/tests/optimizer_rewrite_proof_registry_contracts.rs` | None | `./cargo_full test -p vyre-foundation --test optimizer_rewrite_proof_registry_contracts` |
| `test` | `optimizer_value_dependent_reference_parity` | `vyre-foundation/tests/optimizer_value_dependent_reference_parity.rs` | None | `./cargo_full test -p vyre-foundation --test optimizer_value_dependent_reference_parity` |
| `test` | `output_set_roundtrip` | `vyre-foundation/tests/output_set_roundtrip.rs` | None | `./cargo_full test -p vyre-foundation --test output_set_roundtrip` |
| `test` | `parallelism_subgroup_operand_reads` | `vyre-foundation/tests/parallelism_subgroup_operand_reads.rs` | None | `./cargo_full test -p vyre-foundation --test parallelism_subgroup_operand_reads` |
| `test` | `pipeline_compiles_for_the_adapter_it_was_given` | `vyre-foundation/tests/pipeline_compiles_for_the_adapter_it_was_given.rs` | None | `./cargo_full test -p vyre-foundation --test pipeline_compiles_for_the_adapter_it_was_given` |
| `test` | `program_builder_invariants` | `vyre-foundation/tests/program_builder_invariants.rs` | None | `./cargo_full test -p vyre-foundation --test program_builder_invariants` |
| `test` | `program_canonical_commutative` | `vyre-foundation/tests/program_canonical_commutative.rs` | None | `./cargo_full test -p vyre-foundation --test program_canonical_commutative` |
| `test` | `program_graph_analysis_contract` | `vyre-foundation/tests/program_graph_analysis_contract.rs` | None | `./cargo_full test -p vyre-foundation --test program_graph_analysis_contract` |
| `test` | `program_graph_contract` | `vyre-foundation/tests/program_graph_contract.rs` | None | `./cargo_full test -p vyre-foundation --test program_graph_contract` |
| `test` | `program_graph_from_program` | `vyre-foundation/tests/program_graph_from_program.rs` | None | `./cargo_full test -p vyre-foundation --test program_graph_from_program` |
| `test` | `program_graph_identity_contract` | `vyre-foundation/tests/program_graph_identity_contract.rs` | None | `./cargo_full test -p vyre-foundation --test program_graph_identity_contract` |
| `test` | `program_meta_surface` | `vyre-foundation/tests/program_meta_surface.rs` | None | `./cargo_full test -p vyre-foundation --test program_meta_surface` |
| `test` | `program_rebuild_preserves_metadata` | `vyre-foundation/tests/program_rebuild_preserves_metadata.rs` | None | `./cargo_full test -p vyre-foundation --test program_rebuild_preserves_metadata` |
| `test` | `program_soa_facts` | `vyre-foundation/tests/program_soa_facts.rs` | None | `./cargo_full test -p vyre-foundation --test program_soa_facts` |
| `test` | `program_stats_proptest` | `vyre-foundation/tests/program_stats_proptest.rs` | None | `./cargo_full test -p vyre-foundation --test program_stats_proptest` |
| `test` | `program_wire_property_contracts` | `vyre-foundation/tests/program_wire_property_contracts.rs` | None | `./cargo_full test -p vyre-foundation --test program_wire_property_contracts` |
| `test` | `quantized_datatype_wire` | `vyre-foundation/tests/quantized_datatype_wire.rs` | None | `./cargo_full test -p vyre-foundation --test quantized_datatype_wire` |
| `test` | `read_only_load_hoist_scope` | `vyre-foundation/tests/read_only_load_hoist_scope.rs` | None | `./cargo_full test -p vyre-foundation --test read_only_load_hoist_scope` |
| `test` | `region_chain_adversarial` | `vyre-foundation/tests/region_chain_adversarial.rs` | None | `./cargo_full test -p vyre-foundation --test region_chain_adversarial` |
| `test` | `region_inline_invalidates` | `vyre-foundation/tests/region_inline_invalidates.rs` | None | `./cargo_full test -p vyre-foundation --test region_inline_invalidates` |
| `test` | `region_inline_scope` | `vyre-foundation/tests/region_inline_scope.rs` | None | `./cargo_full test -p vyre-foundation --test region_inline_scope` |
| `test` | `registry_closure` | `vyre-foundation/tests/registry_closure.rs` | None | `./cargo_full test -p vyre-foundation --test registry_closure` |
| `test` | `resource_exhaustion_adversarial` | `vyre-foundation/tests/resource_exhaustion_adversarial.rs` | None | `./cargo_full test -p vyre-foundation --test resource_exhaustion_adversarial` |
| `test` | `rewrite_driver_descends_into_async_offset` | `vyre-foundation/tests/rewrite_driver_descends_into_async_offset.rs` | None | `./cargo_full test -p vyre-foundation --test rewrite_driver_descends_into_async_offset` |
| `test` | `scalar_operator_agreement` | `vyre-foundation/tests/scalar_operator_agreement.rs` | None | `./cargo_full test -p vyre-foundation --test scalar_operator_agreement` |
| `test` | `scan_database_wire_contract` | `vyre-foundation/tests/scan_database_wire_contract.rs` | None | `./cargo_full test -p vyre-foundation --test scan_database_wire_contract` |
| `test` | `schedule_ir_contracts` | `vyre-foundation/tests/schedule_ir_contracts.rs` | None | `./cargo_full test -p vyre-foundation --test schedule_ir_contracts` |
| `test` | `schedule_lowering_contracts` | `vyre-foundation/tests/schedule_lowering_contracts.rs` | None | `./cargo_full test -p vyre-foundation --test schedule_lowering_contracts` |
| `test` | `scope_cow` | `vyre-foundation/tests/scope_cow.rs` | None | `./cargo_full test -p vyre-foundation --test scope_cow` |
| `test` | `scope_rewrite_owner_contract` | `vyre-foundation/tests/scope_rewrite_owner_contract.rs` | None | `./cargo_full test -p vyre-foundation --test scope_rewrite_owner_contract` |
| `test` | `section_190_compiler_bounds_determinism_autodiff_concurrency` | `vyre-foundation/tests/section_190_compiler_bounds_determinism_autodiff_concurrency.rs` | None | `./cargo_full test -p vyre-foundation --test section_190_compiler_bounds_determinism_autodiff_concurrency` |
| `test` | `serial_envelope` | `vyre-foundation/tests/serial_envelope.rs` | None | `./cargo_full test -p vyre-foundation --test serial_envelope` |
| `test` | `serial_envelope_boundary` | `vyre-foundation/tests/serial_envelope_boundary.rs` | None | `./cargo_full test -p vyre-foundation --test serial_envelope_boundary` |
| `test` | `serial_envelope_corruption` | `vyre-foundation/tests/serial_envelope_corruption.rs` | None | `./cargo_full test -p vyre-foundation --test serial_envelope_corruption` |
| `test` | `shape_predicate_evaluation` | `vyre-foundation/tests/shape_predicate_evaluation.rs` | None | `./cargo_full test -p vyre-foundation --test shape_predicate_evaluation` |
| `test` | `source_tree_digest` | `vyre-foundation/tests/source_tree_digest.rs` | None | `./cargo_full test -p vyre-foundation --test source_tree_digest` |
| `test` | `store_to_load_forward_value_invalidation` | `vyre-foundation/tests/store_to_load_forward_value_invalidation.rs` | None | `./cargo_full test -p vyre-foundation --test store_to_load_forward_value_invalidation` |
| `test` | `strength_reduce_shift_fusion_overflow` | `vyre-foundation/tests/strength_reduce_shift_fusion_overflow.rs` | None | `./cargo_full test -p vyre-foundation --test strength_reduce_shift_fusion_overflow` |
| `test` | `subst_preserves_subgroup_reduce_op` | `vyre-foundation/tests/subst_preserves_subgroup_reduce_op.rs` | None | `./cargo_full test -p vyre-foundation --test subst_preserves_subgroup_reduce_op` |
| `test` | `sweep_validation_rejection_oracle_matrix` | `vyre-foundation/tests/sweep_validation_rejection_oracle_matrix.rs` | None | `./cargo_full test -p vyre-foundation --test sweep_validation_rejection_oracle_matrix` |
| `test` | `sweep_validation_rejection_volume_oracle_matrix` | `vyre-foundation/tests/sweep_validation_rejection_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-foundation --test sweep_validation_rejection_volume_oracle_matrix` |
| `test` | `tail_duplication_scope` | `vyre-foundation/tests/tail_duplication_scope.rs` | None | `./cargo_full test -p vyre-foundation --test tail_duplication_scope` |
| `test` | `terminal_wire_round_trip` | `vyre-foundation/tests/terminal_wire_round_trip.rs` | None | `./cargo_full test -p vyre-foundation --test terminal_wire_round_trip` |
| `test` | `text_format_boundary` | `vyre-foundation/tests/text_format_boundary.rs` | None | `./cargo_full test -p vyre-foundation --test text_format_boundary` |
| `test` | `tile_nodes_contracts` | `vyre-foundation/tests/tile_nodes_contracts.rs` | None | `./cargo_full test -p vyre-foundation --test tile_nodes_contracts` |
| `test` | `transform_contract_classification` | `vyre-foundation/tests/transform_contract_classification.rs` | None | `./cargo_full test -p vyre-foundation --test transform_contract_classification` |
| `test` | `transform_rewrites_still_fire` | `vyre-foundation/tests/transform_rewrites_still_fire.rs` | None | `./cargo_full test -p vyre-foundation --test transform_rewrites_still_fire` |
| `test` | `type_boundary_adversarial` | `vyre-foundation/tests/type_boundary_adversarial.rs` | None | `./cargo_full test -p vyre-foundation --test type_boundary_adversarial` |
| `test` | `v055_uniform_exit` | `vyre-foundation/tests/v055_uniform_exit.rs` | None | `./cargo_full test -p vyre-foundation --test v055_uniform_exit` |
| `test` | `validation_contract_gaps` | `vyre-foundation/tests/validation_contract_gaps.rs` | None | `./cargo_full test -p vyre-foundation --test validation_contract_gaps` |
| `test` | `validation_depth_limits` | `vyre-foundation/tests/validation_depth_limits.rs` | None | `./cargo_full test -p vyre-foundation --test validation_depth_limits` |
| `test` | `validation_edge_cases` | `vyre-foundation/tests/validation_edge_cases.rs` | None | `./cargo_full test -p vyre-foundation --test validation_edge_cases` |
| `test` | `validation_output_markers` | `vyre-foundation/tests/validation_output_markers.rs` | None | `./cargo_full test -p vyre-foundation --test validation_output_markers` |
| `test` | `validation_rejection_contract` | `vyre-foundation/tests/validation_rejection_contract.rs` | None | `./cargo_full test -p vyre-foundation --test validation_rejection_contract` |
| `test` | `validator_error_docs` | `vyre-foundation/tests/validator_error_docs.rs` | None | `./cargo_full test -p vyre-foundation --test validator_error_docs` |
| `test` | `validator_limits_and_options` | `vyre-foundation/tests/validator_limits_and_options.rs` | None | `./cargo_full test -p vyre-foundation --test validator_limits_and_options` |
| `test` | `validator_uniformity` | `vyre-foundation/tests/validator_uniformity.rs` | None | `./cargo_full test -p vyre-foundation --test validator_uniformity` |
| `test` | `vast_invariants` | `vyre-foundation/tests/vast_invariants.rs` | None | `./cargo_full test -p vyre-foundation --test vast_invariants` |
| `test` | `vast_layout_overflow_contracts` | `vyre-foundation/tests/vast_layout_overflow_contracts.rs` | None | `./cargo_full test -p vyre-foundation --test vast_layout_overflow_contracts` |
| `test` | `vast_proptest` | `vyre-foundation/tests/vast_proptest.rs` | None | `./cargo_full test -p vyre-foundation --test vast_proptest` |
| `test` | `visitor_walk` | `vyre-foundation/tests/visitor_walk.rs` | None | `./cargo_full test -p vyre-foundation --test visitor_walk` |
| `test` | `wire_adversarial` | `vyre-foundation/tests/wire_adversarial.rs` | None | `./cargo_full test -p vyre-foundation --test wire_adversarial` |
| `test` | `wire_buffer_ref_round_trip` | `vyre-foundation/tests/wire_buffer_ref_round_trip.rs` | None | `./cargo_full test -p vyre-foundation --test wire_buffer_ref_round_trip` |
| `test` | `wire_decode_corruption` | `vyre-foundation/tests/wire_decode_corruption.rs` | None | `./cargo_full test -p vyre-foundation --test wire_decode_corruption` |
| `test` | `wire_decode_hostile_inputs` | `vyre-foundation/tests/wire_decode_hostile_inputs.rs` | None | `./cargo_full test -p vyre-foundation --test wire_decode_hostile_inputs` |
| `test` | `wire_decode_oom_guard` | `vyre-foundation/tests/wire_decode_oom_guard.rs` | None | `./cargo_full test -p vyre-foundation --test wire_decode_oom_guard` |
| `test` | `wire_format_corpus` | `vyre-foundation/tests/wire_format_corpus.rs` | None | `./cargo_full test -p vyre-foundation --test wire_format_corpus` |
| `test` | `wire_fuzz_infra_contracts` | `vyre-foundation/tests/wire_fuzz_infra_contracts.rs` | None | `./cargo_full test -p vyre-foundation --test wire_fuzz_infra_contracts` |
| `test` | `wire_generated_hostile_inputs` | `vyre-foundation/tests/wire_generated_hostile_inputs.rs` | None | `./cargo_full test -p vyre-foundation --test wire_generated_hostile_inputs` |
| `test` | `wire_roundtrip_exhaustive` | `vyre-foundation/tests/wire_roundtrip_exhaustive.rs` | None | `./cargo_full test -p vyre-foundation --test wire_roundtrip_exhaustive` |
| `test` | `wire_roundtrip_non_composable` | `vyre-foundation/tests/wire_roundtrip_non_composable.rs` | None | `./cargo_full test -p vyre-foundation --test wire_roundtrip_non_composable` |
| `test` | `wire_roundtrip_proptest` | `vyre-foundation/tests/wire_roundtrip_proptest.rs` | None | `./cargo_full test -p vyre-foundation --test wire_roundtrip_proptest` |
| `test` | `wire_version_mismatch` | `vyre-foundation/tests/wire_version_mismatch.rs` | None | `./cargo_full test -p vyre-foundation --test wire_version_mismatch` |
| `test` | `workspace_naming_footguns` | `vyre-foundation/tests/workspace_naming_footguns.rs` | None | `./cargo_full test -p vyre-foundation --test workspace_naming_footguns` |

## Test classes

- IR construction and serialization contracts
- Validation and optimizer semantics
- Adversarial, property, and compatibility tests

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
