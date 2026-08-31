# Testing `vyre-reference`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-reference
```

The only crate permitted to compute on the CPU: the pure-Rust IR oracle. Not a backend and not a fallback.

The crate lives at `vyre-reference`. The `reference-semantics` owner maintains its
`semantics` testing contract.

## Commands

```console
./cargo_full test -p vyre-reference
```

```console
./cargo_full test -p vyre-reference --all-features
```

## Feature sets

- Default feature members: `subgroup-ops`
- Available manifest features: `default`, `subgroup-ops`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `vyre_reference_release_surface` | `vyre-reference/examples/vyre_reference_release_surface.rs` | None | `./cargo_full test -p vyre-reference --example vyre_reference_release_surface` |
| `lib` | `vyre_reference` | `vyre-reference/src/lib.rs` | None | `./cargo_full test -p vyre-reference` |
| `test` | `adversarial_empty` | `vyre-reference/tests/adversarial_empty.rs` | None | `./cargo_full test -p vyre-reference --test adversarial_empty` |
| `test` | `adversarial_gaps` | `vyre-reference/tests/adversarial_gaps.rs` | None | `./cargo_full test -p vyre-reference --test adversarial_gaps` |
| `test` | `assign_semantics` | `vyre-reference/tests/assign_semantics.rs` | None | `./cargo_full test -p vyre-reference --test assign_semantics` |
| `test` | `atomic_law_property_contracts` | `vyre-reference/tests/atomic_law_property_contracts.rs` | None | `./cargo_full test -p vyre-reference --test atomic_law_property_contracts` |
| `test` | `atomic_oracle_contract` | `vyre-reference/tests/atomic_oracle_contract.rs` | None | `./cargo_full test -p vyre-reference --test atomic_oracle_contract` |
| `test` | `atomic_property_contracts` | `vyre-reference/tests/atomic_property_contracts.rs` | None | `./cargo_full test -p vyre-reference --test atomic_property_contracts` |
| `test` | `byte_prefix_property_contracts` | `vyre-reference/tests/byte_prefix_property_contracts.rs` | None | `./cargo_full test -p vyre-reference --test byte_prefix_property_contracts` |
| `test` | `composition_witness_contracts` | `vyre-reference/tests/composition_witness_contracts.rs` | None | `./cargo_full test -p vyre-reference --test composition_witness_contracts` |
| `test` | `composition_witness_geometry_contracts` | `vyre-reference/tests/composition_witness_geometry_contracts.rs` | None | `./cargo_full test -p vyre-reference --test composition_witness_geometry_contracts` |
| `test` | `composition_witness_parsing_contracts` | `vyre-reference/tests/composition_witness_parsing_contracts.rs` | None | `./cargo_full test -p vyre-reference --test composition_witness_parsing_contracts` |
| `test` | `composition_witness_reasoning_contracts` | `vyre-reference/tests/composition_witness_reasoning_contracts.rs` | None | `./cargo_full test -p vyre-reference --test composition_witness_reasoning_contracts` |
| `test` | `composition_witness_scheduling_contracts` | `vyre-reference/tests/composition_witness_scheduling_contracts.rs` | None | `./cargo_full test -p vyre-reference --test composition_witness_scheduling_contracts` |
| `test` | `core_contracts` | `vyre-reference/tests/core_contracts/mod.rs` | None | `./cargo_full test -p vyre-reference --test core_contracts` |
| `test` | `dual_arith_reference_contracts` | `vyre-reference/tests/dual_arith_reference_contracts.rs` | None | `./cargo_full test -p vyre-reference --test dual_arith_reference_contracts` |
| `test` | `dual_reference_parity` | `vyre-reference/tests/dual_reference_parity.rs` | None | `./cargo_full test -p vyre-reference --test dual_reference_parity` |
| `test` | `dual_reference_property_contracts` | `vyre-reference/tests/dual_reference_property_contracts.rs` | None | `./cargo_full test -p vyre-reference --test dual_reference_property_contracts` |
| `test` | `dual_registry_adversarial_contract` | `vyre-reference/tests/dual_registry_adversarial_contract.rs` | None | `./cargo_full test -p vyre-reference --test dual_registry_adversarial_contract` |
| `test` | `dual_scalar_evaluator_matrix` | `vyre-reference/tests/dual_scalar_evaluator_matrix.rs` | None | `./cargo_full test -p vyre-reference --test dual_scalar_evaluator_matrix` |
| `test` | `expr_adversarial_proptest` | `vyre-reference/tests/expr_adversarial_proptest.rs` | None | `./cargo_full test -p vyre-reference --test expr_adversarial_proptest` |
| `test` | `f32_comparison_property_contracts` | `vyre-reference/tests/f32_comparison_property_contracts.rs` | None | `./cargo_full test -p vyre-reference --test f32_comparison_property_contracts` |
| `test` | `fixed_width_value_property_contracts` | `vyre-reference/tests/fixed_width_value_property_contracts.rs` | None | `./cargo_full test -p vyre-reference --test fixed_width_value_property_contracts` |
| `test` | `flat_cpu_input_contract` | `vyre-reference/tests/flat_cpu_input_contract.rs` | None | `./cargo_full test -p vyre-reference --test flat_cpu_input_contract` |
| `test` | `gap_transcendentals_parity` | `vyre-reference/tests/gap_transcendentals_parity.rs` | None | `./cargo_full test -p vyre-reference --test gap_transcendentals_parity` |
| `test` | `hashmap_async_and_indirect_contracts` | `vyre-reference/tests/hashmap_async_and_indirect_contracts.rs` | None | `./cargo_full test -p vyre-reference --test hashmap_async_and_indirect_contracts` |
| `test` | `hashmap_buffer_size_contracts` | `vyre-reference/tests/hashmap_buffer_size_contracts.rs` | None | `./cargo_full test -p vyre-reference --test hashmap_buffer_size_contracts` |
| `test` | `hashmap_invocation_size_contracts` | `vyre-reference/tests/hashmap_invocation_size_contracts.rs` | None | `./cargo_full test -p vyre-reference --test hashmap_invocation_size_contracts` |
| `test` | `logical_execution_markers` | `vyre-reference/tests/logical_execution_markers.rs` | None | `./cargo_full test -p vyre-reference --test logical_execution_markers` |
| `test` | `oracle_program_edges` | `vyre-reference/tests/oracle_program_edges.rs` | None | `./cargo_full test -p vyre-reference --test oracle_program_edges` |
| `test` | `quantized_buffer_contract` | `vyre-reference/tests/quantized_buffer_contract.rs` | None | `./cargo_full test -p vyre-reference --test quantized_buffer_contract` |
| `test` | `reference_abi_predicates` | `vyre-reference/tests/reference_abi_predicates.rs` | None | `./cargo_full test -p vyre-reference --test reference_abi_predicates` |
| `test` | `reference_error_contract` | `vyre-reference/tests/reference_error_contract.rs` | None | `./cargo_full test -p vyre-reference --test reference_error_contract` |
| `test` | `reference_eval_fma_select_generated` | `vyre-reference/tests/reference_eval_fma_select_generated.rs` | None | `./cargo_full test -p vyre-reference --test reference_eval_fma_select_generated` |
| `test` | `reference_output_byte_stability` | `vyre-reference/tests/reference_output_byte_stability.rs` | None | `./cargo_full test -p vyre-reference --test reference_output_byte_stability` |
| `test` | `region_frame_lifetime` | `vyre-reference/tests/region_frame_lifetime.rs` | None | `./cargo_full test -p vyre-reference --test region_frame_lifetime` |
| `test` | `region_gate` | `vyre-reference/tests/region_gate.rs` | None | `./cargo_full test -p vyre-reference --test region_gate` |
| `test` | `saturating_binops_contract` | `vyre-reference/tests/saturating_binops_contract.rs` | None | `./cargo_full test -p vyre-reference --test saturating_binops_contract` |
| `test` | `single_rank_collective_reference` | `vyre-reference/tests/single_rank_collective_reference.rs` | None | `./cargo_full test -p vyre-reference --test single_rank_collective_reference` |
| `test` | `storage_graph_generated_adversarial` | `vyre-reference/tests/storage_graph_generated_adversarial.rs` | None | `./cargo_full test -p vyre-reference --test storage_graph_generated_adversarial` |
| `test` | `storage_graph_scalar_matrix` | `vyre-reference/tests/storage_graph_scalar_matrix.rs` | None | `./cargo_full test -p vyre-reference --test storage_graph_scalar_matrix` |
| `test` | `subgroup_collectives_are_lane_identified` | `vyre-reference/tests/subgroup_collectives_are_lane_identified.rs` | None | `./cargo_full test -p vyre-reference --test subgroup_collectives_are_lane_identified` |
| `test` | `subgroup_edge_contract` | `vyre-reference/tests/subgroup_edge_contract.rs` | None | `./cargo_full test -p vyre-reference --test subgroup_edge_contract` |
| `test` | `subnormal_contract` | `vyre-reference/tests/subnormal_contract.rs` | None | `./cargo_full test -p vyre-reference --test subnormal_contract` |
| `test` | `sweep_dual_arith_oracle_matrix` | `vyre-reference/tests/sweep_dual_arith_oracle_matrix.rs` | None | `./cargo_full test -p vyre-reference --test sweep_dual_arith_oracle_matrix` |
| `test` | `sweep_dual_bitwise_and_volume_oracle_matrix` | `vyre-reference/tests/sweep_dual_bitwise_and_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-reference --test sweep_dual_bitwise_and_volume_oracle_matrix` |
| `test` | `sweep_dual_bitwise_clz_volume_oracle_matrix` | `vyre-reference/tests/sweep_dual_bitwise_clz_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-reference --test sweep_dual_bitwise_clz_volume_oracle_matrix` |
| `test` | `sweep_dual_bitwise_not_volume_oracle_matrix` | `vyre-reference/tests/sweep_dual_bitwise_not_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-reference --test sweep_dual_bitwise_not_volume_oracle_matrix` |
| `test` | `sweep_dual_bitwise_or_volume_oracle_matrix` | `vyre-reference/tests/sweep_dual_bitwise_or_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-reference --test sweep_dual_bitwise_or_volume_oracle_matrix` |
| `test` | `sweep_dual_bitwise_popcount_volume_oracle_matrix` | `vyre-reference/tests/sweep_dual_bitwise_popcount_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-reference --test sweep_dual_bitwise_popcount_volume_oracle_matrix` |
| `test` | `sweep_dual_bitwise_shift_left_volume_oracle_matrix` | `vyre-reference/tests/sweep_dual_bitwise_shift_left_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-reference --test sweep_dual_bitwise_shift_left_volume_oracle_matrix` |
| `test` | `sweep_dual_bitwise_shift_right_volume_oracle_matrix` | `vyre-reference/tests/sweep_dual_bitwise_shift_right_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-reference --test sweep_dual_bitwise_shift_right_volume_oracle_matrix` |
| `test` | `sweep_dual_bitwise_xor_volume_oracle_matrix` | `vyre-reference/tests/sweep_dual_bitwise_xor_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-reference --test sweep_dual_bitwise_xor_volume_oracle_matrix` |
| `test` | `sweep_dual_compare_eq_volume_oracle_matrix` | `vyre-reference/tests/sweep_dual_compare_eq_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-reference --test sweep_dual_compare_eq_volume_oracle_matrix` |
| `test` | `sweep_dual_compare_lt_volume_oracle_matrix` | `vyre-reference/tests/sweep_dual_compare_lt_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-reference --test sweep_dual_compare_lt_volume_oracle_matrix` |
| `test` | `test_fnv1a32_zero` | `vyre-reference/tests/fnv1a32_zero.rs` | None | `./cargo_full test -p vyre-reference --test test_fnv1a32_zero` |
| `test` | `tile_reference_contracts` | `vyre-reference/tests/tile_reference_contracts.rs` | None | `./cargo_full test -p vyre-reference --test tile_reference_contracts` |
| `test` | `typed_validation_source` | `vyre-reference/tests/typed_validation_source.rs` | None | `./cargo_full test -p vyre-reference --test typed_validation_source` |
| `test` | `value_array_property_contracts` | `vyre-reference/tests/value_array_property_contracts.rs` | None | `./cargo_full test -p vyre-reference --test value_array_property_contracts` |
| `test` | `value_byte_property_contracts` | `vyre-reference/tests/value_byte_property_contracts.rs` | None | `./cargo_full test -p vyre-reference --test value_byte_property_contracts` |
| `test` | `value_datatype_generated_matrix` | `vyre-reference/tests/value_datatype_generated_matrix.rs` | None | `./cargo_full test -p vyre-reference --test value_datatype_generated_matrix` |
| `test` | `value_encoding_contract` | `vyre-reference/tests/value_encoding_contract.rs` | None | `./cargo_full test -p vyre-reference --test value_encoding_contract` |
| `test` | `value_extend_bytes_width_generated` | `vyre-reference/tests/value_extend_bytes_width_generated.rs` | None | `./cargo_full test -p vyre-reference --test value_extend_bytes_width_generated` |
| `test` | `value_float_property_contracts` | `vyre-reference/tests/value_float_property_contracts.rs` | None | `./cargo_full test -p vyre-reference --test value_float_property_contracts` |
| `test` | `value_narrowing_property_contracts` | `vyre-reference/tests/value_narrowing_property_contracts.rs` | None | `./cargo_full test -p vyre-reference --test value_narrowing_property_contracts` |
| `test` | `value_signed_narrowing_property_contracts` | `vyre-reference/tests/value_signed_narrowing_property_contracts.rs` | None | `./cargo_full test -p vyre-reference --test value_signed_narrowing_property_contracts` |
| `test` | `value_truthiness_property_contracts` | `vyre-reference/tests/value_truthiness_property_contracts.rs` | None | `./cargo_full test -p vyre-reference --test value_truthiness_property_contracts` |
| `test` | `value_write_bytes_width_generated` | `vyre-reference/tests/value_write_bytes_width_generated.rs` | None | `./cargo_full test -p vyre-reference --test value_write_bytes_width_generated` |
| `test` | `vector_cast_generated_matrix` | `vyre-reference/tests/vector_cast_generated_matrix.rs` | None | `./cargo_full test -p vyre-reference --test vector_cast_generated_matrix` |

## Test classes

- Reference execution semantics
- Exact oracle and witness contracts
- Adversarial and property parity tests

## Hardware requirements

No accelerator is required for the default suite.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
