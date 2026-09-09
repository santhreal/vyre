//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/dual_volume/mod.rs`.
#[allow(dead_code, missing_docs)]
#[path = "dual_volume/mod.rs"]
pub mod dual_volume;

/// Shared fixture module from `tests/flat_expr_eval/mod.rs`.
#[allow(dead_code, missing_docs)]
#[path = "flat_expr_eval/mod.rs"]
pub mod flat_expr_eval;

/// Shared fixture module from `tests/wire_words/mod.rs`.
#[allow(dead_code, missing_docs)]
#[path = "wire_words/mod.rs"]
pub mod wire_words;

/// Integration tests from `tests/adversarial_empty.rs`.
#[path = "adversarial_empty.rs"]
pub mod adversarial_empty;

/// Integration tests from `tests/adversarial_gaps.rs`.
#[path = "adversarial_gaps.rs"]
pub mod adversarial_gaps;

/// Integration tests from `tests/assign_semantics.rs`.
#[path = "assign_semantics.rs"]
pub mod assign_semantics;

/// Integration tests from `tests/atomic_law_property_contracts.rs`.
#[path = "atomic_law_property_contracts.rs"]
pub mod atomic_law_property_contracts;

/// Integration tests from `tests/atomic_oracle_contract.rs`.
#[path = "atomic_oracle_contract.rs"]
pub mod atomic_oracle_contract;

/// Integration tests from `tests/atomic_property_contracts.rs`.
#[path = "atomic_property_contracts.rs"]
pub mod atomic_property_contracts;

/// Integration tests from `tests/byte_prefix_property_contracts.rs`.
#[path = "byte_prefix_property_contracts.rs"]
pub mod byte_prefix_property_contracts;

/// Integration tests from `tests/composition_witness_contracts.rs`.
#[path = "composition_witness_contracts.rs"]
pub mod composition_witness_contracts;

/// Integration tests from `tests/composition_witness_geometry_contracts.rs`.
#[path = "composition_witness_geometry_contracts.rs"]
pub mod composition_witness_geometry_contracts;

/// Integration tests from `tests/composition_witness_parsing_contracts.rs`.
#[path = "composition_witness_parsing_contracts.rs"]
pub mod composition_witness_parsing_contracts;

/// Integration tests from `tests/composition_witness_reasoning_contracts.rs`.
#[path = "composition_witness_reasoning_contracts.rs"]
pub mod composition_witness_reasoning_contracts;

/// Integration tests from `tests/composition_witness_scheduling_contracts.rs`.
#[path = "composition_witness_scheduling_contracts.rs"]
pub mod composition_witness_scheduling_contracts;

/// Integration tests from `tests/core_contracts/mod.rs`.
#[path = "core_contracts/mod.rs"]
pub mod core_contracts;

/// Integration tests from `tests/dual_arith_reference_contracts.rs`.
#[path = "dual_arith_reference_contracts.rs"]
pub mod dual_arith_reference_contracts;

/// Integration tests from `tests/dual_reference_parity.rs`.
#[path = "dual_reference_parity.rs"]
pub mod dual_reference_parity;

/// Integration tests from `tests/dual_reference_property_contracts.rs`.
#[path = "dual_reference_property_contracts.rs"]
pub mod dual_reference_property_contracts;

/// Integration tests from `tests/dual_registry_adversarial_contract.rs`.
#[path = "dual_registry_adversarial_contract.rs"]
pub mod dual_registry_adversarial_contract;

/// Integration tests from `tests/dual_scalar_evaluator_matrix.rs`.
#[path = "dual_scalar_evaluator_matrix.rs"]
pub mod dual_scalar_evaluator_matrix;

/// Integration tests from `tests/expr_adversarial_proptest.rs`.
#[allow(dead_code)]
#[path = "expr_adversarial_proptest.rs"]
pub mod expr_adversarial_proptest;

/// Integration tests from `tests/f32_comparison_property_contracts.rs`.
#[path = "f32_comparison_property_contracts.rs"]
pub mod f32_comparison_property_contracts;

/// Integration tests from `tests/fixed_width_value_property_contracts.rs`.
#[path = "fixed_width_value_property_contracts.rs"]
pub mod fixed_width_value_property_contracts;

/// Integration tests from `tests/flat_cpu_input_contract.rs`.
#[path = "flat_cpu_input_contract.rs"]
pub mod flat_cpu_input_contract;

/// Integration tests from `tests/fnv1a32_zero.rs`.
#[allow(missing_docs)]
#[path = "fnv1a32_zero.rs"]
pub mod fnv1a32_zero;

/// Integration tests from `tests/gap_transcendentals_parity.rs`.
#[path = "gap_transcendentals_parity.rs"]
pub mod gap_transcendentals_parity;

/// Integration tests from `tests/grid_fence_oracle_contracts.rs`.
#[path = "grid_fence_oracle_contracts.rs"]
pub mod grid_fence_oracle_contracts;
/// Integration tests from `tests/interleaving_race_freedom_contracts.rs`.
#[path = "interleaving_race_freedom_contracts.rs"]
pub mod interleaving_race_freedom_contracts;

/// Integration tests from `tests/hashmap_async_and_indirect_contracts.rs`.
#[path = "hashmap_async_and_indirect_contracts.rs"]
pub mod hashmap_async_and_indirect_contracts;

/// Integration tests from `tests/hashmap_buffer_size_contracts.rs`.
#[path = "hashmap_buffer_size_contracts.rs"]
pub mod hashmap_buffer_size_contracts;

/// Integration tests from `tests/hashmap_invocation_size_contracts.rs`.
#[path = "hashmap_invocation_size_contracts.rs"]
pub mod hashmap_invocation_size_contracts;

/// Integration tests from `tests/logical_execution_markers.rs`.
#[path = "logical_execution_markers.rs"]
pub mod logical_execution_markers;

/// Integration tests from `tests/oracle_program_edges.rs`.
#[path = "oracle_program_edges.rs"]
pub mod oracle_program_edges;

/// Integration tests from `tests/quantized_buffer_contract.rs`.
#[path = "quantized_buffer_contract.rs"]
pub mod quantized_buffer_contract;

/// Integration tests from `tests/reference_abi_predicates.rs`.
#[path = "reference_abi_predicates.rs"]
pub mod reference_abi_predicates;

/// Integration tests from `tests/reference_error_contract.rs`.
#[path = "reference_error_contract.rs"]
pub mod reference_error_contract;

/// Integration tests from `tests/reference_eval_fma_select_generated.rs`.
#[path = "reference_eval_fma_select_generated.rs"]
pub mod reference_eval_fma_select_generated;

/// Integration tests from `tests/reference_output_byte_stability.rs`.
#[path = "reference_output_byte_stability.rs"]
pub mod reference_output_byte_stability;

/// Integration tests from `tests/region_frame_lifetime.rs`.
#[path = "region_frame_lifetime.rs"]
pub mod region_frame_lifetime;

/// Integration tests from `tests/region_gate.rs`.
#[path = "region_gate.rs"]
pub mod region_gate;

/// Integration tests from `tests/saturating_binops_contract.rs`.
#[path = "saturating_binops_contract.rs"]
pub mod saturating_binops_contract;

/// Integration tests from `tests/single_rank_collective_reference.rs`.
#[path = "single_rank_collective_reference.rs"]
pub mod single_rank_collective_reference;

/// Integration tests from `tests/step_ceiling_contract.rs` (work ceiling contract).
#[path = "step_ceiling_contract.rs"]
pub mod step_ceiling_contract;

/// Integration tests from `tests/storage_graph_generated_adversarial.rs`.
#[path = "storage_graph_generated_adversarial.rs"]
pub mod storage_graph_generated_adversarial;

/// Integration tests from `tests/storage_graph_scalar_matrix.rs`.
#[path = "storage_graph_scalar_matrix.rs"]
pub mod storage_graph_scalar_matrix;

/// Integration tests from `tests/strict_transcendental_accuracy.rs`.
#[path = "strict_transcendental_accuracy.rs"]
pub mod strict_transcendental_accuracy;

/// Integration tests from `tests/subgroup_collectives_are_lane_identified.rs`.
#[cfg(feature = "subgroup-ops")]
#[path = "subgroup_collectives_are_lane_identified.rs"]
pub mod subgroup_collectives_are_lane_identified;

/// Integration tests from `tests/subgroup_edge_contract.rs`.
#[path = "subgroup_edge_contract.rs"]
pub mod subgroup_edge_contract;

/// Integration tests from `tests/subnormal_contract.rs`.
#[path = "subnormal_contract.rs"]
pub mod subnormal_contract;

/// Integration tests from `tests/sweep_dual_arith_oracle_matrix.rs`.
#[path = "sweep_dual_arith_oracle_matrix.rs"]
pub mod sweep_dual_arith_oracle_matrix;

/// Integration tests from `tests/sweep_dual_bitwise_and_volume_oracle_matrix.rs`.
#[path = "sweep_dual_bitwise_and_volume_oracle_matrix.rs"]
pub mod sweep_dual_bitwise_and_volume_oracle_matrix;

/// Integration tests from `tests/sweep_dual_bitwise_clz_volume_oracle_matrix.rs`.
#[path = "sweep_dual_bitwise_clz_volume_oracle_matrix.rs"]
pub mod sweep_dual_bitwise_clz_volume_oracle_matrix;

/// Integration tests from `tests/sweep_dual_bitwise_not_volume_oracle_matrix.rs`.
#[path = "sweep_dual_bitwise_not_volume_oracle_matrix.rs"]
pub mod sweep_dual_bitwise_not_volume_oracle_matrix;

/// Integration tests from `tests/sweep_dual_bitwise_or_volume_oracle_matrix.rs`.
#[path = "sweep_dual_bitwise_or_volume_oracle_matrix.rs"]
pub mod sweep_dual_bitwise_or_volume_oracle_matrix;

/// Integration tests from `tests/sweep_dual_bitwise_popcount_volume_oracle_matrix.rs`.
#[path = "sweep_dual_bitwise_popcount_volume_oracle_matrix.rs"]
pub mod sweep_dual_bitwise_popcount_volume_oracle_matrix;

/// Integration tests from `tests/sweep_dual_bitwise_shift_left_volume_oracle_matrix.rs`.
#[path = "sweep_dual_bitwise_shift_left_volume_oracle_matrix.rs"]
pub mod sweep_dual_bitwise_shift_left_volume_oracle_matrix;

/// Integration tests from `tests/sweep_dual_bitwise_shift_right_volume_oracle_matrix.rs`.
#[path = "sweep_dual_bitwise_shift_right_volume_oracle_matrix.rs"]
pub mod sweep_dual_bitwise_shift_right_volume_oracle_matrix;

/// Integration tests from `tests/sweep_dual_bitwise_xor_volume_oracle_matrix.rs`.
#[path = "sweep_dual_bitwise_xor_volume_oracle_matrix.rs"]
pub mod sweep_dual_bitwise_xor_volume_oracle_matrix;

/// Integration tests from `tests/sweep_dual_compare_eq_volume_oracle_matrix.rs`.
#[path = "sweep_dual_compare_eq_volume_oracle_matrix.rs"]
pub mod sweep_dual_compare_eq_volume_oracle_matrix;

/// Integration tests from `tests/sweep_dual_compare_lt_volume_oracle_matrix.rs`.
#[path = "sweep_dual_compare_lt_volume_oracle_matrix.rs"]
pub mod sweep_dual_compare_lt_volume_oracle_matrix;

/// Integration tests from `tests/tile_reference_contracts.rs`.
#[path = "tile_reference_contracts.rs"]
pub mod tile_reference_contracts;

/// Integration tests from `tests/typed_validation_source.rs`.
#[path = "typed_validation_source.rs"]
pub mod typed_validation_source;

/// Integration tests from `tests/value_array_property_contracts.rs`.
#[path = "value_array_property_contracts.rs"]
pub mod value_array_property_contracts;

/// Integration tests from `tests/value_byte_property_contracts.rs`.
#[path = "value_byte_property_contracts.rs"]
pub mod value_byte_property_contracts;

/// Integration tests from `tests/value_datatype_generated_matrix.rs`.
#[path = "value_datatype_generated_matrix.rs"]
pub mod value_datatype_generated_matrix;

/// Integration tests from `tests/value_encoding_contract.rs`.
#[path = "value_encoding_contract.rs"]
pub mod value_encoding_contract;

/// Integration tests from `tests/value_extend_bytes_width_generated.rs`.
#[path = "value_extend_bytes_width_generated.rs"]
pub mod value_extend_bytes_width_generated;

/// Integration tests from `tests/value_float_property_contracts.rs`.
#[path = "value_float_property_contracts.rs"]
pub mod value_float_property_contracts;

/// Integration tests from `tests/value_narrowing_property_contracts.rs`.
#[path = "value_narrowing_property_contracts.rs"]
pub mod value_narrowing_property_contracts;

/// Integration tests from `tests/value_signed_narrowing_property_contracts.rs`.
#[path = "value_signed_narrowing_property_contracts.rs"]
pub mod value_signed_narrowing_property_contracts;

/// Integration tests from `tests/value_truthiness_property_contracts.rs`.
#[path = "value_truthiness_property_contracts.rs"]
pub mod value_truthiness_property_contracts;

/// Integration tests from `tests/value_write_bytes_width_generated.rs`.
#[path = "value_write_bytes_width_generated.rs"]
pub mod value_write_bytes_width_generated;

/// Integration tests from `tests/vector_cast_generated_matrix.rs`.
#[path = "vector_cast_generated_matrix.rs"]
pub mod vector_cast_generated_matrix;
