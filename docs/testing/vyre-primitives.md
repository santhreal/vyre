# Testing `vyre-primitives`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives
```

Own reusable Tier 2.5 program builders shared by higher-level libraries and runtimes.

The crate lives at `vyre-primitives`. The `primitive-library` owner maintains its
`primitives` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives
```

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --all-features
```

## Feature sets

- Default feature members: None
- Available manifest features: `all-lego`, `bitset`, `cat`, `cpu-parity`, `decode`, `default`, `dnnf`, `effects`, `fixpoint`, `geom`, `gpu`, `graph`, `hardware`, `hash`, `inventory-registry`, `label`, `matching`, `math`, `nfa`, `nn`, `opt`, `parsing`, `predicate`, `reduce`, `text`, `topology`, `types`, `visual`, `vyre-foundation`, `zx`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bench` | `wire_throughput` | `vyre-primitives/benches/wire_throughput.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --bench wire_throughput` |
| `example` | `dominator_tree_e2e` | `vyre-primitives/examples/dominator_tree_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --example dominator_tree_e2e` |
| `example` | `dominator_tree_e2e` | `vyre-primitives/examples/dominator_tree_e2e.rs` | `cpu-parity`, `graph` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --example dominator_tree_e2e` |
| `example` | `vyre_primitives_release_surface` | `vyre-primitives/examples/vyre_primitives_release_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --example vyre_primitives_release_surface` |
| `example` | `wire_harness_smoke` | `vyre-primitives/examples/wire_harness_smoke.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --example wire_harness_smoke` |
| `lib` | `vyre_primitives` | `vyre-primitives/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives` |
| `test` | `adaptive_four_russians_dense_generated` | `vyre-primitives/tests/adaptive_four_russians_dense_generated.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adaptive_four_russians_dense_generated` |
| `test` | `adaptive_four_russians_dense_generated` | `vyre-primitives/tests/adaptive_four_russians_dense_generated.rs` | `cpu-parity`, `graph` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adaptive_four_russians_dense_generated` |
| `test` | `adversarial_bitset_contains` | `vyre-primitives/tests/adversarial_bitset_contains.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_bitset_contains` |
| `test` | `adversarial_bitset_contains` | `vyre-primitives/tests/adversarial_bitset_contains.rs` | `bitset`, `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_bitset_contains` |
| `test` | `adversarial_bitset_ops` | `vyre-primitives/tests/adversarial_bitset_ops.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_bitset_ops` |
| `test` | `adversarial_bitset_ops` | `vyre-primitives/tests/adversarial_bitset_ops.rs` | `bitset`, `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_bitset_ops` |
| `test` | `adversarial_bitset_reduce_matrix` | `vyre-primitives/tests/adversarial_bitset_reduce_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_bitset_reduce_matrix` |
| `test` | `adversarial_bitset_reduce_matrix` | `vyre-primitives/tests/adversarial_bitset_reduce_matrix.rs` | `bitset`, `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_bitset_reduce_matrix` |
| `test` | `adversarial_boolean_packing_four_russians_readiness` | `vyre-primitives/tests/adversarial_boolean_packing_four_russians_readiness.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_boolean_packing_four_russians_readiness` |
| `test` | `adversarial_boolean_packing_four_russians_readiness` | `vyre-primitives/tests/adversarial_boolean_packing_four_russians_readiness.rs` | `bitset`, `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_boolean_packing_four_russians_readiness` |
| `test` | `adversarial_decode` | `vyre-primitives/tests/adversarial_decode.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_decode` |
| `test` | `adversarial_fixpoint` | `vyre-primitives/tests/adversarial_fixpoint.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_fixpoint` |
| `test` | `adversarial_frontier_queue_clear` | `vyre-primitives/tests/adversarial_frontier_queue_clear.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_frontier_queue_clear` |
| `test` | `adversarial_graph` | `vyre-primitives/tests/adversarial_graph.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_graph` |
| `test` | `adversarial_graph_csr_validation_contracts` | `vyre-primitives/tests/adversarial_graph_csr_validation_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_graph_csr_validation_contracts` |
| `test` | `adversarial_graph_csr_validation_contracts` | `vyre-primitives/tests/adversarial_graph_csr_validation_contracts.rs` | `graph` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_graph_csr_validation_contracts` |
| `test` | `adversarial_graph_ops` | `vyre-primitives/tests/adversarial_graph_ops.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_graph_ops` |
| `test` | `adversarial_graph_ops` | `vyre-primitives/tests/adversarial_graph_ops.rs` | `cpu-parity`, `graph` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_graph_ops` |
| `test` | `adversarial_graph_reachability_fixpoint` | `vyre-primitives/tests/adversarial_graph_reachability_fixpoint.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_graph_reachability_fixpoint` |
| `test` | `adversarial_graph_reachability_fixpoint` | `vyre-primitives/tests/adversarial_graph_reachability_fixpoint.rs` | `cpu-parity`, `fixpoint`, `graph`, `math` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_graph_reachability_fixpoint` |
| `test` | `adversarial_hash` | `vyre-primitives/tests/adversarial_hash.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_hash` |
| `test` | `adversarial_label` | `vyre-primitives/tests/adversarial_label.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_label` |
| `test` | `adversarial_matching` | `vyre-primitives/tests/adversarial_matching.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_matching` |
| `test` | `adversarial_math` | `vyre-primitives/tests/adversarial_math.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_math` |
| `test` | `adversarial_math` | `vyre-primitives/tests/adversarial_math.rs` | `cpu-parity`, `math` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_math` |
| `test` | `adversarial_nfa` | `vyre-primitives/tests/adversarial_nfa.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_nfa` |
| `test` | `adversarial_reduce_gather` | `vyre-primitives/tests/adversarial_reduce_gather.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_reduce_gather` |
| `test` | `adversarial_reduce_gather` | `vyre-primitives/tests/adversarial_reduce_gather.rs` | `reduce` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_reduce_gather` |
| `test` | `adversarial_reduce_histogram` | `vyre-primitives/tests/adversarial_reduce_histogram.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_reduce_histogram` |
| `test` | `adversarial_reduce_histogram` | `vyre-primitives/tests/adversarial_reduce_histogram.rs` | `reduce` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_reduce_histogram` |
| `test` | `adversarial_reduce_radix_sort` | `vyre-primitives/tests/adversarial_reduce_radix_sort.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_reduce_radix_sort` |
| `test` | `adversarial_reduce_radix_sort` | `vyre-primitives/tests/adversarial_reduce_radix_sort.rs` | `reduce` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_reduce_radix_sort` |
| `test` | `adversarial_reduce_scatter` | `vyre-primitives/tests/adversarial_reduce_scatter.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_reduce_scatter` |
| `test` | `adversarial_reduce_scatter` | `vyre-primitives/tests/adversarial_reduce_scatter.rs` | `reduce` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_reduce_scatter` |
| `test` | `adversarial_reduce_segment_reduce` | `vyre-primitives/tests/adversarial_reduce_segment_reduce.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_reduce_segment_reduce` |
| `test` | `adversarial_reduce_segment_reduce` | `vyre-primitives/tests/adversarial_reduce_segment_reduce.rs` | `reduce` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_reduce_segment_reduce` |
| `test` | `adversarial_text_char_class` | `vyre-primitives/tests/adversarial_text_char_class.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_text_char_class` |
| `test` | `adversarial_text_char_class` | `vyre-primitives/tests/adversarial_text_char_class.rs` | `text` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_text_char_class` |
| `test` | `adversarial_text_extra` | `vyre-primitives/tests/adversarial_text_extra.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_text_extra` |
| `test` | `adversarial_text_extra` | `vyre-primitives/tests/adversarial_text_extra.rs` | `cpu-parity`, `text` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_text_extra` |
| `test` | `adversarial_text_line_index` | `vyre-primitives/tests/adversarial_text_line_index.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_text_line_index` |
| `test` | `adversarial_text_line_index` | `vyre-primitives/tests/adversarial_text_line_index.rs` | `text` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_text_line_index` |
| `test` | `adversarial_text_utf8_validate` | `vyre-primitives/tests/adversarial_text_utf8_validate.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_text_utf8_validate` |
| `test` | `adversarial_text_utf8_validate` | `vyre-primitives/tests/adversarial_text_utf8_validate.rs` | `text` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test adversarial_text_utf8_validate` |
| `test` | `amg_v_cycle_ir_parity` | `vyre-primitives/tests/amg_v_cycle_ir_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test amg_v_cycle_ir_parity` |
| `test` | `argmax_of_marginals_ir_parity_proptest` | `vyre-primitives/tests/argmax_of_marginals_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test argmax_of_marginals_ir_parity_proptest` |
| `test` | `bellman_oob_edge_parity` | `vyre-primitives/tests/bellman_oob_edge_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test bellman_oob_edge_parity` |
| `test` | `bigint_add_carry_ir_parity_proptest` | `vyre-primitives/tests/bigint_add_carry_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test bigint_add_carry_ir_parity_proptest` |
| `test` | `bitset_fixpoint_warm_start_parity` | `vyre-primitives/tests/bitset_fixpoint_warm_start_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test bitset_fixpoint_warm_start_parity` |
| `test` | `bitset_law_support` | `vyre-primitives/tests/bitset_law_support.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test bitset_law_support` |
| `test` | `bitset_scalar_ir_parity_proptest` | `vyre-primitives/tests/bitset_scalar_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test bitset_scalar_ir_parity_proptest` |
| `test` | `bitset_word_contracts` | `vyre-primitives/tests/bitset_word_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test bitset_word_contracts` |
| `test` | `bitset_word_contracts` | `vyre-primitives/tests/bitset_word_contracts.rs` | `bitset`, `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test bitset_word_contracts` |
| `test` | `bitset_words_sizing_contracts` | `vyre-primitives/tests/bitset_words_sizing_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test bitset_words_sizing_contracts` |
| `test` | `blake3_program` | `vyre-primitives/tests/blake3_program.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test blake3_program` |
| `test` | `bracket_match_proptest` | `vyre-primitives/tests/bracket_match_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test bracket_match_proptest` |
| `test` | `clifford_geometric_product_program_parity` | `vyre-primitives/tests/clifford_geometric_product_program_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test clifford_geometric_product_program_parity` |
| `test` | `consumer_boundary` | `vyre-primitives/tests/consumer_boundary.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test consumer_boundary` |
| `test` | `crc32_map_reduce_generated` | `vyre-primitives/tests/crc32_map_reduce_generated.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test crc32_map_reduce_generated` |
| `test` | `csr_backward_or_changed_ir_fixpoint` | `vyre-primitives/tests/csr_backward_or_changed_ir_fixpoint.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test csr_backward_or_changed_ir_fixpoint` |
| `test` | `csr_backward_traverse_ir_parity_proptest` | `vyre-primitives/tests/csr_backward_traverse_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test csr_backward_traverse_ir_parity_proptest` |
| `test` | `csr_certificates` | `vyre-primitives/tests/csr_certificates.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test csr_certificates` |
| `test` | `csr_forward_traverse_ir_parity_proptest` | `vyre-primitives/tests/csr_forward_traverse_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test csr_forward_traverse_ir_parity_proptest` |
| `test` | `csr_frontier_degree_sum_ir_parity_proptest` | `vyre-primitives/tests/csr_frontier_degree_sum_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test csr_frontier_degree_sum_ir_parity_proptest` |
| `test` | `csr_queue_strided_ir_parity_proptest` | `vyre-primitives/tests/csr_queue_strided_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test csr_queue_strided_ir_parity_proptest` |
| `test` | `csr_traversal_clone_family_equality` | `vyre-primitives/tests/csr_traversal_clone_family_equality.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test csr_traversal_clone_family_equality` |
| `test` | `delegating_builder_equivalence` | `vyre-primitives/tests/delegating_builder_equivalence.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test delegating_builder_equivalence` |
| `test` | `dfa_wire_contracts` | `vyre-primitives/tests/dfa_wire_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test dfa_wire_contracts` |
| `test` | `do_calculus_rule2_value_parity` | `vyre-primitives/tests/do_calculus_rule2_value_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test do_calculus_rule2_value_parity` |
| `test` | `dominator_tree_pristine` | `vyre-primitives/tests/dominator_tree_pristine.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test dominator_tree_pristine` |
| `test` | `dominator_tree_pristine` | `vyre-primitives/tests/dominator_tree_pristine.rs` | `cpu-parity`, `graph` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test dominator_tree_pristine` |
| `test` | `dominator_tree_proptest` | `vyre-primitives/tests/dominator_tree_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test dominator_tree_proptest` |
| `test` | `dominator_tree_proptest` | `vyre-primitives/tests/dominator_tree_proptest.rs` | `cpu-parity`, `graph` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test dominator_tree_proptest` |
| `test` | `dominator_tree_scale_gate` | `vyre-primitives/tests/dominator_tree_scale_gate.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test dominator_tree_scale_gate` |
| `test` | `dominator_tree_scale_gate` | `vyre-primitives/tests/dominator_tree_scale_gate.rs` | `cpu-parity`, `graph` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test dominator_tree_scale_gate` |
| `test` | `dp_clip_signed_newton_parity` | `vyre-primitives/tests/dp_clip_signed_newton_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test dp_clip_signed_newton_parity` |
| `test` | `fmm_program_parity` | `vyre-primitives/tests/fmm_program_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test fmm_program_parity` |
| `test` | `fnv1a64_u32_lane_parity` | `vyre-primitives/tests/fnv1a64_u32_lane_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test fnv1a64_u32_lane_parity` |
| `test` | `fnv1a64_u32_lane_parity` | `vyre-primitives/tests/fnv1a64_u32_lane_parity.rs` | `hash` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test fnv1a64_u32_lane_parity` |
| `test` | `fnv1a_dyn_parity` | `vyre-primitives/tests/fnv1a_dyn_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test fnv1a_dyn_parity` |
| `test` | `four_russians_dense_matvec_generated` | `vyre-primitives/tests/four_russians_dense_matvec_generated.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test four_russians_dense_matvec_generated` |
| `test` | `four_russians_dense_matvec_generated` | `vyre-primitives/tests/four_russians_dense_matvec_generated.rs` | `bitset`, `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test four_russians_dense_matvec_generated` |
| `test` | `frontier_absorb_parity` | `vyre-primitives/tests/frontier_absorb_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test frontier_absorb_parity` |
| `test` | `frontier_load_balancing_policies` | `vyre-primitives/tests/frontier_load_balancing_policies.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test frontier_load_balancing_policies` |
| `test` | `frontier_to_queue_multi_workgroup_span` | `vyre-primitives/tests/frontier_to_queue_multi_workgroup_span.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test frontier_to_queue_multi_workgroup_span` |
| `test` | `functor_apply_ir_parity_proptest` | `vyre-primitives/tests/functor_apply_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test functor_apply_ir_parity_proptest` |
| `test` | `generated_hardware_f32_matrix` | `vyre-primitives/tests/generated_hardware_f32_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test generated_hardware_f32_matrix` |
| `test` | `generated_hardware_f32_matrix` | `vyre-primitives/tests/generated_hardware_f32_matrix.rs` | `hardware` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test generated_hardware_f32_matrix` |
| `test` | `generated_hardware_registry_matrix` | `vyre-primitives/tests/generated_hardware_registry_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test generated_hardware_registry_matrix` |
| `test` | `generated_hardware_registry_matrix` | `vyre-primitives/tests/generated_hardware_registry_matrix.rs` | `hardware` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test generated_hardware_registry_matrix` |
| `test` | `generated_hardware_u32_matrix` | `vyre-primitives/tests/generated_hardware_u32_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test generated_hardware_u32_matrix` |
| `test` | `generated_hardware_u32_matrix` | `vyre-primitives/tests/generated_hardware_u32_matrix.rs` | `hardware` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test generated_hardware_u32_matrix` |
| `test` | `graph_builders_emit_valid_ir` | `vyre-primitives/tests/graph_builders_emit_valid_ir.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test graph_builders_emit_valid_ir` |
| `test` | `graph_fixpoint_adversarial_generated` | `vyre-primitives/tests/graph_fixpoint_adversarial_generated.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test graph_fixpoint_adversarial_generated` |
| `test` | `graph_primitive_binding_contracts` | `vyre-primitives/tests/graph_primitive_binding_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test graph_primitive_binding_contracts` |
| `test` | `graph_primitive_binding_contracts` | `vyre-primitives/tests/graph_primitive_binding_contracts.rs` | `graph`, `inventory-registry` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test graph_primitive_binding_contracts` |
| `test` | `hardware_conform` | `vyre-primitives/tests/hardware_conform.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test hardware_conform` |
| `test` | `hardware_conform` | `vyre-primitives/tests/hardware_conform.rs` | `hardware` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test hardware_conform` |
| `test` | `hardware_registration_safety_rules` | `vyre-primitives/tests/hardware_registration_safety_rules.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test hardware_registration_safety_rules` |
| `test` | `hardware_registry_contract` | `vyre-primitives/tests/hardware_registry_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test hardware_registry_contract` |
| `test` | `hardware_registry_contract` | `vyre-primitives/tests/hardware_registry_contract.rs` | `hardware` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test hardware_registry_contract` |
| `test` | `hash_crc32_ir_parity_proptest` | `vyre-primitives/tests/hash_crc32_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test hash_crc32_ir_parity_proptest` |
| `test` | `hash_incremental_adversarial_generated` | `vyre-primitives/tests/hash_incremental_adversarial_generated.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test hash_incremental_adversarial_generated` |
| `test` | `hash_registration_witnesses` | `vyre-primitives/tests/hash_registration_witnesses.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test hash_registration_witnesses` |
| `test` | `hash_registration_witnesses` | `vyre-primitives/tests/hash_registration_witnesses.rs` | `hash`, `inventory-registry` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test hash_registration_witnesses` |
| `test` | `hash_stream_ir_parity_proptest` | `vyre-primitives/tests/hash_stream_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test hash_stream_ir_parity_proptest` |
| `test` | `histogram_atomic_scatter_parity` | `vyre-primitives/tests/histogram_atomic_scatter_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test histogram_atomic_scatter_parity` |
| `test` | `homotopy_euler_signed_parity` | `vyre-primitives/tests/homotopy_euler_signed_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test homotopy_euler_signed_parity` |
| `test` | `hypervector_ir_parity_proptest` | `vyre-primitives/tests/hypervector_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test hypervector_ir_parity_proptest` |
| `test` | `iht_threshold_ir_parity_proptest` | `vyre-primitives/tests/iht_threshold_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test iht_threshold_ir_parity_proptest` |
| `test` | `indexed_move_gather_oob_parity` | `vyre-primitives/tests/indexed_move_gather_oob_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test indexed_move_gather_oob_parity` |
| `test` | `inflate_program` | `vyre-primitives/tests/inflate_program.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test inflate_program` |
| `test` | `inflate_program` | `vyre-primitives/tests/inflate_program.rs` | `decode` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test inflate_program` |
| `test` | `inflate_stored_ir_parity_proptest` | `vyre-primitives/tests/inflate_stored_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test inflate_stored_ir_parity_proptest` |
| `test` | `integration` | `vyre-primitives/tests/integration.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test integration` |
| `test` | `integration` | `vyre-primitives/tests/integration.rs` | `hash`, `inventory-registry` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test integration` |
| `test` | `jacobi_serial_body_matches_per_lane` | `vyre-primitives/tests/jacobi_serial_body_matches_per_lane.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test jacobi_serial_body_matches_per_lane` |
| `test` | `kfac_block_inverse_proptest` | `vyre-primitives/tests/kfac_block_inverse_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test kfac_block_inverse_proptest` |
| `test` | `line_splice_classify_roundtrip` | `vyre-primitives/tests/line_splice_classify_roundtrip.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test line_splice_classify_roundtrip` |
| `test` | `loop_back_edge_audit` | `vyre-primitives/tests/loop_back_edge_audit.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test loop_back_edge_audit` |
| `test` | `matroid_intersection_full_proptest` | `vyre-primitives/tests/matroid_intersection_full_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test matroid_intersection_full_proptest` |
| `test` | `matroid_intersection_full_value_parity` | `vyre-primitives/tests/matroid_intersection_full_value_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test matroid_intersection_full_value_parity` |
| `test` | `motif_ir_parity_proptest` | `vyre-primitives/tests/motif_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test motif_ir_parity_proptest` |
| `test` | `multi_block_prefix_scan_carry_parity` | `vyre-primitives/tests/multi_block_prefix_scan_carry_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test multi_block_prefix_scan_carry_parity` |
| `test` | `node_kind_eq_ir_parity_proptest` | `vyre-primitives/tests/node_kind_eq_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test node_kind_eq_ir_parity_proptest` |
| `test` | `node_kind_eq_ir_parity_proptest` | `vyre-primitives/tests/node_kind_eq_ir_parity_proptest.rs` | `cpu-parity`, `predicate` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test node_kind_eq_ir_parity_proptest` |
| `test` | `padic_hensel_signed_parity` | `vyre-primitives/tests/padic_hensel_signed_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test padic_hensel_signed_parity` |
| `test` | `persistent_fixpoint_grid_contracts` | `vyre-primitives/tests/persistent_fixpoint_grid_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test persistent_fixpoint_grid_contracts` |
| `test` | `persistent_fixpoint_grid_contracts` | `vyre-primitives/tests/persistent_fixpoint_grid_contracts.rs` | `cpu-parity`, `fixpoint` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test persistent_fixpoint_grid_contracts` |
| `test` | `persistent_fixpoint_loop_contracts` | `vyre-primitives/tests/persistent_fixpoint_loop_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test persistent_fixpoint_loop_contracts` |
| `test` | `planar_rewrite_ir_parity_proptest` | `vyre-primitives/tests/planar_rewrite_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test planar_rewrite_ir_parity_proptest` |
| `test` | `production_ir_parity` | `vyre-primitives/tests/production_ir_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test production_ir_parity` |
| `test` | `proptest_base64_decode` | `vyre-primitives/tests/proptest_base64_decode.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_base64_decode` |
| `test` | `proptest_bitset_and_laws` | `vyre-primitives/tests/proptest_bitset_and_laws.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_bitset_and_laws` |
| `test` | `proptest_bitset_any` | `vyre-primitives/tests/proptest_bitset_any.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_bitset_any` |
| `test` | `proptest_bitset_boolean_algebra` | `vyre-primitives/tests/proptest_bitset_boolean_algebra.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_bitset_boolean_algebra` |
| `test` | `proptest_bitset_contains` | `vyre-primitives/tests/proptest_bitset_contains.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_bitset_contains` |
| `test` | `proptest_bitset_copy` | `vyre-primitives/tests/proptest_bitset_copy.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_bitset_copy` |
| `test` | `proptest_bitset_equal` | `vyre-primitives/tests/proptest_bitset_equal.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_bitset_equal` |
| `test` | `proptest_bitset_not_involution` | `vyre-primitives/tests/proptest_bitset_not_involution.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_bitset_not_involution` |
| `test` | `proptest_bitset_not_laws` | `vyre-primitives/tests/proptest_bitset_not_laws.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_bitset_not_laws` |
| `test` | `proptest_bitset_or_laws` | `vyre-primitives/tests/proptest_bitset_or_laws.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_bitset_or_laws` |
| `test` | `proptest_bitset_popcount` | `vyre-primitives/tests/proptest_bitset_popcount.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_bitset_popcount` |
| `test` | `proptest_bitset_popcount_laws` | `vyre-primitives/tests/proptest_bitset_popcount_laws.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_bitset_popcount_laws` |
| `test` | `proptest_bitset_subset_of` | `vyre-primitives/tests/proptest_bitset_subset_of.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_bitset_subset_of` |
| `test` | `proptest_bitset_words` | `vyre-primitives/tests/proptest_bitset_words.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_bitset_words` |
| `test` | `proptest_bitset_xor_laws` | `vyre-primitives/tests/proptest_bitset_xor_laws.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_bitset_xor_laws` |
| `test` | `proptest_bitset_zero` | `vyre-primitives/tests/proptest_bitset_zero.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_bitset_zero` |
| `test` | `proptest_csr_forward_traverse` | `vyre-primitives/tests/proptest_csr_forward_traverse.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_csr_forward_traverse` |
| `test` | `proptest_csr_frontier_queue` | `vyre-primitives/tests/proptest_csr_frontier_queue.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_csr_frontier_queue` |
| `test` | `proptest_csr_frontier_queue` | `vyre-primitives/tests/proptest_csr_frontier_queue.rs` | `cpu-parity`, `graph` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_csr_frontier_queue` |
| `test` | `proptest_csr_frontier_shard` | `vyre-primitives/tests/proptest_csr_frontier_shard.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_csr_frontier_shard` |
| `test` | `proptest_csr_frontier_shard` | `vyre-primitives/tests/proptest_csr_frontier_shard.rs` | `cpu-parity`, `graph` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_csr_frontier_shard` |
| `test` | `proptest_csr_queue_split` | `vyre-primitives/tests/proptest_csr_queue_split.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_csr_queue_split` |
| `test` | `proptest_csr_queue_split` | `vyre-primitives/tests/proptest_csr_queue_split.rs` | `cpu-parity`, `graph` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_csr_queue_split` |
| `test` | `proptest_csr_queue_strided` | `vyre-primitives/tests/proptest_csr_queue_strided.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_csr_queue_strided` |
| `test` | `proptest_csr_queue_strided` | `vyre-primitives/tests/proptest_csr_queue_strided.rs` | `cpu-parity`, `graph` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_csr_queue_strided` |
| `test` | `proptest_dispatch_pack_roundtrip` | `vyre-primitives/tests/proptest_dispatch_pack_roundtrip.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_dispatch_pack_roundtrip` |
| `test` | `proptest_dominator_frontier` | `vyre-primitives/tests/proptest_dominator_frontier.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_dominator_frontier` |
| `test` | `proptest_graph_reachable` | `vyre-primitives/tests/proptest_graph_reachable.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_graph_reachable` |
| `test` | `proptest_hash_crc32` | `vyre-primitives/tests/proptest_hash_crc32.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_hash_crc32` |
| `test` | `proptest_hash_fnv1a` | `vyre-primitives/tests/proptest_hash_fnv1a.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_hash_fnv1a` |
| `test` | `proptest_hex_decode` | `vyre-primitives/tests/proptest_hex_decode.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_hex_decode` |
| `test` | `proptest_multi_block_prefix_scan` | `vyre-primitives/tests/proptest_multi_block_prefix_scan.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_multi_block_prefix_scan` |
| `test` | `proptest_prefix_scan_large` | `vyre-primitives/tests/proptest_prefix_scan_large.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_prefix_scan_large` |
| `test` | `proptest_prefix_scan_large` | `vyre-primitives/tests/proptest_prefix_scan_large.rs` | `cpu-parity`, `math` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_prefix_scan_large` |
| `test` | `proptest_reduce_all` | `vyre-primitives/tests/proptest_reduce_all.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_reduce_all` |
| `test` | `proptest_reduce_any` | `vyre-primitives/tests/proptest_reduce_any.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_reduce_any` |
| `test` | `proptest_reduce_any_all` | `vyre-primitives/tests/proptest_reduce_any_all.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_reduce_any_all` |
| `test` | `proptest_reduce_count_laws` | `vyre-primitives/tests/proptest_reduce_count_laws.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_reduce_count_laws` |
| `test` | `proptest_reduce_count_non_zero` | `vyre-primitives/tests/proptest_reduce_count_non_zero.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_reduce_count_non_zero` |
| `test` | `proptest_reduce_min_max_laws` | `vyre-primitives/tests/proptest_reduce_min_max_laws.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_reduce_min_max_laws` |
| `test` | `proptest_reduce_sum_laws` | `vyre-primitives/tests/proptest_reduce_sum_laws.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_reduce_sum_laws` |
| `test` | `proptest_text_byte_histogram` | `vyre-primitives/tests/proptest_text_byte_histogram.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_text_byte_histogram` |
| `test` | `proptest_text_char_class` | `vyre-primitives/tests/proptest_text_char_class.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_text_char_class` |
| `test` | `proptest_text_encoding_classify` | `vyre-primitives/tests/proptest_text_encoding_classify.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_text_encoding_classify` |
| `test` | `proptest_text_encoding_classify` | `vyre-primitives/tests/proptest_text_encoding_classify.rs` | `cpu-parity`, `text` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_text_encoding_classify` |
| `test` | `proptest_text_line_index` | `vyre-primitives/tests/proptest_text_line_index.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_text_line_index` |
| `test` | `proptest_text_line_index` | `vyre-primitives/tests/proptest_text_line_index.rs` | `cpu-parity`, `text` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_text_line_index` |
| `test` | `proptest_text_utf8_validate` | `vyre-primitives/tests/proptest_text_utf8_validate.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_text_utf8_validate` |
| `test` | `proptest_text_utf8_validate` | `vyre-primitives/tests/proptest_text_utf8_validate.rs` | `cpu-parity`, `text` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_text_utf8_validate` |
| `test` | `proptest_toposort_dag` | `vyre-primitives/tests/proptest_toposort_dag.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_toposort_dag` |
| `test` | `proptest_wire_roundtrip` | `vyre-primitives/tests/proptest_wire_roundtrip.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_wire_roundtrip` |
| `test` | `proptest_ziftsieve` | `vyre-primitives/tests/proptest_ziftsieve.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test proptest_ziftsieve` |
| `test` | `quantized_packing_contracts` | `vyre-primitives/tests/quantized_packing_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test quantized_packing_contracts` |
| `test` | `randomized_svd_signed_parity` | `vyre-primitives/tests/randomized_svd_signed_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test randomized_svd_signed_parity` |
| `test` | `range_counts_ir_parity_proptest` | `vyre-primitives/tests/range_counts_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test range_counts_ir_parity_proptest` |
| `test` | `reduce_atomic_ir_parity_proptest` | `vyre-primitives/tests/reduce_atomic_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test reduce_atomic_ir_parity_proptest` |
| `test` | `reduction_route_parity` | `vyre-primitives/tests/reduction_route_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test reduction_route_parity` |
| `test` | `region_adversarial` | `vyre-primitives/tests/region_adversarial.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test region_adversarial` |
| `test` | `region_adversarial` | `vyre-primitives/tests/region_adversarial.rs` | `matching` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test region_adversarial` |
| `test` | `region_dedup_property` | `vyre-primitives/tests/region_dedup_property.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test region_dedup_property` |
| `test` | `region_gpu_flag_contracts` | `vyre-primitives/tests/region_gpu_flag_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test region_gpu_flag_contracts` |
| `test` | `registry_closure` | `vyre-primitives/tests/registry_closure.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test registry_closure` |
| `test` | `registry_oob_clean` | `vyre-primitives/tests/registry_oob_clean.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test registry_oob_clean` |
| `test` | `resolve_family_ir_parity_proptest` | `vyre-primitives/tests/resolve_family_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test resolve_family_ir_parity_proptest` |
| `test` | `resolve_family_ir_parity_proptest` | `vyre-primitives/tests/resolve_family_ir_parity_proptest.rs` | `cpu-parity`, `label` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test resolve_family_ir_parity_proptest` |
| `test` | `rle_segment_lengths_contracts` | `vyre-primitives/tests/rle_segment_lengths_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test rle_segment_lengths_contracts` |
| `test` | `rle_segment_lengths_ir_parity_proptest` | `vyre-primitives/tests/rle_segment_lengths_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test rle_segment_lengths_ir_parity_proptest` |
| `test` | `scallop_join_ir_parity` | `vyre-primitives/tests/scallop_join_ir_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test scallop_join_ir_parity` |
| `test` | `score_denoise_signed_parity` | `vyre-primitives/tests/score_denoise_signed_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test score_denoise_signed_parity` |
| `test` | `segment_reduce_ir_parity_proptest` | `vyre-primitives/tests/segment_reduce_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test segment_reduce_ir_parity_proptest` |
| `test` | `semiring_gemm_wide_parity` | `vyre-primitives/tests/semiring_gemm_wide_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test semiring_gemm_wide_parity` |
| `test` | `semiring_registry` | `vyre-primitives/tests/semiring_registry.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test semiring_registry` |
| `test` | `set_domain_selector` | `vyre-primitives/tests/set_domain_selector.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test set_domain_selector` |
| `test` | `sheaf_diffusion_step_signed_parity` | `vyre-primitives/tests/sheaf_diffusion_step_signed_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sheaf_diffusion_step_signed_parity` |
| `test` | `sheaf_laplacian_eigenvalue_dispatch_parity` | `vyre-primitives/tests/sheaf_laplacian_eigenvalue_dispatch_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sheaf_laplacian_eigenvalue_dispatch_parity` |
| `test` | `simplicial_triangle_message_fixed_point_parity` | `vyre-primitives/tests/simplicial_triangle_message_fixed_point_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test simplicial_triangle_message_fixed_point_parity` |
| `test` | `sinkhorn_iterate_ir_parity` | `vyre-primitives/tests/sinkhorn_iterate_ir_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sinkhorn_iterate_ir_parity` |
| `test` | `sinkhorn_scale_ir_parity_proptest` | `vyre-primitives/tests/sinkhorn_scale_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sinkhorn_scale_ir_parity_proptest` |
| `test` | `sos_gram_construct_proptest` | `vyre-primitives/tests/sos_gram_construct_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sos_gram_construct_proptest` |
| `test` | `sos_gram_oob_parity` | `vyre-primitives/tests/sos_gram_oob_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sos_gram_oob_parity` |
| `test` | `ssa_dominance_phi_overflow_parity` | `vyre-primitives/tests/ssa_dominance_phi_overflow_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test ssa_dominance_phi_overflow_parity` |
| `test` | `stream_compact_proptest` | `vyre-primitives/tests/stream_compact_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test stream_compact_proptest` |
| `test` | `subgroup_nfa_ir_parity_proptest` | `vyre-primitives/tests/subgroup_nfa_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test subgroup_nfa_ir_parity_proptest` |
| `test` | `sum_product_signed_parity` | `vyre-primitives/tests/sum_product_signed_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sum_product_signed_parity` |
| `test` | `sweep_bitset_oracle_matrix` | `vyre-primitives/tests/sweep_bitset_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_bitset_oracle_matrix` |
| `test` | `sweep_bitset_oracle_matrix` | `vyre-primitives/tests/sweep_bitset_oracle_matrix.rs` | `bitset`, `cpu-parity` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_bitset_oracle_matrix` |
| `test` | `sweep_decode_base64_volume_oracle_matrix` | `vyre-primitives/tests/sweep_decode_base64_volume_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_decode_base64_volume_oracle_matrix` |
| `test` | `sweep_decode_base64_volume_oracle_matrix` | `vyre-primitives/tests/sweep_decode_base64_volume_oracle_matrix.rs` | `decode` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_decode_base64_volume_oracle_matrix` |
| `test` | `sweep_decode_hex_primitives_volume_oracle_matrix` | `vyre-primitives/tests/sweep_decode_hex_primitives_volume_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_decode_hex_primitives_volume_oracle_matrix` |
| `test` | `sweep_decode_hex_primitives_volume_oracle_matrix` | `vyre-primitives/tests/sweep_decode_hex_primitives_volume_oracle_matrix.rs` | `decode` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_decode_hex_primitives_volume_oracle_matrix` |
| `test` | `sweep_graph_csr_backward_traverse_volume_oracle_matrix` | `vyre-primitives/tests/sweep_graph_csr_backward_traverse_volume_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_graph_csr_backward_traverse_volume_oracle_matrix` |
| `test` | `sweep_graph_csr_backward_traverse_volume_oracle_matrix` | `vyre-primitives/tests/sweep_graph_csr_backward_traverse_volume_oracle_matrix.rs` | `cpu-parity`, `graph` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_graph_csr_backward_traverse_volume_oracle_matrix` |
| `test` | `sweep_graph_csr_bidirectional_oracle_matrix` | `vyre-primitives/tests/sweep_graph_csr_bidirectional_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_graph_csr_bidirectional_oracle_matrix` |
| `test` | `sweep_graph_csr_bidirectional_oracle_matrix` | `vyre-primitives/tests/sweep_graph_csr_bidirectional_oracle_matrix.rs` | `cpu-parity`, `graph` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_graph_csr_bidirectional_oracle_matrix` |
| `test` | `sweep_graph_csr_forward_or_changed_volume_oracle_matrix` | `vyre-primitives/tests/sweep_graph_csr_forward_or_changed_volume_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_graph_csr_forward_or_changed_volume_oracle_matrix` |
| `test` | `sweep_graph_csr_forward_or_changed_volume_oracle_matrix` | `vyre-primitives/tests/sweep_graph_csr_forward_or_changed_volume_oracle_matrix.rs` | `cpu-parity`, `graph` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_graph_csr_forward_or_changed_volume_oracle_matrix` |
| `test` | `sweep_graph_csr_forward_traverse_volume_oracle_matrix` | `vyre-primitives/tests/sweep_graph_csr_forward_traverse_volume_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_graph_csr_forward_traverse_volume_oracle_matrix` |
| `test` | `sweep_graph_csr_forward_traverse_volume_oracle_matrix` | `vyre-primitives/tests/sweep_graph_csr_forward_traverse_volume_oracle_matrix.rs` | `cpu-parity`, `graph` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_graph_csr_forward_traverse_volume_oracle_matrix` |
| `test` | `sweep_graph_motif_oracle_matrix` | `vyre-primitives/tests/sweep_graph_motif_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_graph_motif_oracle_matrix` |
| `test` | `sweep_graph_motif_oracle_matrix` | `vyre-primitives/tests/sweep_graph_motif_oracle_matrix.rs` | `cpu-parity`, `graph` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_graph_motif_oracle_matrix` |
| `test` | `sweep_graph_path_reconstruct_oracle_matrix` | `vyre-primitives/tests/sweep_graph_path_reconstruct_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_graph_path_reconstruct_oracle_matrix` |
| `test` | `sweep_graph_path_reconstruct_oracle_matrix` | `vyre-primitives/tests/sweep_graph_path_reconstruct_oracle_matrix.rs` | `cpu-parity`, `graph` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_graph_path_reconstruct_oracle_matrix` |
| `test` | `sweep_graph_persistent_bfs_volume_oracle_matrix` | `vyre-primitives/tests/sweep_graph_persistent_bfs_volume_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_graph_persistent_bfs_volume_oracle_matrix` |
| `test` | `sweep_graph_persistent_bfs_volume_oracle_matrix` | `vyre-primitives/tests/sweep_graph_persistent_bfs_volume_oracle_matrix.rs` | `cpu-parity`, `graph` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_graph_persistent_bfs_volume_oracle_matrix` |
| `test` | `sweep_graph_reachable_volume_oracle_matrix` | `vyre-primitives/tests/sweep_graph_reachable_volume_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_graph_reachable_volume_oracle_matrix` |
| `test` | `sweep_graph_reachable_volume_oracle_matrix` | `vyre-primitives/tests/sweep_graph_reachable_volume_oracle_matrix.rs` | `cpu-parity`, `graph` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_graph_reachable_volume_oracle_matrix` |
| `test` | `sweep_graph_scc_decompose_volume_oracle_matrix` | `vyre-primitives/tests/sweep_graph_scc_decompose_volume_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_graph_scc_decompose_volume_oracle_matrix` |
| `test` | `sweep_graph_scc_decompose_volume_oracle_matrix` | `vyre-primitives/tests/sweep_graph_scc_decompose_volume_oracle_matrix.rs` | `cpu-parity`, `graph` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_graph_scc_decompose_volume_oracle_matrix` |
| `test` | `sweep_hash_adler32_volume_oracle_matrix` | `vyre-primitives/tests/sweep_hash_adler32_volume_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_hash_adler32_volume_oracle_matrix` |
| `test` | `sweep_hash_adler32_volume_oracle_matrix` | `vyre-primitives/tests/sweep_hash_adler32_volume_oracle_matrix.rs` | `hash` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_hash_adler32_volume_oracle_matrix` |
| `test` | `sweep_hash_blake3_g_volume_oracle_matrix` | `vyre-primitives/tests/sweep_hash_blake3_g_volume_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_hash_blake3_g_volume_oracle_matrix` |
| `test` | `sweep_hash_blake3_g_volume_oracle_matrix` | `vyre-primitives/tests/sweep_hash_blake3_g_volume_oracle_matrix.rs` | `cpu-parity`, `hash` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_hash_blake3_g_volume_oracle_matrix` |
| `test` | `sweep_hash_blake3_round_volume_oracle_matrix` | `vyre-primitives/tests/sweep_hash_blake3_round_volume_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_hash_blake3_round_volume_oracle_matrix` |
| `test` | `sweep_hash_blake3_round_volume_oracle_matrix` | `vyre-primitives/tests/sweep_hash_blake3_round_volume_oracle_matrix.rs` | `cpu-parity`, `hash` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_hash_blake3_round_volume_oracle_matrix` |
| `test` | `sweep_hash_crc32_volume_oracle_matrix` | `vyre-primitives/tests/sweep_hash_crc32_volume_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_hash_crc32_volume_oracle_matrix` |
| `test` | `sweep_hash_crc32_volume_oracle_matrix` | `vyre-primitives/tests/sweep_hash_crc32_volume_oracle_matrix.rs` | `hash` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_hash_crc32_volume_oracle_matrix` |
| `test` | `sweep_hash_crc_oracle_matrix` | `vyre-primitives/tests/sweep_hash_crc_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_hash_crc_oracle_matrix` |
| `test` | `sweep_hash_crc_oracle_matrix` | `vyre-primitives/tests/sweep_hash_crc_oracle_matrix.rs` | `hash` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_hash_crc_oracle_matrix` |
| `test` | `sweep_hash_fnv1a_oracle_matrix` | `vyre-primitives/tests/sweep_hash_fnv1a_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_hash_fnv1a_oracle_matrix` |
| `test` | `sweep_hash_fnv1a_oracle_matrix` | `vyre-primitives/tests/sweep_hash_fnv1a_oracle_matrix.rs` | `hash` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_hash_fnv1a_oracle_matrix` |
| `test` | `sweep_hash_multi_hash_volume_oracle_matrix` | `vyre-primitives/tests/sweep_hash_multi_hash_volume_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_hash_multi_hash_volume_oracle_matrix` |
| `test` | `sweep_hash_multi_hash_volume_oracle_matrix` | `vyre-primitives/tests/sweep_hash_multi_hash_volume_oracle_matrix.rs` | `hash` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_hash_multi_hash_volume_oracle_matrix` |
| `test` | `sweep_hash_volume_oracle_matrix` | `vyre-primitives/tests/sweep_hash_volume_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_hash_volume_oracle_matrix` |
| `test` | `sweep_hash_volume_oracle_matrix` | `vyre-primitives/tests/sweep_hash_volume_oracle_matrix.rs` | `hash` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_hash_volume_oracle_matrix` |
| `test` | `sweep_math_prefix_scan_exclusive_volume_oracle_matrix` | `vyre-primitives/tests/sweep_math_prefix_scan_exclusive_volume_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_math_prefix_scan_exclusive_volume_oracle_matrix` |
| `test` | `sweep_math_prefix_scan_exclusive_volume_oracle_matrix` | `vyre-primitives/tests/sweep_math_prefix_scan_exclusive_volume_oracle_matrix.rs` | `cpu-parity`, `math` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_math_prefix_scan_exclusive_volume_oracle_matrix` |
| `test` | `sweep_math_prefix_scan_inclusive_volume_oracle_matrix` | `vyre-primitives/tests/sweep_math_prefix_scan_inclusive_volume_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_math_prefix_scan_inclusive_volume_oracle_matrix` |
| `test` | `sweep_math_prefix_scan_inclusive_volume_oracle_matrix` | `vyre-primitives/tests/sweep_math_prefix_scan_inclusive_volume_oracle_matrix.rs` | `cpu-parity`, `math` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_math_prefix_scan_inclusive_volume_oracle_matrix` |
| `test` | `sweep_predicate_node_kind_oracle_matrix` | `vyre-primitives/tests/sweep_predicate_node_kind_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_predicate_node_kind_oracle_matrix` |
| `test` | `sweep_predicate_node_kind_oracle_matrix` | `vyre-primitives/tests/sweep_predicate_node_kind_oracle_matrix.rs` | `cpu-parity`, `predicate` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_predicate_node_kind_oracle_matrix` |
| `test` | `sweep_radix_sort_oracle_matrix` | `vyre-primitives/tests/sweep_radix_sort_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_radix_sort_oracle_matrix` |
| `test` | `sweep_radix_sort_oracle_matrix` | `vyre-primitives/tests/sweep_radix_sort_oracle_matrix.rs` | `cpu-parity`, `reduce` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_radix_sort_oracle_matrix` |
| `test` | `sweep_reduce_oracle_matrix` | `vyre-primitives/tests/sweep_reduce_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_reduce_oracle_matrix` |
| `test` | `sweep_reduce_oracle_matrix` | `vyre-primitives/tests/sweep_reduce_oracle_matrix.rs` | `cpu-parity`, `reduce` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_reduce_oracle_matrix` |
| `test` | `sweep_segment_reduce_oracle_matrix` | `vyre-primitives/tests/sweep_segment_reduce_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_segment_reduce_oracle_matrix` |
| `test` | `sweep_segment_reduce_oracle_matrix` | `vyre-primitives/tests/sweep_segment_reduce_oracle_matrix.rs` | `cpu-parity`, `reduce` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_segment_reduce_oracle_matrix` |
| `test` | `sweep_toposort_oracle_matrix` | `vyre-primitives/tests/sweep_toposort_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_toposort_oracle_matrix` |
| `test` | `sweep_toposort_oracle_matrix` | `vyre-primitives/tests/sweep_toposort_oracle_matrix.rs` | `cpu-parity`, `graph` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test sweep_toposort_oracle_matrix` |
| `test` | `symmetric_eigen_jacobi_parity` | `vyre-primitives/tests/symmetric_eigen_jacobi_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test symmetric_eigen_jacobi_parity` |
| `test` | `symmetric_eigen_jacobi_registration` | `vyre-primitives/tests/symmetric_eigen_jacobi_registration.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test symmetric_eigen_jacobi_registration` |
| `test` | `syntax_motif_frontier_compiler` | `vyre-primitives/tests/syntax_motif_frontier_compiler.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test syntax_motif_frontier_compiler` |
| `test` | `tensor_scc_value_parity` | `vyre-primitives/tests/tensor_scc_value_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test tensor_scc_value_parity` |
| `test` | `tensor_train_contract_signed_parity` | `vyre-primitives/tests/tensor_train_contract_signed_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test tensor_train_contract_signed_parity` |
| `test` | `tensor_train_decompose_eigen_contract` | `vyre-primitives/tests/tensor_train_decompose_eigen_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test tensor_train_decompose_eigen_contract` |
| `test` | `tensor_train_decompose_step_parity` | `vyre-primitives/tests/tensor_train_decompose_step_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test tensor_train_decompose_step_parity` |
| `test` | `text_char_class_support` | `vyre-primitives/tests/text_char_class_support.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test text_char_class_support` |
| `test` | `tfn_scalar_mix_signed_parity` | `vyre-primitives/tests/tfn_scalar_mix_signed_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test tfn_scalar_mix_signed_parity` |
| `test` | `toposort_program_value_parity` | `vyre-primitives/tests/toposort_program_value_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test toposort_program_value_parity` |
| `test` | `union_find_connectivity_parity` | `vyre-primitives/tests/union_find_connectivity_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test union_find_connectivity_parity` |
| `test` | `utf8_shape_counts_ir_parity_proptest` | `vyre-primitives/tests/utf8_shape_counts_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test utf8_shape_counts_ir_parity_proptest` |
| `test` | `vast_tree_walk_ir_parity_proptest` | `vyre-primitives/tests/vast_tree_walk_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test vast_tree_walk_ir_parity_proptest` |
| `test` | `wire_differential_std_io` | `vyre-primitives/tests/wire_differential_std_io.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test wire_differential_std_io` |
| `test` | `wire_harness_smoke_test` | `vyre-primitives/tests/wire_harness_smoke_test.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test wire_harness_smoke_test` |
| `test` | `wire_pack_into_contracts` | `vyre-primitives/tests/wire_pack_into_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test wire_pack_into_contracts` |
| `test` | `workgroup_any_ir_parity_proptest` | `vyre-primitives/tests/workgroup_any_ir_parity_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-primitives --test workgroup_any_ir_parity_proptest` |

## Test classes

- Primitive builder semantics
- Reference and backend parity
- Boundary, property, and composition contracts

## Hardware requirements

Builder and reference suites are host-capable. Concrete backend parity tests require the selected device and fail visibly when unavailable.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
