# Testing `vyre-libs`

Run the default crate suite from the workspace root:

```console
./cargo_full test -p vyre-libs
```

Own every composition in the workspace: consumer dialects and compiler-internal solvers, encoding, analysis, scheduling, and reasoning. Returns Programs. No backend, no emitter, no host rewrite of IR.

The crate lives at `vyre-libs`. The `product-libraries` owner maintains its
`libraries` testing contract.

## Commands

```console
./cargo_full test -p vyre-libs
```

```console
./cargo_full test -p vyre-libs --all-features
```

## Feature sets

- Default feature members: `math-linalg`, `math-scan`, `math-broadcast`, `nn-activation`, `nn-linear`, `nn-norm`, `matching-substring`, `matching-dfa`, `hash`, `decode`
- Available manifest features: `analysis`, `bitset`, `cat-a-builder-options`, `cpu-parity`, `crypto`, `crypto-blake3`, `decode`, `default`, `device`, `encoding`, `fixpoint`, `full`, `geom`, `go-parser`, `graph`, `graph-dispatch`, `hash`, `intern`, `label`, `logical`, `matching`, `matching-dfa`, `matching-kernels`, `matching-nfa`, `matching-regex`, `matching-substring`, `math`, `math-algebra`, `math-broadcast`, `math-dialect`, `math-kernels`, `math-linalg`, `math-scan`, `math-succinct`, `nfa`, `nn`, `nn-activation`, `nn-attention`, `nn-inference`, `nn-kernels`, `nn-linear`, `nn-linear-4bit`, `nn-moe`, `nn-norm`, `opt`, `parsing`, `parsing-kernels`, `predicate`, `python-parser`, `reasoning`, `reduce`, `rule`, `scheduling`, `security`, `solvers`, `telemetry`, `text`, `topology`, `visual`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `dominator_tree_e2e` | `vyre-libs/examples/dominator_tree_e2e.rs` | None | `./cargo_full test -p vyre-libs --example dominator_tree_e2e` |
| `example` | `dominator_tree_e2e` | `vyre-libs/examples/dominator_tree_e2e.rs` | `cpu-parity`, `graph` | `./cargo_full test -p vyre-libs --example dominator_tree_e2e` |
| `example` | `prefix_sum_megakernel` | `vyre-libs/examples/prefix_sum_megakernel.rs` | None | `./cargo_full test -p vyre-libs --example prefix_sum_megakernel` |
| `example` | `prefix_sum_megakernel` | `vyre-libs/examples/prefix_sum_megakernel.rs` | `math-scan` | `./cargo_full test -p vyre-libs --example prefix_sum_megakernel` |
| `example` | `select1_optimizer_parity` | `vyre-libs/examples/select1_optimizer_parity.rs` | None | `./cargo_full test -p vyre-libs --example select1_optimizer_parity` |
| `lib` | `vyre_libs` | `vyre-libs/src/lib.rs` | None | `./cargo_full test -p vyre-libs` |
| `test` | `ac_count_suffix3_naga_validation` | `vyre-libs/tests/ac_count_suffix3_naga_validation.rs` | None | `./cargo_full test -p vyre-libs --test ac_count_suffix3_naga_validation` |
| `test` | `adaptive_four_russians_dense_generated` | `vyre-libs/tests/adaptive_four_russians_dense_generated.rs` | None | `./cargo_full test -p vyre-libs --test adaptive_four_russians_dense_generated` |
| `test` | `adaptive_four_russians_dense_generated` | `vyre-libs/tests/adaptive_four_russians_dense_generated.rs` | `cpu-parity`, `graph` | `./cargo_full test -p vyre-libs --test adaptive_four_russians_dense_generated` |
| `test` | `adversarial` | `vyre-libs/tests/adversarial.rs` | None | `./cargo_full test -p vyre-libs --test adversarial` |
| `test` | `adversarial_bitset_contains` | `vyre-libs/tests/adversarial_bitset_contains.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_bitset_contains` |
| `test` | `adversarial_bitset_contains` | `vyre-libs/tests/adversarial_bitset_contains.rs` | `bitset`, `cpu-parity` | `./cargo_full test -p vyre-libs --test adversarial_bitset_contains` |
| `test` | `adversarial_bitset_ops` | `vyre-libs/tests/adversarial_bitset_ops.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_bitset_ops` |
| `test` | `adversarial_bitset_ops` | `vyre-libs/tests/adversarial_bitset_ops.rs` | `bitset`, `cpu-parity` | `./cargo_full test -p vyre-libs --test adversarial_bitset_ops` |
| `test` | `adversarial_bitset_reduce_matrix` | `vyre-libs/tests/adversarial_bitset_reduce_matrix.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_bitset_reduce_matrix` |
| `test` | `adversarial_bitset_reduce_matrix` | `vyre-libs/tests/adversarial_bitset_reduce_matrix.rs` | `bitset`, `cpu-parity` | `./cargo_full test -p vyre-libs --test adversarial_bitset_reduce_matrix` |
| `test` | `adversarial_boolean_packing_four_russians_readiness` | `vyre-libs/tests/adversarial_boolean_packing_four_russians_readiness.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_boolean_packing_four_russians_readiness` |
| `test` | `adversarial_boolean_packing_four_russians_readiness` | `vyre-libs/tests/adversarial_boolean_packing_four_russians_readiness.rs` | `bitset`, `cpu-parity` | `./cargo_full test -p vyre-libs --test adversarial_boolean_packing_four_russians_readiness` |
| `test` | `adversarial_decode` | `vyre-libs/tests/adversarial_decode.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_decode` |
| `test` | `adversarial_fixpoint` | `vyre-libs/tests/adversarial_fixpoint.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_fixpoint` |
| `test` | `adversarial_frontier_queue_clear` | `vyre-libs/tests/adversarial_frontier_queue_clear.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_frontier_queue_clear` |
| `test` | `adversarial_graph` | `vyre-libs/tests/adversarial_graph.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_graph` |
| `test` | `adversarial_graph_csr_validation_contracts` | `vyre-libs/tests/adversarial_graph_csr_validation_contracts.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_graph_csr_validation_contracts` |
| `test` | `adversarial_graph_csr_validation_contracts` | `vyre-libs/tests/adversarial_graph_csr_validation_contracts.rs` | `graph` | `./cargo_full test -p vyre-libs --test adversarial_graph_csr_validation_contracts` |
| `test` | `adversarial_graph_ops` | `vyre-libs/tests/adversarial_graph_ops.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_graph_ops` |
| `test` | `adversarial_graph_ops` | `vyre-libs/tests/adversarial_graph_ops.rs` | `cpu-parity`, `graph` | `./cargo_full test -p vyre-libs --test adversarial_graph_ops` |
| `test` | `adversarial_graph_reachability_fixpoint` | `vyre-libs/tests/adversarial_graph_reachability_fixpoint.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_graph_reachability_fixpoint` |
| `test` | `adversarial_graph_reachability_fixpoint` | `vyre-libs/tests/adversarial_graph_reachability_fixpoint.rs` | `cpu-parity`, `fixpoint`, `graph`, `math-kernels` | `./cargo_full test -p vyre-libs --test adversarial_graph_reachability_fixpoint` |
| `test` | `adversarial_hash` | `vyre-libs/tests/adversarial_hash.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_hash` |
| `test` | `adversarial_label` | `vyre-libs/tests/adversarial_label.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_label` |
| `test` | `adversarial_matching` | `vyre-libs/tests/adversarial_matching.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_matching` |
| `test` | `adversarial_math` | `vyre-libs/tests/adversarial_math.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_math` |
| `test` | `adversarial_math` | `vyre-libs/tests/adversarial_math.rs` | `cpu-parity`, `math-kernels` | `./cargo_full test -p vyre-libs --test adversarial_math` |
| `test` | `adversarial_nfa` | `vyre-libs/tests/adversarial_nfa.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_nfa` |
| `test` | `adversarial_reduce_gather` | `vyre-libs/tests/adversarial_reduce_gather.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_reduce_gather` |
| `test` | `adversarial_reduce_gather` | `vyre-libs/tests/adversarial_reduce_gather.rs` | `reduce` | `./cargo_full test -p vyre-libs --test adversarial_reduce_gather` |
| `test` | `adversarial_reduce_histogram` | `vyre-libs/tests/adversarial_reduce_histogram.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_reduce_histogram` |
| `test` | `adversarial_reduce_histogram` | `vyre-libs/tests/adversarial_reduce_histogram.rs` | `reduce` | `./cargo_full test -p vyre-libs --test adversarial_reduce_histogram` |
| `test` | `adversarial_reduce_radix_sort` | `vyre-libs/tests/adversarial_reduce_radix_sort.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_reduce_radix_sort` |
| `test` | `adversarial_reduce_radix_sort` | `vyre-libs/tests/adversarial_reduce_radix_sort.rs` | `reduce` | `./cargo_full test -p vyre-libs --test adversarial_reduce_radix_sort` |
| `test` | `adversarial_reduce_scatter` | `vyre-libs/tests/adversarial_reduce_scatter.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_reduce_scatter` |
| `test` | `adversarial_reduce_scatter` | `vyre-libs/tests/adversarial_reduce_scatter.rs` | `reduce` | `./cargo_full test -p vyre-libs --test adversarial_reduce_scatter` |
| `test` | `adversarial_reduce_segment_reduce` | `vyre-libs/tests/adversarial_reduce_segment_reduce.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_reduce_segment_reduce` |
| `test` | `adversarial_reduce_segment_reduce` | `vyre-libs/tests/adversarial_reduce_segment_reduce.rs` | `reduce` | `./cargo_full test -p vyre-libs --test adversarial_reduce_segment_reduce` |
| `test` | `adversarial_text_byte_histogram` | `vyre-libs/tests/adversarial_text_byte_histogram.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_text_byte_histogram` |
| `test` | `adversarial_text_byte_histogram` | `vyre-libs/tests/adversarial_text_byte_histogram.rs` | `cpu-parity`, `text` | `./cargo_full test -p vyre-libs --test adversarial_text_byte_histogram` |
| `test` | `adversarial_text_char_class` | `vyre-libs/tests/adversarial_text_char_class.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_text_char_class` |
| `test` | `adversarial_text_char_class` | `vyre-libs/tests/adversarial_text_char_class.rs` | `text` | `./cargo_full test -p vyre-libs --test adversarial_text_char_class` |
| `test` | `adversarial_text_line_index` | `vyre-libs/tests/adversarial_text_line_index.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_text_line_index` |
| `test` | `adversarial_text_line_index` | `vyre-libs/tests/adversarial_text_line_index.rs` | `text` | `./cargo_full test -p vyre-libs --test adversarial_text_line_index` |
| `test` | `adversarial_text_utf8_shape_counts` | `vyre-libs/tests/adversarial_text_utf8_shape_counts.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_text_utf8_shape_counts` |
| `test` | `adversarial_text_utf8_shape_counts` | `vyre-libs/tests/adversarial_text_utf8_shape_counts.rs` | `cpu-parity`, `text` | `./cargo_full test -p vyre-libs --test adversarial_text_utf8_shape_counts` |
| `test` | `adversarial_text_utf8_validate` | `vyre-libs/tests/adversarial_text_utf8_validate.rs` | None | `./cargo_full test -p vyre-libs --test adversarial_text_utf8_validate` |
| `test` | `adversarial_text_utf8_validate` | `vyre-libs/tests/adversarial_text_utf8_validate.rs` | `text` | `./cargo_full test -p vyre-libs --test adversarial_text_utf8_validate` |
| `test` | `aho_corasick_kat` | `vyre-libs/tests/aho_corasick_kat.rs` | None | `./cargo_full test -p vyre-libs --test aho_corasick_kat` |
| `test` | `algebra_lattice_semiring_contracts` | `vyre-libs/tests/algebra_lattice_semiring_contracts.rs` | None | `./cargo_full test -p vyre-libs --test algebra_lattice_semiring_contracts` |
| `test` | `amg_v_cycle_ir_parity` | `vyre-libs/tests/amg_v_cycle_ir_parity.rs` | None | `./cargo_full test -p vyre-libs --test amg_v_cycle_ir_parity` |
| `test` | `analysis_fact_schema` | `vyre-libs/tests/analysis_fact_schema.rs` | None | `./cargo_full test -p vyre-libs --test analysis_fact_schema` |
| `test` | `arg_of_slot_precision` | `vyre-libs/tests/arg_of_slot_precision.rs` | None | `./cargo_full test -p vyre-libs --test arg_of_slot_precision` |
| `test` | `arg_of_slot_precision` | `vyre-libs/tests/arg_of_slot_precision.rs` | `cpu-parity`, `predicate` | `./cargo_full test -p vyre-libs --test arg_of_slot_precision` |
| `test` | `argmax_of_marginals_ir_parity_proptest` | `vyre-libs/tests/argmax_of_marginals_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test argmax_of_marginals_ir_parity_proptest` |
| `test` | `ast_shunting_yard` | `vyre-libs/tests/ast_shunting_yard.rs` | None | `./cargo_full test -p vyre-libs --test ast_shunting_yard` |
| `test` | `attention_head_to_token_contract` | `vyre-libs/tests/attention_head_to_token_contract.rs` | None | `./cargo_full test -p vyre-libs --test attention_head_to_token_contract` |
| `test` | `attention_head_to_token_contract` | `vyre-libs/tests/attention_head_to_token_contract.rs` | `nn-attention` | `./cargo_full test -p vyre-libs --test attention_head_to_token_contract` |
| `test` | `bellman_oob_edge_parity` | `vyre-libs/tests/bellman_oob_edge_parity.rs` | None | `./cargo_full test -p vyre-libs --test bellman_oob_edge_parity` |
| `test` | `bellman_shortest_path_via_reference_parity` | `vyre-libs/tests/bellman_shortest_path_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test bellman_shortest_path_via_reference_parity` |
| `test` | `bellman_shortest_path_via_reference_parity` | `vyre-libs/tests/bellman_shortest_path_via_reference_parity.rs` | `cpu-parity`, `solvers` | `./cargo_full test -p vyre-libs --test bellman_shortest_path_via_reference_parity` |
| `test` | `bigint_add_carry_ir_parity_proptest` | `vyre-libs/tests/bigint_add_carry_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test bigint_add_carry_ir_parity_proptest` |
| `test` | `bitset_dense_matvec_pipeline_generated` | `vyre-libs/tests/bitset_dense_matvec_pipeline_generated.rs` | None | `./cargo_full test -p vyre-libs --test bitset_dense_matvec_pipeline_generated` |
| `test` | `bitset_dense_matvec_pipeline_generated` | `vyre-libs/tests/bitset_dense_matvec_pipeline_generated.rs` | `cpu-parity`, `encoding` | `./cargo_full test -p vyre-libs --test bitset_dense_matvec_pipeline_generated` |
| `test` | `bitset_fixpoint_warm_start_parity` | `vyre-libs/tests/bitset_fixpoint_warm_start_parity.rs` | None | `./cargo_full test -p vyre-libs --test bitset_fixpoint_warm_start_parity` |
| `test` | `bitset_law_properties` | `vyre-libs/tests/bitset_law_properties.rs` | None | `./cargo_full test -p vyre-libs --test bitset_law_properties` |
| `test` | `bitset_mask_algebra_via_reference_parity` | `vyre-libs/tests/bitset_mask_algebra_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test bitset_mask_algebra_via_reference_parity` |
| `test` | `bitset_mask_algebra_via_reference_parity` | `vyre-libs/tests/bitset_mask_algebra_via_reference_parity.rs` | `cpu-parity`, `encoding` | `./cargo_full test -p vyre-libs --test bitset_mask_algebra_via_reference_parity` |
| `test` | `bitset_scalar_ir_parity_proptest` | `vyre-libs/tests/bitset_scalar_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test bitset_scalar_ir_parity_proptest` |
| `test` | `bitset_summary_via_reference_parity` | `vyre-libs/tests/bitset_summary_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test bitset_summary_via_reference_parity` |
| `test` | `bitset_summary_via_reference_parity` | `vyre-libs/tests/bitset_summary_via_reference_parity.rs` | `cpu-parity`, `encoding` | `./cargo_full test -p vyre-libs --test bitset_summary_via_reference_parity` |
| `test` | `bitset_word_contracts` | `vyre-libs/tests/bitset_word_contracts.rs` | None | `./cargo_full test -p vyre-libs --test bitset_word_contracts` |
| `test` | `bitset_word_contracts` | `vyre-libs/tests/bitset_word_contracts.rs` | `bitset`, `cpu-parity` | `./cargo_full test -p vyre-libs --test bitset_word_contracts` |
| `test` | `bitset_words_sizing_contracts` | `vyre-libs/tests/bitset_words_sizing_contracts.rs` | None | `./cargo_full test -p vyre-libs --test bitset_words_sizing_contracts` |
| `test` | `blake3_compress_optimizer_idempotence_contract` | `vyre-libs/tests/blake3_compress_optimizer_idempotence_contract.rs` | None | `./cargo_full test -p vyre-libs --test blake3_compress_optimizer_idempotence_contract` |
| `test` | `blake3_kat` | `vyre-libs/tests/blake3_kat.rs` | None | `./cargo_full test -p vyre-libs --test blake3_kat` |
| `test` | `blake3_program` | `vyre-libs/tests/blake3_program.rs` | None | `./cargo_full test -p vyre-libs --test blake3_program` |
| `test` | `blake3_wrong_size` | `vyre-libs/tests/blake3_wrong_size.rs` | None | `./cargo_full test -p vyre-libs --test blake3_wrong_size` |
| `test` | `bracket_match_proptest` | `vyre-libs/tests/bracket_match_proptest.rs` | None | `./cargo_full test -p vyre-libs --test bracket_match_proptest` |
| `test` | `buffer_name_cross_family` | `vyre-libs/tests/buffer_name_cross_family.rs` | None | `./cargo_full test -p vyre-libs --test buffer_name_cross_family` |
| `test` | `cat_a_conform` | `vyre-libs/tests/cat_a_conform.rs` | None | `./cargo_full test -p vyre-libs --test cat_a_conform` |
| `test` | `categorical_laws_proptest` | `vyre-libs/tests/categorical_laws_proptest.rs` | None | `./cargo_full test -p vyre-libs --test categorical_laws_proptest` |
| `test` | `categorical_laws_proptest` | `vyre-libs/tests/categorical_laws_proptest.rs` | `cpu-parity`, `reasoning` | `./cargo_full test -p vyre-libs --test categorical_laws_proptest` |
| `test` | `causal_conv_state_transition_contract` | `vyre-libs/tests/causal_conv_state_transition_contract.rs` | None | `./cargo_full test -p vyre-libs --test causal_conv_state_transition_contract` |
| `test` | `causal_conv_state_transition_contract` | `vyre-libs/tests/causal_conv_state_transition_contract.rs` | `nn-inference` | `./cargo_full test -p vyre-libs --test causal_conv_state_transition_contract` |
| `test` | `causal_gqa_contract` | `vyre-libs/tests/causal_gqa_contract.rs` | None | `./cargo_full test -p vyre-libs --test causal_gqa_contract` |
| `test` | `causal_gqa_contract` | `vyre-libs/tests/causal_gqa_contract.rs` | `nn-attention` | `./cargo_full test -p vyre-libs --test causal_gqa_contract` |
| `test` | `causal_gqa_typed_contract` | `vyre-libs/tests/causal_gqa_typed_contract.rs` | None | `./cargo_full test -p vyre-libs --test causal_gqa_typed_contract` |
| `test` | `causal_gqa_typed_contract` | `vyre-libs/tests/causal_gqa_typed_contract.rs` | `nn-attention` | `./cargo_full test -p vyre-libs --test causal_gqa_typed_contract` |
| `test` | `chunked_gated_delta_contract` | `vyre-libs/tests/chunked_gated_delta_contract.rs` | None | `./cargo_full test -p vyre-libs --test chunked_gated_delta_contract` |
| `test` | `chunked_gated_delta_contract` | `vyre-libs/tests/chunked_gated_delta_contract.rs` | `nn-attention` | `./cargo_full test -p vyre-libs --test chunked_gated_delta_contract` |
| `test` | `clifford_geometric_product_program_parity` | `vyre-libs/tests/clifford_geometric_product_program_parity.rs` | None | `./cargo_full test -p vyre-libs --test clifford_geometric_product_program_parity` |
| `test` | `consumer_boundary` | `vyre-libs/tests/consumer_boundary.rs` | None | `./cargo_full test -p vyre-libs --test consumer_boundary` |
| `test` | `corpus_privacy_retention_controls` | `vyre-libs/tests/corpus_privacy_retention_controls.rs` | None | `./cargo_full test -p vyre-libs --test corpus_privacy_retention_controls` |
| `test` | `cost_model_predict_runtime_via_reference_parity` | `vyre-libs/tests/cost_model_predict_runtime_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test cost_model_predict_runtime_via_reference_parity` |
| `test` | `cost_model_predict_runtime_via_reference_parity` | `vyre-libs/tests/cost_model_predict_runtime_via_reference_parity.rs` | `analysis`, `cpu-parity` | `./cargo_full test -p vyre-libs --test cost_model_predict_runtime_via_reference_parity` |
| `test` | `cpu_witnesses` | `vyre-libs/tests/cpu_witnesses.rs` | None | `./cargo_full test -p vyre-libs --test cpu_witnesses` |
| `test` | `crc32_map_reduce_generated` | `vyre-libs/tests/crc32_map_reduce_generated.rs` | None | `./cargo_full test -p vyre-libs --test crc32_map_reduce_generated` |
| `test` | `csr_backward_or_changed_ir_fixpoint` | `vyre-libs/tests/csr_backward_or_changed_ir_fixpoint.rs` | None | `./cargo_full test -p vyre-libs --test csr_backward_or_changed_ir_fixpoint` |
| `test` | `csr_backward_traverse_ir_parity_proptest` | `vyre-libs/tests/csr_backward_traverse_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test csr_backward_traverse_ir_parity_proptest` |
| `test` | `csr_certificates` | `vyre-libs/tests/csr_certificates.rs` | None | `./cargo_full test -p vyre-libs --test csr_certificates` |
| `test` | `csr_closure_argument_bundle_gate` | `vyre-libs/tests/csr_closure_argument_bundle_gate.rs` | None | `./cargo_full test -p vyre-libs --test csr_closure_argument_bundle_gate` |
| `test` | `csr_forward_traverse_ir_parity_proptest` | `vyre-libs/tests/csr_forward_traverse_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test csr_forward_traverse_ir_parity_proptest` |
| `test` | `csr_frontier_degree_sum_ir_parity_proptest` | `vyre-libs/tests/csr_frontier_degree_sum_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test csr_frontier_degree_sum_ir_parity_proptest` |
| `test` | `csr_queue_strided_ir_parity_proptest` | `vyre-libs/tests/csr_queue_strided_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test csr_queue_strided_ir_parity_proptest` |
| `test` | `csr_sweep_arm_coverage` | `vyre-libs/tests/csr_sweep_arm_coverage.rs` | None | `./cargo_full test -p vyre-libs --test csr_sweep_arm_coverage` |
| `test` | `csr_traversal_clone_family_equality` | `vyre-libs/tests/csr_traversal_clone_family_equality.rs` | None | `./cargo_full test -p vyre-libs --test csr_traversal_clone_family_equality` |
| `test` | `decode_primitive_composition_contracts` | `vyre-libs/tests/decode_primitive_composition_contracts.rs` | None | `./cargo_full test -p vyre-libs --test decode_primitive_composition_contracts` |
| `test` | `dedup_conv_ast_walk_family_guard` | `vyre-libs/tests/dedup_conv_ast_walk_family_guard.rs` | None | `./cargo_full test -p vyre-libs --test dedup_conv_ast_walk_family_guard` |
| `test` | `delegating_builder_equivalence` | `vyre-libs/tests/delegating_builder_equivalence.rs` | None | `./cargo_full test -p vyre-libs --test delegating_builder_equivalence` |
| `test` | `delta_flow_arrangements` | `vyre-libs/tests/delta_flow_arrangements.rs` | None | `./cargo_full test -p vyre-libs --test delta_flow_arrangements` |
| `test` | `dense_gated_mlp_graph_contract` | `vyre-libs/tests/dense_gated_mlp_graph_contract.rs` | None | `./cargo_full test -p vyre-libs --test dense_gated_mlp_graph_contract` |
| `test` | `dense_gated_mlp_graph_contract` | `vyre-libs/tests/dense_gated_mlp_graph_contract.rs` | `nn-inference` | `./cargo_full test -p vyre-libs --test dense_gated_mlp_graph_contract` |
| `test` | `depthwise_causal_conv1d_contract` | `vyre-libs/tests/depthwise_causal_conv1d_contract.rs` | None | `./cargo_full test -p vyre-libs --test depthwise_causal_conv1d_contract` |
| `test` | `depthwise_causal_conv1d_contract` | `vyre-libs/tests/depthwise_causal_conv1d_contract.rs` | `nn-inference` | `./cargo_full test -p vyre-libs --test depthwise_causal_conv1d_contract` |
| `test` | `device_resident_token_fact_graph_ownership` | `vyre-libs/tests/device_resident_token_fact_graph_ownership.rs` | None | `./cargo_full test -p vyre-libs --test device_resident_token_fact_graph_ownership` |
| `test` | `dfa_wire_contracts` | `vyre-libs/tests/dfa_wire_contracts.rs` | None | `./cargo_full test -p vyre-libs --test dfa_wire_contracts` |
| `test` | `do_calculus_rule2_value_parity` | `vyre-libs/tests/do_calculus_rule2_value_parity.rs` | None | `./cargo_full test -p vyre-libs --test do_calculus_rule2_value_parity` |
| `test` | `do_calculus_surgery_via_reference_parity` | `vyre-libs/tests/do_calculus_surgery_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test do_calculus_surgery_via_reference_parity` |
| `test` | `do_calculus_surgery_via_reference_parity` | `vyre-libs/tests/do_calculus_surgery_via_reference_parity.rs` | `cpu-parity`, `reasoning` | `./cargo_full test -p vyre-libs --test do_calculus_surgery_via_reference_parity` |
| `test` | `dominator_tree_composition` | `vyre-libs/tests/dominator_tree_composition.rs` | None | `./cargo_full test -p vyre-libs --test dominator_tree_composition` |
| `test` | `dominator_tree_pristine` | `vyre-libs/tests/dominator_tree_pristine.rs` | None | `./cargo_full test -p vyre-libs --test dominator_tree_pristine` |
| `test` | `dominator_tree_pristine` | `vyre-libs/tests/dominator_tree_pristine.rs` | `cpu-parity`, `graph` | `./cargo_full test -p vyre-libs --test dominator_tree_pristine` |
| `test` | `dominator_tree_proptest` | `vyre-libs/tests/dominator_tree_proptest.rs` | None | `./cargo_full test -p vyre-libs --test dominator_tree_proptest` |
| `test` | `dominator_tree_proptest` | `vyre-libs/tests/dominator_tree_proptest.rs` | `cpu-parity`, `graph` | `./cargo_full test -p vyre-libs --test dominator_tree_proptest` |
| `test` | `dominator_tree_scale_gate` | `vyre-libs/tests/dominator_tree_scale_gate.rs` | None | `./cargo_full test -p vyre-libs --test dominator_tree_scale_gate` |
| `test` | `dominator_tree_scale_gate` | `vyre-libs/tests/dominator_tree_scale_gate.rs` | `cpu-parity`, `graph` | `./cargo_full test -p vyre-libs --test dominator_tree_scale_gate` |
| `test` | `dp_clip_signed_newton_parity` | `vyre-libs/tests/dp_clip_signed_newton_parity.rs` | None | `./cargo_full test -p vyre-libs --test dp_clip_signed_newton_parity` |
| `test` | `f32_adversarial` | `vyre-libs/tests/f32_adversarial.rs` | None | `./cargo_full test -p vyre-libs --test f32_adversarial` |
| `test` | `family_duplication_budget` | `vyre-libs/tests/family_duplication_budget.rs` | None | `./cargo_full test -p vyre-libs --test family_duplication_budget` |
| `test` | `filesystem_path_archive_policies` | `vyre-libs/tests/filesystem_path_archive_policies.rs` | None | `./cargo_full test -p vyre-libs --test filesystem_path_archive_policies` |
| `test` | `fingerprint_lock` | `vyre-libs/tests/fingerprint_lock.rs` | None | `./cargo_full test -p vyre-libs --test fingerprint_lock` |
| `test` | `fingerprint_lock` | `vyre-libs/tests/fingerprint_lock.rs` | `nn-activation`, `nn-attention`, `nn-linear`, `nn-norm` | `./cargo_full test -p vyre-libs --test fingerprint_lock` |
| `test` | `flow_precision_planner` | `vyre-libs/tests/flow_precision_planner.rs` | None | `./cargo_full test -p vyre-libs --test flow_precision_planner` |
| `test` | `fmm_compress_pairwise_via_reference_parity` | `vyre-libs/tests/fmm_compress_pairwise_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test fmm_compress_pairwise_via_reference_parity` |
| `test` | `fmm_compress_pairwise_via_reference_parity` | `vyre-libs/tests/fmm_compress_pairwise_via_reference_parity.rs` | `cpu-parity`, `solvers` | `./cargo_full test -p vyre-libs --test fmm_compress_pairwise_via_reference_parity` |
| `test` | `fmm_polyhedral_via_reference_parity` | `vyre-libs/tests/fmm_polyhedral_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test fmm_polyhedral_via_reference_parity` |
| `test` | `fmm_polyhedral_via_reference_parity` | `vyre-libs/tests/fmm_polyhedral_via_reference_parity.rs` | `cpu-parity`, `solvers` | `./cargo_full test -p vyre-libs --test fmm_polyhedral_via_reference_parity` |
| `test` | `fmm_program_parity` | `vyre-libs/tests/fmm_program_parity.rs` | None | `./cargo_full test -p vyre-libs --test fmm_program_parity` |
| `test` | `fnv1a64_builder_parity` | `vyre-libs/tests/fnv1a64_builder_parity.rs` | None | `./cargo_full test -p vyre-libs --test fnv1a64_builder_parity` |
| `test` | `fnv1a64_builder_parity` | `vyre-libs/tests/fnv1a64_builder_parity.rs` | `hash` | `./cargo_full test -p vyre-libs --test fnv1a64_builder_parity` |
| `test` | `fnv1a_dyn_parity` | `vyre-libs/tests/fnv1a_dyn_parity.rs` | None | `./cargo_full test -p vyre-libs --test fnv1a_dyn_parity` |
| `test` | `four_russians_dense_matvec_generated` | `vyre-libs/tests/four_russians_dense_matvec_generated.rs` | None | `./cargo_full test -p vyre-libs --test four_russians_dense_matvec_generated` |
| `test` | `four_russians_dense_matvec_generated` | `vyre-libs/tests/four_russians_dense_matvec_generated.rs` | `bitset`, `cpu-parity` | `./cargo_full test -p vyre-libs --test four_russians_dense_matvec_generated` |
| `test` | `frontend_dialect_contracts` | `vyre-libs/tests/frontend_dialect_contracts.rs` | None | `./cargo_full test -p vyre-libs --test frontend_dialect_contracts` |
| `test` | `frontier_absorb_parity` | `vyre-libs/tests/frontier_absorb_parity.rs` | None | `./cargo_full test -p vyre-libs --test frontier_absorb_parity` |
| `test` | `frontier_load_balancing_policies` | `vyre-libs/tests/frontier_load_balancing_policies.rs` | None | `./cargo_full test -p vyre-libs --test frontier_load_balancing_policies` |
| `test` | `frontier_to_queue_multi_workgroup_span` | `vyre-libs/tests/frontier_to_queue_multi_workgroup_span.rs` | None | `./cargo_full test -p vyre-libs --test frontier_to_queue_multi_workgroup_span` |
| `test` | `functor_apply_ir_parity_proptest` | `vyre-libs/tests/functor_apply_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test functor_apply_ir_parity_proptest` |
| `test` | `functor_apply_via_reference_parity` | `vyre-libs/tests/functor_apply_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test functor_apply_via_reference_parity` |
| `test` | `functor_apply_via_reference_parity` | `vyre-libs/tests/functor_apply_via_reference_parity.rs` | `cpu-parity`, `reasoning` | `./cargo_full test -p vyre-libs --test functor_apply_via_reference_parity` |
| `test` | `fuse_decode_scan_error` | `vyre-libs/tests/fuse_decode_scan_error.rs` | None | `./cargo_full test -p vyre-libs --test fuse_decode_scan_error` |
| `test` | `fusion_scores_via_reference_parity` | `vyre-libs/tests/fusion_scores_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test fusion_scores_via_reference_parity` |
| `test` | `fusion_scores_via_reference_parity` | `vyre-libs/tests/fusion_scores_via_reference_parity.rs` | `cpu-parity`, `scheduling` | `./cargo_full test -p vyre-libs --test fusion_scores_via_reference_parity` |
| `test` | `fuzz_target_inventory` | `vyre-libs/tests/fuzz_target_inventory.rs` | None | `./cargo_full test -p vyre-libs --test fuzz_target_inventory` |
| `test` | `gated_rms_norm_contract` | `vyre-libs/tests/gated_rms_norm_contract.rs` | None | `./cargo_full test -p vyre-libs --test gated_rms_norm_contract` |
| `test` | `gated_rms_norm_contract` | `vyre-libs/tests/gated_rms_norm_contract.rs` | `nn-norm` | `./cargo_full test -p vyre-libs --test gated_rms_norm_contract` |
| `test` | `go_channel_creation_parity` | `vyre-libs/tests/go_channel_creation_parity.rs` | None | `./cargo_full test -p vyre-libs --test go_channel_creation_parity` |
| `test` | `go_frontend_corpus` | `vyre-libs/tests/go_frontend_corpus.rs` | None | `./cargo_full test -p vyre-libs --test go_frontend_corpus` |
| `test` | `go_tokenizer_semantics` | `vyre-libs/tests/go_tokenizer_semantics.rs` | None | `./cargo_full test -p vyre-libs --test go_tokenizer_semantics` |
| `test` | `gpu_columnar_string_ingress` | `vyre-libs/tests/gpu_columnar_string_ingress.rs` | None | `./cargo_full test -p vyre-libs --test gpu_columnar_string_ingress` |
| `test` | `gqa_attention_primitive_composition_contracts` | `vyre-libs/tests/gqa_attention_primitive_composition_contracts.rs` | None | `./cargo_full test -p vyre-libs --test gqa_attention_primitive_composition_contracts` |
| `test` | `gqa_attention_primitive_composition_contracts` | `vyre-libs/tests/gqa_attention_primitive_composition_contracts.rs` | `nn-attention` | `./cargo_full test -p vyre-libs --test gqa_attention_primitive_composition_contracts` |
| `test` | `graph_builders_emit_valid_ir` | `vyre-libs/tests/graph_builders_emit_valid_ir.rs` | None | `./cargo_full test -p vyre-libs --test graph_builders_emit_valid_ir` |
| `test` | `graph_fixpoint_adversarial_generated` | `vyre-libs/tests/graph_fixpoint_adversarial_generated.rs` | None | `./cargo_full test -p vyre-libs --test graph_fixpoint_adversarial_generated` |
| `test` | `graph_primitive_binding_contracts` | `vyre-libs/tests/graph_primitive_binding_contracts.rs` | None | `./cargo_full test -p vyre-libs --test graph_primitive_binding_contracts` |
| `test` | `graph_primitive_binding_contracts` | `vyre-libs/tests/graph_primitive_binding_contracts.rs` | `graph` | `./cargo_full test -p vyre-libs --test graph_primitive_binding_contracts` |
| `test` | `graph_single_source_contracts` | `vyre-libs/tests/graph_single_source_contracts.rs` | None | `./cargo_full test -p vyre-libs --test graph_single_source_contracts` |
| `test` | `graph_single_source_contracts` | `vyre-libs/tests/graph_single_source_contracts.rs` | `cpu-parity`, `graph-dispatch` | `./cargo_full test -p vyre-libs --test graph_single_source_contracts` |
| `test` | `hash_crc32_ir_parity_proptest` | `vyre-libs/tests/hash_crc32_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test hash_crc32_ir_parity_proptest` |
| `test` | `hash_incremental_adversarial_generated` | `vyre-libs/tests/hash_incremental_adversarial_generated.rs` | None | `./cargo_full test -p vyre-libs --test hash_incremental_adversarial_generated` |
| `test` | `hash_registration_witnesses` | `vyre-libs/tests/hash_registration_witnesses.rs` | None | `./cargo_full test -p vyre-libs --test hash_registration_witnesses` |
| `test` | `hash_registration_witnesses` | `vyre-libs/tests/hash_registration_witnesses.rs` | `hash` | `./cargo_full test -p vyre-libs --test hash_registration_witnesses` |
| `test` | `hash_stream_ir_parity_proptest` | `vyre-libs/tests/hash_stream_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test hash_stream_ir_parity_proptest` |
| `test` | `head_to_token_typed_contract` | `vyre-libs/tests/head_to_token_typed_contract.rs` | None | `./cargo_full test -p vyre-libs --test head_to_token_typed_contract` |
| `test` | `head_to_token_typed_contract` | `vyre-libs/tests/head_to_token_typed_contract.rs` | `nn-attention` | `./cargo_full test -p vyre-libs --test head_to_token_typed_contract` |
| `test` | `hex_decode_scan_fused` | `vyre-libs/tests/hex_decode_scan_fused.rs` | None | `./cargo_full test -p vyre-libs --test hex_decode_scan_fused` |
| `test` | `histogram_atomic_scatter_parity` | `vyre-libs/tests/histogram_atomic_scatter_parity.rs` | None | `./cargo_full test -p vyre-libs --test histogram_atomic_scatter_parity` |
| `test` | `homotopy_euler_signed_parity` | `vyre-libs/tests/homotopy_euler_signed_parity.rs` | None | `./cargo_full test -p vyre-libs --test homotopy_euler_signed_parity` |
| `test` | `host_dispatch_is_parity_only` | `vyre-libs/tests/host_dispatch_is_parity_only.rs` | None | `./cargo_full test -p vyre-libs --test host_dispatch_is_parity_only` |
| `test` | `hypervector_ir_parity_proptest` | `vyre-libs/tests/hypervector_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test hypervector_ir_parity_proptest` |
| `test` | `iht_threshold_ir_parity_proptest` | `vyre-libs/tests/iht_threshold_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test iht_threshold_ir_parity_proptest` |
| `test` | `indexed_map_composition_contracts` | `vyre-libs/tests/indexed_map_composition_contracts.rs` | None | `./cargo_full test -p vyre-libs --test indexed_map_composition_contracts` |
| `test` | `indexed_map_composition_contracts` | `vyre-libs/tests/indexed_map_composition_contracts.rs` | `nn-activation` | `./cargo_full test -p vyre-libs --test indexed_map_composition_contracts` |
| `test` | `indexed_move_gather_oob_parity` | `vyre-libs/tests/indexed_move_gather_oob_parity.rs` | None | `./cargo_full test -p vyre-libs --test indexed_move_gather_oob_parity` |
| `test` | `inflate_program` | `vyre-libs/tests/inflate_program.rs` | None | `./cargo_full test -p vyre-libs --test inflate_program` |
| `test` | `inflate_program` | `vyre-libs/tests/inflate_program.rs` | `decode` | `./cargo_full test -p vyre-libs --test inflate_program` |
| `test` | `inflate_stored_ir_parity_proptest` | `vyre-libs/tests/inflate_stored_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test inflate_stored_ir_parity_proptest` |
| `test` | `int4_primitive_composition` | `vyre-libs/tests/int4_primitive_composition.rs` | None | `./cargo_full test -p vyre-libs --test int4_primitive_composition` |
| `test` | `int4_primitive_composition` | `vyre-libs/tests/int4_primitive_composition.rs` | `nn-activation` | `./cargo_full test -p vyre-libs --test int4_primitive_composition` |
| `test` | `integration` | `vyre-libs/tests/integration.rs` | None | `./cargo_full test -p vyre-libs --test integration` |
| `test` | `integration` | `vyre-libs/tests/integration.rs` | `hash`, `matching`, `math`, `nn-activation`, `nn-linear` | `./cargo_full test -p vyre-libs --test integration` |
| `test` | `ir_aliasing` | `vyre-libs/tests/ir_aliasing.rs` | None | `./cargo_full test -p vyre-libs --test ir_aliasing` |
| `test` | `ir_aliasing` | `vyre-libs/tests/ir_aliasing.rs` | `decode`, `parsing` | `./cargo_full test -p vyre-libs --test ir_aliasing` |
| `test` | `jacobi_serial_body_matches_per_lane` | `vyre-libs/tests/jacobi_serial_body_matches_per_lane.rs` | None | `./cargo_full test -p vyre-libs --test jacobi_serial_body_matches_per_lane` |
| `test` | `kfac_block_inverse_proptest` | `vyre-libs/tests/kfac_block_inverse_proptest.rs` | None | `./cargo_full test -p vyre-libs --test kfac_block_inverse_proptest` |
| `test` | `kfac_via_reference_parity` | `vyre-libs/tests/kfac_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test kfac_via_reference_parity` |
| `test` | `kfac_via_reference_parity` | `vyre-libs/tests/kfac_via_reference_parity.rs` | `cpu-parity`, `solvers` | `./cargo_full test -p vyre-libs --test kfac_via_reference_parity` |
| `test` | `kv_cache_append_contract` | `vyre-libs/tests/kv_cache_append_contract.rs` | None | `./cargo_full test -p vyre-libs --test kv_cache_append_contract` |
| `test` | `kv_cache_append_contract` | `vyre-libs/tests/kv_cache_append_contract.rs` | `nn-attention` | `./cargo_full test -p vyre-libs --test kv_cache_append_contract` |
| `test` | `kv_cache_typed_contract` | `vyre-libs/tests/kv_cache_typed_contract.rs` | None | `./cargo_full test -p vyre-libs --test kv_cache_typed_contract` |
| `test` | `kv_cache_typed_contract` | `vyre-libs/tests/kv_cache_typed_contract.rs` | `nn-attention` | `./cargo_full test -p vyre-libs --test kv_cache_typed_contract` |
| `test` | `last_dim_l2_norm_contract` | `vyre-libs/tests/last_dim_l2_norm_contract.rs` | None | `./cargo_full test -p vyre-libs --test last_dim_l2_norm_contract` |
| `test` | `last_dim_l2_norm_contract` | `vyre-libs/tests/last_dim_l2_norm_contract.rs` | `nn-norm` | `./cargo_full test -p vyre-libs --test last_dim_l2_norm_contract` |
| `test` | `library_operation_provenance` | `vyre-libs/tests/library_operation_provenance.rs` | None | `./cargo_full test -p vyre-libs --test library_operation_provenance` |
| `test` | `line_splice_classify_roundtrip` | `vyre-libs/tests/line_splice_classify_roundtrip.rs` | None | `./cargo_full test -p vyre-libs --test line_splice_classify_roundtrip` |
| `test` | `linear_rows_contract` | `vyre-libs/tests/linear_rows_contract.rs` | None | `./cargo_full test -p vyre-libs --test linear_rows_contract` |
| `test` | `linear_rows_contract` | `vyre-libs/tests/linear_rows_contract.rs` | `nn-linear` | `./cargo_full test -p vyre-libs --test linear_rows_contract` |
| `test` | `literal_set_presence_and_positions_reference` | `vyre-libs/tests/literal_set_presence_and_positions_reference.rs` | None | `./cargo_full test -p vyre-libs --test literal_set_presence_and_positions_reference` |
| `test` | `literal_set_presence_by_region_ground_truth` | `vyre-libs/tests/literal_set_presence_by_region_ground_truth.rs` | None | `./cargo_full test -p vyre-libs --test literal_set_presence_by_region_ground_truth` |
| `test` | `literal_set_presence_reference` | `vyre-libs/tests/literal_set_presence_reference.rs` | None | `./cargo_full test -p vyre-libs --test literal_set_presence_reference` |
| `test` | `logical_proptest` | `vyre-libs/tests/logical_proptest.rs` | None | `./cargo_full test -p vyre-libs --test logical_proptest` |
| `test` | `logical_should_panic` | `vyre-libs/tests/logical_should_panic.rs` | None | `./cargo_full test -p vyre-libs --test logical_should_panic` |
| `test` | `loop_back_edge_audit` | `vyre-libs/tests/loop_back_edge_audit.rs` | None | `./cargo_full test -p vyre-libs --test loop_back_edge_audit` |
| `test` | `loop_unroll_trip1_idempotence` | `vyre-libs/tests/loop_unroll_trip1_idempotence.rs` | None | `./cargo_full test -p vyre-libs --test loop_unroll_trip1_idempotence` |
| `test` | `lr_tables_contracts` | `vyre-libs/tests/lr_tables_contracts.rs` | None | `./cargo_full test -p vyre-libs --test lr_tables_contracts` |
| `test` | `match_motif_via_reference_parity` | `vyre-libs/tests/match_motif_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test match_motif_via_reference_parity` |
| `test` | `match_motif_via_reference_parity` | `vyre-libs/tests/match_motif_via_reference_parity.rs` | `cpu-parity`, `graph-dispatch` | `./cargo_full test -p vyre-libs --test match_motif_via_reference_parity` |
| `test` | `matching_diagnostic_via_reference_parity` | `vyre-libs/tests/matching_diagnostic_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test matching_diagnostic_via_reference_parity` |
| `test` | `matching_diagnostic_via_reference_parity` | `vyre-libs/tests/matching_diagnostic_via_reference_parity.rs` | `cpu-parity`, `encoding` | `./cargo_full test -p vyre-libs --test matching_diagnostic_via_reference_parity` |
| `test` | `matching_nfa_scan_program_contracts` | `vyre-libs/tests/matching_nfa_scan_program_contracts.rs` | None | `./cargo_full test -p vyre-libs --test matching_nfa_scan_program_contracts` |
| `test` | `matching_post_process_contracts` | `vyre-libs/tests/matching_post_process_contracts.rs` | None | `./cargo_full test -p vyre-libs --test matching_post_process_contracts` |
| `test` | `math_algebra_branchless_contracts` | `vyre-libs/tests/math_algebra_branchless_contracts.rs` | None | `./cargo_full test -p vyre-libs --test math_algebra_branchless_contracts` |
| `test` | `matroid_exact_subset_via_reference_parity` | `vyre-libs/tests/matroid_exact_subset_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test matroid_exact_subset_via_reference_parity` |
| `test` | `matroid_exact_subset_via_reference_parity` | `vyre-libs/tests/matroid_exact_subset_via_reference_parity.rs` | `cpu-parity`, `encoding` | `./cargo_full test -p vyre-libs --test matroid_exact_subset_via_reference_parity` |
| `test` | `matroid_intersection_full_proptest` | `vyre-libs/tests/matroid_intersection_full_proptest.rs` | None | `./cargo_full test -p vyre-libs --test matroid_intersection_full_proptest` |
| `test` | `matroid_intersection_full_value_parity` | `vyre-libs/tests/matroid_intersection_full_value_parity.rs` | None | `./cargo_full test -p vyre-libs --test matroid_intersection_full_value_parity` |
| `test` | `mlp_4x_leaky_sq_multi_workgroup_span` | `vyre-libs/tests/mlp_4x_leaky_sq_multi_workgroup_span.rs` | None | `./cargo_full test -p vyre-libs --test mlp_4x_leaky_sq_multi_workgroup_span` |
| `test` | `mlp_4x_leaky_sq_multi_workgroup_span` | `vyre-libs/tests/mlp_4x_leaky_sq_multi_workgroup_span.rs` | `nn-activation` | `./cargo_full test -p vyre-libs --test mlp_4x_leaky_sq_multi_workgroup_span` |
| `test` | `motif_ir_parity_proptest` | `vyre-libs/tests/motif_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test motif_ir_parity_proptest` |
| `test` | `multi_block_prefix_scan_carry_parity` | `vyre-libs/tests/multi_block_prefix_scan_carry_parity.rs` | None | `./cargo_full test -p vyre-libs --test multi_block_prefix_scan_carry_parity` |
| `test` | `multigrid_matroid_via_reference_parity` | `vyre-libs/tests/multigrid_matroid_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test multigrid_matroid_via_reference_parity` |
| `test` | `multigrid_matroid_via_reference_parity` | `vyre-libs/tests/multigrid_matroid_via_reference_parity.rs` | `cpu-parity`, `solvers` | `./cargo_full test -p vyre-libs --test multigrid_matroid_via_reference_parity` |
| `test` | `mz_project_via_reference_parity` | `vyre-libs/tests/mz_project_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test mz_project_via_reference_parity` |
| `test` | `mz_project_via_reference_parity` | `vyre-libs/tests/mz_project_via_reference_parity.rs` | `cpu-parity`, `solvers` | `./cargo_full test -p vyre-libs --test mz_project_via_reference_parity` |
| `test` | `name_collision` | `vyre-libs/tests/name_collision.rs` | None | `./cargo_full test -p vyre-libs --test name_collision` |
| `test` | `name_collision` | `vyre-libs/tests/name_collision.rs` | `math`, `nn-attention` | `./cargo_full test -p vyre-libs --test name_collision` |
| `test` | `natural_config_gradient_via_reference_parity` | `vyre-libs/tests/natural_config_gradient_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test natural_config_gradient_via_reference_parity` |
| `test` | `natural_config_gradient_via_reference_parity` | `vyre-libs/tests/natural_config_gradient_via_reference_parity.rs` | `cpu-parity`, `solvers` | `./cargo_full test -p vyre-libs --test natural_config_gradient_via_reference_parity` |
| `test` | `natural_gradient_via_reference_parity` | `vyre-libs/tests/natural_gradient_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test natural_gradient_via_reference_parity` |
| `test` | `natural_gradient_via_reference_parity` | `vyre-libs/tests/natural_gradient_via_reference_parity.rs` | `cpu-parity`, `solvers` | `./cargo_full test -p vyre-libs --test natural_gradient_via_reference_parity` |
| `test` | `nfa_plan_contracts` | `vyre-libs/tests/nfa_plan_contracts.rs` | None | `./cargo_full test -p vyre-libs --test nfa_plan_contracts` |
| `test` | `nn_attention_clone_family_ir_invariance` | `vyre-libs/tests/nn_attention_clone_family_ir_invariance.rs` | None | `./cargo_full test -p vyre-libs --test nn_attention_clone_family_ir_invariance` |
| `test` | `nn_attention_clone_family_ir_invariance` | `vyre-libs/tests/nn_attention_clone_family_ir_invariance.rs` | `nn-attention` | `./cargo_full test -p vyre-libs --test nn_attention_clone_family_ir_invariance` |
| `test` | `node_kind_eq_ir_parity_proptest` | `vyre-libs/tests/node_kind_eq_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test node_kind_eq_ir_parity_proptest` |
| `test` | `node_kind_eq_ir_parity_proptest` | `vyre-libs/tests/node_kind_eq_ir_parity_proptest.rs` | `cpu-parity`, `predicate` | `./cargo_full test -p vyre-libs --test node_kind_eq_ir_parity_proptest` |
| `test` | `op_boundaries` | `vyre-libs/tests/op_boundaries.rs` | None | `./cargo_full test -p vyre-libs --test op_boundaries` |
| `test` | `op_boundaries` | `vyre-libs/tests/op_boundaries.rs` | `nn-activation`, `nn-linear` | `./cargo_full test -p vyre-libs --test op_boundaries` |
| `test` | `operation_registry` | `vyre-libs/tests/operation_registry.rs` | None | `./cargo_full test -p vyre-libs --test operation_registry` |
| `test` | `operation_registry` | `vyre-libs/tests/operation_registry.rs` | `math`, `math-linalg`, `nn-activation`, `nn-attention`, `nn-linear`, `nn-norm` | `./cargo_full test -p vyre-libs --test operation_registry` |
| `test` | `operator_reporting_interchange` | `vyre-libs/tests/operator_reporting_interchange.rs` | None | `./cargo_full test -p vyre-libs --test operator_reporting_interchange` |
| `test` | `optimized_programs` | `vyre-libs/tests/optimized_programs.rs` | None | `./cargo_full test -p vyre-libs --test optimized_programs` |
| `test` | `optimized_programs` | `vyre-libs/tests/optimized_programs.rs` | `nn-attention`, `nn-linear`, `nn-norm` | `./cargo_full test -p vyre-libs --test optimized_programs` |
| `test` | `output_encoding_unicode_policies` | `vyre-libs/tests/output_encoding_unicode_policies.rs` | None | `./cargo_full test -p vyre-libs --test output_encoding_unicode_policies` |
| `test` | `overflow_guards` | `vyre-libs/tests/overflow_guards.rs` | None | `./cargo_full test -p vyre-libs --test overflow_guards` |
| `test` | `overflow_guards` | `vyre-libs/tests/overflow_guards.rs` | `nn-attention` | `./cargo_full test -p vyre-libs --test overflow_guards` |
| `test` | `padic_hensel_signed_parity` | `vyre-libs/tests/padic_hensel_signed_parity.rs` | None | `./cargo_full test -p vyre-libs --test padic_hensel_signed_parity` |
| `test` | `parser_edit_delta_contracts` | `vyre-libs/tests/parser_edit_delta_contracts.rs` | None | `./cargo_full test -p vyre-libs --test parser_edit_delta_contracts` |
| `test` | `parser_graph_navigation_contracts` | `vyre-libs/tests/parser_graph_navigation_contracts.rs` | None | `./cargo_full test -p vyre-libs --test parser_graph_navigation_contracts` |
| `test` | `parser_recovery_corpus_registry` | `vyre-libs/tests/parser_recovery_corpus_registry.rs` | None | `./cargo_full test -p vyre-libs --test parser_recovery_corpus_registry` |
| `test` | `parsing_walker_clone_family` | `vyre-libs/tests/parsing_walker_clone_family.rs` | None | `./cargo_full test -p vyre-libs --test parsing_walker_clone_family` |
| `test` | `parsing_walker_clone_family` | `vyre-libs/tests/parsing_walker_clone_family.rs` | `parsing` | `./cargo_full test -p vyre-libs --test parsing_walker_clone_family` |
| `test` | `partial_rope_offset_contract` | `vyre-libs/tests/partial_rope_offset_contract.rs` | None | `./cargo_full test -p vyre-libs --test partial_rope_offset_contract` |
| `test` | `partial_rope_offset_contract` | `vyre-libs/tests/partial_rope_offset_contract.rs` | `nn-attention` | `./cargo_full test -p vyre-libs --test partial_rope_offset_contract` |
| `test` | `partial_rope_typed_contract` | `vyre-libs/tests/partial_rope_typed_contract.rs` | None | `./cargo_full test -p vyre-libs --test partial_rope_typed_contract` |
| `test` | `partial_rope_typed_contract` | `vyre-libs/tests/partial_rope_typed_contract.rs` | `nn-attention` | `./cargo_full test -p vyre-libs --test partial_rope_typed_contract` |
| `test` | `pass_research_trace_artifacts` | `vyre-libs/tests/pass_research_trace_artifacts.rs` | None | `./cargo_full test -p vyre-libs --test pass_research_trace_artifacts` |
| `test` | `persistent_fixpoint_grid_contracts` | `vyre-libs/tests/persistent_fixpoint_grid_contracts.rs` | None | `./cargo_full test -p vyre-libs --test persistent_fixpoint_grid_contracts` |
| `test` | `persistent_fixpoint_grid_contracts` | `vyre-libs/tests/persistent_fixpoint_grid_contracts.rs` | `cpu-parity`, `fixpoint` | `./cargo_full test -p vyre-libs --test persistent_fixpoint_grid_contracts` |
| `test` | `persistent_fixpoint_loop_contracts` | `vyre-libs/tests/persistent_fixpoint_loop_contracts.rs` | None | `./cargo_full test -p vyre-libs --test persistent_fixpoint_loop_contracts` |
| `test` | `planar_rewrite_ir_parity_proptest` | `vyre-libs/tests/planar_rewrite_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test planar_rewrite_ir_parity_proptest` |
| `test` | `planar_rewrite_via_reference_parity` | `vyre-libs/tests/planar_rewrite_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test planar_rewrite_via_reference_parity` |
| `test` | `planar_rewrite_via_reference_parity` | `vyre-libs/tests/planar_rewrite_via_reference_parity.rs` | `cpu-parity`, `scheduling` | `./cargo_full test -p vyre-libs --test planar_rewrite_via_reference_parity` |
| `test` | `predict_impact_via_reference_parity` | `vyre-libs/tests/predict_impact_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test predict_impact_via_reference_parity` |
| `test` | `predict_impact_via_reference_parity` | `vyre-libs/tests/predict_impact_via_reference_parity.rs` | `cpu-parity`, `reasoning` | `./cargo_full test -p vyre-libs --test predict_impact_via_reference_parity` |
| `test` | `prefix_scan_contract` | `vyre-libs/tests/prefix_scan_contract.rs` | None | `./cargo_full test -p vyre-libs --test prefix_scan_contract` |
| `test` | `prefix_scan_contract` | `vyre-libs/tests/prefix_scan_contract.rs` | `cpu-parity`, `math-kernels` | `./cargo_full test -p vyre-libs --test prefix_scan_contract` |
| `test` | `primitive_surface_contracts` | `vyre-libs/tests/primitive_surface_contracts.rs` | None | `./cargo_full test -p vyre-libs --test primitive_surface_contracts` |
| `test` | `primitive_vs_consumer` | `vyre-libs/tests/primitive_vs_consumer.rs` | None | `./cargo_full test -p vyre-libs --test primitive_vs_consumer` |
| `test` | `primitive_vs_consumer` | `vyre-libs/tests/primitive_vs_consumer.rs` | `analysis`, `cpu-parity`, `encoding` | `./cargo_full test -p vyre-libs --test primitive_vs_consumer` |
| `test` | `production_ir_parity` | `vyre-libs/tests/production_ir_parity.rs` | None | `./cargo_full test -p vyre-libs --test production_ir_parity` |
| `test` | `property` | `vyre-libs/tests/property.rs` | None | `./cargo_full test -p vyre-libs --test property` |
| `test` | `property_differential_oracles` | `vyre-libs/tests/property_differential_oracles.rs` | None | `./cargo_full test -p vyre-libs --test property_differential_oracles` |
| `test` | `proptest_base64_decode` | `vyre-libs/tests/proptest_base64_decode.rs` | None | `./cargo_full test -p vyre-libs --test proptest_base64_decode` |
| `test` | `proptest_bitset_and_laws` | `vyre-libs/tests/proptest_bitset_and_laws.rs` | None | `./cargo_full test -p vyre-libs --test proptest_bitset_and_laws` |
| `test` | `proptest_bitset_any` | `vyre-libs/tests/proptest_bitset_any.rs` | None | `./cargo_full test -p vyre-libs --test proptest_bitset_any` |
| `test` | `proptest_bitset_boolean_algebra` | `vyre-libs/tests/proptest_bitset_boolean_algebra.rs` | None | `./cargo_full test -p vyre-libs --test proptest_bitset_boolean_algebra` |
| `test` | `proptest_bitset_contains` | `vyre-libs/tests/proptest_bitset_contains.rs` | None | `./cargo_full test -p vyre-libs --test proptest_bitset_contains` |
| `test` | `proptest_bitset_copy` | `vyre-libs/tests/proptest_bitset_copy.rs` | None | `./cargo_full test -p vyre-libs --test proptest_bitset_copy` |
| `test` | `proptest_bitset_equal` | `vyre-libs/tests/proptest_bitset_equal.rs` | None | `./cargo_full test -p vyre-libs --test proptest_bitset_equal` |
| `test` | `proptest_bitset_not_involution` | `vyre-libs/tests/proptest_bitset_not_involution.rs` | None | `./cargo_full test -p vyre-libs --test proptest_bitset_not_involution` |
| `test` | `proptest_bitset_not_laws` | `vyre-libs/tests/proptest_bitset_not_laws.rs` | None | `./cargo_full test -p vyre-libs --test proptest_bitset_not_laws` |
| `test` | `proptest_bitset_or_laws` | `vyre-libs/tests/proptest_bitset_or_laws.rs` | None | `./cargo_full test -p vyre-libs --test proptest_bitset_or_laws` |
| `test` | `proptest_bitset_popcount` | `vyre-libs/tests/proptest_bitset_popcount.rs` | None | `./cargo_full test -p vyre-libs --test proptest_bitset_popcount` |
| `test` | `proptest_bitset_popcount_laws` | `vyre-libs/tests/proptest_bitset_popcount_laws.rs` | None | `./cargo_full test -p vyre-libs --test proptest_bitset_popcount_laws` |
| `test` | `proptest_bitset_subset_of` | `vyre-libs/tests/proptest_bitset_subset_of.rs` | None | `./cargo_full test -p vyre-libs --test proptest_bitset_subset_of` |
| `test` | `proptest_bitset_words` | `vyre-libs/tests/proptest_bitset_words.rs` | None | `./cargo_full test -p vyre-libs --test proptest_bitset_words` |
| `test` | `proptest_bitset_xor_laws` | `vyre-libs/tests/proptest_bitset_xor_laws.rs` | None | `./cargo_full test -p vyre-libs --test proptest_bitset_xor_laws` |
| `test` | `proptest_bitset_zero` | `vyre-libs/tests/proptest_bitset_zero.rs` | None | `./cargo_full test -p vyre-libs --test proptest_bitset_zero` |
| `test` | `proptest_csr_forward_traverse` | `vyre-libs/tests/proptest_csr_forward_traverse.rs` | None | `./cargo_full test -p vyre-libs --test proptest_csr_forward_traverse` |
| `test` | `proptest_csr_frontier_queue` | `vyre-libs/tests/proptest_csr_frontier_queue.rs` | None | `./cargo_full test -p vyre-libs --test proptest_csr_frontier_queue` |
| `test` | `proptest_csr_frontier_queue` | `vyre-libs/tests/proptest_csr_frontier_queue.rs` | `cpu-parity`, `graph` | `./cargo_full test -p vyre-libs --test proptest_csr_frontier_queue` |
| `test` | `proptest_csr_frontier_shard` | `vyre-libs/tests/proptest_csr_frontier_shard.rs` | None | `./cargo_full test -p vyre-libs --test proptest_csr_frontier_shard` |
| `test` | `proptest_csr_frontier_shard` | `vyre-libs/tests/proptest_csr_frontier_shard.rs` | `cpu-parity`, `graph` | `./cargo_full test -p vyre-libs --test proptest_csr_frontier_shard` |
| `test` | `proptest_csr_queue_split` | `vyre-libs/tests/proptest_csr_queue_split.rs` | None | `./cargo_full test -p vyre-libs --test proptest_csr_queue_split` |
| `test` | `proptest_csr_queue_split` | `vyre-libs/tests/proptest_csr_queue_split.rs` | `cpu-parity`, `graph` | `./cargo_full test -p vyre-libs --test proptest_csr_queue_split` |
| `test` | `proptest_csr_queue_strided` | `vyre-libs/tests/proptest_csr_queue_strided.rs` | None | `./cargo_full test -p vyre-libs --test proptest_csr_queue_strided` |
| `test` | `proptest_csr_queue_strided` | `vyre-libs/tests/proptest_csr_queue_strided.rs` | `cpu-parity`, `graph` | `./cargo_full test -p vyre-libs --test proptest_csr_queue_strided` |
| `test` | `proptest_dispatch_pack_roundtrip` | `vyre-libs/tests/proptest_dispatch_pack_roundtrip.rs` | None | `./cargo_full test -p vyre-libs --test proptest_dispatch_pack_roundtrip` |
| `test` | `proptest_dominator_frontier` | `vyre-libs/tests/proptest_dominator_frontier.rs` | None | `./cargo_full test -p vyre-libs --test proptest_dominator_frontier` |
| `test` | `proptest_graph_reachable` | `vyre-libs/tests/proptest_graph_reachable.rs` | None | `./cargo_full test -p vyre-libs --test proptest_graph_reachable` |
| `test` | `proptest_hash_crc32` | `vyre-libs/tests/proptest_hash_crc32.rs` | None | `./cargo_full test -p vyre-libs --test proptest_hash_crc32` |
| `test` | `proptest_hash_fnv1a` | `vyre-libs/tests/proptest_hash_fnv1a.rs` | None | `./cargo_full test -p vyre-libs --test proptest_hash_fnv1a` |
| `test` | `proptest_hex_decode` | `vyre-libs/tests/proptest_hex_decode.rs` | None | `./cargo_full test -p vyre-libs --test proptest_hex_decode` |
| `test` | `proptest_multi_block_prefix_scan` | `vyre-libs/tests/proptest_multi_block_prefix_scan.rs` | None | `./cargo_full test -p vyre-libs --test proptest_multi_block_prefix_scan` |
| `test` | `proptest_reduce_all` | `vyre-libs/tests/proptest_reduce_all.rs` | None | `./cargo_full test -p vyre-libs --test proptest_reduce_all` |
| `test` | `proptest_reduce_any` | `vyre-libs/tests/proptest_reduce_any.rs` | None | `./cargo_full test -p vyre-libs --test proptest_reduce_any` |
| `test` | `proptest_reduce_any_all` | `vyre-libs/tests/proptest_reduce_any_all.rs` | None | `./cargo_full test -p vyre-libs --test proptest_reduce_any_all` |
| `test` | `proptest_reduce_count_laws` | `vyre-libs/tests/proptest_reduce_count_laws.rs` | None | `./cargo_full test -p vyre-libs --test proptest_reduce_count_laws` |
| `test` | `proptest_reduce_count_non_zero` | `vyre-libs/tests/proptest_reduce_count_non_zero.rs` | None | `./cargo_full test -p vyre-libs --test proptest_reduce_count_non_zero` |
| `test` | `proptest_reduce_min_max_laws` | `vyre-libs/tests/proptest_reduce_min_max_laws.rs` | None | `./cargo_full test -p vyre-libs --test proptest_reduce_min_max_laws` |
| `test` | `proptest_reduce_sum_laws` | `vyre-libs/tests/proptest_reduce_sum_laws.rs` | None | `./cargo_full test -p vyre-libs --test proptest_reduce_sum_laws` |
| `test` | `proptest_text_byte_histogram` | `vyre-libs/tests/proptest_text_byte_histogram.rs` | None | `./cargo_full test -p vyre-libs --test proptest_text_byte_histogram` |
| `test` | `proptest_text_char_class` | `vyre-libs/tests/proptest_text_char_class.rs` | None | `./cargo_full test -p vyre-libs --test proptest_text_char_class` |
| `test` | `proptest_text_encoding_classify` | `vyre-libs/tests/proptest_text_encoding_classify.rs` | None | `./cargo_full test -p vyre-libs --test proptest_text_encoding_classify` |
| `test` | `proptest_text_encoding_classify` | `vyre-libs/tests/proptest_text_encoding_classify.rs` | `cpu-parity`, `text` | `./cargo_full test -p vyre-libs --test proptest_text_encoding_classify` |
| `test` | `proptest_text_line_index` | `vyre-libs/tests/proptest_text_line_index.rs` | None | `./cargo_full test -p vyre-libs --test proptest_text_line_index` |
| `test` | `proptest_text_line_index` | `vyre-libs/tests/proptest_text_line_index.rs` | `cpu-parity`, `text` | `./cargo_full test -p vyre-libs --test proptest_text_line_index` |
| `test` | `proptest_text_utf8_validate` | `vyre-libs/tests/proptest_text_utf8_validate.rs` | None | `./cargo_full test -p vyre-libs --test proptest_text_utf8_validate` |
| `test` | `proptest_text_utf8_validate` | `vyre-libs/tests/proptest_text_utf8_validate.rs` | `cpu-parity`, `text` | `./cargo_full test -p vyre-libs --test proptest_text_utf8_validate` |
| `test` | `proptest_toposort_dag` | `vyre-libs/tests/proptest_toposort_dag.rs` | None | `./cargo_full test -p vyre-libs --test proptest_toposort_dag` |
| `test` | `proptest_ziftsieve` | `vyre-libs/tests/proptest_ziftsieve.rs` | None | `./cargo_full test -p vyre-libs --test proptest_ziftsieve` |
| `test` | `provenance_closure` | `vyre-libs/tests/provenance_closure.rs` | None | `./cargo_full test -p vyre-libs --test provenance_closure` |
| `test` | `provenance_closure` | `vyre-libs/tests/provenance_closure.rs` | `cpu-parity`, `encoding` | `./cargo_full test -p vyre-libs --test provenance_closure` |
| `test` | `qk_gain_shape_overflow_contracts` | `vyre-libs/tests/qk_gain_shape_overflow_contracts.rs` | None | `./cargo_full test -p vyre-libs --test qk_gain_shape_overflow_contracts` |
| `test` | `qk_gain_shape_overflow_contracts` | `vyre-libs/tests/qk_gain_shape_overflow_contracts.rs` | `nn-attention` | `./cargo_full test -p vyre-libs --test qk_gain_shape_overflow_contracts` |
| `test` | `qk_gain_zero_shape_contracts` | `vyre-libs/tests/qk_gain_zero_shape_contracts.rs` | None | `./cargo_full test -p vyre-libs --test qk_gain_zero_shape_contracts` |
| `test` | `qk_gain_zero_shape_contracts` | `vyre-libs/tests/qk_gain_zero_shape_contracts.rs` | `nn-attention` | `./cargo_full test -p vyre-libs --test qk_gain_zero_shape_contracts` |
| `test` | `quantized_dispatch_variant_coverage_gate` | `vyre-libs/tests/quantized_dispatch_variant_coverage_gate.rs` | None | `./cargo_full test -p vyre-libs --test quantized_dispatch_variant_coverage_gate` |
| `test` | `quantized_dispatch_variant_coverage_gate` | `vyre-libs/tests/quantized_dispatch_variant_coverage_gate.rs` | `solvers` | `./cargo_full test -p vyre-libs --test quantized_dispatch_variant_coverage_gate` |
| `test` | `quantized_linear_affine_fma` | `vyre-libs/tests/quantized_linear_affine_fma.rs` | None | `./cargo_full test -p vyre-libs --test quantized_linear_affine_fma` |
| `test` | `quantized_linear_affine_fma` | `vyre-libs/tests/quantized_linear_affine_fma.rs` | `nn-linear` | `./cargo_full test -p vyre-libs --test quantized_linear_affine_fma` |
| `test` | `quantized_packing_contracts` | `vyre-libs/tests/quantized_packing_contracts.rs` | None | `./cargo_full test -p vyre-libs --test quantized_packing_contracts` |
| `test` | `quantized_via_reference_parity` | `vyre-libs/tests/quantized_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test quantized_via_reference_parity` |
| `test` | `quantized_via_reference_parity` | `vyre-libs/tests/quantized_via_reference_parity.rs` | `cpu-parity`, `solvers` | `./cargo_full test -p vyre-libs --test quantized_via_reference_parity` |
| `test` | `randomized_svd_signed_parity` | `vyre-libs/tests/randomized_svd_signed_parity.rs` | None | `./cargo_full test -p vyre-libs --test randomized_svd_signed_parity` |
| `test` | `range_counts_ir_parity_proptest` | `vyre-libs/tests/range_counts_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test range_counts_ir_parity_proptest` |
| `test` | `reconstruct_path_via_reference_parity` | `vyre-libs/tests/reconstruct_path_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test reconstruct_path_via_reference_parity` |
| `test` | `reconstruct_path_via_reference_parity` | `vyre-libs/tests/reconstruct_path_via_reference_parity.rs` | `cpu-parity`, `graph-dispatch` | `./cargo_full test -p vyre-libs --test reconstruct_path_via_reference_parity` |
| `test` | `recurrent_gated_delta_contract` | `vyre-libs/tests/recurrent_gated_delta_contract.rs` | None | `./cargo_full test -p vyre-libs --test recurrent_gated_delta_contract` |
| `test` | `recurrent_gated_delta_contract` | `vyre-libs/tests/recurrent_gated_delta_contract.rs` | `nn-attention` | `./cargo_full test -p vyre-libs --test recurrent_gated_delta_contract` |
| `test` | `reduce_atomic_ir_parity_proptest` | `vyre-libs/tests/reduce_atomic_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test reduce_atomic_ir_parity_proptest` |
| `test` | `reduction_metrics_via_reference_parity` | `vyre-libs/tests/reduction_metrics_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test reduction_metrics_via_reference_parity` |
| `test` | `reduction_metrics_via_reference_parity` | `vyre-libs/tests/reduction_metrics_via_reference_parity.rs` | `cpu-parity`, `encoding` | `./cargo_full test -p vyre-libs --test reduction_metrics_via_reference_parity` |
| `test` | `reduction_route_parity` | `vyre-libs/tests/reduction_route_parity.rs` | None | `./cargo_full test -p vyre-libs --test reduction_route_parity` |
| `test` | `regex_adversarial_class_catalog` | `vyre-libs/tests/regex_adversarial_class_catalog.rs` | None | `./cargo_full test -p vyre-libs --test regex_adversarial_class_catalog` |
| `test` | `regex_capture_mode_contracts` | `vyre-libs/tests/regex_capture_mode_contracts.rs` | None | `./cargo_full test -p vyre-libs --test regex_capture_mode_contracts` |
| `test` | `regex_columnar_output_contracts` | `vyre-libs/tests/regex_columnar_output_contracts.rs` | None | `./cargo_full test -p vyre-libs --test regex_columnar_output_contracts` |
| `test` | `regex_compile_adversarial` | `vyre-libs/tests/regex_compile_adversarial.rs` | None | `./cargo_full test -p vyre-libs --test regex_compile_adversarial` |
| `test` | `regex_compile_ascii_class_contracts` | `vyre-libs/tests/regex_compile_ascii_class_contracts.rs` | None | `./cargo_full test -p vyre-libs --test regex_compile_ascii_class_contracts` |
| `test` | `regex_compile_property` | `vyre-libs/tests/regex_compile_property.rs` | None | `./cargo_full test -p vyre-libs --test regex_compile_property` |
| `test` | `regex_dfa_anchoring_differential` | `vyre-libs/tests/regex_dfa_anchoring_differential.rs` | None | `./cargo_full test -p vyre-libs --test regex_dfa_anchoring_differential` |
| `test` | `regex_dfa_char_class_exhaustive` | `vyre-libs/tests/regex_dfa_char_class_exhaustive.rs` | None | `./cargo_full test -p vyre-libs --test regex_dfa_char_class_exhaustive` |
| `test` | `regex_dfa_leftmost_longest_differential` | `vyre-libs/tests/regex_dfa_leftmost_longest_differential.rs` | None | `./cargo_full test -p vyre-libs --test regex_dfa_leftmost_longest_differential` |
| `test` | `regex_dialect_lattice` | `vyre-libs/tests/regex_dialect_lattice.rs` | None | `./cargo_full test -p vyre-libs --test regex_dialect_lattice` |
| `test` | `regex_leftmost_longest_bounded_repeat` | `vyre-libs/tests/regex_leftmost_longest_bounded_repeat.rs` | None | `./cargo_full test -p vyre-libs --test regex_leftmost_longest_bounded_repeat` |
| `test` | `regex_logical_pattern_planner` | `vyre-libs/tests/regex_logical_pattern_planner.rs` | None | `./cargo_full test -p vyre-libs --test regex_logical_pattern_planner` |
| `test` | `regex_match_policy_contracts` | `vyre-libs/tests/regex_match_policy_contracts.rs` | None | `./cargo_full test -p vyre-libs --test regex_match_policy_contracts` |
| `test` | `regex_prefilter_planner_registry` | `vyre-libs/tests/regex_prefilter_planner_registry.rs` | None | `./cargo_full test -p vyre-libs --test regex_prefilter_planner_registry` |
| `test` | `regex_streaming_state_ledger` | `vyre-libs/tests/regex_streaming_state_ledger.rs` | None | `./cargo_full test -p vyre-libs --test regex_streaming_state_ledger` |
| `test` | `regex_unicode_profiles` | `vyre-libs/tests/regex_unicode_profiles.rs` | None | `./cargo_full test -p vyre-libs --test regex_unicode_profiles` |
| `test` | `regex_unsupported_diagnostic_registry` | `vyre-libs/tests/regex_unsupported_diagnostic_registry.rs` | None | `./cargo_full test -p vyre-libs --test regex_unsupported_diagnostic_registry` |
| `test` | `region_adversarial` | `vyre-libs/tests/region_adversarial.rs` | None | `./cargo_full test -p vyre-libs --test region_adversarial` |
| `test` | `region_adversarial` | `vyre-libs/tests/region_adversarial.rs` | `matching-kernels` | `./cargo_full test -p vyre-libs --test region_adversarial` |
| `test` | `region_chain_discipline` | `vyre-libs/tests/region_chain_discipline.rs` | None | `./cargo_full test -p vyre-libs --test region_chain_discipline` |
| `test` | `region_chain_invariant` | `vyre-libs/tests/region_chain_invariant.rs` | None | `./cargo_full test -p vyre-libs --test region_chain_invariant` |
| `test` | `region_dedup_property` | `vyre-libs/tests/region_dedup_property.rs` | None | `./cargo_full test -p vyre-libs --test region_dedup_property` |
| `test` | `region_gpu_flag_contracts` | `vyre-libs/tests/region_gpu_flag_contracts.rs` | None | `./cargo_full test -p vyre-libs --test region_gpu_flag_contracts` |
| `test` | `region_inline_let_scope` | `vyre-libs/tests/region_inline_let_scope.rs` | None | `./cargo_full test -p vyre-libs --test region_inline_let_scope` |
| `test` | `registration_drift` | `vyre-libs/tests/registration_drift.rs` | None | `./cargo_full test -p vyre-libs --test registration_drift` |
| `test` | `registry_closure` | `vyre-libs/tests/registry_closure.rs` | None | `./cargo_full test -p vyre-libs --test registry_closure` |
| `test` | `resolve_family_ir_parity_proptest` | `vyre-libs/tests/resolve_family_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test resolve_family_ir_parity_proptest` |
| `test` | `resolve_family_ir_parity_proptest` | `vyre-libs/tests/resolve_family_ir_parity_proptest.rs` | `cpu-parity`, `label` | `./cargo_full test -p vyre-libs --test resolve_family_ir_parity_proptest` |
| `test` | `resource_budget_complexity_policies` | `vyre-libs/tests/resource_budget_complexity_policies.rs` | None | `./cargo_full test -p vyre-libs --test resource_budget_complexity_policies` |
| `test` | `rle_segment_lengths_contracts` | `vyre-libs/tests/rle_segment_lengths_contracts.rs` | None | `./cargo_full test -p vyre-libs --test rle_segment_lengths_contracts` |
| `test` | `rle_segment_lengths_ir_parity_proptest` | `vyre-libs/tests/rle_segment_lengths_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test rle_segment_lengths_ir_parity_proptest` |
| `test` | `rule_condition_program_frame_contract` | `vyre-libs/tests/rule_condition_program_frame_contract.rs` | None | `./cargo_full test -p vyre-libs --test rule_condition_program_frame_contract` |
| `test` | `rule_condition_program_frame_contract` | `vyre-libs/tests/rule_condition_program_frame_contract.rs` | `rule` | `./cargo_full test -p vyre-libs --test rule_condition_program_frame_contract` |
| `test` | `scallop_join_grid_contract` | `vyre-libs/tests/scallop_join_grid_contract.rs` | None | `./cargo_full test -p vyre-libs --test scallop_join_grid_contract` |
| `test` | `scallop_join_grid_contract` | `vyre-libs/tests/scallop_join_grid_contract.rs` | `cpu-parity`, `fixpoint`, `math-kernels` | `./cargo_full test -p vyre-libs --test scallop_join_grid_contract` |
| `test` | `scallop_join_ir_parity` | `vyre-libs/tests/scallop_join_ir_parity.rs` | None | `./cargo_full test -p vyre-libs --test scallop_join_ir_parity` |
| `test` | `scallop_provenance_via_reference_parity` | `vyre-libs/tests/scallop_provenance_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test scallop_provenance_via_reference_parity` |
| `test` | `scallop_provenance_via_reference_parity` | `vyre-libs/tests/scallop_provenance_via_reference_parity.rs` | `cpu-parity`, `encoding` | `./cargo_full test -p vyre-libs --test scallop_provenance_via_reference_parity` |
| `test` | `scan_ac_transition_walk_single_owner` | `vyre-libs/tests/scan_ac_transition_walk_single_owner.rs` | None | `./cargo_full test -p vyre-libs --test scan_ac_transition_walk_single_owner` |
| `test` | `scan_ac_transition_walk_single_owner` | `vyre-libs/tests/scan_ac_transition_walk_single_owner.rs` | `matching-regex` | `./cargo_full test -p vyre-libs --test scan_ac_transition_walk_single_owner` |
| `test` | `scan_cpu_api_boundary` | `vyre-libs/tests/scan_cpu_api_boundary.rs` | None | `./cargo_full test -p vyre-libs --test scan_cpu_api_boundary` |
| `test` | `scan_hit_buffer_layout_contracts` | `vyre-libs/tests/scan_hit_buffer_layout_contracts.rs` | None | `./cargo_full test -p vyre-libs --test scan_hit_buffer_layout_contracts` |
| `test` | `scan_hit_buffer_layout_contracts` | `vyre-libs/tests/scan_hit_buffer_layout_contracts.rs` | `matching-substring` | `./cargo_full test -p vyre-libs --test scan_hit_buffer_layout_contracts` |
| `test` | `scan_prefilter_width_closure` | `vyre-libs/tests/scan_prefilter_width_closure.rs` | None | `./cargo_full test -p vyre-libs --test scan_prefilter_width_closure` |
| `test` | `scan_prefix_sum_size_contract` | `vyre-libs/tests/scan_prefix_sum_size_contract.rs` | None | `./cargo_full test -p vyre-libs --test scan_prefix_sum_size_contract` |
| `test` | `scan_prefix_sum_size_contract` | `vyre-libs/tests/scan_prefix_sum_size_contract.rs` | `cpu-parity`, `math-scan` | `./cargo_full test -p vyre-libs --test scan_prefix_sum_size_contract` |
| `test` | `score_denoise_signed_parity` | `vyre-libs/tests/score_denoise_signed_parity.rs` | None | `./cargo_full test -p vyre-libs --test score_denoise_signed_parity` |
| `test` | `secret_crypto_policies` | `vyre-libs/tests/secret_crypto_policies.rs` | None | `./cargo_full test -p vyre-libs --test secret_crypto_policies` |
| `test` | `security_external_ifds` | `vyre-libs/tests/security_external_ifds.rs` | None | `./cargo_full test -p vyre-libs --test security_external_ifds` |
| `test` | `security_flow_skeleton_family_guard` | `vyre-libs/tests/security_flow_skeleton_family_guard.rs` | None | `./cargo_full test -p vyre-libs --test security_flow_skeleton_family_guard` |
| `test` | `security_flow_skeleton_family_guard` | `vyre-libs/tests/security_flow_skeleton_family_guard.rs` | `security` | `./cargo_full test -p vyre-libs --test security_flow_skeleton_family_guard` |
| `test` | `security_flows_to_alias_only_parity` | `vyre-libs/tests/security_flows_to_alias_only_parity.rs` | None | `./cargo_full test -p vyre-libs --test security_flows_to_alias_only_parity` |
| `test` | `security_privacy_path_corpus_guards` | `vyre-libs/tests/security_privacy_path_corpus_guards.rs` | None | `./cargo_full test -p vyre-libs --test security_privacy_path_corpus_guards` |
| `test` | `segment_reduce_ir_parity_proptest` | `vyre-libs/tests/segment_reduce_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test segment_reduce_ir_parity_proptest` |
| `test` | `self_consumer_conform` | `vyre-libs/tests/self_consumer_conform.rs` | None | `./cargo_full test -p vyre-libs --test self_consumer_conform` |
| `test` | `self_consumer_conform` | `vyre-libs/tests/self_consumer_conform.rs` | `cpu-parity`, `encoding` | `./cargo_full test -p vyre-libs --test self_consumer_conform` |
| `test` | `semiring_gemm_via_reference_parity` | `vyre-libs/tests/semiring_gemm_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test semiring_gemm_via_reference_parity` |
| `test` | `semiring_gemm_via_reference_parity` | `vyre-libs/tests/semiring_gemm_via_reference_parity.rs` | `analysis`, `cpu-parity` | `./cargo_full test -p vyre-libs --test semiring_gemm_via_reference_parity` |
| `test` | `semiring_gemm_wide_parity` | `vyre-libs/tests/semiring_gemm_wide_parity.rs` | None | `./cargo_full test -p vyre-libs --test semiring_gemm_wide_parity` |
| `test` | `semiring_registry` | `vyre-libs/tests/semiring_registry.rs` | None | `./cargo_full test -p vyre-libs --test semiring_registry` |
| `test` | `set_domain_selector` | `vyre-libs/tests/set_domain_selector.rs` | None | `./cargo_full test -p vyre-libs --test set_domain_selector` |
| `test` | `shape_spectrum_via_reference_parity` | `vyre-libs/tests/shape_spectrum_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test shape_spectrum_via_reference_parity` |
| `test` | `shape_spectrum_via_reference_parity` | `vyre-libs/tests/shape_spectrum_via_reference_parity.rs` | `cpu-parity`, `scheduling` | `./cargo_full test -p vyre-libs --test shape_spectrum_via_reference_parity` |
| `test` | `shared_emitter_artifact_schema` | `vyre-libs/tests/shared_emitter_artifact_schema.rs` | None | `./cargo_full test -p vyre-libs --test shared_emitter_artifact_schema` |
| `test` | `shared_owner_closure` | `vyre-libs/tests/shared_owner_closure.rs` | None | `./cargo_full test -p vyre-libs --test shared_owner_closure` |
| `test` | `sheaf_diffusion_step_signed_parity` | `vyre-libs/tests/sheaf_diffusion_step_signed_parity.rs` | None | `./cargo_full test -p vyre-libs --test sheaf_diffusion_step_signed_parity` |
| `test` | `sheaf_heterophilic_via_reference_parity` | `vyre-libs/tests/sheaf_heterophilic_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test sheaf_heterophilic_via_reference_parity` |
| `test` | `sheaf_heterophilic_via_reference_parity` | `vyre-libs/tests/sheaf_heterophilic_via_reference_parity.rs` | `cpu-parity`, `solvers` | `./cargo_full test -p vyre-libs --test sheaf_heterophilic_via_reference_parity` |
| `test` | `sheaf_laplacian_eigenvalue_dispatch_parity` | `vyre-libs/tests/sheaf_laplacian_eigenvalue_dispatch_parity.rs` | None | `./cargo_full test -p vyre-libs --test sheaf_laplacian_eigenvalue_dispatch_parity` |
| `test` | `sheaf_spectrum_via_reference_parity` | `vyre-libs/tests/sheaf_spectrum_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test sheaf_spectrum_via_reference_parity` |
| `test` | `sheaf_spectrum_via_reference_parity` | `vyre-libs/tests/sheaf_spectrum_via_reference_parity.rs` | `cpu-parity`, `solvers` | `./cargo_full test -p vyre-libs --test sheaf_spectrum_via_reference_parity` |
| `test` | `sigmoid_gate_typed_contract` | `vyre-libs/tests/sigmoid_gate_typed_contract.rs` | None | `./cargo_full test -p vyre-libs --test sigmoid_gate_typed_contract` |
| `test` | `sigmoid_gate_typed_contract` | `vyre-libs/tests/sigmoid_gate_typed_contract.rs` | `nn-activation` | `./cargo_full test -p vyre-libs --test sigmoid_gate_typed_contract` |
| `test` | `simplicial_triangle_message_fixed_point_parity` | `vyre-libs/tests/simplicial_triangle_message_fixed_point_parity.rs` | None | `./cargo_full test -p vyre-libs --test simplicial_triangle_message_fixed_point_parity` |
| `test` | `sinkhorn_iterate_ir_parity` | `vyre-libs/tests/sinkhorn_iterate_ir_parity.rs` | None | `./cargo_full test -p vyre-libs --test sinkhorn_iterate_ir_parity` |
| `test` | `sinkhorn_scale_ir_parity_proptest` | `vyre-libs/tests/sinkhorn_scale_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test sinkhorn_scale_ir_parity_proptest` |
| `test` | `sinkhorn_via_reference_parity` | `vyre-libs/tests/sinkhorn_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test sinkhorn_via_reference_parity` |
| `test` | `sinkhorn_via_reference_parity` | `vyre-libs/tests/sinkhorn_via_reference_parity.rs` | `cpu-parity`, `solvers` | `./cargo_full test -p vyre-libs --test sinkhorn_via_reference_parity` |
| `test` | `smooth_latency_trace_via_reference_parity` | `vyre-libs/tests/smooth_latency_trace_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test smooth_latency_trace_via_reference_parity` |
| `test` | `smooth_latency_trace_via_reference_parity` | `vyre-libs/tests/smooth_latency_trace_via_reference_parity.rs` | `cpu-parity`, `solvers` | `./cargo_full test -p vyre-libs --test smooth_latency_trace_via_reference_parity` |
| `test` | `smooth_matroid_flow_via_reference_parity` | `vyre-libs/tests/smooth_matroid_flow_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test smooth_matroid_flow_via_reference_parity` |
| `test` | `smooth_matroid_flow_via_reference_parity` | `vyre-libs/tests/smooth_matroid_flow_via_reference_parity.rs` | `cpu-parity`, `solvers` | `./cargo_full test -p vyre-libs --test smooth_matroid_flow_via_reference_parity` |
| `test` | `softmax_pick_config_via_reference_parity` | `vyre-libs/tests/softmax_pick_config_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test softmax_pick_config_via_reference_parity` |
| `test` | `softmax_pick_config_via_reference_parity` | `vyre-libs/tests/softmax_pick_config_via_reference_parity.rs` | `cpu-parity`, `solvers` | `./cargo_full test -p vyre-libs --test softmax_pick_config_via_reference_parity` |
| `test` | `solvers_dispatch_softmax_contract` | `vyre-libs/tests/solvers_dispatch_softmax_contract.rs` | None | `./cargo_full test -p vyre-libs --test solvers_dispatch_softmax_contract` |
| `test` | `solvers_dispatch_softmax_contract` | `vyre-libs/tests/solvers_dispatch_softmax_contract.rs` | `solvers` | `./cargo_full test -p vyre-libs --test solvers_dispatch_softmax_contract` |
| `test` | `sos_gram_construct_proptest` | `vyre-libs/tests/sos_gram_construct_proptest.rs` | None | `./cargo_full test -p vyre-libs --test sos_gram_construct_proptest` |
| `test` | `sos_gram_oob_parity` | `vyre-libs/tests/sos_gram_oob_parity.rs` | None | `./cargo_full test -p vyre-libs --test sos_gram_oob_parity` |
| `test` | `source_span_witness_records` | `vyre-libs/tests/source_span_witness_records.rs` | None | `./cargo_full test -p vyre-libs --test source_span_witness_records` |
| `test` | `ssa_dominance_phi_overflow_parity` | `vyre-libs/tests/ssa_dominance_phi_overflow_parity.rs` | None | `./cargo_full test -p vyre-libs --test ssa_dominance_phi_overflow_parity` |
| `test` | `stream_compact_proptest` | `vyre-libs/tests/stream_compact_proptest.rs` | None | `./cargo_full test -p vyre-libs --test stream_compact_proptest` |
| `test` | `string_diagram_via_reference_parity` | `vyre-libs/tests/string_diagram_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test string_diagram_via_reference_parity` |
| `test` | `string_diagram_via_reference_parity` | `vyre-libs/tests/string_diagram_via_reference_parity.rs` | `cpu-parity`, `reasoning` | `./cargo_full test -p vyre-libs --test string_diagram_via_reference_parity` |
| `test` | `subgroup_nfa_ir_parity_proptest` | `vyre-libs/tests/subgroup_nfa_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test subgroup_nfa_ir_parity_proptest` |
| `test` | `submodular_retention_via_reference_parity` | `vyre-libs/tests/submodular_retention_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test submodular_retention_via_reference_parity` |
| `test` | `submodular_retention_via_reference_parity` | `vyre-libs/tests/submodular_retention_via_reference_parity.rs` | `cpu-parity`, `scheduling` | `./cargo_full test -p vyre-libs --test submodular_retention_via_reference_parity` |
| `test` | `succinct_rank_contracts` | `vyre-libs/tests/succinct_rank_contracts.rs` | None | `./cargo_full test -p vyre-libs --test succinct_rank_contracts` |
| `test` | `succinct_rank_select_adversarial_contracts` | `vyre-libs/tests/succinct_rank_select_adversarial_contracts.rs` | None | `./cargo_full test -p vyre-libs --test succinct_rank_select_adversarial_contracts` |
| `test` | `sum_product_signed_parity` | `vyre-libs/tests/sum_product_signed_parity.rs` | None | `./cargo_full test -p vyre-libs --test sum_product_signed_parity` |
| `test` | `surface_contracts` | `vyre-libs/tests/surface_contracts.rs` | None | `./cargo_full test -p vyre-libs --test surface_contracts` |
| `test` | `surface_contracts` | `vyre-libs/tests/surface_contracts.rs` | `nn-attention`, `nn-norm` | `./cargo_full test -p vyre-libs --test surface_contracts` |
| `test` | `sweep_bitset_oracle_matrix` | `vyre-libs/tests/sweep_bitset_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_bitset_oracle_matrix` |
| `test` | `sweep_bitset_oracle_matrix` | `vyre-libs/tests/sweep_bitset_oracle_matrix.rs` | `bitset`, `cpu-parity` | `./cargo_full test -p vyre-libs --test sweep_bitset_oracle_matrix` |
| `test` | `sweep_decode_base64_volume_oracle_matrix` | `vyre-libs/tests/sweep_decode_base64_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_decode_base64_volume_oracle_matrix` |
| `test` | `sweep_decode_base64_volume_oracle_matrix` | `vyre-libs/tests/sweep_decode_base64_volume_oracle_matrix.rs` | `decode` | `./cargo_full test -p vyre-libs --test sweep_decode_base64_volume_oracle_matrix` |
| `test` | `sweep_decode_hex_oracle_matrix` | `vyre-libs/tests/sweep_decode_hex_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_decode_hex_oracle_matrix` |
| `test` | `sweep_decode_hex_oracle_matrix` | `vyre-libs/tests/sweep_decode_hex_oracle_matrix.rs` | `decode` | `./cargo_full test -p vyre-libs --test sweep_decode_hex_oracle_matrix` |
| `test` | `sweep_decode_hex_primitives_volume_oracle_matrix` | `vyre-libs/tests/sweep_decode_hex_primitives_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_decode_hex_primitives_volume_oracle_matrix` |
| `test` | `sweep_decode_hex_primitives_volume_oracle_matrix` | `vyre-libs/tests/sweep_decode_hex_primitives_volume_oracle_matrix.rs` | `decode` | `./cargo_full test -p vyre-libs --test sweep_decode_hex_primitives_volume_oracle_matrix` |
| `test` | `sweep_graph_cpu_oracle_matrix` | `vyre-libs/tests/sweep_graph_cpu_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_graph_cpu_oracle_matrix` |
| `test` | `sweep_graph_cpu_oracle_matrix` | `vyre-libs/tests/sweep_graph_cpu_oracle_matrix.rs` | `cpu-parity`, `graph-dispatch` | `./cargo_full test -p vyre-libs --test sweep_graph_cpu_oracle_matrix` |
| `test` | `sweep_graph_csr_backward_traverse_volume_oracle_matrix` | `vyre-libs/tests/sweep_graph_csr_backward_traverse_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_graph_csr_backward_traverse_volume_oracle_matrix` |
| `test` | `sweep_graph_csr_backward_traverse_volume_oracle_matrix` | `vyre-libs/tests/sweep_graph_csr_backward_traverse_volume_oracle_matrix.rs` | `cpu-parity`, `graph` | `./cargo_full test -p vyre-libs --test sweep_graph_csr_backward_traverse_volume_oracle_matrix` |
| `test` | `sweep_graph_csr_bidirectional_oracle_matrix` | `vyre-libs/tests/sweep_graph_csr_bidirectional_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_graph_csr_bidirectional_oracle_matrix` |
| `test` | `sweep_graph_csr_bidirectional_oracle_matrix` | `vyre-libs/tests/sweep_graph_csr_bidirectional_oracle_matrix.rs` | `cpu-parity`, `graph` | `./cargo_full test -p vyre-libs --test sweep_graph_csr_bidirectional_oracle_matrix` |
| `test` | `sweep_graph_csr_forward_or_changed_volume_oracle_matrix` | `vyre-libs/tests/sweep_graph_csr_forward_or_changed_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_graph_csr_forward_or_changed_volume_oracle_matrix` |
| `test` | `sweep_graph_csr_forward_or_changed_volume_oracle_matrix` | `vyre-libs/tests/sweep_graph_csr_forward_or_changed_volume_oracle_matrix.rs` | `cpu-parity`, `graph` | `./cargo_full test -p vyre-libs --test sweep_graph_csr_forward_or_changed_volume_oracle_matrix` |
| `test` | `sweep_graph_csr_forward_traverse_volume_oracle_matrix` | `vyre-libs/tests/sweep_graph_csr_forward_traverse_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_graph_csr_forward_traverse_volume_oracle_matrix` |
| `test` | `sweep_graph_csr_forward_traverse_volume_oracle_matrix` | `vyre-libs/tests/sweep_graph_csr_forward_traverse_volume_oracle_matrix.rs` | `cpu-parity`, `graph` | `./cargo_full test -p vyre-libs --test sweep_graph_csr_forward_traverse_volume_oracle_matrix` |
| `test` | `sweep_graph_motif_oracle_matrix` | `vyre-libs/tests/sweep_graph_motif_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_graph_motif_oracle_matrix` |
| `test` | `sweep_graph_motif_oracle_matrix` | `vyre-libs/tests/sweep_graph_motif_oracle_matrix.rs` | `cpu-parity`, `graph` | `./cargo_full test -p vyre-libs --test sweep_graph_motif_oracle_matrix` |
| `test` | `sweep_graph_path_reconstruct_oracle_matrix` | `vyre-libs/tests/sweep_graph_path_reconstruct_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_graph_path_reconstruct_oracle_matrix` |
| `test` | `sweep_graph_path_reconstruct_oracle_matrix` | `vyre-libs/tests/sweep_graph_path_reconstruct_oracle_matrix.rs` | `cpu-parity`, `graph` | `./cargo_full test -p vyre-libs --test sweep_graph_path_reconstruct_oracle_matrix` |
| `test` | `sweep_graph_persistent_bfs_volume_oracle_matrix` | `vyre-libs/tests/sweep_graph_persistent_bfs_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_graph_persistent_bfs_volume_oracle_matrix` |
| `test` | `sweep_graph_persistent_bfs_volume_oracle_matrix` | `vyre-libs/tests/sweep_graph_persistent_bfs_volume_oracle_matrix.rs` | `cpu-parity`, `graph` | `./cargo_full test -p vyre-libs --test sweep_graph_persistent_bfs_volume_oracle_matrix` |
| `test` | `sweep_graph_reachable_volume_oracle_matrix` | `vyre-libs/tests/sweep_graph_reachable_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_graph_reachable_volume_oracle_matrix` |
| `test` | `sweep_graph_reachable_volume_oracle_matrix` | `vyre-libs/tests/sweep_graph_reachable_volume_oracle_matrix.rs` | `cpu-parity`, `graph` | `./cargo_full test -p vyre-libs --test sweep_graph_reachable_volume_oracle_matrix` |
| `test` | `sweep_graph_scc_decompose_volume_oracle_matrix` | `vyre-libs/tests/sweep_graph_scc_decompose_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_graph_scc_decompose_volume_oracle_matrix` |
| `test` | `sweep_graph_scc_decompose_volume_oracle_matrix` | `vyre-libs/tests/sweep_graph_scc_decompose_volume_oracle_matrix.rs` | `cpu-parity`, `graph` | `./cargo_full test -p vyre-libs --test sweep_graph_scc_decompose_volume_oracle_matrix` |
| `test` | `sweep_hash_adler32_volume_oracle_matrix` | `vyre-libs/tests/sweep_hash_adler32_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_hash_adler32_volume_oracle_matrix` |
| `test` | `sweep_hash_adler32_volume_oracle_matrix` | `vyre-libs/tests/sweep_hash_adler32_volume_oracle_matrix.rs` | `hash` | `./cargo_full test -p vyre-libs --test sweep_hash_adler32_volume_oracle_matrix` |
| `test` | `sweep_hash_blake3_g_volume_oracle_matrix` | `vyre-libs/tests/sweep_hash_blake3_g_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_hash_blake3_g_volume_oracle_matrix` |
| `test` | `sweep_hash_blake3_g_volume_oracle_matrix` | `vyre-libs/tests/sweep_hash_blake3_g_volume_oracle_matrix.rs` | `cpu-parity`, `hash` | `./cargo_full test -p vyre-libs --test sweep_hash_blake3_g_volume_oracle_matrix` |
| `test` | `sweep_hash_blake3_round_volume_oracle_matrix` | `vyre-libs/tests/sweep_hash_blake3_round_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_hash_blake3_round_volume_oracle_matrix` |
| `test` | `sweep_hash_blake3_round_volume_oracle_matrix` | `vyre-libs/tests/sweep_hash_blake3_round_volume_oracle_matrix.rs` | `cpu-parity`, `hash` | `./cargo_full test -p vyre-libs --test sweep_hash_blake3_round_volume_oracle_matrix` |
| `test` | `sweep_hash_crc32_reference_matrix` | `vyre-libs/tests/sweep_hash_crc32_reference_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_hash_crc32_reference_matrix` |
| `test` | `sweep_hash_crc32_reference_matrix` | `vyre-libs/tests/sweep_hash_crc32_reference_matrix.rs` | `hash` | `./cargo_full test -p vyre-libs --test sweep_hash_crc32_reference_matrix` |
| `test` | `sweep_hash_crc32_volume_oracle_matrix` | `vyre-libs/tests/sweep_hash_crc32_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_hash_crc32_volume_oracle_matrix` |
| `test` | `sweep_hash_crc32_volume_oracle_matrix` | `vyre-libs/tests/sweep_hash_crc32_volume_oracle_matrix.rs` | `hash` | `./cargo_full test -p vyre-libs --test sweep_hash_crc32_volume_oracle_matrix` |
| `test` | `sweep_hash_crc_oracle_matrix` | `vyre-libs/tests/sweep_hash_crc_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_hash_crc_oracle_matrix` |
| `test` | `sweep_hash_crc_oracle_matrix` | `vyre-libs/tests/sweep_hash_crc_oracle_matrix.rs` | `hash` | `./cargo_full test -p vyre-libs --test sweep_hash_crc_oracle_matrix` |
| `test` | `sweep_hash_fnv1a_oracle_matrix` | `vyre-libs/tests/sweep_hash_fnv1a_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_hash_fnv1a_oracle_matrix` |
| `test` | `sweep_hash_fnv1a_oracle_matrix` | `vyre-libs/tests/sweep_hash_fnv1a_oracle_matrix.rs` | `hash` | `./cargo_full test -p vyre-libs --test sweep_hash_fnv1a_oracle_matrix` |
| `test` | `sweep_hash_multi_hash_volume_oracle_matrix` | `vyre-libs/tests/sweep_hash_multi_hash_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_hash_multi_hash_volume_oracle_matrix` |
| `test` | `sweep_hash_multi_hash_volume_oracle_matrix` | `vyre-libs/tests/sweep_hash_multi_hash_volume_oracle_matrix.rs` | `hash` | `./cargo_full test -p vyre-libs --test sweep_hash_multi_hash_volume_oracle_matrix` |
| `test` | `sweep_hash_volume_oracle_matrix` | `vyre-libs/tests/sweep_hash_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_hash_volume_oracle_matrix` |
| `test` | `sweep_hash_volume_oracle_matrix` | `vyre-libs/tests/sweep_hash_volume_oracle_matrix.rs` | `hash` | `./cargo_full test -p vyre-libs --test sweep_hash_volume_oracle_matrix` |
| `test` | `sweep_logical_reference_matrix` | `vyre-libs/tests/sweep_logical_reference_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_logical_reference_matrix` |
| `test` | `sweep_logical_reference_matrix` | `vyre-libs/tests/sweep_logical_reference_matrix.rs` | `logical` | `./cargo_full test -p vyre-libs --test sweep_logical_reference_matrix` |
| `test` | `sweep_math_prefix_scan_exclusive_volume_oracle_matrix` | `vyre-libs/tests/sweep_math_prefix_scan_exclusive_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_math_prefix_scan_exclusive_volume_oracle_matrix` |
| `test` | `sweep_math_prefix_scan_exclusive_volume_oracle_matrix` | `vyre-libs/tests/sweep_math_prefix_scan_exclusive_volume_oracle_matrix.rs` | `cpu-parity`, `math-kernels` | `./cargo_full test -p vyre-libs --test sweep_math_prefix_scan_exclusive_volume_oracle_matrix` |
| `test` | `sweep_math_prefix_scan_inclusive_volume_oracle_matrix` | `vyre-libs/tests/sweep_math_prefix_scan_inclusive_volume_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_math_prefix_scan_inclusive_volume_oracle_matrix` |
| `test` | `sweep_math_prefix_scan_inclusive_volume_oracle_matrix` | `vyre-libs/tests/sweep_math_prefix_scan_inclusive_volume_oracle_matrix.rs` | `cpu-parity`, `math-kernels` | `./cargo_full test -p vyre-libs --test sweep_math_prefix_scan_inclusive_volume_oracle_matrix` |
| `test` | `sweep_predicate_node_kind_oracle_matrix` | `vyre-libs/tests/sweep_predicate_node_kind_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_predicate_node_kind_oracle_matrix` |
| `test` | `sweep_predicate_node_kind_oracle_matrix` | `vyre-libs/tests/sweep_predicate_node_kind_oracle_matrix.rs` | `cpu-parity`, `predicate` | `./cargo_full test -p vyre-libs --test sweep_predicate_node_kind_oracle_matrix` |
| `test` | `sweep_radix_sort_oracle_matrix` | `vyre-libs/tests/sweep_radix_sort_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_radix_sort_oracle_matrix` |
| `test` | `sweep_radix_sort_oracle_matrix` | `vyre-libs/tests/sweep_radix_sort_oracle_matrix.rs` | `cpu-parity`, `reduce` | `./cargo_full test -p vyre-libs --test sweep_radix_sort_oracle_matrix` |
| `test` | `sweep_reduce_oracle_matrix` | `vyre-libs/tests/sweep_reduce_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_reduce_oracle_matrix` |
| `test` | `sweep_reduce_oracle_matrix` | `vyre-libs/tests/sweep_reduce_oracle_matrix.rs` | `cpu-parity`, `reduce` | `./cargo_full test -p vyre-libs --test sweep_reduce_oracle_matrix` |
| `test` | `sweep_segment_reduce_oracle_matrix` | `vyre-libs/tests/sweep_segment_reduce_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_segment_reduce_oracle_matrix` |
| `test` | `sweep_segment_reduce_oracle_matrix` | `vyre-libs/tests/sweep_segment_reduce_oracle_matrix.rs` | `cpu-parity`, `reduce` | `./cargo_full test -p vyre-libs --test sweep_segment_reduce_oracle_matrix` |
| `test` | `sweep_text_utf8_oracle_matrix` | `vyre-libs/tests/sweep_text_utf8_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_text_utf8_oracle_matrix` |
| `test` | `sweep_text_utf8_oracle_matrix` | `vyre-libs/tests/sweep_text_utf8_oracle_matrix.rs` | `text` | `./cargo_full test -p vyre-libs --test sweep_text_utf8_oracle_matrix` |
| `test` | `sweep_toposort_oracle_matrix` | `vyre-libs/tests/sweep_toposort_oracle_matrix.rs` | None | `./cargo_full test -p vyre-libs --test sweep_toposort_oracle_matrix` |
| `test` | `sweep_toposort_oracle_matrix` | `vyre-libs/tests/sweep_toposort_oracle_matrix.rs` | `cpu-parity`, `graph` | `./cargo_full test -p vyre-libs --test sweep_toposort_oracle_matrix` |
| `test` | `symmetric_eigen_jacobi_parity` | `vyre-libs/tests/symmetric_eigen_jacobi_parity.rs` | None | `./cargo_full test -p vyre-libs --test symmetric_eigen_jacobi_parity` |
| `test` | `symmetric_eigen_jacobi_registration` | `vyre-libs/tests/symmetric_eigen_jacobi_registration.rs` | None | `./cargo_full test -p vyre-libs --test symmetric_eigen_jacobi_registration` |
| `test` | `syntax_motif_frontier_compiler` | `vyre-libs/tests/syntax_motif_frontier_compiler.rs` | None | `./cargo_full test -p vyre-libs --test syntax_motif_frontier_compiler` |
| `test` | `taint_pollution_grid_sync_planner_cut` | `vyre-libs/tests/taint_pollution_grid_sync_planner_cut.rs` | None | `./cargo_full test -p vyre-libs --test taint_pollution_grid_sync_planner_cut` |
| `test` | `target_instruction_capabilities` | `vyre-libs/tests/target_instruction_capabilities.rs` | None | `./cargo_full test -p vyre-libs --test target_instruction_capabilities` |
| `test` | `tensor_scc_value_parity` | `vyre-libs/tests/tensor_scc_value_parity.rs` | None | `./cargo_full test -p vyre-libs --test tensor_scc_value_parity` |
| `test` | `tensor_train_chain_fusion_via_reference_parity` | `vyre-libs/tests/tensor_train_chain_fusion_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test tensor_train_chain_fusion_via_reference_parity` |
| `test` | `tensor_train_chain_fusion_via_reference_parity` | `vyre-libs/tests/tensor_train_chain_fusion_via_reference_parity.rs` | `cpu-parity`, `solvers` | `./cargo_full test -p vyre-libs --test tensor_train_chain_fusion_via_reference_parity` |
| `test` | `tensor_train_compress_via_reference_parity` | `vyre-libs/tests/tensor_train_compress_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test tensor_train_compress_via_reference_parity` |
| `test` | `tensor_train_compress_via_reference_parity` | `vyre-libs/tests/tensor_train_compress_via_reference_parity.rs` | `cpu-parity`, `solvers` | `./cargo_full test -p vyre-libs --test tensor_train_compress_via_reference_parity` |
| `test` | `tensor_train_contract_signed_parity` | `vyre-libs/tests/tensor_train_contract_signed_parity.rs` | None | `./cargo_full test -p vyre-libs --test tensor_train_contract_signed_parity` |
| `test` | `tensor_train_decompose_eigen_contract` | `vyre-libs/tests/tensor_train_decompose_eigen_contract.rs` | None | `./cargo_full test -p vyre-libs --test tensor_train_decompose_eigen_contract` |
| `test` | `tensor_train_decompose_step_parity` | `vyre-libs/tests/tensor_train_decompose_step_parity.rs` | None | `./cargo_full test -p vyre-libs --test tensor_train_decompose_step_parity` |
| `test` | `text_char_class_runner` | `vyre-libs/tests/text_char_class_runner.rs` | None | `./cargo_full test -p vyre-libs --test text_char_class_runner` |
| `test` | `tfn_scalar_mix_signed_parity` | `vyre-libs/tests/tfn_scalar_mix_signed_parity.rs` | None | `./cargo_full test -p vyre-libs --test tfn_scalar_mix_signed_parity` |
| `test` | `toposort_program_value_parity` | `vyre-libs/tests/toposort_program_value_parity.rs` | None | `./cargo_full test -p vyre-libs --test toposort_program_value_parity` |
| `test` | `transport_residual_via_reference_parity` | `vyre-libs/tests/transport_residual_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test transport_residual_via_reference_parity` |
| `test` | `transport_residual_via_reference_parity` | `vyre-libs/tests/transport_residual_via_reference_parity.rs` | `cpu-parity`, `solvers` | `./cargo_full test -p vyre-libs --test transport_residual_via_reference_parity` |
| `test` | `union_find_alias_via_reference_parity` | `vyre-libs/tests/union_find_alias_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test union_find_alias_via_reference_parity` |
| `test` | `union_find_alias_via_reference_parity` | `vyre-libs/tests/union_find_alias_via_reference_parity.rs` | `cpu-parity`, `graph-dispatch` | `./cargo_full test -p vyre-libs --test union_find_alias_via_reference_parity` |
| `test` | `union_find_connectivity_parity` | `vyre-libs/tests/union_find_connectivity_parity.rs` | None | `./cargo_full test -p vyre-libs --test union_find_connectivity_parity` |
| `test` | `universal_harness` | `vyre-libs/tests/universal_harness.rs` | None | `./cargo_full test -p vyre-libs --test universal_harness` |
| `test` | `unsafe_ffi_policies` | `vyre-libs/tests/unsafe_ffi_policies.rs` | None | `./cargo_full test -p vyre-libs --test unsafe_ffi_policies` |
| `test` | `url_network_security_policies` | `vyre-libs/tests/url_network_security_policies.rs` | None | `./cargo_full test -p vyre-libs --test url_network_security_policies` |
| `test` | `utf8_shape_counts_ir_parity_proptest` | `vyre-libs/tests/utf8_shape_counts_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test utf8_shape_counts_ir_parity_proptest` |
| `test` | `vast_tree_walk_ir_parity_proptest` | `vyre-libs/tests/vast_tree_walk_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test vast_tree_walk_ir_parity_proptest` |
| `test` | `vietoris_rips_via_reference_parity` | `vyre-libs/tests/vietoris_rips_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test vietoris_rips_via_reference_parity` |
| `test` | `vietoris_rips_via_reference_parity` | `vyre-libs/tests/vietoris_rips_via_reference_parity.rs` | `cpu-parity`, `solvers` | `./cargo_full test -p vyre-libs --test vietoris_rips_via_reference_parity` |
| `test` | `visual_compositions` | `vyre-libs/tests/visual_compositions.rs` | None | `./cargo_full test -p vyre-libs --test visual_compositions` |
| `test` | `vsa_fingerprint_via_reference_parity` | `vyre-libs/tests/vsa_fingerprint_via_reference_parity.rs` | None | `./cargo_full test -p vyre-libs --test vsa_fingerprint_via_reference_parity` |
| `test` | `vsa_fingerprint_via_reference_parity` | `vyre-libs/tests/vsa_fingerprint_via_reference_parity.rs` | `cpu-parity`, `encoding` | `./cargo_full test -p vyre-libs --test vsa_fingerprint_via_reference_parity` |
| `test` | `wire_cross_crate_compat` | `vyre-libs/tests/wire_cross_crate_compat.rs` | None | `./cargo_full test -p vyre-libs --test wire_cross_crate_compat` |
| `test` | `workgroup_any_ir_parity_proptest` | `vyre-libs/tests/workgroup_any_ir_parity_proptest.rs` | None | `./cargo_full test -p vyre-libs --test workgroup_any_ir_parity_proptest` |
| `test` | `workgroup_cooperative_tiling` | `vyre-libs/tests/workgroup_cooperative_tiling.rs` | None | `./cargo_full test -p vyre-libs --test workgroup_cooperative_tiling` |
| `test` | `workgroup_cooperative_tiling` | `vyre-libs/tests/workgroup_cooperative_tiling.rs` | `nn-attention`, `nn-norm` | `./cargo_full test -p vyre-libs --test workgroup_cooperative_tiling` |

## Test classes

- Product-library exact behavior
- Primitive-to-library composition
- Reference and backend parity

## Hardware requirements

Reference and builder suites are host-capable. Tests that request concrete backend parity require that device and fail visibly when unavailable.

## Evidence outputs

- No persistent release artifact. The command status and exact behavioral assertions are the proof.

## Skips and failures

The default command does not run tests marked `#[ignore]`. No executed test may silently treat a missing requested backend or device as success.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
