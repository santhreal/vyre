# Testing `vyre-libs`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs
```

Own product-facing Tier 3 program compositions built from neutral primitives and contracts.

The crate lives at `vyre-libs`. The `product-libraries` owner maintains its
`libraries` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs
```

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --all-features
```

## Feature sets

- Default feature members: `math-linalg`, `math-scan`, `math-broadcast`, `nn-activation`, `nn-linear`, `nn-norm`, `matching-substring`, `matching-dfa`, `hash`, `decode`
- Available manifest features: `bench`, `c-parser`, `cpu-parity`, `crypto`, `crypto-blake3`, `crypto-fnv`, `decode`, `default`, `full`, `go-parser`, `hash`, `intern`, `logical`, `matching`, `matching-dfa`, `matching-nfa`, `matching-regex`, `matching-substring`, `math`, `math-algebra`, `math-broadcast`, `math-linalg`, `math-scan`, `math-succinct`, `nn`, `nn-activation`, `nn-attention`, `nn-inference`, `nn-linear`, `nn-linear-4bit`, `nn-moe`, `nn-norm`, `parsing`, `python-parser`, `rule`, `security`, `test-fixtures`, `text`, `visual`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `check_select1` | `vyre-libs/examples/check_select1.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --example check_select1` |
| `example` | `scan_corpus_fast_path` | `vyre-libs/examples/scan_corpus_fast_path.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --example scan_corpus_fast_path` |
| `example` | `scan_paged_corpus` | `vyre-libs/examples/scan_paged_corpus.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --example scan_paged_corpus` |
| `lib` | `vyre_libs` | `vyre-libs/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs` |
| `test` | `ac_count_suffix3_naga_validation` | `vyre-libs/tests/ac_count_suffix3_naga_validation.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test ac_count_suffix3_naga_validation` |
| `test` | `adversarial` | `vyre-libs/tests/adversarial.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test adversarial` |
| `test` | `aho_corasick_kat` | `vyre-libs/tests/aho_corasick_kat.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test aho_corasick_kat` |
| `test` | `algebra_lattice_semiring_contracts` | `vyre-libs/tests/algebra_lattice_semiring_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test algebra_lattice_semiring_contracts` |
| `test` | `analysis_fact_schema` | `vyre-libs/tests/analysis_fact_schema.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test analysis_fact_schema` |
| `test` | `ast_shunting_yard` | `vyre-libs/tests/ast_shunting_yard.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test ast_shunting_yard` |
| `test` | `ast_shunting_yard` | `vyre-libs/tests/ast_shunting_yard.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test ast_shunting_yard` |
| `test` | `attention_head_to_token_contract` | `vyre-libs/tests/attention_head_to_token_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test attention_head_to_token_contract` |
| `test` | `blake3_compress_optimizer_idempotence_contract` | `vyre-libs/tests/blake3_compress_optimizer_idempotence_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test blake3_compress_optimizer_idempotence_contract` |
| `test` | `blake3_kat` | `vyre-libs/tests/blake3_kat.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test blake3_kat` |
| `test` | `blake3_wrong_size` | `vyre-libs/tests/blake3_wrong_size.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test blake3_wrong_size` |
| `test` | `buffer_name_cross_family` | `vyre-libs/tests/buffer_name_cross_family.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test buffer_name_cross_family` |
| `test` | `c11_function_extractor_contracts` | `vyre-libs/tests/c11_function_extractor_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c11_function_extractor_contracts` |
| `test` | `c11_function_extractor_contracts` | `vyre-libs/tests/c11_function_extractor_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c11_function_extractor_contracts` |
| `test` | `c11_keyword` | `vyre-libs/tests/c11_keyword.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c11_keyword` |
| `test` | `c11_keyword` | `vyre-libs/tests/c11_keyword.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c11_keyword` |
| `test` | `c11_lexer_naga_validation` | `vyre-libs/tests/c11_lexer_naga_validation.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c11_lexer_naga_validation` |
| `test` | `c11_lexer_naga_validation` | `vyre-libs/tests/c11_lexer_naga_validation.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c11_lexer_naga_validation` |
| `test` | `c11_statement_bounds_long_statement_span` | `vyre-libs/tests/c11_statement_bounds_long_statement_span.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c11_statement_bounds_long_statement_span` |
| `test` | `c_annotate_typedef_oracle_parity` | `vyre-libs/tests/c_annotate_typedef_oracle_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_annotate_typedef_oracle_parity` |
| `test` | `c_ast_c99_for_do_macro_e2e` | `vyre-libs/tests/c_ast_c99_for_do_macro_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_c99_for_do_macro_e2e` |
| `test` | `c_ast_c99_for_do_macro_e2e` | `vyre-libs/tests/c_ast_c99_for_do_macro_e2e.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_c99_for_do_macro_e2e` |
| `test` | `c_ast_container_of_e2e` | `vyre-libs/tests/c_ast_container_of_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_container_of_e2e` |
| `test` | `c_ast_container_of_e2e` | `vyre-libs/tests/c_ast_container_of_e2e.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_container_of_e2e` |
| `test` | `c_ast_gnu_asm_decomposition_and_attribute_kinds` | `vyre-libs/tests/c_ast_gnu_asm_decomposition_and_attribute_kinds.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_gnu_asm_decomposition_and_attribute_kinds` |
| `test` | `c_ast_gnu_asm_decomposition_and_attribute_kinds` | `vyre-libs/tests/c_ast_gnu_asm_decomposition_and_attribute_kinds.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_gnu_asm_decomposition_and_attribute_kinds` |
| `test` | `c_ast_gnu_builtin_vast_contracts` | `vyre-libs/tests/c_ast_gnu_builtin_vast_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_gnu_builtin_vast_contracts` |
| `test` | `c_ast_gnu_builtin_vast_contracts` | `vyre-libs/tests/c_ast_gnu_builtin_vast_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_gnu_builtin_vast_contracts` |
| `test` | `c_ast_linux_grade_gnu_and_c11_construct_coverage` | `vyre-libs/tests/c_ast_linux_grade_gnu_and_c11_construct_coverage.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_linux_grade_gnu_and_c11_construct_coverage` |
| `test` | `c_ast_linux_grade_gnu_and_c11_construct_coverage` | `vyre-libs/tests/c_ast_linux_grade_gnu_and_c11_construct_coverage.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_linux_grade_gnu_and_c11_construct_coverage` |
| `test` | `c_ast_linux_style_raw_source_contracts` | `vyre-libs/tests/c_ast_linux_style_raw_source_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_linux_style_raw_source_contracts` |
| `test` | `c_ast_linux_style_raw_source_contracts` | `vyre-libs/tests/c_ast_linux_style_raw_source_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_linux_style_raw_source_contracts` |
| `test` | `c_conditional_range_policy` | `vyre-libs/tests/c_conditional_range_policy.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_conditional_range_policy` |
| `test` | `c_global_typedef_annotate_parity` | `vyre-libs/tests/c_global_typedef_annotate_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_global_typedef_annotate_parity` |
| `test` | `c_lexer_preprocessor_hash_contracts` | `vyre-libs/tests/c_lexer_preprocessor_hash_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_lexer_preprocessor_hash_contracts` |
| `test` | `c_lexer_preprocessor_hash_contracts` | `vyre-libs/tests/c_lexer_preprocessor_hash_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_lexer_preprocessor_hash_contracts` |
| `test` | `c_lexer_regular_variant_parity` | `vyre-libs/tests/c_lexer_regular_variant_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_lexer_regular_variant_parity` |
| `test` | `c_lower_semantic_graph_control_resolution_parity` | `vyre-libs/tests/c_lower_semantic_graph_control_resolution_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_lower_semantic_graph_control_resolution_parity` |
| `test` | `c_packed_haystack_semantic_parity` | `vyre-libs/tests/c_packed_haystack_semantic_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_packed_haystack_semantic_parity` |
| `test` | `c_packed_haystack_semantic_parity` | `vyre-libs/tests/c_packed_haystack_semantic_parity.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_packed_haystack_semantic_parity` |
| `test` | `c_parser_pipeline_lexer_adversarial_contracts` | `vyre-libs/tests/c_parser_pipeline_lexer_adversarial_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_parser_pipeline_lexer_adversarial_contracts` |
| `test` | `c_parser_pipeline_lexer_adversarial_contracts` | `vyre-libs/tests/c_parser_pipeline_lexer_adversarial_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_parser_pipeline_lexer_adversarial_contracts` |
| `test` | `c_parser_pipeline_malformed_stream_contracts` | `vyre-libs/tests/c_parser_pipeline_malformed_stream_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_parser_pipeline_malformed_stream_contracts` |
| `test` | `c_parser_pipeline_malformed_stream_contracts` | `vyre-libs/tests/c_parser_pipeline_malformed_stream_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_parser_pipeline_malformed_stream_contracts` |
| `test` | `c_preprocess_certificates` | `vyre-libs/tests/c_preprocess_certificates.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_certificates` |
| `test` | `c_preprocess_classified_memory_cache_contract` | `vyre-libs/tests/c_preprocess_classified_memory_cache_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_classified_memory_cache_contract` |
| `test` | `c_preprocess_directive_count_contract` | `vyre-libs/tests/c_preprocess_directive_count_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_directive_count_contract` |
| `test` | `c_preprocess_directive_count_contract` | `vyre-libs/tests/c_preprocess_directive_count_contract.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_directive_count_contract` |
| `test` | `c_preprocess_directive_staging_allocation_contract` | `vyre-libs/tests/c_preprocess_directive_staging_allocation_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_directive_staging_allocation_contract` |
| `test` | `c_preprocess_dynamic_macro_expansion_contracts` | `vyre-libs/tests/c_preprocess_dynamic_macro_expansion_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_dynamic_macro_expansion_contracts` |
| `test` | `c_preprocess_dynamic_macro_expansion_contracts` | `vyre-libs/tests/c_preprocess_dynamic_macro_expansion_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_dynamic_macro_expansion_contracts` |
| `test` | `c_preprocess_dynamic_macro_optimizer_idempotence_contract` | `vyre-libs/tests/c_preprocess_dynamic_macro_optimizer_idempotence_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_dynamic_macro_optimizer_idempotence_contract` |
| `test` | `c_preprocess_dynamic_macro_optimizer_idempotence_contract` | `vyre-libs/tests/c_preprocess_dynamic_macro_optimizer_idempotence_contract.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_dynamic_macro_optimizer_idempotence_contract` |
| `test` | `c_preprocess_gpu_buffer_allocation_contract` | `vyre-libs/tests/c_preprocess_gpu_buffer_allocation_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_gpu_buffer_allocation_contract` |
| `test` | `c_preprocess_gpu_resident_state_contracts` | `vyre-libs/tests/c_preprocess_gpu_resident_state_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_gpu_resident_state_contracts` |
| `test` | `c_preprocess_gpu_resident_state_contracts` | `vyre-libs/tests/c_preprocess_gpu_resident_state_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_gpu_resident_state_contracts` |
| `test` | `c_preprocess_macro_expansion_cache_contract` | `vyre-libs/tests/c_preprocess_macro_expansion_cache_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_macro_expansion_cache_contract` |
| `test` | `c_preprocess_macro_table_allocation_contract` | `vyre-libs/tests/c_preprocess_macro_table_allocation_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_macro_table_allocation_contract` |
| `test` | `c_preprocess_named_macro_expansion_contracts` | `vyre-libs/tests/c_preprocess_named_macro_expansion_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_named_macro_expansion_contracts` |
| `test` | `c_preprocess_named_macro_expansion_contracts` | `vyre-libs/tests/c_preprocess_named_macro_expansion_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_named_macro_expansion_contracts` |
| `test` | `c_preprocess_pipeline_contracts` | `vyre-libs/tests/c_preprocess_pipeline_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_pipeline_contracts` |
| `test` | `c_preprocess_pipeline_contracts` | `vyre-libs/tests/c_preprocess_pipeline_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_pipeline_contracts` |
| `test` | `c_preprocess_prefix_scan_allocation_contract` | `vyre-libs/tests/c_preprocess_prefix_scan_allocation_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_prefix_scan_allocation_contract` |
| `test` | `c_preprocess_replacement_token_cache_contract` | `vyre-libs/tests/c_preprocess_replacement_token_cache_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_replacement_token_cache_contract` |
| `test` | `c_reference_decode_contracts` | `vyre-libs/tests/c_reference_decode_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_reference_decode_contracts` |
| `test` | `c_reference_decode_contracts` | `vyre-libs/tests/c_reference_decode_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_reference_decode_contracts` |
| `test` | `c_sema_scope_optimizer_idempotence_contract` | `vyre-libs/tests/c_sema_scope_optimizer_idempotence_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_sema_scope_optimizer_idempotence_contract` |
| `test` | `c_sema_scope_optimizer_idempotence_contract` | `vyre-libs/tests/c_sema_scope_optimizer_idempotence_contract.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_sema_scope_optimizer_idempotence_contract` |
| `test` | `c_typedef_precomputed_variant_parity` | `vyre-libs/tests/c_typedef_precomputed_variant_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_typedef_precomputed_variant_parity` |
| `test` | `c_vast_classify_wire_depth_contract` | `vyre-libs/tests/c_vast_classify_wire_depth_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_vast_classify_wire_depth_contract` |
| `test` | `c_vast_classify_wire_depth_contract` | `vyre-libs/tests/c_vast_classify_wire_depth_contract.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_vast_classify_wire_depth_contract` |
| `test` | `cache_key_collision` | `vyre-libs/tests/cache_key_collision.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test cache_key_collision` |
| `test` | `cat_a_conform` | `vyre-libs/tests/cat_a_conform.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test cat_a_conform` |
| `test` | `causal_conv_state_transition_contract` | `vyre-libs/tests/causal_conv_state_transition_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test causal_conv_state_transition_contract` |
| `test` | `causal_gqa_contract` | `vyre-libs/tests/causal_gqa_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test causal_gqa_contract` |
| `test` | `causal_gqa_typed_contract` | `vyre-libs/tests/causal_gqa_typed_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test causal_gqa_typed_contract` |
| `test` | `chunked_gated_delta_contract` | `vyre-libs/tests/chunked_gated_delta_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test chunked_gated_delta_contract` |
| `test` | `consumer_boundary` | `vyre-libs/tests/consumer_boundary.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test consumer_boundary` |
| `test` | `corpus_privacy_retention_controls` | `vyre-libs/tests/corpus_privacy_retention_controls.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test corpus_privacy_retention_controls` |
| `test` | `cpu_witnesses` | `vyre-libs/tests/cpu_witnesses.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test cpu_witnesses` |
| `test` | `cross_layer_parity` | `vyre-libs/tests/cross_layer_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test cross_layer_parity` |
| `test` | `decode_primitive_composition_contracts` | `vyre-libs/tests/decode_primitive_composition_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test decode_primitive_composition_contracts` |
| `test` | `delta_flow_arrangements` | `vyre-libs/tests/delta_flow_arrangements.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test delta_flow_arrangements` |
| `test` | `dense_gated_mlp_graph_contract` | `vyre-libs/tests/dense_gated_mlp_graph_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test dense_gated_mlp_graph_contract` |
| `test` | `depthwise_causal_conv1d_contract` | `vyre-libs/tests/depthwise_causal_conv1d_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test depthwise_causal_conv1d_contract` |
| `test` | `f32_adversarial` | `vyre-libs/tests/f32_adversarial.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test f32_adversarial` |
| `test` | `filesystem_path_archive_policies` | `vyre-libs/tests/filesystem_path_archive_policies.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test filesystem_path_archive_policies` |
| `test` | `fingerprint_lock` | `vyre-libs/tests/fingerprint_lock.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test fingerprint_lock` |
| `test` | `flow_precision_planner` | `vyre-libs/tests/flow_precision_planner.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test flow_precision_planner` |
| `test` | `frontend_dialect_contracts` | `vyre-libs/tests/frontend_dialect_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test frontend_dialect_contracts` |
| `test` | `fuse_decode_scan_error` | `vyre-libs/tests/fuse_decode_scan_error.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test fuse_decode_scan_error` |
| `test` | `fuzz_target_inventory` | `vyre-libs/tests/fuzz_target_inventory.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test fuzz_target_inventory` |
| `test` | `gated_rms_norm_contract` | `vyre-libs/tests/gated_rms_norm_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gated_rms_norm_contract` |
| `test` | `go_channel_creation_parity` | `vyre-libs/tests/go_channel_creation_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test go_channel_creation_parity` |
| `test` | `go_frontend_corpus` | `vyre-libs/tests/go_frontend_corpus.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test go_frontend_corpus` |
| `test` | `go_tokenizer_semantics` | `vyre-libs/tests/go_tokenizer_semantics.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test go_tokenizer_semantics` |
| `test` | `gpu_char_constant_scan_roundtrip` | `vyre-libs/tests/gpu_char_constant_scan_roundtrip.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_char_constant_scan_roundtrip` |
| `test` | `gpu_char_constant_scan_roundtrip` | `vyre-libs/tests/gpu_char_constant_scan_roundtrip.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_char_constant_scan_roundtrip` |
| `test` | `gpu_columnar_string_ingress` | `vyre-libs/tests/gpu_columnar_string_ingress.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_columnar_string_ingress` |
| `test` | `gpu_comment_strip_mask_roundtrip` | `vyre-libs/tests/gpu_comment_strip_mask_roundtrip.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_comment_strip_mask_roundtrip` |
| `test` | `gpu_comment_strip_mask_roundtrip` | `vyre-libs/tests/gpu_comment_strip_mask_roundtrip.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_comment_strip_mask_roundtrip` |
| `test` | `gpu_define_parse_roundtrip` | `vyre-libs/tests/gpu_define_parse_roundtrip.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_define_parse_roundtrip` |
| `test` | `gpu_define_parse_roundtrip` | `vyre-libs/tests/gpu_define_parse_roundtrip.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_define_parse_roundtrip` |
| `test` | `gpu_directive_metadata_roundtrip` | `vyre-libs/tests/gpu_directive_metadata_roundtrip.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_directive_metadata_roundtrip` |
| `test` | `gpu_directive_metadata_roundtrip` | `vyre-libs/tests/gpu_directive_metadata_roundtrip.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_directive_metadata_roundtrip` |
| `test` | `gpu_if_expression_roundtrip` | `vyre-libs/tests/gpu_if_expression_roundtrip.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_if_expression_roundtrip` |
| `test` | `gpu_if_expression_roundtrip` | `vyre-libs/tests/gpu_if_expression_roundtrip.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_if_expression_roundtrip` |
| `test` | `gpu_ifdef_value_roundtrip` | `vyre-libs/tests/gpu_ifdef_value_roundtrip.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_ifdef_value_roundtrip` |
| `test` | `gpu_ifdef_value_roundtrip` | `vyre-libs/tests/gpu_ifdef_value_roundtrip.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_ifdef_value_roundtrip` |
| `test` | `gpu_include_parse_roundtrip` | `vyre-libs/tests/gpu_include_parse_roundtrip.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_include_parse_roundtrip` |
| `test` | `gpu_include_parse_roundtrip` | `vyre-libs/tests/gpu_include_parse_roundtrip.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_include_parse_roundtrip` |
| `test` | `gpu_int_literal_scan_roundtrip` | `vyre-libs/tests/gpu_int_literal_scan_roundtrip.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_int_literal_scan_roundtrip` |
| `test` | `gpu_int_literal_scan_roundtrip` | `vyre-libs/tests/gpu_int_literal_scan_roundtrip.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_int_literal_scan_roundtrip` |
| `test` | `gpu_pipeline_driver_roundtrip` | `vyre-libs/tests/gpu_pipeline_driver_roundtrip.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_pipeline_driver_roundtrip` |
| `test` | `gpu_pipeline_driver_roundtrip` | `vyre-libs/tests/gpu_pipeline_driver_roundtrip.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_pipeline_driver_roundtrip` |
| `test` | `gpu_pipeline_extract_payloads_roundtrip` | `vyre-libs/tests/gpu_pipeline_extract_payloads_roundtrip.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_pipeline_extract_payloads_roundtrip` |
| `test` | `gpu_pipeline_extract_payloads_roundtrip` | `vyre-libs/tests/gpu_pipeline_extract_payloads_roundtrip.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_pipeline_extract_payloads_roundtrip` |
| `test` | `gpu_pipeline_filter_roundtrip` | `vyre-libs/tests/gpu_pipeline_filter_roundtrip.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_pipeline_filter_roundtrip` |
| `test` | `gpu_pipeline_filter_roundtrip` | `vyre-libs/tests/gpu_pipeline_filter_roundtrip.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_pipeline_filter_roundtrip` |
| `test` | `gpu_pipeline_lex_classify_roundtrip` | `vyre-libs/tests/gpu_pipeline_lex_classify_roundtrip.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_pipeline_lex_classify_roundtrip` |
| `test` | `gpu_pipeline_lex_classify_roundtrip` | `vyre-libs/tests/gpu_pipeline_lex_classify_roundtrip.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_pipeline_lex_classify_roundtrip` |
| `test` | `gpu_undef_parse_roundtrip` | `vyre-libs/tests/gpu_undef_parse_roundtrip.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_undef_parse_roundtrip` |
| `test` | `gpu_undef_parse_roundtrip` | `vyre-libs/tests/gpu_undef_parse_roundtrip.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gpu_undef_parse_roundtrip` |
| `test` | `gqa_attention_primitive_composition_contracts` | `vyre-libs/tests/gqa_attention_primitive_composition_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gqa_attention_primitive_composition_contracts` |
| `test` | `head_to_token_typed_contract` | `vyre-libs/tests/head_to_token_typed_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test head_to_token_typed_contract` |
| `test` | `hex_decode_scan_fused` | `vyre-libs/tests/hex_decode_scan_fused.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test hex_decode_scan_fused` |
| `test` | `indexed_map_composition_contracts` | `vyre-libs/tests/indexed_map_composition_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test indexed_map_composition_contracts` |
| `test` | `int4_primitive_composition` | `vyre-libs/tests/int4_primitive_composition.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test int4_primitive_composition` |
| `test` | `integration` | `vyre-libs/tests/integration.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test integration` |
| `test` | `ir_aliasing` | `vyre-libs/tests/ir_aliasing.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test ir_aliasing` |
| `test` | `ir_aliasing` | `vyre-libs/tests/ir_aliasing.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test ir_aliasing` |
| `test` | `kv_cache_append_contract` | `vyre-libs/tests/kv_cache_append_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test kv_cache_append_contract` |
| `test` | `kv_cache_typed_contract` | `vyre-libs/tests/kv_cache_typed_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test kv_cache_typed_contract` |
| `test` | `last_dim_l2_norm_contract` | `vyre-libs/tests/last_dim_l2_norm_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test last_dim_l2_norm_contract` |
| `test` | `linear_rows_contract` | `vyre-libs/tests/linear_rows_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test linear_rows_contract` |
| `test` | `literal_set_async_two_batch_pipeline` | `vyre-libs/tests/literal_set_async_two_batch_pipeline.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_async_two_batch_pipeline` |
| `test` | `literal_set_case_insensitive` | `vyre-libs/tests/literal_set_case_insensitive.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_case_insensitive` |
| `test` | `literal_set_count_generated` | `vyre-libs/tests/literal_set_count_generated.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_count_generated` |
| `test` | `literal_set_presence_and_positions_by_region_async` | `vyre-libs/tests/literal_set_presence_and_positions_by_region_async.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_presence_and_positions_by_region_async` |
| `test` | `literal_set_presence_and_positions_by_region_timed` | `vyre-libs/tests/literal_set_presence_and_positions_by_region_timed.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_presence_and_positions_by_region_timed` |
| `test` | `literal_set_presence_and_positions_gpu` | `vyre-libs/tests/literal_set_presence_and_positions_gpu.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_presence_and_positions_gpu` |
| `test` | `literal_set_presence_and_positions_reference` | `vyre-libs/tests/literal_set_presence_and_positions_reference.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_presence_and_positions_reference` |
| `test` | `literal_set_presence_async` | `vyre-libs/tests/literal_set_presence_async.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_presence_async` |
| `test` | `literal_set_presence_by_region_async` | `vyre-libs/tests/literal_set_presence_by_region_async.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_presence_by_region_async` |
| `test` | `literal_set_presence_by_region_gpu_ground_truth` | `vyre-libs/tests/literal_set_presence_by_region_gpu_ground_truth.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_presence_by_region_gpu_ground_truth` |
| `test` | `literal_set_presence_by_region_ground_truth` | `vyre-libs/tests/literal_set_presence_by_region_ground_truth.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_presence_by_region_ground_truth` |
| `test` | `literal_set_presence_by_region_timed` | `vyre-libs/tests/literal_set_presence_by_region_timed.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_presence_by_region_timed` |
| `test` | `literal_set_presence_gpu` | `vyre-libs/tests/literal_set_presence_gpu.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_presence_gpu` |
| `test` | `literal_set_presence_reference` | `vyre-libs/tests/literal_set_presence_reference.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_presence_reference` |
| `test` | `literal_set_presence_timed` | `vyre-libs/tests/literal_set_presence_timed.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_presence_timed` |
| `test` | `literal_set_resident_fused` | `vyre-libs/tests/literal_set_resident_fused.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_resident_fused` |
| `test` | `literal_set_resident_presence` | `vyre-libs/tests/literal_set_resident_presence.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_resident_presence` |
| `test` | `literal_set_resident_scan` | `vyre-libs/tests/literal_set_resident_scan.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_resident_scan` |
| `test` | `literal_set_scan_all` | `vyre-libs/tests/literal_set_scan_all.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_scan_all` |
| `test` | `literal_set_scan_all_timed` | `vyre-libs/tests/literal_set_scan_all_timed.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_scan_all_timed` |
| `test` | `literal_set_scan_differential_proptest` | `vyre-libs/tests/literal_set_scan_differential_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_scan_differential_proptest` |
| `test` | `literal_set_scan_into_async` | `vyre-libs/tests/literal_set_scan_into_async.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_scan_into_async` |
| `test` | `literal_set_scan_into_timed` | `vyre-libs/tests/literal_set_scan_into_timed.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_scan_into_timed` |
| `test` | `literal_set_wire_contracts` | `vyre-libs/tests/literal_set_wire_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_wire_contracts` |
| `test` | `literal_set_wire_versioning` | `vyre-libs/tests/literal_set_wire_versioning.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_wire_versioning` |
| `test` | `logical_proptest` | `vyre-libs/tests/logical_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test logical_proptest` |
| `test` | `logical_should_panic` | `vyre-libs/tests/logical_should_panic.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test logical_should_panic` |
| `test` | `loop_unroll_trip1_idempotence` | `vyre-libs/tests/loop_unroll_trip1_idempotence.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test loop_unroll_trip1_idempotence` |
| `test` | `lr_tables_contracts` | `vyre-libs/tests/lr_tables_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test lr_tables_contracts` |
| `test` | `matching_nfa_scan_program_contracts` | `vyre-libs/tests/matching_nfa_scan_program_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test matching_nfa_scan_program_contracts` |
| `test` | `matching_post_process_contracts` | `vyre-libs/tests/matching_post_process_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test matching_post_process_contracts` |
| `test` | `math_algebra_branchless_contracts` | `vyre-libs/tests/math_algebra_branchless_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test math_algebra_branchless_contracts` |
| `test` | `mlp_4x_leaky_sq_multi_workgroup_span` | `vyre-libs/tests/mlp_4x_leaky_sq_multi_workgroup_span.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test mlp_4x_leaky_sq_multi_workgroup_span` |
| `test` | `name_collision` | `vyre-libs/tests/name_collision.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test name_collision` |
| `test` | `nfa_plan_contracts` | `vyre-libs/tests/nfa_plan_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test nfa_plan_contracts` |
| `test` | `op_boundaries` | `vyre-libs/tests/op_boundaries.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test op_boundaries` |
| `test` | `operation_registry` | `vyre-libs/tests/operation_registry.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test operation_registry` |
| `test` | `operator_reporting_interchange` | `vyre-libs/tests/operator_reporting_interchange.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test operator_reporting_interchange` |
| `test` | `optimized_programs` | `vyre-libs/tests/optimized_programs.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test optimized_programs` |
| `test` | `output_encoding_unicode_policies` | `vyre-libs/tests/output_encoding_unicode_policies.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test output_encoding_unicode_policies` |
| `test` | `overflow_guards` | `vyre-libs/tests/overflow_guards.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test overflow_guards` |
| `test` | `parser_edit_delta_contracts` | `vyre-libs/tests/parser_edit_delta_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test parser_edit_delta_contracts` |
| `test` | `parser_graph_navigation_contracts` | `vyre-libs/tests/parser_graph_navigation_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test parser_graph_navigation_contracts` |
| `test` | `parser_recovery_corpus_registry` | `vyre-libs/tests/parser_recovery_corpus_registry.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test parser_recovery_corpus_registry` |
| `test` | `partial_rope_offset_contract` | `vyre-libs/tests/partial_rope_offset_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test partial_rope_offset_contract` |
| `test` | `partial_rope_typed_contract` | `vyre-libs/tests/partial_rope_typed_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test partial_rope_typed_contract` |
| `test` | `pass_research_trace_artifacts` | `vyre-libs/tests/pass_research_trace_artifacts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test pass_research_trace_artifacts` |
| `test` | `preprocess_cpu_api_boundary` | `vyre-libs/tests/preprocess_cpu_api_boundary.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test preprocess_cpu_api_boundary` |
| `test` | `primitive_surface_contracts` | `vyre-libs/tests/primitive_surface_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test primitive_surface_contracts` |
| `test` | `property` | `vyre-libs/tests/property.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test property` |
| `test` | `property_differential_oracles` | `vyre-libs/tests/property_differential_oracles.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test property_differential_oracles` |
| `test` | `qk_gain_shape_overflow_contracts` | `vyre-libs/tests/qk_gain_shape_overflow_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test qk_gain_shape_overflow_contracts` |
| `test` | `qk_gain_zero_shape_contracts` | `vyre-libs/tests/qk_gain_zero_shape_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test qk_gain_zero_shape_contracts` |
| `test` | `quantized_linear_affine_fma` | `vyre-libs/tests/quantized_linear_affine_fma.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test quantized_linear_affine_fma` |
| `test` | `recurrent_gated_delta_contract` | `vyre-libs/tests/recurrent_gated_delta_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test recurrent_gated_delta_contract` |
| `test` | `regex_adversarial_class_catalog` | `vyre-libs/tests/regex_adversarial_class_catalog.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test regex_adversarial_class_catalog` |
| `test` | `regex_anchored_window_gpu` | `vyre-libs/tests/regex_anchored_window_gpu.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test regex_anchored_window_gpu` |
| `test` | `regex_capture_mode_contracts` | `vyre-libs/tests/regex_capture_mode_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test regex_capture_mode_contracts` |
| `test` | `regex_columnar_output_contracts` | `vyre-libs/tests/regex_columnar_output_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test regex_columnar_output_contracts` |
| `test` | `regex_compile_adversarial` | `vyre-libs/tests/regex_compile_adversarial.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test regex_compile_adversarial` |
| `test` | `regex_compile_ascii_class_contracts` | `vyre-libs/tests/regex_compile_ascii_class_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test regex_compile_ascii_class_contracts` |
| `test` | `regex_compile_property` | `vyre-libs/tests/regex_compile_property.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test regex_compile_property` |
| `test` | `regex_dfa_anchoring_differential` | `vyre-libs/tests/regex_dfa_anchoring_differential.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test regex_dfa_anchoring_differential` |
| `test` | `regex_dfa_char_class_exhaustive` | `vyre-libs/tests/regex_dfa_char_class_exhaustive.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test regex_dfa_char_class_exhaustive` |
| `test` | `regex_dfa_leftmost_longest_differential` | `vyre-libs/tests/regex_dfa_leftmost_longest_differential.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test regex_dfa_leftmost_longest_differential` |
| `test` | `regex_dialect_lattice` | `vyre-libs/tests/regex_dialect_lattice.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test regex_dialect_lattice` |
| `test` | `regex_leftmost_longest_bounded_repeat` | `vyre-libs/tests/regex_leftmost_longest_bounded_repeat.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test regex_leftmost_longest_bounded_repeat` |
| `test` | `regex_logical_pattern_planner` | `vyre-libs/tests/regex_logical_pattern_planner.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test regex_logical_pattern_planner` |
| `test` | `regex_match_policy_contracts` | `vyre-libs/tests/regex_match_policy_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test regex_match_policy_contracts` |
| `test` | `regex_prefilter_planner_registry` | `vyre-libs/tests/regex_prefilter_planner_registry.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test regex_prefilter_planner_registry` |
| `test` | `regex_replay_extent_contract` | `vyre-libs/tests/regex_replay_extent_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test regex_replay_extent_contract` |
| `test` | `regex_streaming_state_ledger` | `vyre-libs/tests/regex_streaming_state_ledger.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test regex_streaming_state_ledger` |
| `test` | `regex_unicode_profiles` | `vyre-libs/tests/regex_unicode_profiles.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test regex_unicode_profiles` |
| `test` | `regex_unsupported_diagnostic_registry` | `vyre-libs/tests/regex_unsupported_diagnostic_registry.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test regex_unsupported_diagnostic_registry` |
| `test` | `region_chain_discipline` | `vyre-libs/tests/region_chain_discipline.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test region_chain_discipline` |
| `test` | `region_chain_invariant` | `vyre-libs/tests/region_chain_invariant.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test region_chain_invariant` |
| `test` | `region_inline_let_scope` | `vyre-libs/tests/region_inline_let_scope.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test region_inline_let_scope` |
| `test` | `registration_drift` | `vyre-libs/tests/registration_drift.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test registration_drift` |
| `test` | `resource_budget_complexity_policies` | `vyre-libs/tests/resource_budget_complexity_policies.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test resource_budget_complexity_policies` |
| `test` | `scan_conformance_matrix` | `vyre-libs/tests/scan_conformance_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test scan_conformance_matrix` |
| `test` | `scan_cpu_api_boundary` | `vyre-libs/tests/scan_cpu_api_boundary.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test scan_cpu_api_boundary` |
| `test` | `secret_crypto_policies` | `vyre-libs/tests/secret_crypto_policies.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test secret_crypto_policies` |
| `test` | `security_external_ifds` | `vyre-libs/tests/security_external_ifds.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test security_external_ifds` |
| `test` | `security_flows_to_alias_only_parity` | `vyre-libs/tests/security_flows_to_alias_only_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test security_flows_to_alias_only_parity` |
| `test` | `security_privacy_path_corpus_guards` | `vyre-libs/tests/security_privacy_path_corpus_guards.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test security_privacy_path_corpus_guards` |
| `test` | `shared_emitter_artifact_schema` | `vyre-libs/tests/shared_emitter_artifact_schema.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test shared_emitter_artifact_schema` |
| `test` | `sigmoid_gate_typed_contract` | `vyre-libs/tests/sigmoid_gate_typed_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sigmoid_gate_typed_contract` |
| `test` | `skill_md_examples` | `vyre-libs/tests/skill_md_examples.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test skill_md_examples` |
| `test` | `source_span_witness_records` | `vyre-libs/tests/source_span_witness_records.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test source_span_witness_records` |
| `test` | `statement_bounds_launch_contract` | `vyre-libs/tests/statement_bounds_launch_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test statement_bounds_launch_contract` |
| `test` | `succinct_rank_contracts` | `vyre-libs/tests/succinct_rank_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test succinct_rank_contracts` |
| `test` | `succinct_rank_select_adversarial_contracts` | `vyre-libs/tests/succinct_rank_select_adversarial_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test succinct_rank_select_adversarial_contracts` |
| `test` | `surface_contracts` | `vyre-libs/tests/surface_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test surface_contracts` |
| `test` | `sweep_decode_hex_oracle_matrix` | `vyre-libs/tests/sweep_decode_hex_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sweep_decode_hex_oracle_matrix` |
| `test` | `sweep_decode_hex_oracle_matrix` | `vyre-libs/tests/sweep_decode_hex_oracle_matrix.rs` | `decode` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sweep_decode_hex_oracle_matrix` |
| `test` | `sweep_hash_crc32_reference_matrix` | `vyre-libs/tests/sweep_hash_crc32_reference_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sweep_hash_crc32_reference_matrix` |
| `test` | `sweep_hash_crc32_reference_matrix` | `vyre-libs/tests/sweep_hash_crc32_reference_matrix.rs` | `hash` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sweep_hash_crc32_reference_matrix` |
| `test` | `sweep_logical_reference_matrix` | `vyre-libs/tests/sweep_logical_reference_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sweep_logical_reference_matrix` |
| `test` | `sweep_logical_reference_matrix` | `vyre-libs/tests/sweep_logical_reference_matrix.rs` | `logical` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sweep_logical_reference_matrix` |
| `test` | `sweep_text_utf8_oracle_matrix` | `vyre-libs/tests/sweep_text_utf8_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sweep_text_utf8_oracle_matrix` |
| `test` | `sweep_text_utf8_oracle_matrix` | `vyre-libs/tests/sweep_text_utf8_oracle_matrix.rs` | `text` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sweep_text_utf8_oracle_matrix` |
| `test` | `target_instruction_capabilities` | `vyre-libs/tests/target_instruction_capabilities.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test target_instruction_capabilities` |
| `test` | `universal_harness` | `vyre-libs/tests/universal_harness.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test universal_harness` |
| `test` | `unsafe_ffi_policies` | `vyre-libs/tests/unsafe_ffi_policies.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test unsafe_ffi_policies` |
| `test` | `url_network_security_policies` | `vyre-libs/tests/url_network_security_policies.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test url_network_security_policies` |
| `test` | `vast_builder_oob_guard_regression` | `vyre-libs/tests/vast_builder_oob_guard_regression.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test vast_builder_oob_guard_regression` |
| `test` | `vast_builder_oob_guard_regression` | `vyre-libs/tests/vast_builder_oob_guard_regression.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test vast_builder_oob_guard_regression` |
| `test` | `visual_compositions` | `vyre-libs/tests/visual_compositions.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test visual_compositions` |
| `test` | `wire_cross_crate_compat` | `vyre-libs/tests/wire_cross_crate_compat.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test wire_cross_crate_compat` |
| `test` | `wire_format_fuzz` | `vyre-libs/tests/wire_format_fuzz.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test wire_format_fuzz` |
| `test` | `workgroup_cooperative_tiling` | `vyre-libs/tests/workgroup_cooperative_tiling.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test workgroup_cooperative_tiling` |

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
