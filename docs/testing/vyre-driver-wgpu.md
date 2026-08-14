# Testing `vyre-driver-wgpu`

Run the default crate suite from the workspace root:

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu
```

Own pure WGSL target compilation, portable GPU acquisition, materialization, dispatch, graph execution, and backend evidence.

The crate lives at `vyre-driver-wgpu`. The `portable-driver` owner maintains its
`concrete-backend` testing contract.

## Commands

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu
```

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --all-features
```

```console
CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu -- --ignored --nocapture
```

## Feature sets

- Default feature members: None
- Available manifest features: `c-parser`, `default`, `matching-dfa`, `matching-nfa`, `matching-substring`, `math-linalg`, `math-scan`, `nn-attention`, `parity-testing`, `wgpu`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `bin` | `vyre` | `vyre-driver-wgpu/src/bin/vyre.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --bin vyre` |
| `bin` | `vyre-wgpu` | `vyre-driver-wgpu/src/bin/vyre.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --bin vyre-wgpu` |
| `example` | `wgpu_release_surface` | `vyre-driver-wgpu/examples/wgpu_release_surface.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --example wgpu_release_surface` |
| `lib` | `vyre_driver_wgpu` | `vyre-driver-wgpu/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu` |
| `test` | `_probe_matmul_wgsl` | `vyre-driver-wgpu/tests/_probe_matmul_wgsl.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test _probe_matmul_wgsl` |
| `test` | `adapter_limits_not_defaults` | `vyre-driver-wgpu/tests/adapter_limits_not_defaults.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test adapter_limits_not_defaults` |
| `test` | `adler32_gpu_parity` | `vyre-driver-wgpu/tests/adler32_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test adler32_gpu_parity` |
| `test` | `async_capability_innovation` | `vyre-driver-wgpu/tests/async_capability_innovation.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test async_capability_innovation` |
| `test` | `async_dispatch_contract` | `vyre-driver-wgpu/tests/async_dispatch_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test async_dispatch_contract` |
| `test` | `async_dispatch_non_blocking` | `vyre-driver-wgpu/tests/async_dispatch_non_blocking.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test async_dispatch_non_blocking` |
| `test` | `binding_layout_drift` | `vyre-driver-wgpu/tests/binding_layout_drift.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test binding_layout_drift` |
| `test` | `binop_parity_support` | `vyre-driver-wgpu/tests/binop_parity_support.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test binop_parity_support` |
| `test` | `bitset_zero_gpu_parity` | `vyre-driver-wgpu/tests/bitset_zero_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test bitset_zero_gpu_parity` |
| `test` | `blake3_compress_gpu_parity` | `vyre-driver-wgpu/tests/blake3_compress_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test blake3_compress_gpu_parity` |
| `test` | `buf_len_array_length` | `vyre-driver-wgpu/tests/buf_len_array_length.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test buf_len_array_length` |
| `test` | `c11_ast_corpus_complete_constructs` | `vyre-driver-wgpu/tests/c11_ast_corpus_complete_constructs.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c11_ast_corpus_complete_constructs` |
| `test` | `c11_build_vast_nodes` | `vyre-driver-wgpu/tests/c11_build_vast_nodes.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c11_build_vast_nodes` |
| `test` | `c11_parser_hostile_full_c` | `vyre-driver-wgpu/tests/c11_parser_hostile_full_c.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c11_parser_hostile_full_c` |
| `test` | `c11_parser_integration` | `vyre-driver-wgpu/tests/c11_parser_integration.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c11_parser_integration` |
| `test` | `c11_parser_typedef_contracts` | `vyre-driver-wgpu/tests/c11_parser_typedef_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c11_parser_typedef_contracts` |
| `test` | `c11_sema_scope` | `vyre-driver-wgpu/tests/c11_sema_scope.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c11_sema_scope` |
| `test` | `c11_typedef_annotations` | `vyre-driver-wgpu/tests/c11_typedef_annotations.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c11_typedef_annotations` |
| `test` | `c_ast_asm_extended_operand_goto_label_pg_lowering_contracts` | `vyre-driver-wgpu/tests/c_ast_asm_extended_operand_goto_label_pg_lowering_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_asm_extended_operand_goto_label_pg_lowering_contracts` |
| `test` | `c_ast_bitfield_and_abstract_declarator_contracts` | `vyre-driver-wgpu/tests/c_ast_bitfield_and_abstract_declarator_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_bitfield_and_abstract_declarator_contracts` |
| `test` | `c_ast_builtin_offsetof_object_size_prefetch_unreachable_contracts` | `vyre-driver-wgpu/tests/c_ast_builtin_offsetof_object_size_prefetch_unreachable_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_builtin_offsetof_object_size_prefetch_unreachable_contracts` |
| `test` | `c_ast_c11_atomic_and_generic_e2e` | `vyre-driver-wgpu/tests/c_ast_c11_atomic_and_generic_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_c11_atomic_and_generic_e2e` |
| `test` | `c_ast_compound_literal_designated_init_nested_pg_lowering_contracts` | `vyre-driver-wgpu/tests/c_ast_compound_literal_designated_init_nested_pg_lowering_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_compound_literal_designated_init_nested_pg_lowering_contracts` |
| `test` | `c_ast_control_flow_e2e` | `vyre-driver-wgpu/tests/c_ast_control_flow_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_control_flow_e2e` |
| `test` | `c_ast_declaration_advanced_contracts` | `vyre-driver-wgpu/tests/c_ast_declaration_advanced_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_declaration_advanced_contracts` |
| `test` | `c_ast_declaration_container_nodes` | `vyre-driver-wgpu/tests/c_ast_declaration_container_nodes.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_declaration_container_nodes` |
| `test` | `c_ast_declaration_container_nodes_e2e` | `vyre-driver-wgpu/tests/c_ast_declaration_container_nodes_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_declaration_container_nodes_e2e` |
| `test` | `c_ast_declarator_edge_cases` | `vyre-driver-wgpu/tests/c_ast_declarator_edge_cases.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_declarator_edge_cases` |
| `test` | `c_ast_declarator_matrix_contracts` | `vyre-driver-wgpu/tests/c_ast_declarator_matrix_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_declarator_matrix_contracts` |
| `test` | `c_ast_declarator_type_shape_contracts` | `vyre-driver-wgpu/tests/c_ast_declarator_type_shape_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_declarator_type_shape_contracts` |
| `test` | `c_ast_expression_member_ptr_access_and_ambiguity_contracts` | `vyre-driver-wgpu/tests/c_ast_expression_member_ptr_access_and_ambiguity_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_expression_member_ptr_access_and_ambiguity_contracts` |
| `test` | `c_ast_expression_operator_ambiguity_contracts` | `vyre-driver-wgpu/tests/c_ast_expression_operator_ambiguity_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_expression_operator_ambiguity_contracts` |
| `test` | `c_ast_expression_operator_builtin_contracts` | `vyre-driver-wgpu/tests/c_ast_expression_operator_builtin_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_expression_operator_builtin_contracts` |
| `test` | `c_ast_expression_operator_initializer_contracts` | `vyre-driver-wgpu/tests/c_ast_expression_operator_initializer_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_expression_operator_initializer_contracts` |
| `test` | `c_ast_expression_operator_postfix_contracts` | `vyre-driver-wgpu/tests/c_ast_expression_operator_postfix_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_expression_operator_postfix_contracts` |
| `test` | `c_ast_expression_operator_precedence_contracts` | `vyre-driver-wgpu/tests/c_ast_expression_operator_precedence_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_expression_operator_precedence_contracts` |
| `test` | `c_ast_expression_precedence_e2e` | `vyre-driver-wgpu/tests/c_ast_expression_precedence_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_expression_precedence_e2e` |
| `test` | `c_ast_expression_shape_gaps_e2e` | `vyre-driver-wgpu/tests/c_ast_expression_shape_gaps_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_expression_shape_gaps_e2e` |
| `test` | `c_ast_gnu_and_kernel_construct_integration` | `vyre-driver-wgpu/tests/c_ast_gnu_and_kernel_construct_integration.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_gnu_and_kernel_construct_integration` |
| `test` | `c_ast_gnu_asm_and_attributes_e2e` | `vyre-driver-wgpu/tests/c_ast_gnu_asm_and_attributes_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_gnu_asm_and_attributes_e2e` |
| `test` | `c_ast_gnu_asm_attribute_deep_contracts` | `vyre-driver-wgpu/tests/c_ast_gnu_asm_attribute_deep_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_gnu_asm_attribute_deep_contracts` |
| `test` | `c_ast_gnu_attribute_statement_pg_lowering_contracts` | `vyre-driver-wgpu/tests/c_ast_gnu_attribute_statement_pg_lowering_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_gnu_attribute_statement_pg_lowering_contracts` |
| `test` | `c_ast_gnu_builtin_control_flow_pg_lowering_contracts` | `vyre-driver-wgpu/tests/c_ast_gnu_builtin_control_flow_pg_lowering_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_gnu_builtin_control_flow_pg_lowering_contracts` |
| `test` | `c_ast_gnu_computed_goto_and_c11_atomic_contracts` | `vyre-driver-wgpu/tests/c_ast_gnu_computed_goto_and_c11_atomic_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_gnu_computed_goto_and_c11_atomic_contracts` |
| `test` | `c_ast_gnu_extensions_e2e` | `vyre-driver-wgpu/tests/c_ast_gnu_extensions_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_gnu_extensions_e2e` |
| `test` | `c_ast_initializer_designator_deep_contracts` | `vyre-driver-wgpu/tests/c_ast_initializer_designator_deep_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_initializer_designator_deep_contracts` |
| `test` | `c_ast_initializer_designator_e2e` | `vyre-driver-wgpu/tests/c_ast_initializer_designator_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_initializer_designator_e2e` |
| `test` | `c_ast_kernel_grade_construct_shape` | `vyre-driver-wgpu/tests/c_ast_kernel_grade_construct_shape.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_kernel_grade_construct_shape` |
| `test` | `c_ast_kernel_style_corpus` | `vyre-driver-wgpu/tests/c_ast_kernel_style_corpus.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_kernel_style_corpus` |
| `test` | `c_ast_label_statement_expression_contracts` | `vyre-driver-wgpu/tests/c_ast_label_statement_expression_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_label_statement_expression_contracts` |
| `test` | `c_ast_linux_corpus_hostile_flow_and_pg_parity_contracts` | `vyre-driver-wgpu/tests/c_ast_linux_corpus_hostile_flow_and_pg_parity_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_linux_corpus_hostile_flow_and_pg_parity_contracts` |
| `test` | `c_ast_linux_corpus_macro_builtin_and_qualifier_contracts` | `vyre-driver-wgpu/tests/c_ast_linux_corpus_macro_builtin_and_qualifier_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_linux_corpus_macro_builtin_and_qualifier_contracts` |
| `test` | `c_ast_linux_corpus_type_memory_and_init_contracts` | `vyre-driver-wgpu/tests/c_ast_linux_corpus_type_memory_and_init_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_linux_corpus_type_memory_and_init_contracts` |
| `test` | `c_ast_linux_gnu_declarations_preprocessor_contracts` | `vyre-driver-wgpu/tests/c_ast_linux_gnu_declarations_preprocessor_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_linux_gnu_declarations_preprocessor_contracts` |
| `test` | `c_ast_macro_call_trailing_comma_e2e` | `vyre-driver-wgpu/tests/c_ast_macro_call_trailing_comma_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_macro_call_trailing_comma_e2e` |
| `test` | `c_ast_nested_initializer_lists_e2e` | `vyre-driver-wgpu/tests/c_ast_nested_initializer_lists_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_nested_initializer_lists_e2e` |
| `test` | `c_ast_pg_expression_shape_e2e` | `vyre-driver-wgpu/tests/c_ast_pg_expression_shape_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_pg_expression_shape_e2e` |
| `test` | `c_ast_pg_lowering_deep_contracts` | `vyre-driver-wgpu/tests/c_ast_pg_lowering_deep_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_pg_lowering_deep_contracts` |
| `test` | `c_ast_pg_lowering_gnu_contracts` | `vyre-driver-wgpu/tests/c_ast_pg_lowering_gnu_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_pg_lowering_gnu_contracts` |
| `test` | `c_ast_preprocessor_token_stream_e2e` | `vyre-driver-wgpu/tests/c_ast_preprocessor_token_stream_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_preprocessor_token_stream_e2e` |
| `test` | `c_ast_property_graph_consistency_contracts` | `vyre-driver-wgpu/tests/c_ast_property_graph_consistency_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_property_graph_consistency_contracts` |
| `test` | `c_ast_property_operator_stream_contracts` | `vyre-driver-wgpu/tests/c_ast_property_operator_stream_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_property_operator_stream_contracts` |
| `test` | `c_ast_property_span_monotonicity_contracts` | `vyre-driver-wgpu/tests/c_ast_property_span_monotonicity_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_property_span_monotonicity_contracts` |
| `test` | `c_ast_property_typedef_annotation_contracts` | `vyre-driver-wgpu/tests/c_ast_property_typedef_annotation_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_property_typedef_annotation_contracts` |
| `test` | `c_ast_real_corpus_harness` | `vyre-driver-wgpu/tests/c_ast_real_corpus_harness.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_real_corpus_harness` |
| `test` | `c_ast_sema_scope_cast_decl_redecl_field_contracts` | `vyre-driver-wgpu/tests/c_ast_sema_scope_cast_decl_redecl_field_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_sema_scope_cast_decl_redecl_field_contracts` |
| `test` | `c_ast_sema_scope_deep_nesting_cpu_gpu_parity_contracts` | `vyre-driver-wgpu/tests/c_ast_sema_scope_deep_nesting_cpu_gpu_parity_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_sema_scope_deep_nesting_cpu_gpu_parity_contracts` |
| `test` | `c_ast_sema_scope_function_parameter_prototype_contracts` | `vyre-driver-wgpu/tests/c_ast_sema_scope_function_parameter_prototype_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_sema_scope_function_parameter_prototype_contracts` |
| `test` | `c_ast_sema_scope_tag_enum_label_contracts` | `vyre-driver-wgpu/tests/c_ast_sema_scope_tag_enum_label_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_sema_scope_tag_enum_label_contracts` |
| `test` | `c_ast_sema_scope_typedef_shadow_restore_contracts` | `vyre-driver-wgpu/tests/c_ast_sema_scope_typedef_shadow_restore_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_sema_scope_typedef_shadow_restore_contracts` |
| `test` | `c_ast_semantic_edge_expectations_gnu_and_control_flow` | `vyre-driver-wgpu/tests/c_ast_semantic_edge_expectations_gnu_and_control_flow.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_semantic_edge_expectations_gnu_and_control_flow` |
| `test` | `c_ast_semantic_gaps_linux_grade` | `vyre-driver-wgpu/tests/c_ast_semantic_gaps_linux_grade.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_semantic_gaps_linux_grade` |
| `test` | `c_ast_semantic_pg_no_host_edge_contracts` | `vyre-driver-wgpu/tests/c_ast_semantic_pg_no_host_edge_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_semantic_pg_no_host_edge_contracts` |
| `test` | `c_ast_statement_construct_gaps_e2e` | `vyre-driver-wgpu/tests/c_ast_statement_construct_gaps_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_statement_construct_gaps_e2e` |
| `test` | `c_ast_string_init_e2e` | `vyre-driver-wgpu/tests/c_ast_string_init_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_string_init_e2e` |
| `test` | `c_ast_switch_case_complex_body_pg_lowering_contracts` | `vyre-driver-wgpu/tests/c_ast_switch_case_complex_body_pg_lowering_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_switch_case_complex_body_pg_lowering_contracts` |
| `test` | `c_ast_typedef_scope_restore_e2e` | `vyre-driver-wgpu/tests/c_ast_typedef_scope_restore_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_typedef_scope_restore_e2e` |
| `test` | `c_ast_typeof_unqual_and_complex_declarators_e2e` | `vyre-driver-wgpu/tests/c_ast_typeof_unqual_and_complex_declarators_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_typeof_unqual_and_complex_declarators_e2e` |
| `test` | `c_ast_typeof_unqual_real_declarator_contracts` | `vyre-driver-wgpu/tests/c_ast_typeof_unqual_real_declarator_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_ast_typeof_unqual_real_declarator_contracts` |
| `test` | `c_lexer_parallelization_contract` | `vyre-driver-wgpu/tests/c_lexer_parallelization_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_lexer_parallelization_contract` |
| `test` | `c_lower_ast_to_pg_nodes` | `vyre-driver-wgpu/tests/c_lower_ast_to_pg_nodes.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_lower_ast_to_pg_nodes` |
| `test` | `c_lower_ast_to_pg_nodes_gpu_parity` | `vyre-driver-wgpu/tests/c_lower_ast_to_pg_nodes_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_lower_ast_to_pg_nodes_gpu_parity` |
| `test` | `c_parser_pipeline_macro_boundary_contracts` | `vyre-driver-wgpu/tests/c_parser_pipeline_macro_boundary_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_parser_pipeline_macro_boundary_contracts` |
| `test` | `c_parser_pipeline_vast_pg_parity_contracts` | `vyre-driver-wgpu/tests/c_parser_pipeline_vast_pg_parity_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_parser_pipeline_vast_pg_parity_contracts` |
| `test` | `c_preprocess_gpu_comment_strip_mask` | `vyre-driver-wgpu/tests/c_preprocess_gpu_comment_strip_mask.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_preprocess_gpu_comment_strip_mask` |
| `test` | `c_preprocess_gpu_if_expression` | `vyre-driver-wgpu/tests/c_preprocess_gpu_if_expression.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_preprocess_gpu_if_expression` |
| `test` | `c_preprocess_macro_deep_contracts` | `vyre-driver-wgpu/tests/c_preprocess_macro_deep_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_preprocess_macro_deep_contracts` |
| `test` | `c_token_support` | `vyre-driver-wgpu/tests/c_token_support.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_token_support` |
| `test` | `c_type_specifier_propagation` | `vyre-driver-wgpu/tests/c_type_specifier_propagation.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test c_type_specifier_propagation` |
| `test` | `capability_contract` | `vyre-driver-wgpu/tests/capability_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test capability_contract` |
| `test` | `capability_drift` | `vyre-driver-wgpu/tests/capability_drift.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test capability_drift` |
| `test` | `cat_a_conform` | `vyre-driver-wgpu/tests/cat_a_conform.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test cat_a_conform` |
| `test` | `cat_a_gpu_differential` | `vyre-driver-wgpu/tests/cat_a_gpu_differential.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test cat_a_gpu_differential` |
| `test` | `cli_contract` | `vyre-driver-wgpu/tests/cli_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test cli_contract` |
| `test` | `crc32_gpu_parity` | `vyre-driver-wgpu/tests/crc32_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test crc32_gpu_parity` |
| `test` | `debug_c11_annotate` | `vyre-driver-wgpu/tests/debug_c11_annotate.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test debug_c11_annotate` |
| `test` | `decode_hex_gpu_parity` | `vyre-driver-wgpu/tests/decode_hex_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test decode_hex_gpu_parity` |
| `test` | `default_workgroup_contract` | `vyre-driver-wgpu/tests/default_workgroup_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test default_workgroup_contract` |
| `test` | `determinism_contract` | `vyre-driver-wgpu/tests/determinism_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test determinism_contract` |
| `test` | `device_lost_recovery` | `vyre-driver-wgpu/tests/device_lost_recovery.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test device_lost_recovery` |
| `test` | `differential_fuzz` | `vyre-driver-wgpu/tests/differential_fuzz.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test differential_fuzz` |
| `test` | `dispatch_adversarial` | `vyre-driver-wgpu/tests/dispatch_adversarial.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test dispatch_adversarial` |
| `test` | `dispatch_allocation_contract` | `vyre-driver-wgpu/tests/dispatch_allocation_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test dispatch_allocation_contract` |
| `test` | `dispatch_async_deferred` | `vyre-driver-wgpu/tests/dispatch_async_deferred.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test dispatch_async_deferred` |
| `test` | `dispatch_grid_shape_contract` | `vyre-driver-wgpu/tests/dispatch_grid_shape_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test dispatch_grid_shape_contract` |
| `test` | `dispatch_hot_path` | `vyre-driver-wgpu/tests/dispatch_hot_path.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test dispatch_hot_path` |
| `test` | `dispatch_never_cpu_fallback` | `vyre-driver-wgpu/tests/dispatch_never_cpu_fallback.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test dispatch_never_cpu_fallback` |
| `test` | `dispatch_preemption` | `vyre-driver-wgpu/tests/dispatch_preemption.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test dispatch_preemption` |
| `test` | `div_zero_shift_mask_parity` | `vyre-driver-wgpu/tests/div_zero_shift_mask_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test div_zero_shift_mask_parity` |
| `test` | `every_op_random_inputs` | `vyre-driver-wgpu/tests/every_op_random_inputs.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test every_op_random_inputs` |
| `test` | `f32_no_contraction_contract` | `vyre-driver-wgpu/tests/f32_no_contraction_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test f32_no_contraction_contract` |
| `test` | `float_to_int_cast_parity` | `vyre-driver-wgpu/tests/float_to_int_cast_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test float_to_int_cast_parity` |
| `test` | `fnv1a32_gpu_parity` | `vyre-driver-wgpu/tests/fnv1a32_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test fnv1a32_gpu_parity` |
| `test` | `fnv1a64_gpu_parity` | `vyre-driver-wgpu/tests/fnv1a64_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test fnv1a64_gpu_parity` |
| `test` | `gap_determinism_contract` | `vyre-driver-wgpu/tests/gap_determinism_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test gap_determinism_contract` |
| `test` | `gap_device_lost_recovery` | `vyre-driver-wgpu/tests/gap_device_lost_recovery.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test gap_device_lost_recovery` |
| `test` | `gap_dispatch_preemption` | `vyre-driver-wgpu/tests/gap_dispatch_preemption.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test gap_dispatch_preemption` |
| `test` | `gap_transcendentals_parity` | `vyre-driver-wgpu/tests/gap_transcendentals_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test gap_transcendentals_parity` |
| `test` | `gemini_c_ast_contracts` | `vyre-driver-wgpu/tests/gemini_c_ast_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test gemini_c_ast_contracts` |
| `test` | `hit_buffer` | `vyre-driver-wgpu/tests/hit_buffer.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test hit_buffer` |
| `test` | `lens_gpu_parity` | `vyre-driver-wgpu/tests/lens_gpu_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test lens_gpu_parity` |
| `test` | `limits_from_adapter_device` | `vyre-driver-wgpu/tests/limits_from_adapter_device.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test limits_from_adapter_device` |
| `test` | `live_capability_honesty` | `vyre-driver-wgpu/tests/live_capability_honesty.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test live_capability_honesty` |
| `test` | `loop_carrier_three_level_if_real_dispatch` | `vyre-driver-wgpu/tests/loop_carrier_three_level_if_real_dispatch.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test loop_carrier_three_level_if_real_dispatch` |
| `test` | `lowering_actionable_errors` | `vyre-driver-wgpu/tests/lowering_actionable_errors.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test lowering_actionable_errors` |
| `test` | `megakernel_emit` | `vyre-driver-wgpu/tests/megakernel_emit.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test megakernel_emit` |
| `test` | `naga_deeper_regressions` | `vyre-driver-wgpu/tests/naga_deeper_regressions.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test naga_deeper_regressions` |
| `test` | `naga_findings_followup` | `vyre-driver-wgpu/tests/naga_findings_followup.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test naga_findings_followup` |
| `test` | `naga_loop_region_followup` | `vyre-driver-wgpu/tests/naga_loop_region_followup.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test naga_loop_region_followup` |
| `test` | `naga_type_buffer_followup` | `vyre-driver-wgpu/tests/naga_type_buffer_followup.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test naga_type_buffer_followup` |
| `test` | `narrowing_cast_parity` | `vyre-driver-wgpu/tests/narrowing_cast_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test narrowing_cast_parity` |
| `test` | `newton_schulz_ir_shape` | `vyre-driver-wgpu/tests/newton_schulz_ir_shape.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test newton_schulz_ir_shape` |
| `test` | `no_cpu_fallback` | `vyre-driver-wgpu/tests/no_cpu_fallback.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test no_cpu_fallback` |
| `test` | `nvme_gpu_ingest_e2e` | `vyre-driver-wgpu/tests/nvme_gpu_ingest_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test nvme_gpu_ingest_e2e` |
| `test` | `op_pairwise` | `vyre-driver-wgpu/tests/op_pairwise.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test op_pairwise` |
| `test` | `oversized_workgroup_fails_loudly` | `vyre-driver-wgpu/tests/oversized_workgroup_fails_loudly.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test oversized_workgroup_fails_loudly` |
| `test` | `pipeline_cache_contract` | `vyre-driver-wgpu/tests/pipeline_cache_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test pipeline_cache_contract` |
| `test` | `pipeline_cache_persistence` | `vyre-driver-wgpu/tests/pipeline_cache_persistence.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test pipeline_cache_persistence` |
| `test` | `preferred_dispatch_backend` | `vyre-driver-wgpu/tests/preferred_dispatch_backend.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test preferred_dispatch_backend` |
| `test` | `readback_ring_liveness_contracts` | `vyre-driver-wgpu/tests/readback_ring_liveness_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test readback_ring_liveness_contracts` |
| `test` | `resident_buffer_contracts` | `vyre-driver-wgpu/tests/resident_buffer_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test resident_buffer_contracts` |
| `test` | `resident_grid_sync_contracts` | `vyre-driver-wgpu/tests/resident_grid_sync_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test resident_grid_sync_contracts` |
| `test` | `resident_output_contracts` | `vyre-driver-wgpu/tests/resident_output_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test resident_output_contracts` |
| `test` | `resident_timed_outputs` | `vyre-driver-wgpu/tests/resident_timed_outputs.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test resident_timed_outputs` |
| `test` | `same_width_store_parity` | `vyre-driver-wgpu/tests/same_width_store_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test same_width_store_parity` |
| `test` | `self_optimizer_canonicalize_e2e` | `vyre-driver-wgpu/tests/self_optimizer_canonicalize_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test self_optimizer_canonicalize_e2e` |
| `test` | `self_optimizer_const_fold_e2e` | `vyre-driver-wgpu/tests/self_optimizer_const_fold_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test self_optimizer_const_fold_e2e` |
| `test` | `self_optimizer_dce_e2e` | `vyre-driver-wgpu/tests/self_optimizer_dce_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test self_optimizer_dce_e2e` |
| `test` | `self_optimizer_pattern_match_e2e` | `vyre-driver-wgpu/tests/self_optimizer_pattern_match_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test self_optimizer_pattern_match_e2e` |
| `test` | `self_optimizer_pipeline_e2e` | `vyre-driver-wgpu/tests/self_optimizer_pipeline_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test self_optimizer_pipeline_e2e` |
| `test` | `self_optimizer_scaling_bench` | `vyre-driver-wgpu/tests/self_optimizer_scaling_bench.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test self_optimizer_scaling_bench` |
| `test` | `shared_backend_contract` | `vyre-driver-wgpu/tests/shared_backend_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test shared_backend_contract` |
| `test` | `signed_int_op_parity` | `vyre-driver-wgpu/tests/signed_int_op_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test signed_int_op_parity` |
| `test` | `signed_modulo_parity` | `vyre-driver-wgpu/tests/signed_modulo_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test signed_modulo_parity` |
| `test` | `sinkhorn_iterate_contract` | `vyre-driver-wgpu/tests/sinkhorn_iterate_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test sinkhorn_iterate_contract` |
| `test` | `stream_shard_public_error_contracts` | `vyre-driver-wgpu/tests/stream_shard_public_error_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test stream_shard_public_error_contracts` |
| `test` | `subgroup_detection` | `vyre-driver-wgpu/tests/subgroup_detection.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test subgroup_detection` |
| `test` | `subgroup_reporting_honesty` | `vyre-driver-wgpu/tests/subgroup_reporting_honesty.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test subgroup_reporting_honesty` |
| `test` | `synthetic_binop_parity` | `vyre-driver-wgpu/tests/synthetic_binop_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test synthetic_binop_parity` |
| `test` | `target_compiler` | `vyre-driver-wgpu/tests/target_compiler.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test target_compiler` |
| `test` | `timed_dispatch_device_ns` | `vyre-driver-wgpu/tests/timed_dispatch_device_ns.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test timed_dispatch_device_ns` |
| `test` | `transcendentals_parity` | `vyre-driver-wgpu/tests/transcendentals_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test transcendentals_parity` |
| `test` | `trap_propagation` | `vyre-driver-wgpu/tests/trap_propagation.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test trap_propagation` |
| `test` | `trap_sidecar` | `vyre-driver-wgpu/tests/trap_sidecar.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test trap_sidecar` |
| `test` | `u32_wrap_arithmetic` | `vyre-driver-wgpu/tests/u32_wrap_arithmetic.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test u32_wrap_arithmetic` |
| `test` | `unary_int_parity` | `vyre-driver-wgpu/tests/unary_int_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test unary_int_parity` |
| `test` | `validation_cross_backend` | `vyre-driver-wgpu/tests/validation_cross_backend.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test validation_cross_backend` |
| `test` | `wgpu_command_reuse_classifier` | `vyre-driver-wgpu/tests/wgpu_command_reuse_classifier.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test wgpu_command_reuse_classifier` |
| `test` | `wgpu_subgroup_capability_diagnostics` | `vyre-driver-wgpu/tests/wgpu_subgroup_capability_diagnostics.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test wgpu_subgroup_capability_diagnostics` |
| `test` | `wgpu_subgroup_scan_plan_registry` | `vyre-driver-wgpu/tests/wgpu_subgroup_scan_plan_registry.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test wgpu_subgroup_scan_plan_registry` |
| `test` | `wgsl_scan_uniformity_certificates` | `vyre-driver-wgpu/tests/wgsl_scan_uniformity_certificates.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test wgsl_scan_uniformity_certificates` |
| `test` | `widening_cast_64_parity` | `vyre-driver-wgpu/tests/widening_cast_64_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-driver-wgpu --test widening_cast_64_parity` |

## Test classes

- Device and capability contracts
- Lowering and artifact semantics
- Dispatch, graph, memory, and backend parity tests

## Hardware requirements

You need a supported physical GPU adapter for device dispatch and ignored physical-adapter tests. A requested adapter that cannot initialize is an error.

## Evidence outputs

- `release/evidence/conformance/release-all-backends-certificate.json`
- Command status and exact portable-backend parity assertions

## Skips and failures

The default command omits only tests marked `#[ignore]`. Run the ignored-test command on a configured GPU host. Backend initialization failures must remain visible.

A failed assertion, build error, backend acquisition error, or malformed fixture returns a nonzero status with the failing test and contract in the diagnostic.
