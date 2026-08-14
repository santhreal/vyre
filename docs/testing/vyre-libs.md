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
- Available manifest features: `analysis`, `bench`, `c-parser`, `cat-a-builder-options`, `cpu-parity`, `crypto`, `crypto-blake3`, `decode`, `default`, `device`, `encoding`, `full`, `go-parser`, `graph-dispatch`, `hash`, `intern`, `logical`, `matching`, `matching-dfa`, `matching-nfa`, `matching-regex`, `matching-substring`, `math`, `math-algebra`, `math-broadcast`, `math-linalg`, `math-scan`, `math-succinct`, `nn`, `nn-activation`, `nn-attention`, `nn-inference`, `nn-linear`, `nn-linear-4bit`, `nn-moe`, `nn-norm`, `parsing`, `python-parser`, `reasoning`, `rule`, `scheduling`, `security`, `solvers`, `telemetry`, `test-fixtures`, `text`, `visual`
- Use the all-features command above to compile every declared feature together.

## Cargo targets

| Kind | Target | Source | Required features | Focused command |
| --- | --- | --- | --- | --- |
| `example` | `check_select1` | `vyre-libs/examples/check_select1.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --example check_select1` |
| `lib` | `vyre_libs` | `vyre-libs/src/lib.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs` |
| `test` | `ac_count_suffix3_naga_validation` | `vyre-libs/tests/ac_count_suffix3_naga_validation.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test ac_count_suffix3_naga_validation` |
| `test` | `adversarial` | `vyre-libs/tests/adversarial.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test adversarial` |
| `test` | `aho_corasick_kat` | `vyre-libs/tests/aho_corasick_kat.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test aho_corasick_kat` |
| `test` | `algebra_lattice_semiring_contracts` | `vyre-libs/tests/algebra_lattice_semiring_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test algebra_lattice_semiring_contracts` |
| `test` | `analysis_fact_schema` | `vyre-libs/tests/analysis_fact_schema.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test analysis_fact_schema` |
| `test` | `ast_shunting_yard` | `vyre-libs/tests/ast_shunting_yard.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test ast_shunting_yard` |
| `test` | `ast_shunting_yard` | `vyre-libs/tests/ast_shunting_yard.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test ast_shunting_yard` |
| `test` | `attention_head_to_token_contract` | `vyre-libs/tests/attention_head_to_token_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test attention_head_to_token_contract` |
| `test` | `attention_head_to_token_contract` | `vyre-libs/tests/attention_head_to_token_contract.rs` | `nn-attention` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test attention_head_to_token_contract` |
| `test` | `bellman_shortest_path_via_reference_parity` | `vyre-libs/tests/bellman_shortest_path_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test bellman_shortest_path_via_reference_parity` |
| `test` | `bellman_shortest_path_via_reference_parity` | `vyre-libs/tests/bellman_shortest_path_via_reference_parity.rs` | `cpu-parity`, `solvers`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test bellman_shortest_path_via_reference_parity` |
| `test` | `bitset_dense_matvec_pipeline_generated` | `vyre-libs/tests/bitset_dense_matvec_pipeline_generated.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test bitset_dense_matvec_pipeline_generated` |
| `test` | `bitset_dense_matvec_pipeline_generated` | `vyre-libs/tests/bitset_dense_matvec_pipeline_generated.rs` | `cpu-parity`, `encoding`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test bitset_dense_matvec_pipeline_generated` |
| `test` | `bitset_mask_algebra_via_reference_parity` | `vyre-libs/tests/bitset_mask_algebra_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test bitset_mask_algebra_via_reference_parity` |
| `test` | `bitset_mask_algebra_via_reference_parity` | `vyre-libs/tests/bitset_mask_algebra_via_reference_parity.rs` | `cpu-parity`, `encoding`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test bitset_mask_algebra_via_reference_parity` |
| `test` | `bitset_summary_via_reference_parity` | `vyre-libs/tests/bitset_summary_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test bitset_summary_via_reference_parity` |
| `test` | `bitset_summary_via_reference_parity` | `vyre-libs/tests/bitset_summary_via_reference_parity.rs` | `cpu-parity`, `encoding`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test bitset_summary_via_reference_parity` |
| `test` | `blake3_compress_optimizer_idempotence_contract` | `vyre-libs/tests/blake3_compress_optimizer_idempotence_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test blake3_compress_optimizer_idempotence_contract` |
| `test` | `blake3_kat` | `vyre-libs/tests/blake3_kat.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test blake3_kat` |
| `test` | `blake3_wrong_size` | `vyre-libs/tests/blake3_wrong_size.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test blake3_wrong_size` |
| `test` | `buffer_name_cross_family` | `vyre-libs/tests/buffer_name_cross_family.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test buffer_name_cross_family` |
| `test` | `c11_ast_corpus_complete_constructs` | `vyre-libs/tests/c11_ast_corpus_complete_constructs.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c11_ast_corpus_complete_constructs` |
| `test` | `c11_ast_corpus_complete_constructs` | `vyre-libs/tests/c11_ast_corpus_complete_constructs.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c11_ast_corpus_complete_constructs` |
| `test` | `c11_build_vast_nodes` | `vyre-libs/tests/c11_build_vast_nodes.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c11_build_vast_nodes` |
| `test` | `c11_build_vast_nodes` | `vyre-libs/tests/c11_build_vast_nodes.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c11_build_vast_nodes` |
| `test` | `c11_function_extractor_contracts` | `vyre-libs/tests/c11_function_extractor_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c11_function_extractor_contracts` |
| `test` | `c11_function_extractor_contracts` | `vyre-libs/tests/c11_function_extractor_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c11_function_extractor_contracts` |
| `test` | `c11_keyword` | `vyre-libs/tests/c11_keyword.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c11_keyword` |
| `test` | `c11_keyword` | `vyre-libs/tests/c11_keyword.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c11_keyword` |
| `test` | `c11_lexer_ir_identity` | `vyre-libs/tests/c11_lexer_ir_identity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c11_lexer_ir_identity` |
| `test` | `c11_lexer_ir_identity` | `vyre-libs/tests/c11_lexer_ir_identity.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c11_lexer_ir_identity` |
| `test` | `c11_lexer_naga_validation` | `vyre-libs/tests/c11_lexer_naga_validation.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c11_lexer_naga_validation` |
| `test` | `c11_lexer_naga_validation` | `vyre-libs/tests/c11_lexer_naga_validation.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c11_lexer_naga_validation` |
| `test` | `c11_parser_typedef_contracts` | `vyre-libs/tests/c11_parser_typedef_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c11_parser_typedef_contracts` |
| `test` | `c11_parser_typedef_contracts` | `vyre-libs/tests/c11_parser_typedef_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c11_parser_typedef_contracts` |
| `test` | `c11_statement_bounds_long_statement_span` | `vyre-libs/tests/c11_statement_bounds_long_statement_span.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c11_statement_bounds_long_statement_span` |
| `test` | `c_annotate_typedef_oracle_parity` | `vyre-libs/tests/c_annotate_typedef_oracle_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_annotate_typedef_oracle_parity` |
| `test` | `c_ast_asm_extended_operand_goto_label_pg_lowering_contracts` | `vyre-libs/tests/c_ast_asm_extended_operand_goto_label_pg_lowering_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_asm_extended_operand_goto_label_pg_lowering_contracts` |
| `test` | `c_ast_asm_extended_operand_goto_label_pg_lowering_contracts` | `vyre-libs/tests/c_ast_asm_extended_operand_goto_label_pg_lowering_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_asm_extended_operand_goto_label_pg_lowering_contracts` |
| `test` | `c_ast_c99_for_do_macro_e2e` | `vyre-libs/tests/c_ast_c99_for_do_macro_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_c99_for_do_macro_e2e` |
| `test` | `c_ast_c99_for_do_macro_e2e` | `vyre-libs/tests/c_ast_c99_for_do_macro_e2e.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_c99_for_do_macro_e2e` |
| `test` | `c_ast_compound_literal_designated_init_nested_pg_lowering_contracts` | `vyre-libs/tests/c_ast_compound_literal_designated_init_nested_pg_lowering_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_compound_literal_designated_init_nested_pg_lowering_contracts` |
| `test` | `c_ast_compound_literal_designated_init_nested_pg_lowering_contracts` | `vyre-libs/tests/c_ast_compound_literal_designated_init_nested_pg_lowering_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_compound_literal_designated_init_nested_pg_lowering_contracts` |
| `test` | `c_ast_container_of_e2e` | `vyre-libs/tests/c_ast_container_of_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_container_of_e2e` |
| `test` | `c_ast_container_of_e2e` | `vyre-libs/tests/c_ast_container_of_e2e.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_container_of_e2e` |
| `test` | `c_ast_declaration_advanced_contracts` | `vyre-libs/tests/c_ast_declaration_advanced_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_declaration_advanced_contracts` |
| `test` | `c_ast_declaration_advanced_contracts` | `vyre-libs/tests/c_ast_declaration_advanced_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_declaration_advanced_contracts` |
| `test` | `c_ast_declaration_container_nodes` | `vyre-libs/tests/c_ast_declaration_container_nodes.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_declaration_container_nodes` |
| `test` | `c_ast_declaration_container_nodes` | `vyre-libs/tests/c_ast_declaration_container_nodes.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_declaration_container_nodes` |
| `test` | `c_ast_declarator_matrix_contracts` | `vyre-libs/tests/c_ast_declarator_matrix_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_declarator_matrix_contracts` |
| `test` | `c_ast_declarator_matrix_contracts` | `vyre-libs/tests/c_ast_declarator_matrix_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_declarator_matrix_contracts` |
| `test` | `c_ast_expression_operator_ambiguity_contracts` | `vyre-libs/tests/c_ast_expression_operator_ambiguity_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_expression_operator_ambiguity_contracts` |
| `test` | `c_ast_expression_operator_ambiguity_contracts` | `vyre-libs/tests/c_ast_expression_operator_ambiguity_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_expression_operator_ambiguity_contracts` |
| `test` | `c_ast_expression_operator_builtin_contracts` | `vyre-libs/tests/c_ast_expression_operator_builtin_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_expression_operator_builtin_contracts` |
| `test` | `c_ast_expression_operator_builtin_contracts` | `vyre-libs/tests/c_ast_expression_operator_builtin_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_expression_operator_builtin_contracts` |
| `test` | `c_ast_expression_operator_postfix_contracts` | `vyre-libs/tests/c_ast_expression_operator_postfix_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_expression_operator_postfix_contracts` |
| `test` | `c_ast_expression_operator_postfix_contracts` | `vyre-libs/tests/c_ast_expression_operator_postfix_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_expression_operator_postfix_contracts` |
| `test` | `c_ast_expression_operator_precedence_contracts` | `vyre-libs/tests/c_ast_expression_operator_precedence_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_expression_operator_precedence_contracts` |
| `test` | `c_ast_expression_operator_precedence_contracts` | `vyre-libs/tests/c_ast_expression_operator_precedence_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_expression_operator_precedence_contracts` |
| `test` | `c_ast_expression_precedence_e2e` | `vyre-libs/tests/c_ast_expression_precedence_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_expression_precedence_e2e` |
| `test` | `c_ast_expression_precedence_e2e` | `vyre-libs/tests/c_ast_expression_precedence_e2e.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_expression_precedence_e2e` |
| `test` | `c_ast_expression_shape_gaps_e2e` | `vyre-libs/tests/c_ast_expression_shape_gaps_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_expression_shape_gaps_e2e` |
| `test` | `c_ast_expression_shape_gaps_e2e` | `vyre-libs/tests/c_ast_expression_shape_gaps_e2e.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_expression_shape_gaps_e2e` |
| `test` | `c_ast_gnu_asm_decomposition_and_attribute_kinds` | `vyre-libs/tests/c_ast_gnu_asm_decomposition_and_attribute_kinds.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_gnu_asm_decomposition_and_attribute_kinds` |
| `test` | `c_ast_gnu_asm_decomposition_and_attribute_kinds` | `vyre-libs/tests/c_ast_gnu_asm_decomposition_and_attribute_kinds.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_gnu_asm_decomposition_and_attribute_kinds` |
| `test` | `c_ast_gnu_attribute_statement_pg_lowering_contracts` | `vyre-libs/tests/c_ast_gnu_attribute_statement_pg_lowering_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_gnu_attribute_statement_pg_lowering_contracts` |
| `test` | `c_ast_gnu_attribute_statement_pg_lowering_contracts` | `vyre-libs/tests/c_ast_gnu_attribute_statement_pg_lowering_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_gnu_attribute_statement_pg_lowering_contracts` |
| `test` | `c_ast_gnu_builtin_control_flow_pg_lowering_contracts` | `vyre-libs/tests/c_ast_gnu_builtin_control_flow_pg_lowering_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_gnu_builtin_control_flow_pg_lowering_contracts` |
| `test` | `c_ast_gnu_builtin_control_flow_pg_lowering_contracts` | `vyre-libs/tests/c_ast_gnu_builtin_control_flow_pg_lowering_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_gnu_builtin_control_flow_pg_lowering_contracts` |
| `test` | `c_ast_gnu_builtin_vast_contracts` | `vyre-libs/tests/c_ast_gnu_builtin_vast_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_gnu_builtin_vast_contracts` |
| `test` | `c_ast_gnu_builtin_vast_contracts` | `vyre-libs/tests/c_ast_gnu_builtin_vast_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_gnu_builtin_vast_contracts` |
| `test` | `c_ast_linux_corpus_macro_builtin_and_qualifier_contracts` | `vyre-libs/tests/c_ast_linux_corpus_macro_builtin_and_qualifier_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_linux_corpus_macro_builtin_and_qualifier_contracts` |
| `test` | `c_ast_linux_corpus_macro_builtin_and_qualifier_contracts` | `vyre-libs/tests/c_ast_linux_corpus_macro_builtin_and_qualifier_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_linux_corpus_macro_builtin_and_qualifier_contracts` |
| `test` | `c_ast_linux_grade_gnu_and_c11_construct_coverage` | `vyre-libs/tests/c_ast_linux_grade_gnu_and_c11_construct_coverage.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_linux_grade_gnu_and_c11_construct_coverage` |
| `test` | `c_ast_linux_grade_gnu_and_c11_construct_coverage` | `vyre-libs/tests/c_ast_linux_grade_gnu_and_c11_construct_coverage.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_linux_grade_gnu_and_c11_construct_coverage` |
| `test` | `c_ast_linux_style_raw_source_contracts` | `vyre-libs/tests/c_ast_linux_style_raw_source_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_linux_style_raw_source_contracts` |
| `test` | `c_ast_linux_style_raw_source_contracts` | `vyre-libs/tests/c_ast_linux_style_raw_source_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_linux_style_raw_source_contracts` |
| `test` | `c_ast_pg_expression_shape_e2e` | `vyre-libs/tests/c_ast_pg_expression_shape_e2e.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_pg_expression_shape_e2e` |
| `test` | `c_ast_pg_expression_shape_e2e` | `vyre-libs/tests/c_ast_pg_expression_shape_e2e.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_pg_expression_shape_e2e` |
| `test` | `c_ast_pg_lowering_deep_contracts` | `vyre-libs/tests/c_ast_pg_lowering_deep_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_pg_lowering_deep_contracts` |
| `test` | `c_ast_pg_lowering_deep_contracts` | `vyre-libs/tests/c_ast_pg_lowering_deep_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_pg_lowering_deep_contracts` |
| `test` | `c_ast_sema_scope_cast_decl_redecl_field_contracts` | `vyre-libs/tests/c_ast_sema_scope_cast_decl_redecl_field_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_sema_scope_cast_decl_redecl_field_contracts` |
| `test` | `c_ast_sema_scope_cast_decl_redecl_field_contracts` | `vyre-libs/tests/c_ast_sema_scope_cast_decl_redecl_field_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_sema_scope_cast_decl_redecl_field_contracts` |
| `test` | `c_ast_sema_scope_function_parameter_prototype_contracts` | `vyre-libs/tests/c_ast_sema_scope_function_parameter_prototype_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_sema_scope_function_parameter_prototype_contracts` |
| `test` | `c_ast_sema_scope_function_parameter_prototype_contracts` | `vyre-libs/tests/c_ast_sema_scope_function_parameter_prototype_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_sema_scope_function_parameter_prototype_contracts` |
| `test` | `c_ast_sema_scope_tag_enum_label_contracts` | `vyre-libs/tests/c_ast_sema_scope_tag_enum_label_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_sema_scope_tag_enum_label_contracts` |
| `test` | `c_ast_sema_scope_tag_enum_label_contracts` | `vyre-libs/tests/c_ast_sema_scope_tag_enum_label_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_sema_scope_tag_enum_label_contracts` |
| `test` | `c_ast_sema_scope_typedef_shadow_restore_contracts` | `vyre-libs/tests/c_ast_sema_scope_typedef_shadow_restore_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_sema_scope_typedef_shadow_restore_contracts` |
| `test` | `c_ast_sema_scope_typedef_shadow_restore_contracts` | `vyre-libs/tests/c_ast_sema_scope_typedef_shadow_restore_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_sema_scope_typedef_shadow_restore_contracts` |
| `test` | `c_ast_semantic_gaps_linux_grade` | `vyre-libs/tests/c_ast_semantic_gaps_linux_grade.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_semantic_gaps_linux_grade` |
| `test` | `c_ast_semantic_gaps_linux_grade` | `vyre-libs/tests/c_ast_semantic_gaps_linux_grade.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_semantic_gaps_linux_grade` |
| `test` | `c_ast_switch_case_complex_body_pg_lowering_contracts` | `vyre-libs/tests/c_ast_switch_case_complex_body_pg_lowering_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_switch_case_complex_body_pg_lowering_contracts` |
| `test` | `c_ast_switch_case_complex_body_pg_lowering_contracts` | `vyre-libs/tests/c_ast_switch_case_complex_body_pg_lowering_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_ast_switch_case_complex_body_pg_lowering_contracts` |
| `test` | `c_conditional_range_policy` | `vyre-libs/tests/c_conditional_range_policy.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_conditional_range_policy` |
| `test` | `c_global_typedef_annotate_parity` | `vyre-libs/tests/c_global_typedef_annotate_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_global_typedef_annotate_parity` |
| `test` | `c_lexer_preprocessor_hash_contracts` | `vyre-libs/tests/c_lexer_preprocessor_hash_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_lexer_preprocessor_hash_contracts` |
| `test` | `c_lexer_preprocessor_hash_contracts` | `vyre-libs/tests/c_lexer_preprocessor_hash_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_lexer_preprocessor_hash_contracts` |
| `test` | `c_lexer_regular_variant_parity` | `vyre-libs/tests/c_lexer_regular_variant_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_lexer_regular_variant_parity` |
| `test` | `c_lower_semantic_graph_control_resolution_parity` | `vyre-libs/tests/c_lower_semantic_graph_control_resolution_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_lower_semantic_graph_control_resolution_parity` |
| `test` | `c_packed_haystack_semantic_parity` | `vyre-libs/tests/c_packed_haystack_semantic_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_packed_haystack_semantic_parity` |
| `test` | `c_packed_haystack_semantic_parity` | `vyre-libs/tests/c_packed_haystack_semantic_parity.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_packed_haystack_semantic_parity` |
| `test` | `c_parser_hostile_malformed_stream_contracts` | `vyre-libs/tests/c_parser_hostile_malformed_stream_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_parser_hostile_malformed_stream_contracts` |
| `test` | `c_parser_hostile_malformed_stream_contracts` | `vyre-libs/tests/c_parser_hostile_malformed_stream_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_parser_hostile_malformed_stream_contracts` |
| `test` | `c_parser_pipeline_lexer_adversarial_contracts` | `vyre-libs/tests/c_parser_pipeline_lexer_adversarial_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_parser_pipeline_lexer_adversarial_contracts` |
| `test` | `c_parser_pipeline_lexer_adversarial_contracts` | `vyre-libs/tests/c_parser_pipeline_lexer_adversarial_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_parser_pipeline_lexer_adversarial_contracts` |
| `test` | `c_parser_pipeline_malformed_stream_contracts` | `vyre-libs/tests/c_parser_pipeline_malformed_stream_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_parser_pipeline_malformed_stream_contracts` |
| `test` | `c_parser_pipeline_malformed_stream_contracts` | `vyre-libs/tests/c_parser_pipeline_malformed_stream_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_parser_pipeline_malformed_stream_contracts` |
| `test` | `c_preprocess_certificates` | `vyre-libs/tests/c_preprocess_certificates.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_certificates` |
| `test` | `c_preprocess_classified_memory_cache_contract` | `vyre-libs/tests/c_preprocess_classified_memory_cache_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_classified_memory_cache_contract` |
| `test` | `c_preprocess_dedup_guard` | `vyre-libs/tests/c_preprocess_dedup_guard.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_dedup_guard` |
| `test` | `c_preprocess_dedup_guard` | `vyre-libs/tests/c_preprocess_dedup_guard.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test c_preprocess_dedup_guard` |
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
| `test` | `cat_a_conform` | `vyre-libs/tests/cat_a_conform.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test cat_a_conform` |
| `test` | `categorical_laws_proptest` | `vyre-libs/tests/categorical_laws_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test categorical_laws_proptest` |
| `test` | `categorical_laws_proptest` | `vyre-libs/tests/categorical_laws_proptest.rs` | `cpu-parity`, `reasoning`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test categorical_laws_proptest` |
| `test` | `causal_conv_state_transition_contract` | `vyre-libs/tests/causal_conv_state_transition_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test causal_conv_state_transition_contract` |
| `test` | `causal_conv_state_transition_contract` | `vyre-libs/tests/causal_conv_state_transition_contract.rs` | `nn-inference` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test causal_conv_state_transition_contract` |
| `test` | `causal_gqa_contract` | `vyre-libs/tests/causal_gqa_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test causal_gqa_contract` |
| `test` | `causal_gqa_contract` | `vyre-libs/tests/causal_gqa_contract.rs` | `nn-attention` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test causal_gqa_contract` |
| `test` | `causal_gqa_typed_contract` | `vyre-libs/tests/causal_gqa_typed_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test causal_gqa_typed_contract` |
| `test` | `causal_gqa_typed_contract` | `vyre-libs/tests/causal_gqa_typed_contract.rs` | `nn-attention` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test causal_gqa_typed_contract` |
| `test` | `chunked_gated_delta_contract` | `vyre-libs/tests/chunked_gated_delta_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test chunked_gated_delta_contract` |
| `test` | `chunked_gated_delta_contract` | `vyre-libs/tests/chunked_gated_delta_contract.rs` | `nn-attention` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test chunked_gated_delta_contract` |
| `test` | `consumer_boundary` | `vyre-libs/tests/consumer_boundary.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test consumer_boundary` |
| `test` | `corpus_privacy_retention_controls` | `vyre-libs/tests/corpus_privacy_retention_controls.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test corpus_privacy_retention_controls` |
| `test` | `cost_model_predict_runtime_via_reference_parity` | `vyre-libs/tests/cost_model_predict_runtime_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test cost_model_predict_runtime_via_reference_parity` |
| `test` | `cost_model_predict_runtime_via_reference_parity` | `vyre-libs/tests/cost_model_predict_runtime_via_reference_parity.rs` | `analysis`, `cpu-parity`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test cost_model_predict_runtime_via_reference_parity` |
| `test` | `cpu_witnesses` | `vyre-libs/tests/cpu_witnesses.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test cpu_witnesses` |
| `test` | `decode_primitive_composition_contracts` | `vyre-libs/tests/decode_primitive_composition_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test decode_primitive_composition_contracts` |
| `test` | `dedup_conv_ast_walk_family_guard` | `vyre-libs/tests/dedup_conv_ast_walk_family_guard.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test dedup_conv_ast_walk_family_guard` |
| `test` | `delta_flow_arrangements` | `vyre-libs/tests/delta_flow_arrangements.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test delta_flow_arrangements` |
| `test` | `dense_gated_mlp_graph_contract` | `vyre-libs/tests/dense_gated_mlp_graph_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test dense_gated_mlp_graph_contract` |
| `test` | `dense_gated_mlp_graph_contract` | `vyre-libs/tests/dense_gated_mlp_graph_contract.rs` | `nn-inference` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test dense_gated_mlp_graph_contract` |
| `test` | `depthwise_causal_conv1d_contract` | `vyre-libs/tests/depthwise_causal_conv1d_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test depthwise_causal_conv1d_contract` |
| `test` | `depthwise_causal_conv1d_contract` | `vyre-libs/tests/depthwise_causal_conv1d_contract.rs` | `nn-inference` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test depthwise_causal_conv1d_contract` |
| `test` | `do_calculus_surgery_via_reference_parity` | `vyre-libs/tests/do_calculus_surgery_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test do_calculus_surgery_via_reference_parity` |
| `test` | `do_calculus_surgery_via_reference_parity` | `vyre-libs/tests/do_calculus_surgery_via_reference_parity.rs` | `cpu-parity`, `reasoning`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test do_calculus_surgery_via_reference_parity` |
| `test` | `f32_adversarial` | `vyre-libs/tests/f32_adversarial.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test f32_adversarial` |
| `test` | `family_duplication_budget` | `vyre-libs/tests/family_duplication_budget.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test family_duplication_budget` |
| `test` | `filesystem_path_archive_policies` | `vyre-libs/tests/filesystem_path_archive_policies.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test filesystem_path_archive_policies` |
| `test` | `fingerprint_lock` | `vyre-libs/tests/fingerprint_lock.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test fingerprint_lock` |
| `test` | `fingerprint_lock` | `vyre-libs/tests/fingerprint_lock.rs` | `nn-activation`, `nn-attention`, `nn-linear`, `nn-norm` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test fingerprint_lock` |
| `test` | `flow_precision_planner` | `vyre-libs/tests/flow_precision_planner.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test flow_precision_planner` |
| `test` | `fmm_compress_pairwise_via_reference_parity` | `vyre-libs/tests/fmm_compress_pairwise_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test fmm_compress_pairwise_via_reference_parity` |
| `test` | `fmm_compress_pairwise_via_reference_parity` | `vyre-libs/tests/fmm_compress_pairwise_via_reference_parity.rs` | `cpu-parity`, `solvers`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test fmm_compress_pairwise_via_reference_parity` |
| `test` | `fmm_polyhedral_via_reference_parity` | `vyre-libs/tests/fmm_polyhedral_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test fmm_polyhedral_via_reference_parity` |
| `test` | `fmm_polyhedral_via_reference_parity` | `vyre-libs/tests/fmm_polyhedral_via_reference_parity.rs` | `cpu-parity`, `solvers`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test fmm_polyhedral_via_reference_parity` |
| `test` | `frontend_dialect_contracts` | `vyre-libs/tests/frontend_dialect_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test frontend_dialect_contracts` |
| `test` | `functor_apply_via_reference_parity` | `vyre-libs/tests/functor_apply_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test functor_apply_via_reference_parity` |
| `test` | `functor_apply_via_reference_parity` | `vyre-libs/tests/functor_apply_via_reference_parity.rs` | `cpu-parity`, `reasoning`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test functor_apply_via_reference_parity` |
| `test` | `fuse_decode_scan_error` | `vyre-libs/tests/fuse_decode_scan_error.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test fuse_decode_scan_error` |
| `test` | `fusion_scores_via_reference_parity` | `vyre-libs/tests/fusion_scores_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test fusion_scores_via_reference_parity` |
| `test` | `fusion_scores_via_reference_parity` | `vyre-libs/tests/fusion_scores_via_reference_parity.rs` | `cpu-parity`, `scheduling`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test fusion_scores_via_reference_parity` |
| `test` | `fuzz_target_inventory` | `vyre-libs/tests/fuzz_target_inventory.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test fuzz_target_inventory` |
| `test` | `gated_rms_norm_contract` | `vyre-libs/tests/gated_rms_norm_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gated_rms_norm_contract` |
| `test` | `gated_rms_norm_contract` | `vyre-libs/tests/gated_rms_norm_contract.rs` | `nn-norm` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gated_rms_norm_contract` |
| `test` | `gemini_c_ast_contracts` | `vyre-libs/tests/gemini_c_ast_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gemini_c_ast_contracts` |
| `test` | `gemini_c_ast_contracts` | `vyre-libs/tests/gemini_c_ast_contracts.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gemini_c_ast_contracts` |
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
| `test` | `gqa_attention_primitive_composition_contracts` | `vyre-libs/tests/gqa_attention_primitive_composition_contracts.rs` | `nn-attention` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test gqa_attention_primitive_composition_contracts` |
| `test` | `graph_single_source_contracts` | `vyre-libs/tests/graph_single_source_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test graph_single_source_contracts` |
| `test` | `graph_single_source_contracts` | `vyre-libs/tests/graph_single_source_contracts.rs` | `cpu-parity`, `graph-dispatch`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test graph_single_source_contracts` |
| `test` | `head_to_token_typed_contract` | `vyre-libs/tests/head_to_token_typed_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test head_to_token_typed_contract` |
| `test` | `head_to_token_typed_contract` | `vyre-libs/tests/head_to_token_typed_contract.rs` | `nn-attention` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test head_to_token_typed_contract` |
| `test` | `hex_decode_scan_fused` | `vyre-libs/tests/hex_decode_scan_fused.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test hex_decode_scan_fused` |
| `test` | `indexed_map_composition_contracts` | `vyre-libs/tests/indexed_map_composition_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test indexed_map_composition_contracts` |
| `test` | `indexed_map_composition_contracts` | `vyre-libs/tests/indexed_map_composition_contracts.rs` | `nn-activation` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test indexed_map_composition_contracts` |
| `test` | `int4_primitive_composition` | `vyre-libs/tests/int4_primitive_composition.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test int4_primitive_composition` |
| `test` | `int4_primitive_composition` | `vyre-libs/tests/int4_primitive_composition.rs` | `nn-activation` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test int4_primitive_composition` |
| `test` | `integration` | `vyre-libs/tests/integration.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test integration` |
| `test` | `integration` | `vyre-libs/tests/integration.rs` | `hash`, `matching`, `math`, `nn-activation`, `nn-linear` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test integration` |
| `test` | `ir_aliasing` | `vyre-libs/tests/ir_aliasing.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test ir_aliasing` |
| `test` | `ir_aliasing` | `vyre-libs/tests/ir_aliasing.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test ir_aliasing` |
| `test` | `kfac_via_reference_parity` | `vyre-libs/tests/kfac_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test kfac_via_reference_parity` |
| `test` | `kfac_via_reference_parity` | `vyre-libs/tests/kfac_via_reference_parity.rs` | `cpu-parity`, `solvers`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test kfac_via_reference_parity` |
| `test` | `kv_cache_append_contract` | `vyre-libs/tests/kv_cache_append_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test kv_cache_append_contract` |
| `test` | `kv_cache_append_contract` | `vyre-libs/tests/kv_cache_append_contract.rs` | `nn-attention` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test kv_cache_append_contract` |
| `test` | `kv_cache_typed_contract` | `vyre-libs/tests/kv_cache_typed_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test kv_cache_typed_contract` |
| `test` | `kv_cache_typed_contract` | `vyre-libs/tests/kv_cache_typed_contract.rs` | `nn-attention` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test kv_cache_typed_contract` |
| `test` | `last_dim_l2_norm_contract` | `vyre-libs/tests/last_dim_l2_norm_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test last_dim_l2_norm_contract` |
| `test` | `last_dim_l2_norm_contract` | `vyre-libs/tests/last_dim_l2_norm_contract.rs` | `nn-norm` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test last_dim_l2_norm_contract` |
| `test` | `linear_rows_contract` | `vyre-libs/tests/linear_rows_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test linear_rows_contract` |
| `test` | `linear_rows_contract` | `vyre-libs/tests/linear_rows_contract.rs` | `nn-linear` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test linear_rows_contract` |
| `test` | `literal_set_presence_and_positions_reference` | `vyre-libs/tests/literal_set_presence_and_positions_reference.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_presence_and_positions_reference` |
| `test` | `literal_set_presence_by_region_ground_truth` | `vyre-libs/tests/literal_set_presence_by_region_ground_truth.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_presence_by_region_ground_truth` |
| `test` | `literal_set_presence_reference` | `vyre-libs/tests/literal_set_presence_reference.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test literal_set_presence_reference` |
| `test` | `logical_proptest` | `vyre-libs/tests/logical_proptest.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test logical_proptest` |
| `test` | `logical_should_panic` | `vyre-libs/tests/logical_should_panic.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test logical_should_panic` |
| `test` | `loop_unroll_trip1_idempotence` | `vyre-libs/tests/loop_unroll_trip1_idempotence.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test loop_unroll_trip1_idempotence` |
| `test` | `lr_tables_contracts` | `vyre-libs/tests/lr_tables_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test lr_tables_contracts` |
| `test` | `match_motif_via_reference_parity` | `vyre-libs/tests/match_motif_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test match_motif_via_reference_parity` |
| `test` | `match_motif_via_reference_parity` | `vyre-libs/tests/match_motif_via_reference_parity.rs` | `cpu-parity`, `graph-dispatch`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test match_motif_via_reference_parity` |
| `test` | `matching_diagnostic_via_reference_parity` | `vyre-libs/tests/matching_diagnostic_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test matching_diagnostic_via_reference_parity` |
| `test` | `matching_diagnostic_via_reference_parity` | `vyre-libs/tests/matching_diagnostic_via_reference_parity.rs` | `cpu-parity`, `encoding`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test matching_diagnostic_via_reference_parity` |
| `test` | `matching_nfa_scan_program_contracts` | `vyre-libs/tests/matching_nfa_scan_program_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test matching_nfa_scan_program_contracts` |
| `test` | `matching_post_process_contracts` | `vyre-libs/tests/matching_post_process_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test matching_post_process_contracts` |
| `test` | `math_algebra_branchless_contracts` | `vyre-libs/tests/math_algebra_branchless_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test math_algebra_branchless_contracts` |
| `test` | `matroid_exact_subset_via_reference_parity` | `vyre-libs/tests/matroid_exact_subset_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test matroid_exact_subset_via_reference_parity` |
| `test` | `matroid_exact_subset_via_reference_parity` | `vyre-libs/tests/matroid_exact_subset_via_reference_parity.rs` | `cpu-parity`, `encoding`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test matroid_exact_subset_via_reference_parity` |
| `test` | `mlp_4x_leaky_sq_multi_workgroup_span` | `vyre-libs/tests/mlp_4x_leaky_sq_multi_workgroup_span.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test mlp_4x_leaky_sq_multi_workgroup_span` |
| `test` | `mlp_4x_leaky_sq_multi_workgroup_span` | `vyre-libs/tests/mlp_4x_leaky_sq_multi_workgroup_span.rs` | `nn-activation` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test mlp_4x_leaky_sq_multi_workgroup_span` |
| `test` | `multigrid_matroid_via_reference_parity` | `vyre-libs/tests/multigrid_matroid_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test multigrid_matroid_via_reference_parity` |
| `test` | `multigrid_matroid_via_reference_parity` | `vyre-libs/tests/multigrid_matroid_via_reference_parity.rs` | `cpu-parity`, `solvers`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test multigrid_matroid_via_reference_parity` |
| `test` | `mz_project_via_reference_parity` | `vyre-libs/tests/mz_project_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test mz_project_via_reference_parity` |
| `test` | `mz_project_via_reference_parity` | `vyre-libs/tests/mz_project_via_reference_parity.rs` | `cpu-parity`, `solvers`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test mz_project_via_reference_parity` |
| `test` | `name_collision` | `vyre-libs/tests/name_collision.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test name_collision` |
| `test` | `name_collision` | `vyre-libs/tests/name_collision.rs` | `math`, `nn-attention` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test name_collision` |
| `test` | `natural_config_gradient_via_reference_parity` | `vyre-libs/tests/natural_config_gradient_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test natural_config_gradient_via_reference_parity` |
| `test` | `natural_config_gradient_via_reference_parity` | `vyre-libs/tests/natural_config_gradient_via_reference_parity.rs` | `cpu-parity`, `solvers`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test natural_config_gradient_via_reference_parity` |
| `test` | `natural_gradient_via_reference_parity` | `vyre-libs/tests/natural_gradient_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test natural_gradient_via_reference_parity` |
| `test` | `natural_gradient_via_reference_parity` | `vyre-libs/tests/natural_gradient_via_reference_parity.rs` | `cpu-parity`, `solvers`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test natural_gradient_via_reference_parity` |
| `test` | `nfa_plan_contracts` | `vyre-libs/tests/nfa_plan_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test nfa_plan_contracts` |
| `test` | `nn_attention_clone_family_ir_invariance` | `vyre-libs/tests/nn_attention_clone_family_ir_invariance.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test nn_attention_clone_family_ir_invariance` |
| `test` | `nn_attention_clone_family_ir_invariance` | `vyre-libs/tests/nn_attention_clone_family_ir_invariance.rs` | `nn-attention` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test nn_attention_clone_family_ir_invariance` |
| `test` | `op_boundaries` | `vyre-libs/tests/op_boundaries.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test op_boundaries` |
| `test` | `op_boundaries` | `vyre-libs/tests/op_boundaries.rs` | `nn-activation`, `nn-linear` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test op_boundaries` |
| `test` | `operation_registry` | `vyre-libs/tests/operation_registry.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test operation_registry` |
| `test` | `operation_registry` | `vyre-libs/tests/operation_registry.rs` | `math`, `math-linalg`, `nn-activation`, `nn-attention`, `nn-linear`, `nn-norm` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test operation_registry` |
| `test` | `operator_reporting_interchange` | `vyre-libs/tests/operator_reporting_interchange.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test operator_reporting_interchange` |
| `test` | `optimized_programs` | `vyre-libs/tests/optimized_programs.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test optimized_programs` |
| `test` | `optimized_programs` | `vyre-libs/tests/optimized_programs.rs` | `nn-attention`, `nn-linear`, `nn-norm` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test optimized_programs` |
| `test` | `output_encoding_unicode_policies` | `vyre-libs/tests/output_encoding_unicode_policies.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test output_encoding_unicode_policies` |
| `test` | `overflow_guards` | `vyre-libs/tests/overflow_guards.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test overflow_guards` |
| `test` | `overflow_guards` | `vyre-libs/tests/overflow_guards.rs` | `nn-attention` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test overflow_guards` |
| `test` | `parser_edit_delta_contracts` | `vyre-libs/tests/parser_edit_delta_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test parser_edit_delta_contracts` |
| `test` | `parser_graph_navigation_contracts` | `vyre-libs/tests/parser_graph_navigation_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test parser_graph_navigation_contracts` |
| `test` | `parser_recovery_corpus_registry` | `vyre-libs/tests/parser_recovery_corpus_registry.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test parser_recovery_corpus_registry` |
| `test` | `parsing_walker_clone_family` | `vyre-libs/tests/parsing_walker_clone_family.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test parsing_walker_clone_family` |
| `test` | `parsing_walker_clone_family` | `vyre-libs/tests/parsing_walker_clone_family.rs` | `parsing` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test parsing_walker_clone_family` |
| `test` | `partial_rope_offset_contract` | `vyre-libs/tests/partial_rope_offset_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test partial_rope_offset_contract` |
| `test` | `partial_rope_offset_contract` | `vyre-libs/tests/partial_rope_offset_contract.rs` | `nn-attention` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test partial_rope_offset_contract` |
| `test` | `partial_rope_typed_contract` | `vyre-libs/tests/partial_rope_typed_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test partial_rope_typed_contract` |
| `test` | `partial_rope_typed_contract` | `vyre-libs/tests/partial_rope_typed_contract.rs` | `nn-attention` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test partial_rope_typed_contract` |
| `test` | `pass_research_trace_artifacts` | `vyre-libs/tests/pass_research_trace_artifacts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test pass_research_trace_artifacts` |
| `test` | `planar_rewrite_via_reference_parity` | `vyre-libs/tests/planar_rewrite_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test planar_rewrite_via_reference_parity` |
| `test` | `planar_rewrite_via_reference_parity` | `vyre-libs/tests/planar_rewrite_via_reference_parity.rs` | `cpu-parity`, `scheduling`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test planar_rewrite_via_reference_parity` |
| `test` | `predict_impact_via_reference_parity` | `vyre-libs/tests/predict_impact_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test predict_impact_via_reference_parity` |
| `test` | `predict_impact_via_reference_parity` | `vyre-libs/tests/predict_impact_via_reference_parity.rs` | `cpu-parity`, `reasoning`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test predict_impact_via_reference_parity` |
| `test` | `preprocess_cpu_api_boundary` | `vyre-libs/tests/preprocess_cpu_api_boundary.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test preprocess_cpu_api_boundary` |
| `test` | `primitive_surface_contracts` | `vyre-libs/tests/primitive_surface_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test primitive_surface_contracts` |
| `test` | `primitive_vs_consumer` | `vyre-libs/tests/primitive_vs_consumer.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test primitive_vs_consumer` |
| `test` | `primitive_vs_consumer` | `vyre-libs/tests/primitive_vs_consumer.rs` | `analysis`, `cpu-parity`, `encoding`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test primitive_vs_consumer` |
| `test` | `property` | `vyre-libs/tests/property.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test property` |
| `test` | `property_differential_oracles` | `vyre-libs/tests/property_differential_oracles.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test property_differential_oracles` |
| `test` | `provenance_closure` | `vyre-libs/tests/provenance_closure.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test provenance_closure` |
| `test` | `provenance_closure` | `vyre-libs/tests/provenance_closure.rs` | `cpu-parity`, `encoding`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test provenance_closure` |
| `test` | `qk_gain_shape_overflow_contracts` | `vyre-libs/tests/qk_gain_shape_overflow_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test qk_gain_shape_overflow_contracts` |
| `test` | `qk_gain_shape_overflow_contracts` | `vyre-libs/tests/qk_gain_shape_overflow_contracts.rs` | `nn-attention` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test qk_gain_shape_overflow_contracts` |
| `test` | `qk_gain_zero_shape_contracts` | `vyre-libs/tests/qk_gain_zero_shape_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test qk_gain_zero_shape_contracts` |
| `test` | `qk_gain_zero_shape_contracts` | `vyre-libs/tests/qk_gain_zero_shape_contracts.rs` | `nn-attention` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test qk_gain_zero_shape_contracts` |
| `test` | `quantized_linear_affine_fma` | `vyre-libs/tests/quantized_linear_affine_fma.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test quantized_linear_affine_fma` |
| `test` | `quantized_linear_affine_fma` | `vyre-libs/tests/quantized_linear_affine_fma.rs` | `nn-linear` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test quantized_linear_affine_fma` |
| `test` | `quantized_via_reference_parity` | `vyre-libs/tests/quantized_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test quantized_via_reference_parity` |
| `test` | `quantized_via_reference_parity` | `vyre-libs/tests/quantized_via_reference_parity.rs` | `cpu-parity`, `solvers`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test quantized_via_reference_parity` |
| `test` | `reconstruct_path_via_reference_parity` | `vyre-libs/tests/reconstruct_path_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test reconstruct_path_via_reference_parity` |
| `test` | `reconstruct_path_via_reference_parity` | `vyre-libs/tests/reconstruct_path_via_reference_parity.rs` | `cpu-parity`, `graph-dispatch`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test reconstruct_path_via_reference_parity` |
| `test` | `recurrent_gated_delta_contract` | `vyre-libs/tests/recurrent_gated_delta_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test recurrent_gated_delta_contract` |
| `test` | `recurrent_gated_delta_contract` | `vyre-libs/tests/recurrent_gated_delta_contract.rs` | `nn-attention` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test recurrent_gated_delta_contract` |
| `test` | `reduction_metrics_via_reference_parity` | `vyre-libs/tests/reduction_metrics_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test reduction_metrics_via_reference_parity` |
| `test` | `reduction_metrics_via_reference_parity` | `vyre-libs/tests/reduction_metrics_via_reference_parity.rs` | `cpu-parity`, `encoding`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test reduction_metrics_via_reference_parity` |
| `test` | `regex_adversarial_class_catalog` | `vyre-libs/tests/regex_adversarial_class_catalog.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test regex_adversarial_class_catalog` |
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
| `test` | `regex_streaming_state_ledger` | `vyre-libs/tests/regex_streaming_state_ledger.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test regex_streaming_state_ledger` |
| `test` | `regex_unicode_profiles` | `vyre-libs/tests/regex_unicode_profiles.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test regex_unicode_profiles` |
| `test` | `regex_unsupported_diagnostic_registry` | `vyre-libs/tests/regex_unsupported_diagnostic_registry.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test regex_unsupported_diagnostic_registry` |
| `test` | `region_chain_discipline` | `vyre-libs/tests/region_chain_discipline.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test region_chain_discipline` |
| `test` | `region_chain_invariant` | `vyre-libs/tests/region_chain_invariant.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test region_chain_invariant` |
| `test` | `region_inline_let_scope` | `vyre-libs/tests/region_inline_let_scope.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test region_inline_let_scope` |
| `test` | `registration_drift` | `vyre-libs/tests/registration_drift.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test registration_drift` |
| `test` | `registry_closure` | `vyre-libs/tests/registry_closure.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test registry_closure` |
| `test` | `resource_budget_complexity_policies` | `vyre-libs/tests/resource_budget_complexity_policies.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test resource_budget_complexity_policies` |
| `test` | `rule_condition_program_frame_contract` | `vyre-libs/tests/rule_condition_program_frame_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test rule_condition_program_frame_contract` |
| `test` | `rule_condition_program_frame_contract` | `vyre-libs/tests/rule_condition_program_frame_contract.rs` | `rule` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test rule_condition_program_frame_contract` |
| `test` | `scallop_provenance_via_reference_parity` | `vyre-libs/tests/scallop_provenance_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test scallop_provenance_via_reference_parity` |
| `test` | `scallop_provenance_via_reference_parity` | `vyre-libs/tests/scallop_provenance_via_reference_parity.rs` | `cpu-parity`, `encoding`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test scallop_provenance_via_reference_parity` |
| `test` | `scan_ac_transition_walk_single_owner` | `vyre-libs/tests/scan_ac_transition_walk_single_owner.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test scan_ac_transition_walk_single_owner` |
| `test` | `scan_ac_transition_walk_single_owner` | `vyre-libs/tests/scan_ac_transition_walk_single_owner.rs` | `matching-regex` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test scan_ac_transition_walk_single_owner` |
| `test` | `scan_conformance_matrix` | `vyre-libs/tests/scan_conformance_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test scan_conformance_matrix` |
| `test` | `scan_cpu_api_boundary` | `vyre-libs/tests/scan_cpu_api_boundary.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test scan_cpu_api_boundary` |
| `test` | `scan_hit_buffer_layout_contracts` | `vyre-libs/tests/scan_hit_buffer_layout_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test scan_hit_buffer_layout_contracts` |
| `test` | `scan_hit_buffer_layout_contracts` | `vyre-libs/tests/scan_hit_buffer_layout_contracts.rs` | `matching-substring` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test scan_hit_buffer_layout_contracts` |
| `test` | `secret_crypto_policies` | `vyre-libs/tests/secret_crypto_policies.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test secret_crypto_policies` |
| `test` | `security_external_ifds` | `vyre-libs/tests/security_external_ifds.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test security_external_ifds` |
| `test` | `security_flow_skeleton_family_guard` | `vyre-libs/tests/security_flow_skeleton_family_guard.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test security_flow_skeleton_family_guard` |
| `test` | `security_flow_skeleton_family_guard` | `vyre-libs/tests/security_flow_skeleton_family_guard.rs` | `security` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test security_flow_skeleton_family_guard` |
| `test` | `security_flows_to_alias_only_parity` | `vyre-libs/tests/security_flows_to_alias_only_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test security_flows_to_alias_only_parity` |
| `test` | `security_privacy_path_corpus_guards` | `vyre-libs/tests/security_privacy_path_corpus_guards.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test security_privacy_path_corpus_guards` |
| `test` | `self_consumer_conform` | `vyre-libs/tests/self_consumer_conform.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test self_consumer_conform` |
| `test` | `self_consumer_conform` | `vyre-libs/tests/self_consumer_conform.rs` | `cpu-parity`, `encoding`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test self_consumer_conform` |
| `test` | `semiring_gemm_via_reference_parity` | `vyre-libs/tests/semiring_gemm_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test semiring_gemm_via_reference_parity` |
| `test` | `semiring_gemm_via_reference_parity` | `vyre-libs/tests/semiring_gemm_via_reference_parity.rs` | `analysis`, `cpu-parity`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test semiring_gemm_via_reference_parity` |
| `test` | `shape_spectrum_via_reference_parity` | `vyre-libs/tests/shape_spectrum_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test shape_spectrum_via_reference_parity` |
| `test` | `shape_spectrum_via_reference_parity` | `vyre-libs/tests/shape_spectrum_via_reference_parity.rs` | `cpu-parity`, `scheduling`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test shape_spectrum_via_reference_parity` |
| `test` | `shared_emitter_artifact_schema` | `vyre-libs/tests/shared_emitter_artifact_schema.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test shared_emitter_artifact_schema` |
| `test` | `sheaf_heterophilic_via_reference_parity` | `vyre-libs/tests/sheaf_heterophilic_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sheaf_heterophilic_via_reference_parity` |
| `test` | `sheaf_heterophilic_via_reference_parity` | `vyre-libs/tests/sheaf_heterophilic_via_reference_parity.rs` | `cpu-parity`, `solvers`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sheaf_heterophilic_via_reference_parity` |
| `test` | `sheaf_spectrum_via_reference_parity` | `vyre-libs/tests/sheaf_spectrum_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sheaf_spectrum_via_reference_parity` |
| `test` | `sheaf_spectrum_via_reference_parity` | `vyre-libs/tests/sheaf_spectrum_via_reference_parity.rs` | `cpu-parity`, `solvers`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sheaf_spectrum_via_reference_parity` |
| `test` | `sigmoid_gate_typed_contract` | `vyre-libs/tests/sigmoid_gate_typed_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sigmoid_gate_typed_contract` |
| `test` | `sigmoid_gate_typed_contract` | `vyre-libs/tests/sigmoid_gate_typed_contract.rs` | `nn-activation` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sigmoid_gate_typed_contract` |
| `test` | `sinkhorn_via_reference_parity` | `vyre-libs/tests/sinkhorn_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sinkhorn_via_reference_parity` |
| `test` | `sinkhorn_via_reference_parity` | `vyre-libs/tests/sinkhorn_via_reference_parity.rs` | `cpu-parity`, `solvers`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sinkhorn_via_reference_parity` |
| `test` | `smooth_latency_trace_via_reference_parity` | `vyre-libs/tests/smooth_latency_trace_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test smooth_latency_trace_via_reference_parity` |
| `test` | `smooth_latency_trace_via_reference_parity` | `vyre-libs/tests/smooth_latency_trace_via_reference_parity.rs` | `cpu-parity`, `solvers`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test smooth_latency_trace_via_reference_parity` |
| `test` | `smooth_matroid_flow_via_reference_parity` | `vyre-libs/tests/smooth_matroid_flow_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test smooth_matroid_flow_via_reference_parity` |
| `test` | `smooth_matroid_flow_via_reference_parity` | `vyre-libs/tests/smooth_matroid_flow_via_reference_parity.rs` | `cpu-parity`, `solvers`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test smooth_matroid_flow_via_reference_parity` |
| `test` | `softmax_pick_config_via_reference_parity` | `vyre-libs/tests/softmax_pick_config_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test softmax_pick_config_via_reference_parity` |
| `test` | `softmax_pick_config_via_reference_parity` | `vyre-libs/tests/softmax_pick_config_via_reference_parity.rs` | `cpu-parity`, `solvers`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test softmax_pick_config_via_reference_parity` |
| `test` | `solvers_dispatch_softmax_contract` | `vyre-libs/tests/solvers_dispatch_softmax_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test solvers_dispatch_softmax_contract` |
| `test` | `solvers_dispatch_softmax_contract` | `vyre-libs/tests/solvers_dispatch_softmax_contract.rs` | `solvers` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test solvers_dispatch_softmax_contract` |
| `test` | `source_span_witness_records` | `vyre-libs/tests/source_span_witness_records.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test source_span_witness_records` |
| `test` | `statement_bounds_launch_contract` | `vyre-libs/tests/statement_bounds_launch_contract.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test statement_bounds_launch_contract` |
| `test` | `string_diagram_via_reference_parity` | `vyre-libs/tests/string_diagram_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test string_diagram_via_reference_parity` |
| `test` | `string_diagram_via_reference_parity` | `vyre-libs/tests/string_diagram_via_reference_parity.rs` | `cpu-parity`, `reasoning`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test string_diagram_via_reference_parity` |
| `test` | `submodular_retention_via_reference_parity` | `vyre-libs/tests/submodular_retention_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test submodular_retention_via_reference_parity` |
| `test` | `submodular_retention_via_reference_parity` | `vyre-libs/tests/submodular_retention_via_reference_parity.rs` | `cpu-parity`, `scheduling`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test submodular_retention_via_reference_parity` |
| `test` | `succinct_rank_contracts` | `vyre-libs/tests/succinct_rank_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test succinct_rank_contracts` |
| `test` | `succinct_rank_select_adversarial_contracts` | `vyre-libs/tests/succinct_rank_select_adversarial_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test succinct_rank_select_adversarial_contracts` |
| `test` | `surface_contracts` | `vyre-libs/tests/surface_contracts.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test surface_contracts` |
| `test` | `surface_contracts` | `vyre-libs/tests/surface_contracts.rs` | `nn-attention`, `nn-norm` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test surface_contracts` |
| `test` | `sweep_decode_hex_oracle_matrix` | `vyre-libs/tests/sweep_decode_hex_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sweep_decode_hex_oracle_matrix` |
| `test` | `sweep_decode_hex_oracle_matrix` | `vyre-libs/tests/sweep_decode_hex_oracle_matrix.rs` | `decode` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sweep_decode_hex_oracle_matrix` |
| `test` | `sweep_graph_cpu_oracle_matrix` | `vyre-libs/tests/sweep_graph_cpu_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sweep_graph_cpu_oracle_matrix` |
| `test` | `sweep_graph_cpu_oracle_matrix` | `vyre-libs/tests/sweep_graph_cpu_oracle_matrix.rs` | `cpu-parity`, `graph-dispatch`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sweep_graph_cpu_oracle_matrix` |
| `test` | `sweep_hash_crc32_reference_matrix` | `vyre-libs/tests/sweep_hash_crc32_reference_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sweep_hash_crc32_reference_matrix` |
| `test` | `sweep_hash_crc32_reference_matrix` | `vyre-libs/tests/sweep_hash_crc32_reference_matrix.rs` | `hash` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sweep_hash_crc32_reference_matrix` |
| `test` | `sweep_logical_reference_matrix` | `vyre-libs/tests/sweep_logical_reference_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sweep_logical_reference_matrix` |
| `test` | `sweep_logical_reference_matrix` | `vyre-libs/tests/sweep_logical_reference_matrix.rs` | `logical` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sweep_logical_reference_matrix` |
| `test` | `sweep_text_utf8_oracle_matrix` | `vyre-libs/tests/sweep_text_utf8_oracle_matrix.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sweep_text_utf8_oracle_matrix` |
| `test` | `sweep_text_utf8_oracle_matrix` | `vyre-libs/tests/sweep_text_utf8_oracle_matrix.rs` | `text` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test sweep_text_utf8_oracle_matrix` |
| `test` | `target_instruction_capabilities` | `vyre-libs/tests/target_instruction_capabilities.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test target_instruction_capabilities` |
| `test` | `tensor_train_chain_fusion_via_reference_parity` | `vyre-libs/tests/tensor_train_chain_fusion_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test tensor_train_chain_fusion_via_reference_parity` |
| `test` | `tensor_train_chain_fusion_via_reference_parity` | `vyre-libs/tests/tensor_train_chain_fusion_via_reference_parity.rs` | `cpu-parity`, `solvers`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test tensor_train_chain_fusion_via_reference_parity` |
| `test` | `tensor_train_compress_via_reference_parity` | `vyre-libs/tests/tensor_train_compress_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test tensor_train_compress_via_reference_parity` |
| `test` | `tensor_train_compress_via_reference_parity` | `vyre-libs/tests/tensor_train_compress_via_reference_parity.rs` | `cpu-parity`, `solvers`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test tensor_train_compress_via_reference_parity` |
| `test` | `transport_residual_via_reference_parity` | `vyre-libs/tests/transport_residual_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test transport_residual_via_reference_parity` |
| `test` | `transport_residual_via_reference_parity` | `vyre-libs/tests/transport_residual_via_reference_parity.rs` | `cpu-parity`, `solvers`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test transport_residual_via_reference_parity` |
| `test` | `typedef_row_phase_witnesses` | `vyre-libs/tests/typedef_row_phase_witnesses.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test typedef_row_phase_witnesses` |
| `test` | `typedef_row_phase_witnesses` | `vyre-libs/tests/typedef_row_phase_witnesses.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test typedef_row_phase_witnesses` |
| `test` | `union_find_alias_via_reference_parity` | `vyre-libs/tests/union_find_alias_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test union_find_alias_via_reference_parity` |
| `test` | `union_find_alias_via_reference_parity` | `vyre-libs/tests/union_find_alias_via_reference_parity.rs` | `cpu-parity`, `graph-dispatch`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test union_find_alias_via_reference_parity` |
| `test` | `universal_harness` | `vyre-libs/tests/universal_harness.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test universal_harness` |
| `test` | `unsafe_ffi_policies` | `vyre-libs/tests/unsafe_ffi_policies.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test unsafe_ffi_policies` |
| `test` | `url_network_security_policies` | `vyre-libs/tests/url_network_security_policies.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test url_network_security_policies` |
| `test` | `vast_builder_oob_guard_regression` | `vyre-libs/tests/vast_builder_oob_guard_regression.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test vast_builder_oob_guard_regression` |
| `test` | `vast_builder_oob_guard_regression` | `vyre-libs/tests/vast_builder_oob_guard_regression.rs` | `c-parser` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test vast_builder_oob_guard_regression` |
| `test` | `vietoris_rips_via_reference_parity` | `vyre-libs/tests/vietoris_rips_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test vietoris_rips_via_reference_parity` |
| `test` | `vietoris_rips_via_reference_parity` | `vyre-libs/tests/vietoris_rips_via_reference_parity.rs` | `cpu-parity`, `solvers`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test vietoris_rips_via_reference_parity` |
| `test` | `visual_compositions` | `vyre-libs/tests/visual_compositions.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test visual_compositions` |
| `test` | `vsa_fingerprint_via_reference_parity` | `vyre-libs/tests/vsa_fingerprint_via_reference_parity.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test vsa_fingerprint_via_reference_parity` |
| `test` | `vsa_fingerprint_via_reference_parity` | `vyre-libs/tests/vsa_fingerprint_via_reference_parity.rs` | `cpu-parity`, `encoding`, `test-fixtures` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test vsa_fingerprint_via_reference_parity` |
| `test` | `wire_cross_crate_compat` | `vyre-libs/tests/wire_cross_crate_compat.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test wire_cross_crate_compat` |
| `test` | `workgroup_cooperative_tiling` | `vyre-libs/tests/workgroup_cooperative_tiling.rs` | None | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test workgroup_cooperative_tiling` |
| `test` | `workgroup_cooperative_tiling` | `vyre-libs/tests/workgroup_cooperative_tiling.rs` | `nn-attention`, `nn-norm` | `CARGO_BUILD_JOBS=1 ./cargo_full test -p vyre-libs --test workgroup_cooperative_tiling` |

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
