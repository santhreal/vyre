use super::*;

#[test]
fn gpu_parity_classifies_c_declarators_initializers_and_fields() {
    let (tok_types, tok_starts, tok_lens) = declarator_initializer_fixture();
    let typed = classified_rows(&tok_types, &tok_starts, &tok_lens);

    assert_kind(&typed, 4, C_AST_KIND_FIELD_DECL);
    assert_kind(&typed, 8, C_AST_KIND_POINTER_DECL);
    assert_kind(&typed, 13, C_AST_KIND_ARRAY_DECL);
    assert_kind(&typed, 22, C_AST_KIND_ENUMERATOR_DECL);
    assert_kind(&typed, 35, C_AST_KIND_FUNCTION_DECLARATOR);
    assert_kind(&typed, 46, C_AST_KIND_CAST_EXPR);
    assert_kind(&typed, 56, C_AST_KIND_COMPOUND_LITERAL_EXPR);
    assert_kind(&typed, 60, C_AST_KIND_INITIALIZER_LIST);
}

#[test]
fn gpu_parity_classifies_nested_function_pointer_array_prototype() {
    let (tok_types, tok_starts, tok_lens) = function_pointer_array_prototype_fixture();
    let typed = classified_rows(&tok_types, &tok_starts, &tok_lens);

    assert_kind(&typed, 2, C_AST_KIND_POINTER_DECL);
    assert_kind(&typed, 4, C_AST_KIND_ARRAY_DECL);
    assert_kind(&typed, 9, C_AST_KIND_FUNCTION_DECLARATOR);
    assert_kind(&typed, 12, C_AST_KIND_POINTER_DECL);
    assert_kind(&typed, 18, C_AST_KIND_ARRAY_DECL);
}

#[test]
fn gpu_parity_classifies_anonymous_aggregate_declarators() {
    let (tok_types, tok_starts, tok_lens) = anonymous_aggregate_declarator_fixture();
    let typed = classified_rows(&tok_types, &tok_starts, &tok_lens);

    assert_kind(&typed, 10, C_AST_KIND_POINTER_DECL);
    assert_kind(&typed, 12, C_AST_KIND_ARRAY_DECL);
    assert_kind(&typed, 16, C_AST_KIND_FUNCTION_DECLARATOR);
    assert_kind(&typed, 28, C_AST_KIND_FIELD_DECL);
    assert_kind(&typed, 39, C_AST_KIND_FIELD_DECL);
}

#[test]
fn gpu_parity_int_main_return_zero_vast_rows() {
    let tok_types = [
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_RPAREN,
        TOK_LBRACE,
        TOK_RETURN,
        TOK_INTEGER,
        TOK_SEMICOLON,
        TOK_RBRACE,
    ];
    let tok_starts = [0u32, 4, 8, 9, 10, 11, 18, 19, 20];
    let tok_lens = [3u32, 4, 1, 1, 1, 6, 1, 1, 1];

    let (rows, count) = arm_raw_vast_with_count(&GpuArm, &tok_types, &tok_starts, &tok_lens);
    assert_eq!(
        rows,
        reference_c11_build_vast_nodes(&tok_types, &tok_starts, &tok_lens),
        "GPU VAST rows must match CPU oracle"
    );
    assert_eq!(
        count,
        tok_types.len() as u32,
        "VAST builder must write exact node count"
    );

    for i in 0..tok_types.len() {
        let row = i * VAST_STRIDE_U32;
        assert_eq!(word_at(&rows, row), tok_types[i], "kind[{i}]");
        assert_eq!(word_at(&rows, row + 5), tok_starts[i], "start[{i}]");
        assert_eq!(word_at(&rows, row + 6), tok_lens[i], "len[{i}]");
    }

    assert_vast_row(&rows, 2, TOK_LPAREN, u32::MAX, 3, 4);
    assert_vast_row(&rows, 4, TOK_LBRACE, u32::MAX, 5, u32::MAX);
    assert_vast_row(&rows, 8, TOK_RBRACE, 4, u32::MAX, u32::MAX);
}
