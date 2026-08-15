use super::*;

#[test]
fn gpu_parity_classifies_gnu_attribute_and_inline_asm_nodes() {
    let tok_types = [
        TOK_STATIC,
        TOK_INLINE,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_RPAREN,
        TOK_GNU_ATTRIBUTE,
        TOK_LPAREN,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_RPAREN,
        TOK_LBRACE,
        TOK_GNU_ASM,
        TOK_VOLATILE,
        TOK_LPAREN,
        TOK_STRING,
        TOK_COLON,
        TOK_COLON,
        TOK_COLON,
        TOK_STRING,
        TOK_RPAREN,
        TOK_SEMICOLON,
        TOK_RETURN,
        TOK_INTEGER,
        TOK_SEMICOLON,
        TOK_RBRACE,
    ];
    let tok_lens = [
        6, 6, 3, 6, 1, 1, 13, 1, 1, 13, 1, 1, 1, 3, 8, 1, 5, 1, 1, 1, 8, 1, 1, 6, 1, 1, 1,
    ];
    let typed = classified_rows(&tok_types, &starts_for_lens(&tok_lens), &tok_lens);

    assert_kind(&typed, 3, C_AST_KIND_FUNCTION_DEFINITION);
    assert_kind(&typed, 6, C_AST_KIND_GNU_ATTRIBUTE);
    assert_kind(&typed, 13, C_AST_KIND_INLINE_ASM);
    assert_kind(&typed, 16, C_AST_KIND_ASM_TEMPLATE);
    assert_kind(&typed, 20, C_AST_KIND_ASM_CLOBBERS_LIST);
}

#[test]
fn gpu_parity_classifies_c_statement_nodes() {
    let tok_types = [
        TOK_IF,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_RETURN,
        TOK_INTEGER,
        TOK_SEMICOLON,
        TOK_ELSE,
        TOK_FOR,
        TOK_LPAREN,
        TOK_SEMICOLON,
        TOK_SEMICOLON,
        TOK_RPAREN,
        TOK_WHILE,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_DO,
        TOK_CONTINUE,
        TOK_SEMICOLON,
        TOK_WHILE,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_SEMICOLON,
        TOK_SWITCH,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_LBRACE,
        TOK_CASE,
        TOK_INTEGER,
        TOK_COLON,
        TOK_BREAK,
        TOK_SEMICOLON,
        TOK_DEFAULT,
        TOK_COLON,
        TOK_GOTO,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_RBRACE,
    ];
    let tok_lens = [
        2, 1, 1, 1, 6, 1, 1, 4, 3, 1, 1, 1, 1, 5, 1, 1, 1, 2, 8, 1, 5, 1, 1, 1, 1, 6, 1, 1, 1, 1,
        4, 1, 1, 5, 1, 7, 1, 4, 3, 1, 1,
    ];
    let typed = classified_rows(&tok_types, &starts_for_lens(&tok_lens), &tok_lens);

    assert_kind(&typed, 0, C_AST_KIND_IF_STMT);
    assert_kind(&typed, 4, C_AST_KIND_RETURN_STMT);
    assert_kind(&typed, 8, C_AST_KIND_FOR_STMT);
    assert_kind(&typed, 17, C_AST_KIND_DO_STMT);
    assert_kind(&typed, 25, C_AST_KIND_SWITCH_STMT);
    assert_kind(&typed, 37, C_AST_KIND_GOTO_STMT);
}

#[test]
fn gpu_parity_classifies_c_expression_operator_nodes() {
    let (tok_types, tok_starts, tok_lens) = expression_operator_fixture();
    let typed = classified_rows(&tok_types, &tok_starts, &tok_lens);

    assert_kind(&typed, 1, C_AST_KIND_MEMBER_ACCESS_EXPR);
    assert_kind(&typed, 3, C_AST_KIND_ASSIGN_EXPR);
    assert_kind(&typed, 4, C_AST_KIND_SIZEOF_EXPR);
    assert_kind(&typed, 6, C_AST_KIND_UNARY_EXPR);
    assert_kind(&typed, 13, node_kind::BINARY);
    assert_kind(&typed, 19, C_AST_KIND_CONDITIONAL_EXPR);
    assert_kind(&typed, 28, C_AST_KIND_ASSIGN_EXPR);
    assert_kind(&typed, 32, node_kind::BINARY);
    assert_kind(&typed, 36, node_kind::BINARY);
    assert_kind(&typed, 40, C_AST_KIND_ARRAY_SUBSCRIPT_EXPR);
    assert_kind(&typed, 46, C_AST_KIND_UNARY_EXPR);
    assert_kind(&typed, 48, node_kind::BINARY);
    assert_kind(&typed, 49, C_AST_KIND_UNARY_EXPR);
    assert_kind(&typed, 52, C_AST_KIND_UNARY_EXPR);
    assert_kind(&typed, 56, node_kind::BINARY);
}

#[test]
fn gpu_parity_builds_c11_expression_semantic_shape_rows() {
    let (tok_types, tok_starts, tok_lens) = expression_shape_fixture();
    let raw = reference_c11_build_vast_nodes(&tok_types, &tok_starts, &tok_lens);
    let typed = reference_c11_classify_vast_node_kinds(&raw);
    let shape = run_gpu_expr_shape(&raw, &typed);

    assert_eq!(
        shape,
        reference_c11_build_expression_shape_nodes(&raw, &typed),
        "GPU expression-shape rows must match CPU oracle"
    );
    assert_expr_shape_row(
        &shape,
        5,
        C_EXPR_SHAPE_CONDITIONAL,
        TOK_QUESTION,
        3,
        C_EXPR_ASSOC_RIGHT,
        1,
        7,
        11,
    );
}
