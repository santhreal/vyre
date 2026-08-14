use super::*;

#[test]
fn cpu_reference_builds_delimiter_tree_for_function_body() {
    let (tok_types, _, tok_lens) =
        c_rows("INT:3 IDENTIFIER:4 LPAREN RPAREN LBRACE RETURN:6 INTEGER SEMICOLON RBRACE");
    let tok_starts = [0u32, 4, 8, 9, 10, 11, 18, 19, 20];
    let rows = reference_c11_build_vast_nodes(&tok_types, &tok_starts, &tok_lens);

    assert_vast_row(&rows, 0, TOK_INT, u32::MAX, u32::MAX, 1);
    assert_vast_row(&rows, 1, TOK_IDENTIFIER, u32::MAX, u32::MAX, 2);
    assert_vast_row(&rows, 2, TOK_LPAREN, u32::MAX, 3, 4);
    assert_vast_row(&rows, 3, TOK_RPAREN, 2, u32::MAX, u32::MAX);
    assert_vast_row(&rows, 4, TOK_LBRACE, u32::MAX, 5, u32::MAX);
    assert_vast_row(&rows, 5, TOK_RETURN, 4, u32::MAX, 6);
    assert_vast_row(&rows, 6, TOK_INTEGER, 4, u32::MAX, 7);
    assert_vast_row(&rows, 7, TOK_SEMICOLON, 4, u32::MAX, 8);
    assert_vast_row(&rows, 8, TOK_RBRACE, 4, u32::MAX, u32::MAX);
}

#[test]
fn cpu_reference_classifies_gnu_c_style_function_definition() {
    let (tok_types, tok_starts, tok_lens) = c_rows(
        "STATIC:6 INLINE:6 LONG:4 STAR IDENTIFIER:9 LPAREN STRUCT:6 IDENTIFIER:6 STAR \
         IDENTIFIER:3 COMMA CONST:5 CHAR_KW:4 IDENTIFIER:6 STAR IDENTIFIER:3 RPAREN LBRACE \
         RETURN:6 IDENTIFIER:6 LPAREN IDENTIFIER:3 RPAREN QUESTION IDENTIFIER:7 LPAREN \
         IDENTIFIER:3 RPAREN COLON INTEGER SEMICOLON RBRACE",
    );
    let raw = reference_c11_build_vast_nodes(&tok_types, &tok_starts, &tok_lens);
    let typed = reference_c11_classify_vast_node_kinds(&raw);

    assert_kind(&typed, 4, C_AST_KIND_FUNCTION_DEFINITION);
    assert_kind(&typed, 17, node_kind::BASIC_BLOCK);
    assert_kind(&typed, 19, node_kind::CALL);
    assert_kind(&typed, 21, node_kind::VARIABLE);
    assert_kind(&typed, 24, node_kind::CALL);
    assert_kind(&typed, 29, node_kind::LITERAL);
    assert_eq!(
        word_at(&typed, 4 * VAST_STRIDE_U32 + 5),
        tok_starts[4],
        "function span start must survive classification"
    );
}

#[test]
fn cpu_reference_classifies_gnu_c_typedef_attributes_blocks_and_calls() {
    let (tok_types, tok_starts, tok_lens) = c_rows(
        "TYPEDEF:7 LONG:4 IDENTIFIER:10 LPAREN STRUCT:6 IDENTIFIER:4 STAR IDENTIFIER COMMA INT:3 \
         IDENTIFIER:5 RPAREN SEMICOLON STATIC:6 INLINE:6 LONG:4 IDENTIFIER:12 LPAREN STRUCT:6 \
         IDENTIFIER:4 STAR IDENTIFIER COMMA CONST:5 CHAR_KW:4 STAR IDENTIFIER:4 COMMA INT:3 \
         IDENTIFIER:5 RPAREN GNU_ATTRIBUTE:13 LPAREN LPAREN IDENTIFIER:13 RPAREN RPAREN LBRACE \
         LBRACE IDENTIFIER:11 LPAREN IDENTIFIER:4 RPAREN SEMICOLON RBRACE IF:2 LPAREN IDENTIFIER:6 \
         LPAREN IDENTIFIER:5 AMP INTEGER RPAREN RPAREN LBRACE GNU_ASM:3 VOLATILE:8 LPAREN STRING:7 \
         COLON COLON COLON STRING:8 RPAREN SEMICOLON RETURN:6 IDENTIFIER:8 LPAREN IDENTIFIER COMMA \
         IDENTIFIER:5 RPAREN SEMICOLON RBRACE RETURN:6 INTEGER SEMICOLON RBRACE",
    );
    let raw = reference_c11_build_vast_nodes(&tok_types, &tok_starts, &tok_lens);
    let typed = reference_c11_classify_vast_node_kinds(&raw);

    let functions = typed_indices(&typed, node_kind::FUNCTION_DECL);
    let function_defs = typed_indices(&typed, C_AST_KIND_FUNCTION_DEFINITION);
    let calls = typed_indices(&typed, node_kind::CALL);
    let blocks = typed_indices(&typed, node_kind::BASIC_BLOCK);
    let literals = typed_indices(&typed, node_kind::LITERAL);
    let attributes = typed_indices(&typed, C_AST_KIND_GNU_ATTRIBUTE);
    let inline_asm = typed_indices(&typed, C_AST_KIND_INLINE_ASM);
    let asm_templates = typed_indices(&typed, C_AST_KIND_ASM_TEMPLATE);
    let asm_clobbers = typed_indices(&typed, C_AST_KIND_ASM_CLOBBERS_LIST);

    assert_eq!(
        functions,
        vec![2],
        "typedef prototype must remain a generic function declaration"
    );
    assert_eq!(
        function_defs,
        vec![16],
        "attributed GNU-C definition with a body must be a first-class function definition"
    );
    assert!(
        calls.len() >= 3,
        "trace_fault, likely, and do_fault calls must be typed; got {calls:?}"
    );
    assert!(
        blocks.len() >= 3,
        "outer function, nested block, and if body must be basic blocks"
    );
    assert!(
        literals.len() >= 2,
        "integer literals must survive classification"
    );
    assert_eq!(
        attributes,
        vec![31],
        "GNU attribute syntax must be a first-class VAST node"
    );
    assert_eq!(
        inline_asm,
        vec![55],
        "inline asm syntax must be a first-class VAST node"
    );
    assert_eq!(
        asm_templates,
        vec![58],
        "inline asm template strings must be first-class VAST nodes"
    );
    assert_eq!(
        asm_clobbers,
        vec![62],
        "inline asm clobber strings must be first-class VAST nodes"
    );
    assert_vast_row(&typed, 37, node_kind::BASIC_BLOCK, u32::MAX, 38, u32::MAX);
    assert_vast_row(&typed, 38, node_kind::BASIC_BLOCK, 37, 39, 45);
    assert_vast_row(&typed, 54, node_kind::BASIC_BLOCK, 37, 55, 74);
    assert_ne!(
        word_at(&typed, 31 * VAST_STRIDE_U32),
        node_kind::FUNCTION_DECL,
        "GNU attribute suffix must not be mistaken for the function declarator"
    );
}

#[test]
fn cpu_reference_classifies_c_statement_keywords_as_first_class_vast_nodes() {
    let (tok_types, tok_starts, tok_lens) = c_rows(
        "IF:2 LPAREN IDENTIFIER RPAREN RETURN:6 INTEGER SEMICOLON ELSE:4 FOR:3 LPAREN SEMICOLON \
         SEMICOLON RPAREN WHILE:5 LPAREN IDENTIFIER RPAREN DO:2 CONTINUE:8 SEMICOLON WHILE:5 LPAREN \
         IDENTIFIER RPAREN SEMICOLON SWITCH:6 LPAREN IDENTIFIER RPAREN LBRACE CASE:4 INTEGER COLON \
         BREAK:5 SEMICOLON DEFAULT:7 COLON GOTO:4 IDENTIFIER:3 SEMICOLON RBRACE",
    );
    let raw = reference_c11_build_vast_nodes(&tok_types, &tok_starts, &tok_lens);
    let typed = reference_c11_classify_vast_node_kinds(&raw);

    assert_kind(&typed, 0, C_AST_KIND_IF_STMT);
    assert_kind(&typed, 4, C_AST_KIND_RETURN_STMT);
    assert_kind(&typed, 7, C_AST_KIND_ELSE_STMT);
    assert_kind(&typed, 8, C_AST_KIND_FOR_STMT);
    assert_kind(&typed, 13, C_AST_KIND_WHILE_STMT);
    assert_kind(&typed, 17, C_AST_KIND_DO_STMT);
    assert_kind(&typed, 18, C_AST_KIND_CONTINUE_STMT);
    assert_kind(&typed, 20, C_AST_KIND_WHILE_STMT);
    assert_kind(&typed, 25, C_AST_KIND_SWITCH_STMT);
    assert_kind(&typed, 30, C_AST_KIND_CASE_STMT);
    assert_kind(&typed, 33, C_AST_KIND_BREAK_STMT);
    assert_kind(&typed, 35, C_AST_KIND_DEFAULT_STMT);
    assert_kind(&typed, 37, C_AST_KIND_GOTO_STMT);
    assert_kind(&typed, 38, node_kind::VARIABLE);
    assert_kind(&typed, 29, node_kind::BASIC_BLOCK);
}
