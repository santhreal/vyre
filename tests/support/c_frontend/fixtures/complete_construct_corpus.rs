//! Token fixtures and the case table for the complete C11 construct corpus.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use crate::c_frontend::spelling::c_rows;
pub(crate) fn fixture_macro_shaped_decl_after_preproc() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "PREPROC:7 GNU_ATTRIBUTE:13 LPAREN LPAREN IDENTIFIER:10 LPAREN STRING:10 RPAREN \
         RPAREN RPAREN INT:3 STAR IDENTIFIER:5 LPAREN INT:3 IDENTIFIER:2 RPAREN SEMICOLON",
    )
}

pub(crate) fn fixture_nested_anonymous_aggregates() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "STRUCT IDENTIFIER LBRACE STRUCT LBRACE INT IDENTIFIER SEMICOLON RBRACE \
         IDENTIFIER SEMICOLON UNION LBRACE FLOAT_KW IDENTIFIER SEMICOLON INT IDENTIFIER \
         SEMICOLON RBRACE IDENTIFIER SEMICOLON ENUM LBRACE IDENTIFIER ASSIGN INTEGER \
         COMMA IDENTIFIER RBRACE IDENTIFIER SEMICOLON INT LPAREN STAR IDENTIFIER LBRACKET \
         INTEGER RBRACKET RPAREN LPAREN INT RPAREN SEMICOLON RBRACE SEMICOLON",
    )
}

pub(crate) fn fixture_function_pointer_array() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "STATIC INT LPAREN STAR CONST IDENTIFIER LBRACKET INTEGER RBRACKET RPAREN LPAREN \
         VOID STAR IDENTIFIER COMMA INT IDENTIFIER RPAREN SEMICOLON",
    )
}

pub(crate) fn fixture_nested_designated_init() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "STRUCT IDENTIFIER IDENTIFIER ASSIGN LBRACE DOT IDENTIFIER ASSIGN STRING COMMA \
         DOT IDENTIFIER ASSIGN LBRACE LBRACKET INTEGER RBRACKET ASSIGN INTEGER COMMA \
         LBRACKET INTEGER RBRACKET ASSIGN LBRACE DOT IDENTIFIER ASSIGN INTEGER COMMA DOT \
         IDENTIFIER ASSIGN INTEGER RBRACE RBRACE COMMA DOT IDENTIFIER LBRACKET INTEGER \
         RBRACKET ASSIGN INTEGER COMMA RBRACE SEMICOLON",
    )
}

pub(crate) fn fixture_attribute_and_asm() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "GNU_ATTRIBUTE LPAREN LPAREN IDENTIFIER RPAREN RPAREN VOID IDENTIFIER LPAREN VOID \
         RPAREN LBRACE GNU_ASM VOLATILE LPAREN STRING COLON COLON COLON STRING RPAREN \
         SEMICOLON RETURN SEMICOLON RBRACE",
    )
}

pub(crate) fn fixture_enum_values() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "ENUM IDENTIFIER LBRACE IDENTIFIER ASSIGN INTEGER COMMA IDENTIFIER ASSIGN INTEGER \
         COMMA IDENTIFIER COMMA IDENTIFIER ASSIGN INTEGER COMMA IDENTIFIER RBRACE \
         SEMICOLON",
    )
}

pub(crate) fn fixture_sizeof_type_vs_expr() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "VOID IDENTIFIER LPAREN VOID RPAREN LBRACE INT IDENTIFIER ASSIGN SIZEOF LPAREN \
         INT RPAREN SEMICOLON INT IDENTIFIER ASSIGN SIZEOF LPAREN IDENTIFIER RPAREN \
         SEMICOLON INT IDENTIFIER ASSIGN SIZEOF LPAREN IDENTIFIER PLUS INTEGER RPAREN \
         SEMICOLON RBRACE",
    )
}

pub(crate) fn fixture_stmt_expr_nesting() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "INT IDENTIFIER LPAREN INT IDENTIFIER RPAREN LBRACE RETURN LPAREN IDENTIFIER GT \
         INTEGER RPAREN QUESTION LPAREN LBRACE IF LPAREN IDENTIFIER GT INTEGER RPAREN \
         RETURN INTEGER SEMICOLON IDENTIFIER SEMICOLON RBRACE RPAREN COLON INTEGER \
         SEMICOLON RBRACE",
    )
}

pub(crate) struct CorpusCase {
    pub(crate) name: &'static str,
    pub(crate) fixture: fn() -> (Vec<u32>, Vec<u32>, Vec<u32>),
}

pub(crate) const CORPUS_CASES: &[CorpusCase] = &[
    CorpusCase {
        name: "macro_shaped_decl_after_preproc",
        fixture: fixture_macro_shaped_decl_after_preproc,
    },
    CorpusCase {
        name: "nested_anonymous_aggregates",
        fixture: fixture_nested_anonymous_aggregates,
    },
    CorpusCase {
        name: "function_pointer_array",
        fixture: fixture_function_pointer_array,
    },
    CorpusCase {
        name: "nested_designated_init",
        fixture: fixture_nested_designated_init,
    },
    CorpusCase {
        name: "attribute_and_asm",
        fixture: fixture_attribute_and_asm,
    },
    CorpusCase {
        name: "enum_values",
        fixture: fixture_enum_values,
    },
    CorpusCase {
        name: "sizeof_type_vs_expr",
        fixture: fixture_sizeof_type_vs_expr,
    },
    CorpusCase {
        name: "stmt_expr_nesting",
        fixture: fixture_stmt_expr_nesting,
    },
];
