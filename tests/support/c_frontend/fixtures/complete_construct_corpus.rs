//! Token fixtures and the case table for the complete C11 construct corpus.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use crate::c_frontend::rows::starts_for_lens;
use vyre_libs::parsing::c::lex::tokens::*;

pub(crate) fn fixture_macro_shaped_decl_after_preproc() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_PREPROC,
        TOK_GNU_ATTRIBUTE,
        TOK_LPAREN,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_STRING,
        TOK_RPAREN,
        TOK_RPAREN,
        TOK_RPAREN,
        TOK_INT,
        TOK_STAR,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![7, 13, 1, 1, 10, 1, 10, 1, 1, 1, 3, 1, 5, 1, 3, 2, 1, 1];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

pub(crate) fn fixture_nested_anonymous_aggregates() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_STRUCT,
        TOK_IDENTIFIER,
        TOK_LBRACE,
        TOK_STRUCT,
        TOK_LBRACE,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_RBRACE,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_UNION,
        TOK_LBRACE,
        TOK_FLOAT_KW,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_RBRACE,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_ENUM,
        TOK_LBRACE,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_IDENTIFIER,
        TOK_RBRACE,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_INT,
        TOK_LPAREN,
        TOK_STAR,
        TOK_IDENTIFIER,
        TOK_LBRACKET,
        TOK_INTEGER,
        TOK_RBRACKET,
        TOK_RPAREN,
        TOK_LPAREN,
        TOK_INT,
        TOK_RPAREN,
        TOK_SEMICOLON,
        TOK_RBRACE,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

pub(crate) fn fixture_function_pointer_array() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_STATIC,
        TOK_INT,
        TOK_LPAREN,
        TOK_STAR,
        TOK_CONST,
        TOK_IDENTIFIER,
        TOK_LBRACKET,
        TOK_INTEGER,
        TOK_RBRACKET,
        TOK_RPAREN,
        TOK_LPAREN,
        TOK_VOID,
        TOK_STAR,
        TOK_IDENTIFIER,
        TOK_COMMA,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

pub(crate) fn fixture_nested_designated_init() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_STRUCT,
        TOK_IDENTIFIER,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_LBRACE,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_STRING,
        TOK_COMMA,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_LBRACE,
        TOK_LBRACKET,
        TOK_INTEGER,
        TOK_RBRACKET,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_LBRACKET,
        TOK_INTEGER,
        TOK_RBRACKET,
        TOK_ASSIGN,
        TOK_LBRACE,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_RBRACE,
        TOK_RBRACE,
        TOK_COMMA,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_LBRACKET,
        TOK_INTEGER,
        TOK_RBRACKET,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_RBRACE,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

pub(crate) fn fixture_attribute_and_asm() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_GNU_ATTRIBUTE,
        TOK_LPAREN,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_RPAREN,
        TOK_VOID,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_VOID,
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
        TOK_SEMICOLON,
        TOK_RBRACE,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

pub(crate) fn fixture_enum_values() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_ENUM,
        TOK_IDENTIFIER,
        TOK_LBRACE,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_IDENTIFIER,
        TOK_COMMA,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_INTEGER,
        TOK_COMMA,
        TOK_IDENTIFIER,
        TOK_RBRACE,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

pub(crate) fn fixture_sizeof_type_vs_expr() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_VOID,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_VOID,
        TOK_RPAREN,
        TOK_LBRACE,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_SIZEOF,
        TOK_LPAREN,
        TOK_INT,
        TOK_RPAREN,
        TOK_SEMICOLON,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_SIZEOF,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_SEMICOLON,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_SIZEOF,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_PLUS,
        TOK_INTEGER,
        TOK_RPAREN,
        TOK_SEMICOLON,
        TOK_RBRACE,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

pub(crate) fn fixture_stmt_expr_nesting() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_LBRACE,
        TOK_RETURN,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_GT,
        TOK_INTEGER,
        TOK_RPAREN,
        TOK_QUESTION,
        TOK_LPAREN,
        TOK_LBRACE,
        TOK_IF,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_GT,
        TOK_INTEGER,
        TOK_RPAREN,
        TOK_RETURN,
        TOK_INTEGER,
        TOK_SEMICOLON,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_RBRACE,
        TOK_RPAREN,
        TOK_COLON,
        TOK_INTEGER,
        TOK_SEMICOLON,
        TOK_RBRACE,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
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
