//! Token fixtures for end-to-end expression-shape and property-graph lowering: assignment chains,
//! compound literals, and labelled switches.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use vyre_libs::parsing::c::lex::tokens::*;

pub(crate) fn expression_chain_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_IDENTIFIER,
        TOK_COMMA,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_IDENTIFIER,
        TOK_COMMA,
        TOK_IDENTIFIER,
        TOK_QUESTION,
        TOK_IDENTIFIER,
        TOK_COLON,
        TOK_IDENTIFIER,
        TOK_COMMA,
        TOK_IDENTIFIER,
        TOK_LBRACKET,
        TOK_IDENTIFIER,
        TOK_RBRACKET,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_MINUS,
        TOK_IDENTIFIER,
        TOK_PLUS,
        TOK_IDENTIFIER,
        TOK_STAR,
        TOK_STAR,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn compound_literal_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_LPAREN,
        TOK_STRUCT,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_LBRACE,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_IDENTIFIER,
        TOK_LBRACKET,
        TOK_IDENTIFIER,
        TOK_RBRACKET,
        TOK_COMMA,
        TOK_DOT,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_IDENTIFIER,
        TOK_QUESTION,
        TOK_IDENTIFIER,
        TOK_COLON,
        TOK_IDENTIFIER,
        TOK_RBRACE,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn label_switch_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_IDENTIFIER,
        TOK_COLON,
        TOK_SWITCH,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_LBRACE,
        TOK_CASE,
        TOK_INTEGER,
        TOK_COLON,
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_DEFAULT,
        TOK_COLON,
        TOK_GOTO,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_RBRACE,
    ];
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}
