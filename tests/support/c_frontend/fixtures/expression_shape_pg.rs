//! Token fixtures for end-to-end expression-shape and property-graph lowering: assignment chains,
//! compound literals, and labelled switches.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use crate::c_frontend::spelling::c_kinds;
pub(crate) fn expression_chain_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = c_kinds(
        "IDENTIFIER ASSIGN IDENTIFIER ASSIGN IDENTIFIER COMMA IDENTIFIER ASSIGN \
         IDENTIFIER COMMA IDENTIFIER QUESTION IDENTIFIER COLON IDENTIFIER COMMA \
         IDENTIFIER LBRACKET IDENTIFIER RBRACKET DOT IDENTIFIER ASSIGN MINUS IDENTIFIER \
         PLUS IDENTIFIER STAR STAR IDENTIFIER SEMICOLON",
    );
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn compound_literal_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = c_kinds(
        "IDENTIFIER ASSIGN LPAREN STRUCT IDENTIFIER RPAREN LBRACE DOT IDENTIFIER ASSIGN \
         IDENTIFIER LBRACKET IDENTIFIER RBRACKET COMMA DOT IDENTIFIER ASSIGN IDENTIFIER \
         QUESTION IDENTIFIER COLON IDENTIFIER RBRACE SEMICOLON",
    );
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn label_switch_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = c_kinds(
        "IDENTIFIER COLON SWITCH LPAREN IDENTIFIER RPAREN LBRACE CASE INTEGER COLON \
         IDENTIFIER ASSIGN IDENTIFIER SEMICOLON DEFAULT COLON GOTO IDENTIFIER SEMICOLON \
         RBRACE",
    );
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}
