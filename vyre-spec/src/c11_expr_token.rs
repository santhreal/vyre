//! Terminal ids for the LR(1) arithmetic expression grammar.
//!
//! The ids are the column order of the packed action table that
//! `vyre_libs::parsing::lr_tables::c11_expr` ships, and the token ids a caller
//! feeds the GPU parser, so they are a wire contract and are owned here.
//! Every reader names this module.

/// Identifier token id.
pub const TOK_ID: u32 = 0;
/// Numeric literal token id.
pub const TOK_NUM: u32 = 1;
/// `+` token id.
pub const TOK_PLUS: u32 = 2;
/// `-` token id.
pub const TOK_MINUS: u32 = 3;
/// `*` token id.
pub const TOK_STAR: u32 = 4;
/// `/` token id.
pub const TOK_SLASH: u32 = 5;
/// `(` token id.
pub const TOK_LPAREN: u32 = 6;
/// `)` token id.
pub const TOK_RPAREN: u32 = 7;
/// End-of-file token id.
pub const TOK_EOF: u32 = 8;
