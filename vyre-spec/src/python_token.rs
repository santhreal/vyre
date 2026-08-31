//! Python lexer token ids: the single owner of the numbering.
//!
//! The ids are the wire contract between the GPU lexer program that writes a
//! sparse token row per byte and every host matcher that reads those rows, so
//! they live in the foundation-layer contract crate rather than beside one
//! consumer. Every reader names this module.

/// Sparse-token sentinel: non-token byte positions stay zeroed.
pub const TOK_NONE: u32 = 0;
/// Identifier token.
pub const TOK_IDENTIFIER: u32 = 1;
/// Number literal token.
pub const TOK_NUMBER: u32 = 2;
/// String literal token.
pub const TOK_STRING: u32 = 3;
/// Newline token.
pub const TOK_NEWLINE: u32 = 4;
/// Comment token.
pub const TOK_COMMENT: u32 = 5;

/// `(` token.
pub const TOK_LPAREN: u32 = 10;
/// `)` token.
pub const TOK_RPAREN: u32 = 11;
/// `[` token.
pub const TOK_LBRACKET: u32 = 12;
/// `]` token.
pub const TOK_RBRACKET: u32 = 13;
/// `{` token.
pub const TOK_LBRACE: u32 = 14;
/// `}` token.
pub const TOK_RBRACE: u32 = 15;
/// `:` token.
pub const TOK_COLON: u32 = 16;
/// `,` token.
pub const TOK_COMMA: u32 = 17;
/// `.` token.
pub const TOK_DOT: u32 = 18;
/// `=` token.
pub const TOK_EQ: u32 = 19;
/// `@` token.
pub const TOK_AT: u32 = 20;
/// `*` token.
pub const TOK_STAR: u32 = 21;

/// `def` keyword token.
pub const TOK_DEF: u32 = 100;
/// `async` keyword token.
pub const TOK_ASYNC: u32 = 101;
/// `class` keyword token.
pub const TOK_CLASS: u32 = 102;
/// `import` keyword token.
pub const TOK_IMPORT: u32 = 103;
/// `from` keyword token.
pub const TOK_FROM: u32 = 104;
/// `as` keyword token.
pub const TOK_AS: u32 = 105;
/// `with` keyword token.
pub const TOK_WITH: u32 = 106;
/// `await` keyword token.
pub const TOK_AWAIT: u32 = 107;
/// `match` keyword token.
pub const TOK_MATCH: u32 = 108;
/// `case` keyword token.
pub const TOK_CASE: u32 = 109;
/// `except` keyword token.
pub const TOK_EXCEPT: u32 = 110;
