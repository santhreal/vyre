//! Go lexer token ids: the single owner of the numbering.
//!
//! The ids are the wire contract between the GPU lexer program that writes a
//! sparse token row per byte and every host matcher that reads those rows, so
//! they live in the foundation-layer contract crate rather than beside one
//! consumer. Every reader names this module.

/// Sparse-token sentinel: non-token byte positions stay zeroed.
pub const TOK_NONE: u32 = 0;
/// Identifier token.
pub const TOK_IDENTIFIER: u32 = 1;
/// Double-quoted string literal token.
pub const TOK_STRING: u32 = 2;
/// `(` token.
pub const TOK_LPAREN: u32 = 10;
/// `)` token.
pub const TOK_RPAREN: u32 = 11;
/// `{` token.
pub const TOK_LBRACE: u32 = 12;
/// `}` token.
pub const TOK_RBRACE: u32 = 13;
/// `[` token.
pub const TOK_LBRACKET: u32 = 14;
/// `]` token.
pub const TOK_RBRACKET: u32 = 15;
/// `,` token.
pub const TOK_COMMA: u32 = 16;
/// `.` token.
pub const TOK_DOT: u32 = 17;
/// `;` token.
pub const TOK_SEMICOLON: u32 = 18;
/// `:` token.
pub const TOK_COLON: u32 = 19;
/// `=` token.
pub const TOK_ASSIGN: u32 = 20;
/// `*` token.
pub const TOK_STAR: u32 = 21;
/// `<-` token.
pub const TOK_ARROW: u32 = 22;
/// Statement terminator: a line break, which Go treats as an implicit semicolon.
///
/// Emitting this is not cosmetic. Go separates statements by newline, and the
/// lexer emits no token for a numeric literal, so without a terminator two
/// consecutive receive statements
///
/// ```go
/// <-input
/// <-output
/// ```
///
/// tokenize as `ARROW IDENT ARROW IDENT`, which is indistinguishable from the
/// single send `input <- output`. The channel matchers then read the second
/// receive as a send. That is exactly how the fixture corpus came to report 35
/// sends and 14 receives where Go has 25 and 24: ten receives had been
/// swallowed by the statement before them.
pub const TOK_NEWLINE: u32 = 23;
