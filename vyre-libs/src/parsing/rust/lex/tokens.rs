//! Token constants for the Rust nano-subset lexer.
//!
//! These mirror the token kinds produced by `rustc_lexer` so that the oracle
//! comparison is one-to-one.  Gaps in numbering are reserved for future
//! tokens; the nano-subset only needs a small surface.

/// End of file.
pub const RUST_TOK_EOF: u16 = 0;

// ─── Literals ──────────────────────────────────────────────────────────────
/// Integer literal (decimal, hex, octal, binary).
pub const RUST_TOK_LITERAL_INT: u16 = 1;
/// Boolean literal (`true` or `false`).
pub const RUST_TOK_LITERAL_BOOL: u16 = 2;

// ─── Identifiers ───────────────────────────────────────────────────────────
/// Identifier or raw identifier.
pub const RUST_TOK_IDENT: u16 = 10;

// ─── Keywords (promoted from IDENT by keyword.rs) ──────────────────────────
pub const RUST_TOK_FN: u16 = 20;
pub const RUST_TOK_LET: u16 = 21;
pub const RUST_TOK_MUT: u16 = 22;
pub const RUST_TOK_IF: u16 = 23;
pub const RUST_TOK_ELSE: u16 = 24;
pub const RUST_TOK_RETURN: u16 = 25;
pub const RUST_TOK_WHILE: u16 = 26;
pub const RUST_TOK_REF: u16 = 27; // `&`

// ─── Types (keywords in this nano-subset) ──────────────────────────────────
pub const RUST_TOK_I32: u16 = 30;
pub const RUST_TOK_BOOL: u16 = 31;
pub const RUST_TOK_UNIT: u16 = 32; // `()` treated as a keyword-like token

// ─── Punctuation ───────────────────────────────────────────────────────────
pub const RUST_TOK_PLUS: u16 = 40;
pub const RUST_TOK_MINUS: u16 = 41;
pub const RUST_TOK_STAR: u16 = 42;
pub const RUST_TOK_SLASH: u16 = 43;
pub const RUST_TOK_EQ: u16 = 44;      // `==`
pub const RUST_TOK_LT: u16 = 45;      // `<`
pub const RUST_TOK_ASSIGN: u16 = 46;  // `=`
pub const RUST_TOK_SEMI: u16 = 47;
pub const RUST_TOK_COLON: u16 = 48;
pub const RUST_TOK_COMMA: u16 = 49;
pub const RUST_TOK_ARROW: u16 = 50;   // `->`
pub const RUST_TOK_AMP: u16 = 51;     // `&` (borrow/ref)
pub const RUST_TOK_AMP_MUT: u16 = 52; // `&mut`
pub const RUST_TOK_BANG: u16 = 53;    // `!`

// ─── Delimiters ────────────────────────────────────────────────────────────
pub const RUST_TOK_LPAREN: u16 = 60;
pub const RUST_TOK_RPAREN: u16 = 61;
pub const RUST_TOK_LBRACE: u16 = 62;
pub const RUST_TOK_RBRACE: u16 = 63;

// ─── Error / sentinel ──────────────────────────────────────────────────────
/// Unrecognized byte sequence.
pub const RUST_TOK_ERROR: u16 = 0xFFFE;
/// Placeholder for tokens outside the nano-subset (so the lexer can
/// continue and the parser can emit a graceful diagnostic).
pub const RUST_TOK_UNSUPPORTED: u16 = 0xFFFF;

/// True if the token id represents a literal.
pub const fn is_literal(tok: u16) -> bool {
    matches!(tok, RUST_TOK_LITERAL_INT | RUST_TOK_LITERAL_BOOL)
}

/// True if the token id represents a binary operator.
pub const fn is_binop(tok: u16) -> bool {
    matches!(
        tok,
        RUST_TOK_PLUS | RUST_TOK_MINUS | RUST_TOK_STAR | RUST_TOK_SLASH | RUST_TOK_EQ | RUST_TOK_LT
    )
}
