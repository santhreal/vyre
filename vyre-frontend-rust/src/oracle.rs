//! Differential testing oracle: validate our output against `rustc_lexer`.
//!
//! Every token emitted by our lexer must match the token produced by
//! `rustc_lexer` for the same source span (modulo unsupported tokens
//! which are mapped to `RUST_TOK_UNSUPPORTED`).

use rustc_lexer::TokenKind;
use vyre_libs::parsing::rust::lex::lexer::core::{Token, lex};
use vyre_libs::parsing::rust::lex::tokens::*;

/// Compare our token stream against `rustc_lexer`.
///
/// Returns `Ok(())` if every token matches.  Returns `Err(diff)` on
/// the first divergence with a human-readable description.
pub fn validate_against_rustc_lexer(source: &[u8]) -> Result<(), String> {
    let our_tokens = lex(source).map_err(|off| format!("our lexer failed at {}", off))?;
    let rustc_tokens = rustc_lexer::tokenize(std::str::from_utf8(source).map_err(|_| "invalid utf8")?);

    let mut rustc_iter = rustc_tokens.iter().peekable();
    let mut our_iter = our_tokens.iter().filter(|t| t.kind != RUST_TOK_EOF).peekable();

    while let (Some(rt), Some(ot)) = (rustc_iter.peek(), our_iter.peek()) {
        let expected = map_rustc_kind(rt.kind);
        if expected == RUST_TOK_UNSUPPORTED {
            // Unsupported token: skip both sides for now
            rustc_iter.next();
            our_iter.next();
            continue;
        }
        if ot.kind != expected {
            return Err(format!(
                "divergence at byte {}: rustc={:?} ({}), ours={} ({})",
                rt.len, rt.kind, expected, ot.kind, ot.start
            ));
        }
        if rt.len as u16 != ot.len {
            return Err(format!(
                "span mismatch at byte {}: rustc len={}, ours len={}",
                ot.start, rt.len, ot.len
            ));
        }
        rustc_iter.next();
        our_iter.next();
    }

    // Check remaining tokens
    if rustc_iter.peek().is_some() {
        return Err("rustc produced more tokens than our lexer".to_string());
    }
    if our_iter.peek().is_some() {
        return Err("our lexer produced more tokens than rustc".to_string());
    }

    Ok(())
}

/// Map a `rustc_lexer::TokenKind` to our token id.
///
/// Returns `RUST_TOK_UNSUPPORTED` for tokens outside the nano-subset.
fn map_rustc_kind(kind: TokenKind) -> u16 {
    use rustc_lexer::TokenKind::*;
    match kind {
        LineComment { .. } | BlockComment { .. } => RUST_TOK_EOF, // we skip comments
        Whitespace => RUST_TOK_EOF, // we skip whitespace
        Ident => RUST_TOK_IDENT,
        RawIdent => RUST_TOK_IDENT,
        Literal { kind: rustc_lexer::LiteralKind::Int { .. }, .. } => RUST_TOK_LITERAL_INT,
        Literal { kind: rustc_lexer::LiteralKind::Float { .. }, .. } => RUST_TOK_UNSUPPORTED,
        Literal { kind: rustc_lexer::LiteralKind::Char { .. }, .. } => RUST_TOK_UNSUPPORTED,
        Literal { kind: rustc_lexer::LiteralKind::Byte { .. }, .. } => RUST_TOK_UNSUPPORTED,
        Literal { kind: rustc_lexer::LiteralKind::Str { .. }, .. } => RUST_TOK_UNSUPPORTED,
        Literal { kind: rustc_lexer::LiteralKind::ByteStr { .. }, .. } => RUST_TOK_UNSUPPORTED,
        Literal { kind: rustc_lexer::LiteralKind::RawStr { .. }, .. } => RUST_TOK_UNSUPPORTED,
        Literal { kind: rustc_lexer::LiteralKind::RawByteStr { .. }, .. } => RUST_TOK_UNSUPPORTED,
        Lifetime { .. } => RUST_TOK_UNSUPPORTED,
        Semi => RUST_TOK_SEMI,
        Comma => RUST_TOK_COMMA,
        Dot => RUST_TOK_UNSUPPORTED,
        OpenParen => RUST_TOK_LPAREN,
        CloseParen => RUST_TOK_RPAREN,
        OpenBrace => RUST_TOK_LBRACE,
        CloseBrace => RUST_TOK_RBRACE,
        OpenBracket => RUST_TOK_UNSUPPORTED,
        CloseBracket => RUST_TOK_UNSUPPORTED,
        At => RUST_TOK_UNSUPPORTED,
        Pound => RUST_TOK_UNSUPPORTED,
        Tilde => RUST_TOK_UNSUPPORTED,
        Question => RUST_TOK_UNSUPPORTED,
        Colon => RUST_TOK_COLON,
        Dollar => RUST_TOK_UNSUPPORTED,
        Eq => RUST_TOK_ASSIGN,
        Bang => RUST_TOK_BANG,
        Lt => RUST_TOK_LT,
        Gt => RUST_TOK_UNSUPPORTED,
        Minus => RUST_TOK_MINUS,
        And => RUST_TOK_AMP,
        Or => RUST_TOK_UNSUPPORTED,
        Plus => RUST_TOK_PLUS,
        Star => RUST_TOK_STAR,
        Slash => RUST_TOK_SLASH,
        Caret => RUST_TOK_UNSUPPORTED,
        Percent => RUST_TOK_UNSUPPORTED,
        Unknown => RUST_TOK_ERROR,
        Eof => RUST_TOK_EOF,
    }
}
