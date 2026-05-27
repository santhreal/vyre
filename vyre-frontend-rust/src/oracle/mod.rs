//! Differential testing oracle framework.
//!
//! The oracle compares the reusable Rust lexer substrate against
//! `rustc_lexer` so frontend changes can be checked against the upstream token
//! contract before parser or lowering work depends on them.

use rustc_lexer::TokenKind;
use vyre_libs::parsing::rust::lex::lexer::core::lex;
use vyre_libs::parsing::rust::lex::tokens as tok;

/// Result of an oracle comparison.
#[derive(Debug, Clone, PartialEq)]
pub enum OracleResult {
    /// Outputs match within the supported subset.
    Match,
    /// Divergence detected with details.
    Mismatch(String),
    /// The construct is unsupported and was skipped.
    SkippedUnsupported,
}

/// Compare our token stream against `rustc_lexer`.
pub fn lexer_parity(source: &[u8]) -> OracleResult {
    let our_tokens = match lex(source) {
        Ok(t) => t,
        Err(off) => return OracleResult::Mismatch(format!("our lexer failed at {}", off)),
    };

    let source_str = match std::str::from_utf8(source) {
        Ok(s) => s,
        Err(_) => return OracleResult::Mismatch("invalid utf8".into()),
    };

    let rustc_tokens: Vec<_> = rustc_lexer::tokenize(source_str).collect();
    let mut r_iter = rustc_tokens.iter().peekable();
    let mut o_iter = our_tokens.iter().filter(|t| t.kind != tok::EOF).peekable();

    while let (Some(rt), Some(ot)) = (r_iter.peek(), o_iter.peek()) {
        let expected = map_rustc_kind(rt.kind);
        if expected == tok::UNSUPPORTED {
            r_iter.next();
            o_iter.next();
            continue;
        }
        if ot.kind != expected {
            return OracleResult::Mismatch(format!(
                "divergence at byte {}: rustc={:?} (mapped={}), ours={} ({})",
                ot.start, rt.kind, expected, ot.kind, ot.start
            ));
        }
        if rt.len as u16 != ot.len {
            return OracleResult::Mismatch(format!(
                "span mismatch at byte {}: rustc len={}, ours len={}",
                ot.start, rt.len, ot.len
            ));
        }
        r_iter.next();
        o_iter.next();
    }

    if r_iter.peek().is_some() {
        return OracleResult::Mismatch("rustc produced more tokens".into());
    }
    if o_iter.peek().is_some() {
        return OracleResult::Mismatch("our lexer produced more tokens".into());
    }

    OracleResult::Match
}

#[allow(non_snake_case)]
fn map_rustc_kind(kind: TokenKind) -> u16 {
    use rustc_lexer::TokenKind::*;
    match kind {
        LineComment | BlockComment { .. } | Whitespace => tok::EOF,
        Ident | RawIdent => tok::IDENT,
        Literal { kind: rustc_lexer::LiteralKind::Int { .. }, .. } => tok::LITERAL_INT,
        Literal { .. } | Lifetime { .. } => tok::UNSUPPORTED,
        Semi => tok::SEMI,
        Comma => tok::COMMA,
        Dot => tok::UNSUPPORTED,
        OpenParen => tok::LPAREN,
        CloseParen => tok::RPAREN,
        OpenBrace => tok::LBRACE,
        CloseBrace => tok::RBRACE,
        OpenBracket | CloseBracket | At | Pound | Tilde | Question | Dollar => tok::UNSUPPORTED,
        Colon => tok::COLON,
        Eq => tok::ASSIGN,
        Not => tok::BANG,
        Lt => tok::LT,
        Gt => tok::UNSUPPORTED,
        Minus => tok::MINUS,
        And => tok::AMP,
        Or => tok::UNSUPPORTED,
        Plus => tok::PLUS,
        Star => tok::STAR,
        Slash => tok::SLASH,
        Caret | Percent => tok::UNSUPPORTED,
        Unknown => tok::ERROR,
    }
}
