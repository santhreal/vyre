//! CPU reference lexer for the Rust nano-subset.
//!
//! This is the oracle.  Every GPU lex result is compared token-for-token
//! against this function's output.  The implementation is deliberately
//! simple (not maximally optimised) so that correctness is obvious.

use crate::parsing::rust::lex::tokens::*;

/// A single token with its source span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
    pub kind: u16,
    pub start: u32,
    pub len: u16,
}

/// Lex a byte slice into a vector of tokens.
///
/// Returns `Err(offset)` on the first unrecognised byte.
pub fn lex(source: &[u8]) -> Result<Vec<Token>, usize> {
    let mut tokens = Vec::new();
    let mut i = 0usize;

    while i < source.len() {
        let b = source[i];

        // Whitespace
        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        // Line comment
        if b == b'/' && i + 1 < source.len() && source[i + 1] == b'/' {
            while i < source.len() && source[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        // Block comment (nano-subset: no nested block comments)
        if b == b'/' && i + 1 < source.len() && source[i + 1] == b'*' {
            i += 2;
            while i + 1 < source.len() {
                if source[i] == b'*' && source[i + 1] == b'/' {
                    i += 2;
                    break;
                }
                i += 1;
            }
            continue;
        }

        let start = i;

        // Identifiers and keywords
        if b == b'_' || b.is_ascii_alphabetic() {
            while i < source.len()
                && (source[i].is_ascii_alphanumeric() || source[i] == b'_')
            {
                i += 1;
            }
            let text = std::str::from_utf8(&source[start..i]).unwrap_or("");
            let kind = match text {
                "fn" => RUST_TOK_FN,
                "let" => RUST_TOK_LET,
                "mut" => RUST_TOK_MUT,
                "if" => RUST_TOK_IF,
                "else" => RUST_TOK_ELSE,
                "return" => RUST_TOK_RETURN,
                "while" => RUST_TOK_WHILE,
                "true" | "false" => RUST_TOK_LITERAL_BOOL,
                "i32" => RUST_TOK_I32,
                "bool" => RUST_TOK_BOOL,
                _ => RUST_TOK_IDENT,
            };
            tokens.push(Token {
                kind,
                start: start as u32,
                len: (i - start) as u16,
            });
            continue;
        }

        // Integer literals (decimal only in nano-subset)
        if b.is_ascii_digit() {
            while i < source.len() && source[i].is_ascii_digit() {
                i += 1;
            }
            tokens.push(Token {
                kind: RUST_TOK_LITERAL_INT,
                start: start as u32,
                len: (i - start) as u16,
            });
            continue;
        }

        // Two-character punctuators
        if i + 1 < source.len() {
            let pair = [b, source[i + 1]];
            let kind = match &pair {
                b"==" => RUST_TOK_EQ,
                b"->" => RUST_TOK_ARROW,
                b"&mut" => {
                    // Special case: `&mut` is three bytes
                    if i + 3 <= source.len() && &source[i + 1..i + 4] == b"mut" {
                        i += 4;
                        tokens.push(Token {
                            kind: RUST_TOK_AMP_MUT,
                            start: start as u32,
                            len: 4,
                        });
                        continue;
                    }
                    RUST_TOK_AMP
                }
                _ => 0xFFFF, // sentinel: not a recognised pair
            };
            if kind != 0xFFFF {
                i += 2;
                tokens.push(Token {
                    kind,
                    start: start as u32,
                    len: 2,
                });
                continue;
            }
        }

        // Single-character tokens
        let kind = match b {
            b'+' => RUST_TOK_PLUS,
            b'-' => RUST_TOK_MINUS,
            b'*' => RUST_TOK_STAR,
            b'/' => RUST_TOK_SLASH,
            b'=' => RUST_TOK_ASSIGN,
            b'<' => RUST_TOK_LT,
            b';' => RUST_TOK_SEMI,
            b':' => RUST_TOK_COLON,
            b',' => RUST_TOK_COMMA,
            b'&' => RUST_TOK_AMP,
            b'!' => RUST_TOK_BANG,
            b'(' => RUST_TOK_LPAREN,
            b')' => RUST_TOK_RPAREN,
            b'{' => RUST_TOK_LBRACE,
            b'}' => RUST_TOK_RBRACE,
            _ => return Err(start),
        };
        i += 1;
        tokens.push(Token {
            kind,
            start: start as u32,
            len: 1,
        });
    }

    tokens.push(Token {
        kind: RUST_TOK_EOF,
        start: source.len() as u32,
        len: 0,
    });

    Ok(tokens)
}
