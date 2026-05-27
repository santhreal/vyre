//! Rust pipeline modules — lex / parse / typeck / borrow.

/// DFA lexer pipeline (lexer, tokens, keywords).
pub mod lex;
/// Structural parser for the supported nano-subset.
pub mod parse;
