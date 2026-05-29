//! Rust pipeline modules - lex / parse / AST.
//!
//! This is the reusable substrate.  All compiler-specific concerns
//! (name resolution, type inference, borrow checking, lowering) live
//! in `vyre-frontend-rust`.

/// DFA lexer pipeline (tokens, keywords, lexer kernels).
pub mod lex;
/// Nano-subset structural parser.
pub mod parse;
