//! GPU DFA lexer pipeline for Rust source text.
//!
//! Reuses `vyre-primitives::text` byte-classifiers and the sparse-dispatch
//! pattern proven in the C11 lexer. Rust is simpler than C at the lexing
//! layer: no preprocessor, no digraphs, no trigraphs, no line-continuation,
//! but adds raw strings, nested block comments, and format-string literals.

/// Token-id constants (`RUST_TOK_*`) shared by every Rust-parser stage.
pub mod tokens;
/// Post-lex keyword promotion (identifier → keyword token id).
pub mod keyword;
/// Maximally-munching DFA-driven lexer kernel.
pub mod lexer;
