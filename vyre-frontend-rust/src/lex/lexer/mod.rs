//! Rust source lexer and backend-neutral lexer IR plan.

/// CPU reference lexer (hand-written, validated against `rustc_lexer`).
pub mod core;

/// Parallel sparse-dispatch lexer IR plan builder.
pub mod plan;
