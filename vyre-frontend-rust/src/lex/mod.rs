//! Rust source lexer and backend-neutral parallel lexer IR.
//!
//! The CPU lexer is validated token-for-token against `rustc_lexer`; the plan
//! module emits equivalent Vyre IR for upper execution harnesses.

/// Post-lex keyword promotion.
pub mod keyword;
/// Lexer implementations (CPU source lexer + parallel IR plan builder).
pub mod lexer;
/// Token constants and predicates.
pub mod tokens;
