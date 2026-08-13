//! Rust source-to-typed-IR frontend for Vyre.
//!
//! This crate owns Rust source ingestion, syntax and semantic analysis, and
//! lowering to backend-neutral [`vyre_foundation::ir::Program`] values. It does
//! not select, compile for, or dispatch an execution backend.
//!
//! Architecture:
//! - `lex/`      - source tokenization and backend-neutral lexer IR plans
//! - `parse/`    - nano-subset AST construction
//! - `sema/`     - name, type, and borrow analysis
//! - `lower/`    - typed AST to backend-neutral Vyre IR
//! - `pipeline/` - source-to-IR stage orchestration
//!
//! The differential oracle against `rustc_lexer` lives under `tests/`;
//! `rustc_lexer` is a dev-dependency, not a normal dependency.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use thiserror::Error;

pub mod api;
pub mod borrowck;
pub mod lex;
pub mod lower;
pub mod parse;
pub mod pipeline;
pub mod sema;

/// Unified error type for the Rust frontend, one variant per pipeline stage.
///
/// Error messages follow the `vyre-frontend-c` convention:
/// `"description. Fix: suggestion."`
#[derive(Debug, Clone, Error)]
pub enum RustFrontendError {
    /// Lexing failed at the given byte offset.
    #[error("Rust frontend lex failed at byte {0}. Fix: check for invalid UTF-8 or unsupported characters in the source.")]
    Lex(usize),
    /// Parsing failed.
    #[error("Rust frontend parse failed at token {token_index}: {message}. Fix: ensure the source uses only the supported nano-subset (fn, let, if/else, return, i32, bool, references).")]
    Parse {
        /// Error message.
        message: String,
        /// Token index.
        token_index: usize,
    },
    /// Name resolution failed (e.g. use of an undefined name; rustc E0425).
    #[error("Rust frontend name resolution failed: {0}. Fix: declare the name before use or correct the identifier.")]
    Resolve(String),
    /// Type checking failed (rustc E0308 / E0061 / E0614).
    #[error("Rust frontend type check failed: {0}. Fix: correct the types so they match.")]
    Typeck(String),
    /// Borrow checking failed, or is incomplete for this program.
    #[error("Rust frontend borrow check failed: {0}. Fix: borrow the place mutably only when it is declared mutable.")]
    Borrow(String),
    /// Lowering to Vyre IR failed.
    #[error("Rust frontend lowering failed: {0}. Fix: see the lowering substrate status.")]
    Lower(String),
    /// The source contains constructs outside the nano-subset.
    #[error(
        "Rust frontend unsupported construct: {0}. Fix: simplify the source to the nano-subset."
    )]
    Unsupported(String),
}

// Re-export AST types so consumers can inspect parsed results.
/// Re-export token type.
pub use crate::lex::lexer::cpu_lexer::Token;
/// Re-export AST types.
pub use crate::parse::{Expr, Function, Module, Stmt, Type};
