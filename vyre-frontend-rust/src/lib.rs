//! GPU-first Rust compiler frontend for Vyre.
//!
//! This crate is the thick driver.  All reusable parsing substrate lives
//! in `vyre-libs::parsing::rust`.
//!
//! Architecture:
//! - `oracle/`  — differential testing against rustc
//! - `pipeline/` — stage dispatch (lex → parse → resolve → typeck → borrow → lower)
//! - `object/`  — evidence object emission

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use thiserror::Error;

pub mod api;
pub mod object;
pub mod oracle;
pub mod pipeline;

/// Unified error type for the Rust frontend.
#[derive(Debug, Clone, Error)]
pub enum RustFrontendError {
    /// Lexing failed at the given byte offset.
    #[error("lex error at byte {0}")]
    Lex(usize),
    /// Parsing failed.
    #[error("parse error at token {token_index}: {message}")]
    Parse {
        /// Error message.
        message: String,
        /// Token index.
        token_index: usize,
    },
    /// The source contains constructs outside the nano-subset.
    #[error("unsupported construct: {0}")]
    Unsupported(String),
    /// GPU backend unavailable.
    #[error("GPU backend unavailable: {0}")]
    Backend(String),
    /// Oracle mismatch: our output diverged from rustc.
    #[error("oracle mismatch: {0}")]
    Oracle(String),
}

// Re-export AST types so consumers can inspect parsed results.
/// Re-export AST types.
pub use vyre_libs::parsing::rust::parse::{Expr, Function, Module, Stmt, Type};
/// Re-export token type.
pub use vyre_libs::parsing::rust::lex::lexer::core::Token;
