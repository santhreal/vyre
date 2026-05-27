//! GPU-first Rust compiler frontend for Vyre.
//!
//! This crate is the thick driver.  All reusable parsing substrate lives
//! in `vyre-libs::parsing::rust`.
//!
//! Architecture:
//! - `api/`     — public entry points (`parse_rust_bytes`)
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
    /// The source contains constructs outside the nano-subset.
    #[error("Rust frontend unsupported construct: {0}. Fix: simplify the source to the nano-subset.")]
    Unsupported(String),
    /// GPU backend unavailable.
    #[error("Rust frontend GPU backend unavailable: {0}. Fix: ensure a CUDA or WGPU backend is installed and detected.")]
    Backend(String),
    /// Oracle mismatch: our output diverged from rustc.
    #[error("Rust frontend oracle mismatch: {0}. Fix: compare token spans and kinds against rustc_lexer output.")]
    Oracle(String),
}

// Re-export AST types so consumers can inspect parsed results.
/// Re-export AST types.
pub use vyre_libs::parsing::rust::parse::{Expr, Function, Module, Stmt, Type};
/// Re-export token type.
pub use vyre_libs::parsing::rust::lex::lexer::core::Token;
