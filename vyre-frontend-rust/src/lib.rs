//! GPU-first Rust compiler frontend for Vyre.
//!
//! This crate is a thin-on-semantics driver: the lexer, parser, semantic
//! analysis, and lowering to Vyre IR all live in `vyre-libs::parsing::rust`.
//! The crate orchestrates those stages and owns the application-level surface
//! (object emission, GPU dispatch, public API), mirroring `vyre-frontend-c`.
//!
//! Architecture:
//! - `api/`      - public entry points (`parse_rust_bytes`)
//! - `pipeline/` - stage orchestration (lex -> parse -> resolve -> typeck -> borrow -> lower)
//! - `object/`   - evidence object emission
//!
//! The differential oracle against `rustc_lexer` lives under `tests/`;
//! `rustc_lexer` is a dev-dependency, not a runtime dependency.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use thiserror::Error;

pub mod api;
pub mod object;
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
    /// Oracle mismatch: frontend output diverged from rustc.
    #[error("Rust frontend oracle mismatch: {0}. Fix: compare token spans against rustc_lexer output.")]
    Oracle(String),
}

// Re-export AST types so consumers can inspect parsed results.
/// Re-export AST types.
pub use vyre_libs::parsing::rust::parse::{Expr, Function, Module, Stmt, Type};
/// Re-export token type.
pub use vyre_libs::parsing::rust::lex::lexer::core::Token;
