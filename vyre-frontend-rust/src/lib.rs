//! GPU-first Rust parser and compile-evidence driver for Vyre.
//!
//! Status: experimental / v0.0.1 scaffold.  This crate is the thin pipeline
//! driver; all reusable GPU primitives live in `vyre-libs::parsing::rust`.
//!
//! The nano-subset supported in v0.0.1:
//! - Functions, `let` bindings, `return`
//! - Types: `i32`, `bool`, `&T`, `&mut T`
//! - Expressions: literals, arithmetic, comparison, `if/else`, blocks
//! - NO: generics, traits, impls, macros, modules, structs, enums, closures

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use thiserror::Error;

pub mod api;
pub mod oracle;
pub mod pipeline;

// Re-export AST types so integration tests can inspect parsed results.
pub use vyre_libs::parsing::rust::parse::{Expr, Function, Module, Stmt, Type};
pub use vyre_libs::parsing::rust::lex::lexer::core::Token;

/// Unified error type for the Rust frontend.
#[derive(Debug, Clone, Error)]
pub enum RustFrontendError {
    /// Lexing failed at the given byte offset.
    #[error("lex error at byte {0}")]
    Lex(usize),
    /// Parsing failed.
    #[error("parse error at token {token_index}: {message}")]
    Parse {
        /// Human-readable message.
        message: String,
        /// Token index in the stream.
        token_index: usize,
    },
    /// The source contains constructs outside the nano-subset.
    #[error("unsupported construct: {0}")]
    Unsupported(String),
    /// GPU backend unavailable and no CPU fallback configured.
    #[error("GPU backend unavailable: {0}")]
    Backend(String),
}
