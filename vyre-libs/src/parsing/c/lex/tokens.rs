//! C11 token type constants for the GPU lexer pipeline.
//!
//! `vyre_spec::c11_token` owns the numbering: it is the wire contract between
//! the host table generator and the GPU parser that decodes its blobs, and
//! both sides depend down onto the foundation-layer spec crate to read it.
//! This module re-exports the vocabulary so the published
//! `vyre_libs::parsing::c::lex::tokens::TOK_*` paths keep resolving.

pub use vyre_spec::c11_token::*;
