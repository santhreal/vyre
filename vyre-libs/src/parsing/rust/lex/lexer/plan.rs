//! GPU lexer plan builder for Rust source text.
//!
//! Builds a `vyre::Program` that implements the same maximally-munching
//! DFA as `core::lex` but executes across workgroups.  The plan is
//! separated from dispatch so that the program can be cached
//! content-addressed (the plan is pure; source bytes are the only input).

use vyre::ir::Program;

/// Builder for the GPU Rust lexer.
pub struct RustLexerPlan;

impl RustLexerPlan {
    /// Create a new plan (no source needed; the lexer kernel is fixed).
    pub fn new() -> Self {
        Self
    }

    /// Build the vyre::Program that lexes a source buffer into a token
    /// stream buffer.
    ///
    /// Inputs (bind group 0):
    /// - `source`: byte array of Rust source text
    ///
    /// Outputs (bind group 0):
    /// - `tokens`: flat array of `(kind: u16, start: u32, len: u16)`
    /// - `token_count`: atomic u32
    ///
    /// TODO(v0.0.1): implement as pure vyre IR over text primitives.
    /// For now this returns a placeholder and the caller must fall back
    /// to `core::lex`.
    pub fn build(&self) -> Program {
        // Placeholder: the conform gate will reject this until it is
        // replaced with a real kernel.
        Program::default()
    }
}

impl Default for RustLexerPlan {
    fn default() -> Self {
        Self::new()
    }
}
