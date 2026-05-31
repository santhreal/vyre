//! GPU lexer plan builder for Rust source text.

mod program;

pub use program::rust_lexer;

use vyre::ir::Program;

/// Builder for the GPU Rust lexer.
pub struct RustLexerPlan;

impl RustLexerPlan {
    /// Create a new lexer plan.
    pub fn new() -> Self {
        Self
    }

    /// Build the Vyre IR program for an empty source buffer.
    #[must_use]
    pub fn build(&self) -> Program {
        self.build_for_len(0)
    }

    /// Build the Vyre IR program that lexes a source buffer of `haystack_len`
    /// bytes into compact token columns. The source buffer stores one byte per
    /// `u32` word.
    #[must_use]
    pub fn build_for_len(&self, haystack_len: u32) -> Program {
        rust_lexer(
            "haystack",
            "out_tok_types",
            "out_tok_starts",
            "out_tok_lens",
            "out_counts",
            haystack_len,
        )
    }
}

impl Default for RustLexerPlan {
    fn default() -> Self {
        Self::new()
    }
}
