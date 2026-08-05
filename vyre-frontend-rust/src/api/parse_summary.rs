//! Result types for Rust frontend entry points.

/// Result of parsing a Rust source file.
#[derive(Debug, Clone)]
pub struct ParseSummary {
    /// The parsed module AST.
    pub module: crate::parse::Module,
    /// Number of tokens.
    pub token_count: usize,
}
