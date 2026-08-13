//! Source tokenization stage for the Rust frontend pipeline.

use crate::lex::lexer::cpu_lexer::{lex as lex_source, Token};
use crate::RustFrontendError;

/// Lex one Rust source buffer on the host.
pub fn lex(source: &[u8]) -> Result<Vec<Token>, RustFrontendError> {
    lex_source(source).map_err(RustFrontendError::Lex)
}

/// Lex independent Rust source buffers while preserving input order.
pub fn lex_batch(sources: &[&[u8]]) -> Vec<Result<Vec<Token>, RustFrontendError>> {
    sources
        .iter()
        .map(|source| lex_source(source).map_err(RustFrontendError::Lex))
        .collect()
}
