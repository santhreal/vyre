//! Public API entry points.

use vyre_libs::parsing::rust::lex::lexer::core::lex;
use vyre_libs::parsing::rust::parse::Module;

use crate::{RustFrontendError, Token};

/// Result of parsing a Rust source file.
#[derive(Debug, Clone)]
pub struct ParseSummary {
    /// The parsed module AST.
    pub module: Module,
    /// Number of tokens.
    pub token_count: usize,
    /// Whether GPU fast-path was used for lexing.
    pub gpu_lex: bool,
}

/// Parse Rust source bytes into a `ParseSummary`.
///
/// 1. Lexes the source (GPU if available, CPU reference otherwise).
/// 2. Parses the token stream into the nano-subset AST.
pub fn parse_rust_bytes(source: &[u8]) -> Result<ParseSummary, RustFrontendError> {
    let tokens = lex(source).map_err(RustFrontendError::Lex)?;
    let module = parse_tokens(source, &tokens)?;
    Ok(ParseSummary {
        token_count: tokens.len(),
        module,
        gpu_lex: false,
    })
}

fn parse_tokens(source: &[u8], tokens: &[Token]) -> Result<Module, RustFrontendError> {
    vyre_libs::parsing::rust::parse::parse(source, tokens)
        .map_err(|e| RustFrontendError::Parse {
            message: e.message,
            token_index: e.token_index,
        })
}
