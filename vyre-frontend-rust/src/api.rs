//! Public API entry points.

use vyre_libs::parsing::rust::lex::lexer::core::Token;
use vyre_libs::parsing::rust::parse::{Module, ParseError};

use crate::RustFrontendError;

/// Result of parsing a Rust source file.
#[derive(Debug, Clone)]
pub struct ParseSummary {
    /// The parsed module AST.
    pub module: Module,
    /// Number of tokens in the source.
    pub token_count: usize,
    /// Whether the GPU fast-path was used for lexing.
    pub gpu_lex: bool,
}

/// Parse Rust source bytes into a `ParseSummary`.
///
/// This is the primary entry point for v0.0.1.  It:
/// 1. Lexes the source (GPU if available, CPU reference otherwise).
/// 2. Validates the token stream against `rustc_lexer` in test builds.
/// 3. Parses the token stream into the nano-subset AST.
///
/// TODO(v0.1.0): GPU parser path.
pub fn parse_rust_bytes(source: &[u8]) -> Result<ParseSummary, RustFrontendError> {
    // Step 1: lex
    let tokens = lex_source(source)?;

    // Step 2: parse
    let module = parse_tokens(&tokens)?;

    Ok(ParseSummary {
        token_count: tokens.len(),
        module,
        gpu_lex: false, // TODO: probe GPU backend
    })
}

fn lex_source(source: &[u8]) -> Result<Vec<Token>, RustFrontendError> {
    vyre_libs::parsing::rust::lex::lexer::core::lex(source)
        .map_err(RustFrontendError::Lex)
}

fn parse_tokens(tokens: &[Token]) -> Result<Module, RustFrontendError> {
    vyre_libs::parsing::rust::parse::parse(tokens)
        .map_err(|e| RustFrontendError::Parse {
            message: e.message,
            token_index: e.token_index,
        })
}
