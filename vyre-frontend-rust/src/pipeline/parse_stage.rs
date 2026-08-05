//! Parse stage: token stream → AST.

use crate::lex::lexer::core::Token;
use crate::parse::Module;

use crate::RustFrontendError;

/// Parse tokens into an AST module.
pub fn parse(source: &[u8], tokens: &[Token]) -> Result<Module, RustFrontendError> {
    crate::parse::parse(source, tokens).map_err(|e| RustFrontendError::Parse {
        message: e.message,
        token_index: e.token_index,
    })
}
