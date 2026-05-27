//! Lowering stage: Rust AST → Vyre IR.

use vyre::ir::Program;

use crate::pipeline::borrow_stage::VerifiedModule;
use crate::RustFrontendError;

/// Lower a verified module to Vyre IR.
pub fn lower(module: &VerifiedModule) -> Result<Program, RustFrontendError> {
    let _ = module;
    Err(RustFrontendError::Unsupported(
        "Rust AST to Vyre IR lowering is not wired yet; do not consume a default empty Program as compiled output"
            .to_string(),
    ))
}
