//! Lowering stage: Rust AST → Vyre IR.
//!
//! TODO(v0.1.0): implement AST → Vyre IR lowering for the nano-subset.

use vyre::ir::Program;

use crate::pipeline::borrow_stage::VerifiedModule;
use crate::RustFrontendError;

/// Lower a verified module to Vyre IR.
pub fn lower(_module: &VerifiedModule) -> Result<Program, RustFrontendError> {
    // Placeholder.
    Ok(Program::default())
}
