//! Lowering stage: thin orchestrator over the `vyre-libs` lowering substrate.

use vyre::ir::Program;
use vyre_libs::parsing::rust::lower as rust_lower;
use vyre_libs::parsing::rust::parse::Module;

use crate::RustFrontendError;

/// Lower a verified module to Vyre IR via the reusable lowering substrate.
pub fn lower(module: &Module) -> Result<Program, RustFrontendError> {
    rust_lower::lower(module).map_err(|e| RustFrontendError::Unsupported(e.to_string()))
}
