//! Borrow-checking stage: thin orchestrator over the `vyre-libs` sema substrate.
//!
//! The borrow analysis (CFG construction plus the dataflow fixed point) is
//! language-specific substrate and lives in `vyre-libs::parsing::rust::sema`.
//! This stage only orchestrates it.

use vyre_libs::parsing::rust::parse::Module;
use vyre_libs::parsing::rust::sema;

use crate::RustFrontendError;

/// Borrow-check a typed module via the reusable sema substrate.
pub fn borrow_check(module: &Module) -> Result<Module, RustFrontendError> {
    sema::borrow_check(module).map_err(|e| RustFrontendError::Unsupported(e.to_string()))
}
