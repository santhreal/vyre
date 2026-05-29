//! Type-checking stage: thin orchestrator over the `vyre-libs` sema substrate.

use vyre_libs::parsing::rust::parse::Module;
use vyre_libs::parsing::rust::sema;

use crate::RustFrontendError;

/// Type-check a resolved module via the reusable sema substrate.
pub fn typeck(module: &Module) -> Result<Module, RustFrontendError> {
    sema::typeck(module).map_err(|e| RustFrontendError::Unsupported(e.to_string()))
}
