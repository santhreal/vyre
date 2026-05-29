//! Name resolution stage: thin orchestrator over the `vyre-libs` sema substrate.

use vyre_libs::parsing::rust::parse::Module;
use vyre_libs::parsing::rust::sema;

use crate::RustFrontendError;

/// Resolve names in a module via the reusable sema substrate.
pub fn resolve(module: &Module) -> Result<Module, RustFrontendError> {
    sema::resolve(module).map_err(|e| RustFrontendError::Unsupported(e.to_string()))
}
