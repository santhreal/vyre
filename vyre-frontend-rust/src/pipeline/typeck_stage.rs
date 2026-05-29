//! Type-checking stage: thin orchestrator over the `vyre-libs` sema substrate.

use vyre_libs::parsing::rust::sema::{self, ResolvedModule};

use crate::RustFrontendError;

/// Type-check a resolved module via the reusable sema substrate.
pub fn typeck(module: &ResolvedModule) -> Result<ResolvedModule, RustFrontendError> {
    sema::typeck(module).map_err(|e| RustFrontendError::Unsupported(e.to_string()))
}
