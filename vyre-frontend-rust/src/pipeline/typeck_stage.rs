//! Type inference and checking stage.

use crate::pipeline::resolve_stage::ResolvedModule;
use crate::RustFrontendError;

/// Module with inferred types.
pub type TypedModule = ResolvedModule;

/// Type-check a resolved module.
pub fn typeck(module: &ResolvedModule) -> Result<TypedModule, RustFrontendError> {
    let _ = module;
    Err(RustFrontendError::Unsupported(
        "type checking is not wired to the Rust nano-subset type environment yet; use parse-only API calls until semantic analysis is enabled"
            .to_string(),
    ))
}
