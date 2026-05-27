//! Type inference and checking stage.
//!
//! TODO(v0.1.0): implement nano-subset type inference.

use crate::pipeline::resolve_stage::ResolvedModule;
use crate::RustFrontendError;

/// Module with inferred types.
pub type TypedModule = ResolvedModule;

/// Type-check a resolved module.
pub fn typeck(module: &ResolvedModule) -> Result<TypedModule, RustFrontendError> {
    // Placeholder.
    Ok(module.clone())
}
