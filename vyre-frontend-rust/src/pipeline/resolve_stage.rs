//! Name resolution stage.
//!
//! TODO(v0.1.0): build a scope tree, resolve identifiers to def-ids,
//! report unresolved names.

use vyre_libs::parsing::rust::parse::Module;

use crate::RustFrontendError;

/// Resolved module (placeholder: currently identity pass).
pub type ResolvedModule = Module;

/// Resolve names in a module.
pub fn resolve(module: &Module) -> Result<ResolvedModule, RustFrontendError> {
    // Placeholder: name resolution is unimplemented.
    Ok(module.clone())
}
