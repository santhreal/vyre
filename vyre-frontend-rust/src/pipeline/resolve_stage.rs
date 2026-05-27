//! Name resolution stage.

use vyre_libs::parsing::rust::parse::Module;

use crate::RustFrontendError;

/// Resolved module for the nano-subset.
pub type ResolvedModule = Module;

/// Resolve names in a module.
pub fn resolve(module: &Module) -> Result<ResolvedModule, RustFrontendError> {
    let _ = module;
    Err(RustFrontendError::Unsupported(
        "name resolution is not wired to a Rust scope graph yet; use `parse_rust_bytes` for parse-only API calls"
            .to_string(),
    ))
}
