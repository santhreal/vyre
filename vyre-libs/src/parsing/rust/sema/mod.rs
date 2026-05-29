//! Rust semantic analysis: name resolution, type inference, borrow checking.
//!
//! Reusable substrate (Tier 3). The algorithms live here, mirroring
//! `vyre-libs::parsing::c::sema`, so any consumer can run typed Rust analysis
//! without depending on the `vyre-frontend-rust` driver crate (and its
//! object/ELF/GPU-dispatch surface). The driver only orchestrates these stages.
//!
//! Name resolution, type inference, and borrow checking are not yet implemented
//! for the nano-subset. Each entry point returns a loud, actionable error
//! rather than a fake success, so a caller never consumes an unanalyzed module
//! as if it were verified.

use thiserror::Error;

use super::parse::Module;

/// Module after name resolution. Alias until a distinct resolved form exists.
pub type ResolvedModule = Module;
/// Module after type inference. Alias until a distinct typed form exists.
pub type TypedModule = Module;
/// Module after borrow checking. Alias until a distinct verified form exists.
pub type VerifiedModule = Module;

/// Errors from the Rust semantic-analysis stages.
#[derive(Debug, Clone, Error)]
pub enum RustSemaError {
    /// Name resolution is not implemented for the nano-subset.
    #[error("name resolution is not wired to a Rust scope graph yet; use parse-only API calls (parse_rust_bytes) until semantic analysis is enabled")]
    ResolveUnavailable,
    /// Type inference and checking are not implemented for the nano-subset.
    #[error("type checking is not wired to the Rust nano-subset type environment yet; use parse-only API calls until semantic analysis is enabled")]
    TypeckUnavailable,
    /// Borrow checking is not implemented for the nano-subset.
    #[error("borrow checking is not wired to a Rust CFG yet; disable borrow checking for parse-only pipeline runs")]
    BorrowUnavailable,
}

/// Resolve names in a parsed module.
///
/// # Errors
/// Returns [`RustSemaError::ResolveUnavailable`] until name resolution is wired.
pub fn resolve(module: &Module) -> Result<ResolvedModule, RustSemaError> {
    let _ = module;
    Err(RustSemaError::ResolveUnavailable)
}

/// Infer and check types for a resolved module.
///
/// # Errors
/// Returns [`RustSemaError::TypeckUnavailable`] until type checking is wired.
pub fn typeck(module: &ResolvedModule) -> Result<TypedModule, RustSemaError> {
    let _ = module;
    Err(RustSemaError::TypeckUnavailable)
}

/// Borrow-check a typed module.
///
/// When implemented this composes the shared dataflow engine over a Rust CFG.
/// It does not import that engine while unimplemented, to avoid a dead
/// dependency edge from the substrate.
///
/// # Errors
/// Returns [`RustSemaError::BorrowUnavailable`] until borrow checking is wired.
pub fn borrow_check(module: &TypedModule) -> Result<VerifiedModule, RustSemaError> {
    let _ = module;
    Err(RustSemaError::BorrowUnavailable)
}
