//! Evidence object emission.
//!
//! When the Rust frontend compiles a translation unit, it emits an
//! object artifact containing:
//! - The parsed AST (for debugging / parity)
//! - The verified Vyre IR program
//! - Borrow-check evidence
//!
//! Format is TBD; modelled after vyre-frontend-c's VYRECOB2 sections.

/// Object artifact builder.
pub struct RustObjectBuilder;

impl RustObjectBuilder {
    /// Create a new object builder.
    pub fn new() -> Self {
        Self
    }
}

impl Default for RustObjectBuilder {
    fn default() -> Self {
        Self::new()
    }
}
