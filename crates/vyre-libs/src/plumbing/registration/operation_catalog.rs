//! Registry-derived library operation catalog.
//!
//! Canonical semantic records are defined in `vyre-foundation` and submitted
//! through `OperationRegistration::library_unconstrained` or an explicit
//! constraint declaration. This projection exposes the library tier to
//! conformance and documentation consumers.
use vyre_foundation::operation::{OperationRegistry, OperationTier, SemanticOperation};

/// Iterate every registration in the library tier.
///
/// Selects on [`OperationTier::Library`], so a Category C hardware intrinsic
/// is not returned here. Intrinsic-tier readers use
/// `vyre_primitives::operation_catalog::intrinsic_entries`.
pub fn library_entries() -> impl Iterator<Item = SemanticOperation> {
    OperationRegistry::global()
        .iter()
        .filter(|entry| entry.tier == OperationTier::Library)
}

/// Ensure all feature-selected library operation registrations are retained by the linker.
///
/// Returns the number of canonical library operations contributed by the active feature set.
#[must_use]
pub fn link_anchor() -> usize {
    library_entries().count()
}

/// Iterate over library operations with complete deterministic execution fixtures.
///
/// Callable composition components remain present in [`library_entries`] for
/// validation, inlining, documentation, and complexity accounting. They are
/// omitted here when their execution is covered through a parent operation's
/// fixture rather than a standalone dispatch shape.
pub fn fixture_entries() -> impl Iterator<Item = SemanticOperation> {
    library_entries().filter(|entry| entry.test_inputs.is_some() && entry.expected_output.is_some())
}
