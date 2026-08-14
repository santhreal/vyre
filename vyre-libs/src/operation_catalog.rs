//! Registry-derived library operation catalog.
//!
//! The canonical semantic records live in `vyre-foundation`, submitted through
//! `OperationRegistration::library`. This read-only projection exposes the
//! library tier and its bounded convergence metadata to conformance and
//! documentation consumers.

use vyre_foundation::operation::{OperationRegistry, OperationTier, SemanticOperation};

pub use crate::region::{reparent_program_children, tag_program, wrap, wrap_anonymous, wrap_child};

/// Iterate over canonical library composition registrations.
pub fn all_entries() -> impl Iterator<Item = SemanticOperation> {
    OperationRegistry::global()
        .iter()
        .filter(|entry| entry.tier == OperationTier::Library)
}

/// Iterate over library operations with complete deterministic execution fixtures.
///
/// Callable composition components remain present in [`all_entries`] for
/// validation, inlining, documentation, and complexity accounting. They are
/// omitted here when their execution is covered through a parent operation's
/// fixture rather than a standalone dispatch shape.
pub fn fixture_entries() -> impl Iterator<Item = SemanticOperation> {
    all_entries().filter(|entry| entry.test_inputs.is_some() && entry.expected_output.is_some())
}

/// Convergence metadata consumed by upper execution harnesses.
#[derive(Clone, Debug)]
pub struct ConvergenceContract {
    /// Stable operation id.
    pub op_id: &'static str,
    /// Explicit iteration ceiling.
    pub max_iterations: u32,
}

inventory::collect!(ConvergenceContract);

/// Look up convergence metadata.
#[must_use]
pub fn convergence_contract(op_id: &str) -> Option<&'static ConvergenceContract> {
    inventory::iter::<ConvergenceContract>().find(|contract| contract.op_id == op_id)
}
