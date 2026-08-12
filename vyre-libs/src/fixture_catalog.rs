//! Library composition fixture metadata.
//!
//! Execution belongs to upper conformance harnesses. This module contains only
//! neutral program builders and deterministic byte fixtures; it has no backend
//! or harness-crate dependency.

use vyre_foundation::operation::{OperationRegistry, OperationTier, SemanticOperation};
/// Floating-point parity policy for upper execution harnesses.
pub mod fp_contract;

pub use crate::region::{reparent_program_children, tag_program, wrap, wrap_anonymous, wrap_child};

/// Deterministic fixture input cases.
pub type InputsFn = vyre_foundation::operation::OperationFixtures;
/// Deterministic expected-output fixtures.
pub type ExpectedFn = vyre_foundation::operation::OperationFixtures;

/// Iterate over canonical library composition registrations.
pub fn all_entries() -> impl Iterator<Item = SemanticOperation> {
    OperationRegistry::global()
        .iter()
        .filter(|entry| entry.tier == OperationTier::Library)
}

/// Fixpoint metadata consumed by upper execution harnesses.
#[derive(Clone, Debug)]
pub struct FixpointContract {
    /// Changed-flag buffer name.
    pub converged_flag_buffer: &'static str,
    /// Explicit iteration ceiling.
    pub max_iterations: u32,
}

/// Associates fixpoint metadata with a neutral composition.
pub struct FixpointRegistration {
    /// Stable operation id.
    pub op_id: &'static str,
    /// Fixpoint contract.
    pub contract: FixpointContract,
}

inventory::collect!(FixpointRegistration);

/// Look up fixpoint metadata.
#[must_use]
pub fn fixpoint_contract(op_id: &str) -> Option<&'static FixpointContract> {
    inventory::iter::<FixpointRegistration>()
        .find(|registration| registration.op_id == op_id)
        .map(|registration| &registration.contract)
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
