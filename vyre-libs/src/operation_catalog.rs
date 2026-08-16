//! Registry-derived library operation catalog.
//!
//! The canonical semantic records live in `vyre-foundation`, submitted through
//! `OperationRegistration::library`. This read-only projection exposes the
//! library tier and its bounded convergence metadata to conformance and
//! documentation consumers.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use vyre_foundation::operation::{OperationRegistry, OperationTier, SemanticOperation};

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

/// Every linked convergence contract, keyed by the operation it bounds.
///
/// Registrations are link-time constants, so the walk happens once and a lookup
/// is a probe rather than a scan of every registration.
static CONTRACTS: LazyLock<BTreeMap<&'static str, &'static ConvergenceContract>> =
    LazyLock::new(|| {
        let mut contracts = BTreeMap::new();
        for contract in inventory::iter::<ConvergenceContract> {
            assert!(
                contracts.insert(contract.op_id, contract).is_none(),
                "duplicate convergence contract for `{}`; keep one iteration ceiling per operation",
                contract.op_id
            );
        }
        contracts
    });

/// Look up convergence metadata.
#[must_use]
pub fn convergence_contract(op_id: &str) -> Option<&'static ConvergenceContract> {
    CONTRACTS.get(op_id).copied()
}
