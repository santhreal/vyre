//! Verification certificate and checked invariant definitions.

use super::super::types::shape::solver::ShapeProofCertificate;
use super::super::types::shape::ShapeInterner;

/// Category of invariant evaluated during semantic verification.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum InvariantCategory {
    /// Well-formed syntax, closed references, and module structural completeness.
    StructuralClosure,
    /// Static single assignment dominance and use-def chain validity.
    DominanceUseDef,
    /// Type, shape, rank, and layout constraint compatibility.
    TypeShapeRank,
    /// Resource capability, exclusive mutability, and alias freedom.
    AliasOwnership,
    /// Memory model ordering, scope visibility, and barrier synchronization.
    EffectsConcurrency,
    /// Buffer extent bounds, dispatch geometry, and loop termination.
    BoundsTermination,
    /// Determinism guarantees, IEEE floating point rules, and precision contracts.
    DeterminismNumeric,
    /// Uniform participation and topology rules for collective communication.
    CollectiveGroups,
    /// Asynchronous transaction lifecycle and stage state machine transitions.
    StateTransitions,
}

/// One specific invariant evaluated and recorded during verification.
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub struct CheckedInvariant {
    /// Invariant category.
    pub category: InvariantCategory,
    /// Stable invariant rule code (e.g. "INV-STRUCT-001", "INV-SHAPE-002").
    pub code: &'static str,
    /// Human-readable explanation of the verified condition.
    pub description: String,
    /// Whether this invariant passed verification.
    pub passed: bool,
}

/// Hardware resource bounds certified by the verifier.
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub struct ResourceBounds {
    /// Maximum workgroup local memory allocated (bytes).
    pub max_workgroup_memory_bytes: u64,
    /// Maximum estimated registers per thread.
    pub max_registers_per_thread: u32,
    /// Maximum total threads per workgroup.
    pub max_threads_per_workgroup: u32,
    /// Launch grid dimensions [x, y, z].
    pub grid_dimensions: [u32; 3],
    /// Number of bound storage and uniform buffers.
    pub buffer_count: usize,
}

impl Default for ResourceBounds {
    fn default() -> Self {
        Self {
            max_workgroup_memory_bytes: 0,
            max_registers_per_thread: 32,
            max_threads_per_workgroup: 256,
            grid_dimensions: [1, 1, 1],
            buffer_count: 0,
        }
    }
}

/// Cryptographic and semantic verification certificate produced by [`DeclarativeVerifier`](super::DeclarativeVerifier).
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct VerificationCertificate {
    /// Certificate schema version.
    pub schema_version: u32,
    /// Declarative verifier version identifier.
    pub verifier_version: &'static str,
    /// Cryptographic digest (Blake3 hex) of the verified semantic module.
    pub input_identity: String,
    /// Exact sequence of invariants actually evaluated and proven for this module.
    pub checked_invariants: Vec<CheckedInvariant>,
    /// Explicit assumptions relied upon during proof generation.
    pub assumptions: Vec<String>,
    /// Solver proof certificates for shape, layout, and divisibility properties.
    pub solver_proofs: Vec<ShapeProofCertificate>,
    /// Certified upper bounds on resource consumption.
    pub resource_bounds: ResourceBounds,
}

impl VerificationCertificate {
    /// Number of verified invariants recorded in this certificate.
    #[must_use]
    pub fn invariant_count(&self) -> usize {
        self.checked_invariants.len()
    }

    /// Whether this certificate contains a check for the given category.
    #[must_use]
    pub fn has_category(&self, category: InvariantCategory) -> bool {
        self.checked_invariants.iter().any(|inv| inv.category == category)
    }

    /// Replay solver proofs recorded in this certificate.
    #[must_use]
    pub fn replay_solver_proofs(&self, interner: &ShapeInterner) -> bool {
        self.solver_proofs.iter().all(|proof| proof.replay(interner))
    }
}
