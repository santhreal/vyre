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
    Effects,
    /// Buffer extent bounds, dispatch geometry, and resource limits.
    Bounds,
    /// Loop termination, trip count bounds, and forward execution progress.
    TerminationProgress,
    /// Determinism guarantees, IEEE floating point rules, and reproducible execution.
    Determinism,
    /// Numerical accuracy contracts, rounding modes, and precision constraints.
    NumericContracts,
    /// Uniform participation and topology rules for collective communication.
    CollectiveGroups,
    /// Asynchronous transaction lifecycle and stage state machine transitions.
    StateTransitions,
    /// External extension obligations and dialect ABI requirements.
    SemanticExtensionObligations,
    /// Legacy composite effect/concurrency category.
    EffectsConcurrency,
    /// Legacy composite bounds/termination category.
    BoundsTermination,
    /// Legacy composite determinism/numeric category.
    DeterminismNumeric,
}

impl InvariantCategory {
    /// The twelve canonical invariant categories.
    pub const CANONICAL: &'static [InvariantCategory] = &[
        InvariantCategory::StructuralClosure,
        InvariantCategory::DominanceUseDef,
        InvariantCategory::TypeShapeRank,
        InvariantCategory::AliasOwnership,
        InvariantCategory::Effects,
        InvariantCategory::Bounds,
        InvariantCategory::TerminationProgress,
        InvariantCategory::Determinism,
        InvariantCategory::NumericContracts,
        InvariantCategory::CollectiveGroups,
        InvariantCategory::StateTransitions,
        InvariantCategory::SemanticExtensionObligations,
    ];

    /// All canonical invariant categories.
    #[must_use]
    pub const fn all() -> &'static [InvariantCategory] {
        Self::CANONICAL
    }

    /// Category code prefix string.
    #[must_use]
    pub const fn code_prefix(&self) -> &'static str {
        match self {
            Self::StructuralClosure => "INV-STRUCT",
            Self::DominanceUseDef => "INV-DOM",
            Self::TypeShapeRank => "INV-TYPE",
            Self::AliasOwnership => "INV-ALIAS",
            Self::Effects | Self::EffectsConcurrency => "INV-EFFECT",
            Self::Bounds | Self::BoundsTermination => "INV-BOUND",
            Self::TerminationProgress => "INV-TERM",
            Self::Determinism => "INV-DET",
            Self::NumericContracts | Self::DeterminismNumeric => "INV-NUM",
            Self::CollectiveGroups => "INV-COLL",
            Self::StateTransitions => "INV-STATE",
            Self::SemanticExtensionObligations => "INV-EXT",
        }
    }
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
    /// Estimated static work units.
    pub estimated_work_units: u64,
}

impl Default for ResourceBounds {
    fn default() -> Self {
        Self {
            max_workgroup_memory_bytes: 0,
            max_registers_per_thread: 32,
            max_threads_per_workgroup: 256,
            grid_dimensions: [1, 1, 1],
            buffer_count: 0,
            estimated_work_units: 1,
        }
    }
}

/// Verification certificate replay failure.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ReplayError {
    /// The certificate's recorded input identity does not match the module's computed identity.
    #[error("MismatchedInputIdentity: certificate expected identity `{expected}` but module computed `{actual}`. Refusing tampered certificate by name.")]
    MismatchedInputIdentity {
        /// Expected identity digest recorded in the certificate.
        expected: String,
        /// Actual identity digest computed from the semantic module.
        actual: String,
    },
    /// The certificate's schema version does not match the expected version.
    #[error("SchemaVersionMismatch: certificate schema version {found} does not match expected {expected}")]
    SchemaVersionMismatch {
        /// Expected schema version.
        expected: u32,
        /// Found schema version.
        found: u32,
    },
    /// The certificate's verifier version does not match the expected version.
    #[error("VerifierVersionMismatch: certificate verifier version `{found}` does not match expected `{expected}`")]
    VerifierVersionMismatch {
        /// Expected verifier version.
        expected: &'static str,
        /// Found verifier version.
        found: String,
    },
    /// An invariant recorded in the certificate was marked as not passed.
    #[error("UnsatisfiedInvariant: invariant `{code}` ({category:?}) was not proven")]
    UnsatisfiedInvariant {
        /// Violated invariant code.
        code: &'static str,
        /// Invariant category.
        category: InvariantCategory,
    },
    /// Solver proof replay failed for a shape/layout proof object.
    #[error(
        "SolverProofReplayFailed: shape or layout proof certificate failed independent replay"
    )]
    SolverProofReplayFailed,
    /// Resource bounds contain invalid dimensions or limits.
    #[error("ResourceBoundsViolation: {0}")]
    ResourceBoundsViolation(String),
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
        self.checked_invariants.iter().any(|inv| {
            inv.category == category
                || match (inv.category, category) {
                    (InvariantCategory::Effects, InvariantCategory::EffectsConcurrency)
                    | (InvariantCategory::EffectsConcurrency, InvariantCategory::Effects) => true,
                    (InvariantCategory::Bounds, InvariantCategory::BoundsTermination)
                    | (InvariantCategory::BoundsTermination, InvariantCategory::Bounds) => true,
                    (InvariantCategory::Determinism, InvariantCategory::DeterminismNumeric)
                    | (InvariantCategory::DeterminismNumeric, InvariantCategory::Determinism) => {
                        true
                    }
                    (
                        InvariantCategory::NumericContracts,
                        InvariantCategory::DeterminismNumeric,
                    )
                    | (
                        InvariantCategory::DeterminismNumeric,
                        InvariantCategory::NumericContracts,
                    ) => true,
                    (
                        InvariantCategory::TerminationProgress,
                        InvariantCategory::BoundsTermination,
                    )
                    | (
                        InvariantCategory::BoundsTermination,
                        InvariantCategory::TerminationProgress,
                    ) => true,
                    _ => false,
                }
        })
    }

    /// Replay solver proofs recorded in this certificate.
    #[must_use]
    pub fn replay_solver_proofs(&self, interner: &ShapeInterner) -> bool {
        self.solver_proofs
            .iter()
            .all(|proof| proof.replay(interner))
    }

    /// Independent lightweight proof replay of this certificate against a semantic module.
    ///
    /// Validates schema version, verifier version, input identity digest matching,
    /// invariant satisfaction, solver proof replay, and resource bounds without
    /// rerunning an ad hoc AST traversal.
    ///
    /// # Errors
    ///
    /// Returns [`ReplayError`] when any certificate invariant, proof, or identity check fails.
    pub fn replay_proof(&self, module: &super::module::SemanticModule) -> Result<(), ReplayError> {
        if self.schema_version != super::declarative::DeclarativeVerifier::SCHEMA_VERSION {
            return Err(ReplayError::SchemaVersionMismatch {
                expected: super::declarative::DeclarativeVerifier::SCHEMA_VERSION,
                found: self.schema_version,
            });
        }
        if self.verifier_version != super::declarative::DeclarativeVerifier::VERSION {
            return Err(ReplayError::VerifierVersionMismatch {
                expected: super::declarative::DeclarativeVerifier::VERSION,
                found: self.verifier_version.to_string(),
            });
        }
        let actual_identity = module.compute_identity();
        if self.input_identity != actual_identity {
            return Err(ReplayError::MismatchedInputIdentity {
                expected: self.input_identity.clone(),
                actual: actual_identity,
            });
        }
        for inv in &self.checked_invariants {
            if !inv.passed {
                return Err(ReplayError::UnsatisfiedInvariant {
                    code: inv.code,
                    category: inv.category,
                });
            }
        }
        if !self.replay_solver_proofs(&module.shape_interner) {
            return Err(ReplayError::SolverProofReplayFailed);
        }
        if self.resource_bounds.grid_dimensions.contains(&0) {
            return Err(ReplayError::ResourceBoundsViolation(
                "grid dimensions cannot contain zero".into(),
            ));
        }
        Ok(())
    }
}
