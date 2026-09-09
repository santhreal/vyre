//! Optional operation-contract metadata shared by signatures and catalogs.

use alloc::string::String;
use alloc::vec::Vec;

use crate::algebraic_law::{GuardedLaw, LawValidationError};
use crate::memory_effect::MemoryEffect;
use crate::numeric_semantics::{InfinityBehavior, NanBehavior};
use crate::op_signature::OpSignature;
/// Backend capability required by an operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
pub struct CapabilityId(pub String);

impl CapabilityId {
    /// Create a capability id from a stable name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Return the stable capability name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Determinism contract for an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
#[non_exhaustive]
pub enum DeterminismClass {
    /// Bit-identical outputs for identical inputs.
    Deterministic,
    /// Deterministic except for backend rounding policy.
    DeterministicModuloRounding,
    /// Backend scheduling or hardware effects may change results.
    NonDeterministic,
}

/// Side-effect class for an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
#[non_exhaustive]
pub enum SideEffectClass {
    /// Pure value computation.
    Pure,
    /// Reads memory through explicit operands.
    ReadsMemory,
    /// Writes memory through explicit operands.
    WritesMemory,
    /// Performs synchronization.
    Synchronizing,
    /// Performs atomic memory effects.
    Atomic,
}

/// Portable cost hint for planning and diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
#[non_exhaustive]
pub enum CostHint {
    /// Cheap scalar or metadata operation.
    Cheap,
    /// Medium-cost operation.
    Medium,
    /// Expensive operation.
    Expensive,
    /// Cost depends on backend or runtime data.
    Unknown,
}

/// Optional contract annotations for operation declarations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
pub struct OperationContract {
    /// Required backend capabilities.
    #[serde(default)]
    pub capability_requirements: Option<smallvec::SmallVec<[CapabilityId; 4]>>,
    /// Determinism class.
    #[serde(default)]
    pub determinism: Option<DeterminismClass>,
    /// Side-effect class.
    #[serde(default)]
    pub side_effect: Option<SideEffectClass>,
    /// Portable cost hint.
    #[serde(default)]
    pub cost_hint: Option<CostHint>,
}

impl OperationContract {
    /// Empty contract for declarations that have not been annotated yet.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            capability_requirements: None,
            determinism: None,
            side_effect: None,
            cost_hint: None,
        }
    }
}

impl Default for OperationContract {
    fn default() -> Self {
        Self::none()
    }
}

/// Memory and buffer aliasing permissions for an operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum AliasingContract {
    /// All input and output buffers must occupy pairwise disjoint address ranges.
    Disjoint,
    /// Input and output buffers may alias arbitrarily.
    MayAlias,
    /// Operands are required to alias identically (e.g. in-place update).
    MustAlias,
    /// Multiple inputs may share read-only overlapping memory regions.
    ReadSharingOnly,
    /// Explicitly declared partition sets that may alias internally.
    ExplicitAliasSets(Vec<String>),
}

impl AliasingContract {
    /// Whether this contract requires pairwise disjoint memory.
    #[must_use]
    pub const fn is_disjoint(&self) -> bool {
        matches!(self, Self::Disjoint)
    }

    /// Whether this contract permits in-place mutation or write aliasing.
    #[must_use]
    pub const fn allows_write_aliasing(&self) -> bool {
        matches!(self, Self::MayAlias | Self::MustAlias)
    }
}

impl Default for AliasingContract {
    fn default() -> Self {
        Self::Disjoint
    }
}

/// Structural relationship between input and output tensor/buffer shapes and index spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum ShapeIndexRelation {
    /// Output shape identical to input shape; 1:1 index map.
    Elementwise,
    /// Output rank/dimension strictly smaller than input (reductions, contractions).
    Contracting,
    /// Output dimension larger than input (broadcast, padding, upsampling).
    Expanding,
    /// Multidimensional broadcasting according to numpy/wgsl rules.
    Broadcast,
    /// Permutation or transposition of coordinate axes with invariant volume.
    Permutation,
    /// Stenciled or windowed index neighborhood (convolutions, pooling).
    Windowed,
    /// Regular strided subsampling or slicing.
    Strided,
    /// Indirect or irregular indexing (gather, scatter, CSR graph traversal).
    Irregular,
    /// Shape determined by runtime data values.
    Dynamic,
    /// Agnostic to shape constraints.
    Agnostic,
}

impl Default for ShapeIndexRelation {
    fn default() -> Self {
        Self::Elementwise
    }
}

/// Shape and indexing contract record.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ShapeIndexContract {
    /// Primary shape relationship.
    pub relation: ShapeIndexRelation,
    /// Whether rank is preserved across inputs and outputs.
    pub rank_preserving: bool,
    /// Closed algebraic invariants on dimensions (e.g. `"dim[0] == out[0]"`).
    pub dimension_invariants: Vec<String>,
}

impl ShapeIndexContract {
    /// Construct an elementwise rank-preserving shape contract.
    #[must_use]
    pub const fn elementwise() -> Self {
        Self {
            relation: ShapeIndexRelation::Elementwise,
            rank_preserving: true,
            dimension_invariants: Vec::new(),
        }
    }

    /// Construct an agnostic shape contract.
    #[must_use]
    pub const fn agnostic() -> Self {
        Self {
            relation: ShapeIndexRelation::Agnostic,
            rank_preserving: false,
            dimension_invariants: Vec::new(),
        }
    }
}

impl Default for ShapeIndexContract {
    fn default() -> Self {
        Self::elementwise()
    }
}

/// Closed numerical behavior model.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum NumericBehavior {
    /// Bit-exact result across all compliant implementations.
    Exact,
    /// Exact integer semantics with wrapping overflow.
    ExactInteger,
    /// IEEE-754 floating point arithmetic with bounded ULP drift.
    IeeeFloatingPoint {
        /// Permitted error budget in Units in the Last Place.
        ulp_budget: u32,
        /// Handling of NaN operands and outputs.
        nan_behavior: NanBehavior,
        /// Handling of infinities.
        infinity_behavior: InfinityBehavior,
    },
    /// Approximate computation with bounded relative error in floating-point bits.
    Approximate {
        /// Maximum relative error expressed as IEEE 754 float bits.
        max_relative_error_fp64_bits: u64,
    },
    /// Saturated arithmetic clamping at domain extrema.
    Saturated,
    /// Modular field arithmetic.
    Modular,
}

impl NumericBehavior {
    /// Whether this behavior is bit-exact.
    #[must_use]
    pub const fn is_exact(&self) -> bool {
        matches!(self, Self::Exact | Self::ExactInteger)
    }

    /// Return the ULP error budget if floating point.
    #[must_use]
    pub const fn ulp_budget(&self) -> Option<u32> {
        match self {
            Self::IeeeFloatingPoint { ulp_budget, .. } => Some(*ulp_budget),
            Self::Exact | Self::ExactInteger => Some(0),
            _ => None,
        }
    }
}

impl Default for NumericBehavior {
    fn default() -> Self {
        Self::Exact
    }
}

/// Range precondition on operand values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum RangePrecondition {
    /// Entire operand type domain admitted.
    Unbounded,
    /// Strictly positive operands (`x > 0`).
    PositiveOnly,
    /// Non-zero operands (`x != 0`).
    NonZero,
    /// Finite values only (no NaN or infinities).
    FiniteOnly,
    /// Values restricted to normalized interval `[0.0, 1.0]`.
    UnitInterval,
    /// Inclusive integer bounds `[lo, hi]`.
    InclusiveRange {
        /// Lower bound.
        lo: i64,
        /// Upper bound.
        hi: i64,
    },
}

impl Default for RangePrecondition {
    fn default() -> Self {
        Self::Unbounded
    }
}

/// Range contract declaring global and per-input preconditions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RangeContract {
    /// Global precondition applied across all operands.
    pub precondition: RangePrecondition,
    /// Positional preconditions for each input operand.
    pub input_bounds: Vec<RangePrecondition>,
}

impl RangeContract {
    /// Construct an unbounded range contract.
    #[must_use]
    pub const fn unbounded() -> Self {
        Self {
            precondition: RangePrecondition::Unbounded,
            input_bounds: Vec::new(),
        }
    }
}

impl Default for RangeContract {
    fn default() -> Self {
        Self::unbounded()
    }
}

/// Resource allocation bounds contract for an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum ResourceBoundsContract {
    /// Static memory allocation of fixed size.
    StaticMemoryBytes(usize),
    /// Dynamically bounded temporary workspace memory.
    DynamicBounded {
        /// Maximum bytes.
        max_bytes: usize,
    },
    /// Bounded register budget per invocation.
    RegistersBounded {
        /// Maximum registers.
        max_registers: u32,
    },
    /// No additional memory or register resources beyond arguments.
    Unbounded,
}

impl Default for ResourceBoundsContract {
    fn default() -> Self {
        Self::Unbounded
    }
}

/// Exhaustive transformation decision: either proof-producing guarded laws or an explicit opaque decision.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum TransformDecision {
    /// Operation exposes one or more verified guarded algebraic laws.
    GuardedLaws(Vec<GuardedLaw>),
    /// Explicit decision that no algebraic transform applies due to opaque primitive semantics.
    Opaque {
        /// Non-empty, concrete explanatory reason.
        reason: String,
    },
    /// Explicit decision that no algebraic transform applies due to non-composable architecture.
    NoTransform {
        /// Non-empty, concrete explanatory reason.
        reason: String,
    },
}

impl TransformDecision {
    /// Whether a decision has been recorded.
    #[must_use]
    pub fn has_decision(&self) -> bool {
        match self {
            Self::GuardedLaws(laws) => !laws.is_empty(),
            Self::Opaque { reason } | Self::NoTransform { reason } => {
                !reason.trim().is_empty() && !is_placeholder_reason(reason)
            }
        }
    }

    /// Whether this is an opaque or no-transform decision.
    #[must_use]
    pub const fn is_opaque(&self) -> bool {
        matches!(self, Self::Opaque { .. } | Self::NoTransform { .. })
    }

    /// Return the opaque or no-transform reason string, if any.
    #[must_use]
    pub fn opaque_reason(&self) -> Option<&str> {
        match self {
            Self::Opaque { reason } | Self::NoTransform { reason } => Some(reason.as_str()),
            Self::GuardedLaws(_) => None,
        }
    }

    /// Return the declared laws, if any.
    #[must_use]
    pub fn laws(&self) -> &[GuardedLaw] {
        match self {
            Self::GuardedLaws(laws) => laws.as_slice(),
            Self::Opaque { .. } | Self::NoTransform { .. } => &[],
        }
    }

    /// Validate the decision: ensures laws have proof evidence and reasons are non-placeholder.
    ///
    /// # Errors
    /// Returns [`ContractValidationError`] if the decision is invalid.
    pub fn validate(&self) -> Result<(), ContractValidationError> {
        match self {
            Self::GuardedLaws(laws) => {
                if laws.is_empty() {
                    return Err(ContractValidationError::EmptyLaws);
                }
                for law in laws {
                    law.validate().map_err(ContractValidationError::LawError)?;
                }
                Ok(())
            }
            Self::Opaque { reason } | Self::NoTransform { reason } => {
                let trimmed = reason.trim();
                if trimmed.len() < 5 || is_placeholder_reason(trimmed) {
                    return Err(ContractValidationError::InvalidOpaqueReason(reason.clone()));
                }
                Ok(())
            }
        }
    }
}

fn is_placeholder_reason(reason: &str) -> bool {
    let mut lower = reason.to_ascii_lowercase();
    lower.retain(|c| !c.is_whitespace());
    matches!(
        lower.as_str(),
        "todo"
            | "tbd"
            | "placeholder"
            | "none"
            | "unimplemented"
            | "opaque"
            | "no-op"
            | "notransform"
            | "notransforms"
            | "notimplemented"
    ) || lower.starts_with("todo:")
        || lower.starts_with("placeholder:")
}

/// Error produced when validating a [`SemanticContractRecord`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ContractValidationError {
    /// Operation id is missing or empty.
    MissingId,
    /// Guarded laws decision contained zero laws.
    EmptyLaws,
    /// Opaque reason was too short or contained a placeholder string.
    InvalidOpaqueReason(String),
    /// A declared law failed formal evidence validation.
    LawError(LawValidationError),
    /// Signature is malformed.
    InvalidSignature(String),
}

impl core::fmt::Display for ContractValidationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingId => write!(f, "operation contract record has empty id"),
            Self::EmptyLaws => write!(f, "guarded laws decision contains zero laws"),
            Self::InvalidOpaqueReason(reason) => {
                write!(f, "invalid or placeholder opaque reason: `{reason}`")
            }
            Self::LawError(err) => write!(f, "law validation error: {err}"),
            Self::InvalidSignature(sig) => write!(f, "invalid signature: {sig}"),
        }
    }
}

/// One canonical semantic contract record defining exact signature, effects, aliasing,
/// shape and index relations, numerical behavior, determinism, range preconditions,
/// resource bounds, and either proof-producing guarded laws or an explicit opaque/no-transform decision.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SemanticContractRecord {
    /// Stable operation identity.
    pub id: String,
    /// Exact typed signature.
    pub signature: Option<OpSignature>,
    /// Memory effect classification.
    pub effects: MemoryEffect,
    /// Memory aliasing contract.
    pub aliasing: AliasingContract,
    /// Shape and index relationships.
    pub shape_index: ShapeIndexContract,
    /// Numerical behavior model.
    pub numerical: NumericBehavior,
    /// Execution determinism classification.
    pub determinism: DeterminismClass,
    /// Range preconditions.
    pub range_preconditions: RangeContract,
    /// Resource allocation bounds.
    pub resource_bounds: ResourceBoundsContract,
    /// Exhaustive transform decision.
    pub decision: TransformDecision,
}

impl SemanticContractRecord {
    /// Construct a contract record with an explicit opaque / no-transform decision.
    #[must_use]
    pub fn opaque(id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            signature: None,
            effects: MemoryEffect::Pure,
            aliasing: AliasingContract::Disjoint,
            shape_index: ShapeIndexContract::elementwise(),
            numerical: NumericBehavior::Exact,
            determinism: DeterminismClass::Deterministic,
            range_preconditions: RangeContract::unbounded(),
            resource_bounds: ResourceBoundsContract::Unbounded,
            decision: TransformDecision::Opaque {
                reason: reason.into(),
            },
        }
    }

    /// Construct a contract record with explicit guarded laws.
    #[must_use]
    pub fn with_laws(id: impl Into<String>, laws: Vec<GuardedLaw>) -> Self {
        Self {
            id: id.into(),
            signature: None,
            effects: MemoryEffect::Pure,
            aliasing: AliasingContract::Disjoint,
            shape_index: ShapeIndexContract::elementwise(),
            numerical: NumericBehavior::Exact,
            determinism: DeterminismClass::Deterministic,
            range_preconditions: RangeContract::unbounded(),
            resource_bounds: ResourceBoundsContract::Unbounded,
            decision: TransformDecision::GuardedLaws(laws),
        }
    }

    /// Validate the contract record: ensures identity, decision, and laws are fully valid.
    ///
    /// # Errors
    /// Returns [`ContractValidationError`] if any component fails validation.
    pub fn validate(&self) -> Result<(), ContractValidationError> {
        if self.id.trim().is_empty() {
            return Err(ContractValidationError::MissingId);
        }
        self.decision.validate()
    }
}
