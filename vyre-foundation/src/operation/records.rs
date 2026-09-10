//! Identity-joined operation records: production semantic descriptor,
//! implementation lowering provider, and conformance-case provider.

use crate::dialect_lookup::Signature;
use crate::geometry::GeometryRequirements;
use crate::ir::Program;
use crate::numeric::NumericContract;
use crate::operation::semantics::{OperationEffects, OperationTier};
use crate::program_caps::RequiredCapabilities;

/// Why an operation records no unconditional algebraic law.
///
/// Two distinct answers were previously carried in one `Option<&str>`, so the
/// registry could not tell a recorded "no legal rewrite exists" apart from a
/// recorded "the semantics were never characterized".
///
/// The decision carries no prose. A justification written beside it is a label
/// with nothing behind it, and 272 registrations shared 29 such strings, one
/// per domain, none of which said anything about the operation it was attached
/// to. What justifies the decision is the executed evidence the conformance
/// disposition ledger records per operation, and that evidence is what the
/// suite holds this field to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AbsenceDecision {
    /// Every law family whose witness this operation's shape admits is refuted
    /// by the reference oracle, so no rewrite in the executable vocabulary
    /// preserves its observable result.
    NoLegalRewrite,
    /// No law family produces a verdict against this operation, so its
    /// algebraic behavior is not characterized.
    Uncharacterized,
}

impl AbsenceDecision {
    /// Every decision. Exhaustive by construction.
    pub const ALL: [Self; 2] = [Self::NoLegalRewrite, Self::Uncharacterized];

    /// Stable wire name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::NoLegalRewrite => "no-legal-rewrite",
            Self::Uncharacterized => "uncharacterized",
        }
    }

    /// The decision `name` spells, or `None` when nothing carries that name.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "no-legal-rewrite" => Some(Self::NoLegalRewrite),
            "uncharacterized" => Some(Self::Uncharacterized),
            _ => None,
        }
    }
}

/// Deterministic fixture input cases. One case contains declaration-ordered buffers.
pub type OperationFixtures = fn() -> Vec<Vec<Vec<u8>>>;

/// Production semantic descriptor containing typed identity, signature, tier, laws,
/// numeric contract, geometry constraints, explicit effects and capabilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SemanticDescriptor {
    /// Stable operation identifier.
    pub id: &'static str,
    /// Semantic schema version.
    pub semantic_version: u32,
    /// Explicit callable signature when the operation is used through `Expr::Call`.
    pub signature: Option<&'static Signature>,
    /// Semantic tier.
    pub tier: OperationTier,
    /// Derived dialect/category namespace.
    pub category: Option<&'static str>,
    /// Algebraic or semantic law identifiers.
    pub laws: &'static [&'static str],
    /// What the result is allowed to be.
    pub numeric: NumericContract,
    /// Recorded target-neutral schedule constraints.
    pub geometry_requirements: GeometryRequirements,
    /// Optional explicit closed effects.
    pub explicit_effects: Option<OperationEffects>,
    /// Optional explicit closed capabilities.
    pub explicit_capabilities: Option<RequiredCapabilities>,
    /// Recorded decision when the operation declares no unconditional law.
    pub absence: Option<AbsenceDecision>,
}

/// Implementation constructor and lowering provider.
#[derive(Clone, Copy, Debug)]
pub struct LoweringProvider {
    /// Stable operation identifier matching the semantic descriptor.
    pub id: &'static str,
    /// Neutral program builder for lowering/inlining.
    pub build: Option<fn() -> Program>,
}

/// Conformance-case provider available only to conformance and tooling packages.
#[derive(Clone, Copy, Debug)]
pub struct ConformanceProvider {
    /// Stable operation identifier matching the semantic descriptor.
    pub id: &'static str,
    /// Deterministic fixture inputs.
    pub test_inputs: Option<OperationFixtures>,
    /// Deterministic fixture outputs.
    pub expected_output: Option<OperationFixtures>,
}

/// Deterministic contract builder function for semantic operations.
pub type OperationContractBuilder = fn() -> vyre_spec::SemanticContractRecord;

/// Contract-record provider for semantic operations.
#[derive(Clone, Copy, Debug)]
pub struct ContractProvider {
    /// Stable operation identifier matching the semantic descriptor.
    pub id: &'static str,
    /// Contract record builder.
    pub contract: Option<OperationContractBuilder>,
}

inventory::collect!(SemanticDescriptor);
inventory::collect!(LoweringProvider);
inventory::collect!(ConformanceProvider);
inventory::collect!(ContractProvider);
