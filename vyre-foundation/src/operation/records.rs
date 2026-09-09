//! Identity-joined operation records: production semantic descriptor,
//! implementation lowering provider, and conformance-case provider.

use crate::dialect_lookup::Signature;
use crate::geometry::GeometryRequirements;
use crate::ir::Program;
use crate::numeric::NumericContract;
use crate::operation::semantics::{OperationEffects, OperationTier};
use crate::program_caps::RequiredCapabilities;

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

inventory::collect!(SemanticDescriptor);
inventory::collect!(LoweringProvider);
inventory::collect!(ConformanceProvider);
