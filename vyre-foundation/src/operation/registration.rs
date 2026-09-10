//! Registration metadata and legacy/in-tree operation submission builder.

use std::panic::Location;

use crate::dialect_lookup::Signature;
use crate::geometry::{GeometryConstraintConflict, GeometryRequirements};
use crate::ir::Program;
use crate::numeric::NumericContract;
use crate::operation::records::{
    AbsenceDecision, ConformanceProvider, ContractProvider, LoweringProvider, OperationFixtures,
    SemanticDescriptor,
};
use crate::operation::semantic_op::{
    canonical_program, composed_schedule_constraints, conformance_provider_of, local_capabilities,
    local_effects, lowering_provider_of, SemanticOperation,
};
use crate::operation::semantics::{OperationEffects, OperationTier};
use crate::program_caps::RequiredCapabilities;

/// One semantic operation identity and all target-neutral catalog policy.
pub struct OperationRegistration {
    /// Stable operation identifier.
    pub id: &'static str,
    /// Semantic schema version.
    pub semantic_version: u32,
    /// Optional explicitly declared signature. When absent, [`Self::program`] is authoritative.
    pub signature: Option<Signature>,
    /// Semantic tier.
    pub tier: OperationTier,
    /// Coarse taxonomy category.
    pub category: Option<&'static str>,
    /// Optional neutral program builder.
    pub build: Option<fn() -> Program>,
    /// Deterministic fixture inputs.
    pub test_inputs: Option<OperationFixtures>,
    /// Optional deterministic fixture outputs or reference-oracle projection.
    pub expected_output: Option<OperationFixtures>,
    /// Algebraic or semantic law identifiers.
    pub laws: &'static [&'static str],
    /// What the result is allowed to be.
    pub numeric: NumericContract,
    /// Recorded target-neutral schedule constraints.
    pub geometry_requirements: GeometryRequirements,
    /// Source file that owns the registration.
    pub source_file: &'static str,
    /// Optional explicit closed effects.
    pub explicit_effects: Option<OperationEffects>,
    /// Optional explicit closed capabilities.
    pub explicit_capabilities: Option<RequiredCapabilities>,
    /// Recorded decision when the operation declares no unconditional law.
    pub absence: Option<AbsenceDecision>,
}

impl OperationRegistration {
    /// Construct an explicitly unconstrained neutral operation registration.
    #[must_use]
    #[track_caller]
    pub const fn new_unconstrained(
        id: &'static str,
        tier: OperationTier,
        build: Option<fn() -> Program>,
        test_inputs: Option<OperationFixtures>,
        expected_output: Option<OperationFixtures>,
    ) -> Self {
        Self {
            id,
            semantic_version: 1,
            signature: None,
            tier,
            category: None,
            build,
            test_inputs,
            expected_output,
            laws: &[],
            numeric: NumericContract::EXACT,
            geometry_requirements: GeometryRequirements::agnostic(),
            source_file: Location::caller().file(),
            explicit_effects: None,
            explicit_capabilities: None,
            absence: None,
        }
    }

    /// Construct an explicitly unconstrained Category A composition registration.
    #[must_use]
    #[track_caller]
    pub const fn library_unconstrained(
        id: &'static str,
        build: fn() -> Program,
        test_inputs: Option<OperationFixtures>,
        expected_output: Option<OperationFixtures>,
    ) -> Self {
        Self::new_unconstrained(
            id,
            OperationTier::Library,
            Some(build),
            test_inputs,
            expected_output,
        )
    }

    /// Construct an explicitly unconstrained Category C intrinsic registration.
    #[must_use]
    #[track_caller]
    pub const fn intrinsic_unconstrained(
        id: &'static str,
        signature: Signature,
        build: Option<fn() -> Program>,
        test_inputs: Option<OperationFixtures>,
        expected_output: Option<OperationFixtures>,
    ) -> Self {
        Self::new_unconstrained(
            id,
            OperationTier::Intrinsic,
            build,
            test_inputs,
            expected_output,
        )
        .with_signature(signature)
        .with_category("hardware")
    }

    /// Construct an explicitly unconstrained Category C primitive registration.
    #[must_use]
    #[track_caller]
    pub const fn primitive_unconstrained(
        id: &'static str,
        build: fn() -> Program,
        test_inputs: Option<OperationFixtures>,
        expected_output: Option<OperationFixtures>,
    ) -> Self {
        Self::new_unconstrained(
            id,
            OperationTier::Intrinsic,
            Some(build),
            test_inputs,
            expected_output,
        )
    }

    /// Attach an explicit signature.
    #[must_use]
    pub const fn with_signature(mut self, signature: Signature) -> Self {
        self.signature = Some(signature);
        self
    }

    /// Attach a coarse category.
    #[must_use]
    pub const fn with_category(mut self, category: &'static str) -> Self {
        self.category = Some(category);
        self
    }

    /// Attach semantic law identifiers.
    #[must_use]
    pub const fn with_laws(mut self, laws: &'static [&'static str]) -> Self {
        self.laws = laws;
        self
    }

    /// Record that no law family produces a verdict against this operation, so
    /// its algebraic behavior is not characterized.
    #[must_use]
    pub const fn with_uncharacterized(mut self) -> Self {
        self.absence = Some(AbsenceDecision::Uncharacterized);
        self
    }

    /// Record that every law family whose witness this operation's shape admits
    /// is refuted, so no rewrite of it is legal.
    #[must_use]
    pub const fn with_no_legal_rewrite(mut self) -> Self {
        self.absence = Some(AbsenceDecision::NoLegalRewrite);
        self
    }

    /// Return the recorded law-absence decision, if any.
    #[must_use]
    pub const fn absence(&self) -> Option<AbsenceDecision> {
        self.absence
    }

    /// Whether this operation records a transform decision: either declared
    /// laws or an explicit law-absence decision.
    #[must_use]
    pub const fn has_transform_decision(&self) -> bool {
        !self.laws.is_empty() || self.absence.is_some()
    }

    /// Attach the source file that owns this registration.
    #[must_use]
    pub const fn with_source_file(mut self, source_file: &'static str) -> Self {
        self.source_file = source_file;
        self
    }

    /// Return the coarse category.
    #[must_use]
    pub const fn category(&self) -> Option<&'static str> {
        self.category
    }

    /// Return the permitted f32 drift in ULPs.
    #[must_use]
    pub fn ulp_budget(&self) -> Option<u32> {
        self.numeric.ulp_budget()
    }

    /// State what the result is allowed to be.
    #[must_use]
    pub const fn with_numeric(mut self, numeric: NumericContract) -> Self {
        self.numeric = numeric;
        self
    }

    /// Attach target-neutral execution geometry requirements.
    #[must_use]
    pub const fn with_geometry_requirements(mut self, requirements: GeometryRequirements) -> Self {
        self.geometry_requirements = requirements;
        self
    }

    /// Attach explicit closed effects.
    #[must_use]
    pub const fn with_explicit_effects(mut self, effects: OperationEffects) -> Self {
        self.explicit_effects = Some(effects);
        self
    }

    /// Attach explicit closed capabilities.
    #[must_use]
    pub const fn with_explicit_capabilities(mut self, capabilities: RequiredCapabilities) -> Self {
        self.explicit_capabilities = Some(capabilities);
        self
    }

    /// Return the recorded schedule-constraint decision.
    #[must_use]
    pub const fn declared_schedule_constraints(&self) -> GeometryRequirements {
        self.geometry_requirements
    }

    /// Build the canonical program and stamp its stable operation identity.
    #[must_use]
    pub fn program(&self) -> Option<Program> {
        canonical_program(self.id, self.build)
    }

    /// Derive the effective neutral schedule constraints from the recorded
    /// decision and canonical program semantics.
    ///
    /// # Errors
    ///
    /// Returns a stable conflict when the recorded decision contradicts semantics.
    pub fn schedule_constraints(&self) -> Result<GeometryRequirements, GeometryConstraintConflict> {
        composed_schedule_constraints(self.geometry_requirements, self.program().as_ref())
    }

    /// Direct (local) required capabilities without call-graph transitive propagation.
    #[must_use]
    pub fn direct_required_capabilities(&self) -> Option<RequiredCapabilities> {
        local_capabilities(self.explicit_capabilities, self.program().as_ref())
    }

    /// Direct (local) memory and synchronization effects without call-graph transitive propagation.
    #[must_use]
    pub fn direct_effects(&self) -> Option<OperationEffects> {
        local_effects(self.explicit_effects, self.program().as_ref())
    }

    /// Derive target-neutral capability requirements from the canonical program.
    #[must_use]
    pub fn required_capabilities(&self) -> Option<RequiredCapabilities> {
        self.direct_required_capabilities()
    }

    /// Derive target-neutral effects from the canonical program.
    #[must_use]
    pub fn effects(&self) -> Option<OperationEffects> {
        self.direct_effects()
    }

    /// Extract the production semantic descriptor from this registration.
    #[must_use]
    pub fn descriptor(&'static self) -> SemanticDescriptor {
        SemanticOperation::from(self).descriptor()
    }

    /// Extract the lowering provider from this registration.
    #[must_use]
    pub fn lowering_provider(&self) -> LoweringProvider {
        lowering_provider_of(self.id, self.build)
    }

    /// Extract the conformance case provider from this registration.
    #[must_use]
    pub fn conformance_provider(&self) -> ConformanceProvider {
        conformance_provider_of(self.id, self.test_inputs, self.expected_output)
    }

    /// Extract the contract provider from this registration.
    #[must_use]
    pub fn contract_provider(&'static self) -> ContractProvider {
        SemanticOperation::from(self).contract_provider()
    }

    /// Construct the canonical semantic contract record.
    #[must_use]
    pub fn contract_record(&self) -> vyre_spec::SemanticContractRecord {
        super::semantic_op::build_contract_record(&super::semantic_op::ContractFacts {
            id: self.id,
            signature: self.signature.as_ref(),
            effects: self.direct_effects(),
            capabilities: self.direct_required_capabilities(),
            numeric: self.numeric,
            laws: self.laws,
            absence: self.absence,
            program: self.program(),
        })
    }
}

impl From<&'static OperationRegistration> for SemanticOperation {
    fn from(registration: &'static OperationRegistration) -> Self {
        Self {
            id: registration.id,
            semantic_version: registration.semantic_version,
            signature: registration.signature.as_ref(),
            tier: registration.tier,
            category: registration.category,
            build: registration.build,
            test_inputs: registration.test_inputs,
            expected_output: registration.expected_output,
            laws: registration.laws,
            numeric: registration.numeric,
            geometry_requirements: registration.geometry_requirements,
            source_file: registration.source_file,
            explicit_effects: registration.explicit_effects,
            explicit_capabilities: registration.explicit_capabilities,
            absence: registration.absence,
        }
    }
}

inventory::collect!(OperationRegistration);
