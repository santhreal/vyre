//! Canonical unified semantic operation record.

use crate::dialect_lookup::Signature;
use crate::geometry::{GeometryConstraintConflict, GeometryRequirements};
use crate::ir::Program;
use crate::numeric::NumericContract;
use crate::operation::records::{ConformanceProvider, LoweringProvider, OperationFixtures, SemanticDescriptor};
use crate::operation::registry::OperationRegistry;
use crate::operation::semantics::{OperationEffects, OperationTier};
use crate::program_caps::{scan as scan_capabilities, RequiredCapabilities};

/// One immutable semantic record used by validation, inlining, conformance,
/// documentation, and target-facet joins.
#[derive(Clone, Copy, Debug)]
pub struct SemanticOperation {
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
    /// Optional neutral program builder.
    pub build: Option<fn() -> Program>,
    /// Deterministic fixture inputs.
    pub test_inputs: Option<OperationFixtures>,
    /// Deterministic fixture outputs.
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
    /// Optional explicit opaque / no-transform reason.
    pub opaque_reason: Option<&'static str>,
}

impl SemanticOperation {
    /// Build the canonical program and stamp its stable operation identity.
    #[must_use]
    pub fn program(self) -> Option<Program> {
        self.build.map(|build| build().with_entry_op_id(self.id))
    }

    /// Derive the effective neutral schedule constraints from the recorded
    /// decision and the canonical program.
    ///
    /// # Errors
    ///
    /// Returns a stable conflict when the recorded decision contradicts semantics.
    pub fn schedule_constraints(self) -> Result<GeometryRequirements, GeometryConstraintConflict> {
        match self.program() {
            Some(program) => self
                .geometry_requirements
                .compose(GeometryRequirements::from_program(&program)?),
            None => Ok(self.geometry_requirements),
        }
    }

    /// Derive target-neutral capability requirements transitively over `Expr::Call`.
    #[must_use]
    pub fn required_capabilities(self) -> Option<RequiredCapabilities> {
        OperationRegistry::global()
            .transitive_capabilities(self.id)
            .or_else(|| self.direct_required_capabilities())
    }

    /// Direct (local) capability requirements without call-graph transitive propagation.
    #[must_use]
    pub fn direct_required_capabilities(self) -> Option<RequiredCapabilities> {
        self.explicit_capabilities
            .or_else(|| self.program().map(|program| scan_capabilities(&program)))
    }

    /// Derive target-neutral effects transitively over `Expr::Call`.
    #[must_use]
    pub fn effects(self) -> Option<OperationEffects> {
        OperationRegistry::global()
            .transitive_effects(self.id)
            .or_else(|| self.direct_effects())
    }

    /// Direct (local) memory and synchronization effects without call-graph transitive propagation.
    #[must_use]
    pub fn direct_effects(self) -> Option<OperationEffects> {
        self.explicit_effects.or_else(|| {
            self.program()
                .map(|program| OperationEffects::from_program(&program))
        })
    }

    /// Direct callees invoked by this operation via `Expr::Call`.
    #[must_use]
    pub fn callees(self) -> Option<&'static [&'static str]> {
        OperationRegistry::global().callees(self.id)
    }

    /// Return the coarse category.
    #[must_use]
    pub const fn category(self) -> Option<&'static str> {
        self.category
    }

    /// Return the explicit opaque / no-transform reason, if one was recorded.
    #[must_use]
    pub const fn opaque_reason(self) -> Option<&'static str> {
        self.opaque_reason
    }

    /// Whether this operation has a recorded transform decision (either laws or an explicit opaque decision).
    #[must_use]
    pub fn has_transform_decision(self) -> bool {
        !self.laws.is_empty() || self.opaque_reason.is_some()
    }

    /// Return the permitted f32 drift in ULPs.
    #[must_use]
    pub fn ulp_budget(&self) -> Option<u32> {
        self.numeric.ulp_budget()
    }

    /// Effective composite semantic version including local schema version, transitive effects,
    /// capabilities, and the global call-graph closure identity.
    #[must_use]
    pub fn composite_version(self) -> u64 {
        OperationRegistry::global()
            .composite_version(self.id)
            .unwrap_or_else(|| {
                let mut hasher = blake3::Hasher::new();
                hasher.update(b"vyre-foundation::semantic_operation::composite_version::v1\n");
                hasher.update(self.id.as_bytes());
                hasher.update(&self.semantic_version.to_le_bytes());
                if let Some(eff) = self.direct_effects() {
                    hasher.update(&[
                        eff.reads as u8,
                        eff.writes as u8,
                        eff.atomics as u8,
                        eff.synchronizes as u8,
                    ]);
                }
                if let Some(caps) = self.direct_required_capabilities() {
                    hasher.update(&[
                        caps.subgroup_ops as u8,
                        caps.f16 as u8,
                        caps.bf16 as u8,
                        caps.f64 as u8,
                        caps.async_dispatch as u8,
                        caps.indirect_dispatch as u8,
                        caps.tensor_ops as u8,
                        caps.trap as u8,
                        caps.distributed_collectives as u8,
                    ]);
                    hasher.update(&caps.static_storage_bytes.to_le_bytes());
                }
                let hash_bytes = hasher.finalize();
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&hash_bytes.as_bytes()[..8]);
                u64::from_le_bytes(bytes)
            })
    }

    /// Extract the production semantic descriptor from this operation.
    #[must_use]
    pub fn descriptor(self) -> SemanticDescriptor {
        SemanticDescriptor {
            id: self.id,
            semantic_version: self.semantic_version,
            signature: self.signature,
            tier: self.tier,
            category: self.category,
            laws: self.laws,
            numeric: self.numeric,
            geometry_requirements: self.geometry_requirements,
            explicit_effects: self.explicit_effects,
            explicit_capabilities: self.explicit_capabilities,
        }
    }

    /// Extract the lowering provider from this operation.
    #[must_use]
    pub fn lowering_provider(self) -> LoweringProvider {
        LoweringProvider {
            id: self.id,
            build: self.build,
        }
    }

    /// Extract the conformance case provider from this operation.
    #[must_use]
    pub fn conformance_provider(self) -> ConformanceProvider {
        ConformanceProvider {
            id: self.id,
            test_inputs: self.test_inputs,
            expected_output: self.expected_output,
        }
    }
}
