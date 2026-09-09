//! Canonical unified semantic operation record.

use crate::dialect_lookup::Signature;
use crate::geometry::{GeometryConstraintConflict, GeometryRequirements};
use crate::ir::Program;
use crate::numeric::NumericContract;
use crate::operation::records::{
    ConformanceProvider, ContractProvider, LoweringProvider, OperationFixtures, SemanticDescriptor,
};
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
            opaque_reason: self.opaque_reason,
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

    /// Extract the contract provider from this operation.
    #[must_use]
    pub fn contract_provider(self) -> ContractProvider {
        ContractProvider {
            id: self.id,
            contract: None,
        }
    }

    /// Construct the canonical semantic contract record.
    #[must_use]
    pub fn contract_record(self) -> vyre_spec::SemanticContractRecord {
        build_contract_record(
            self.id,
            self.signature,
            self.explicit_effects,
            self.numeric,
            self.laws,
            self.opaque_reason,
        )
    }
}

fn parse_datatype(ty: &str) -> vyre_spec::DataType {
    match ty {
        "u8" => vyre_spec::DataType::U8,
        "u16" => vyre_spec::DataType::U16,
        "u32" => vyre_spec::DataType::U32,
        "u64" => vyre_spec::DataType::U64,
        "i8" => vyre_spec::DataType::I8,
        "i16" => vyre_spec::DataType::I16,
        "i32" => vyre_spec::DataType::I32,
        "i64" => vyre_spec::DataType::I64,
        "f16" => vyre_spec::DataType::F16,
        "bf16" => vyre_spec::DataType::BF16,
        "f32" => vyre_spec::DataType::F32,
        "f64" => vyre_spec::DataType::F64,
        "bool" => vyre_spec::DataType::Bool,
        "bytes" => vyre_spec::DataType::Bytes,
        _ => vyre_spec::DataType::U32,
    }
}

fn signature_to_contract_sig(sig: Option<&Signature>) -> Option<vyre_spec::OpSignature> {
    sig.map(|s| {
        let inputs: Vec<vyre_spec::DataType> =
            s.inputs.iter().map(|p| parse_datatype(p.ty)).collect();
        let output = s
            .outputs
            .first()
            .map(|p| parse_datatype(p.ty))
            .unwrap_or(vyre_spec::DataType::U32);
        let input_params = Some(
            s.inputs
                .iter()
                .map(|p| vyre_spec::SignatureParam {
                    name: p.name.to_string(),
                    ty: parse_datatype(p.ty),
                    metadata: None,
                })
                .collect(),
        );
        let output_params = Some(
            s.outputs
                .iter()
                .map(|p| vyre_spec::SignatureParam {
                    name: p.name.to_string(),
                    ty: parse_datatype(p.ty),
                    metadata: None,
                })
                .collect(),
        );
        vyre_spec::OpSignature {
            inputs,
            output,
            input_params,
            output_params,
            contract: None,
        }
    })
}

pub(crate) fn build_contract_record(
    id: &'static str,
    signature: Option<&Signature>,
    explicit_effects: Option<OperationEffects>,
    numeric: NumericContract,
    laws: &'static [&'static str],
    opaque_reason: Option<&'static str>,
) -> vyre_spec::SemanticContractRecord {
    let sig = signature_to_contract_sig(signature);

    let eff = if let Some(e) = explicit_effects {
        if !e.reads && !e.writes && !e.atomics && !e.synchronizes {
            vyre_spec::MemoryEffect::Pure
        } else if e.atomics {
            vyre_spec::MemoryEffect::Atomic
        } else if e.synchronizes {
            vyre_spec::MemoryEffect::Synchronizing
        } else if e.writes {
            vyre_spec::MemoryEffect::Write
        } else {
            vyre_spec::MemoryEffect::Read
        }
    } else {
        vyre_spec::MemoryEffect::Pure
    };

    let num = match numeric.ulp_budget() {
        Some(0) | None => vyre_spec::NumericBehavior::Exact,
        Some(ulps) => vyre_spec::NumericBehavior::IeeeFloatingPoint {
            ulp_budget: ulps,
            nan_behavior: vyre_spec::NanBehavior::CanonicalQuietNan,
            infinity_behavior: vyre_spec::InfinityBehavior::SignedInfinity,
        },
    };

    let decision = if !laws.is_empty() {
        let mut guarded = Vec::new();
        for &law_name in laws {
            let law = match law_name {
                "commutative" => vyre_spec::AlgebraicLaw::Commutative,
                "associative" => vyre_spec::AlgebraicLaw::Associative,
                "identity" => vyre_spec::AlgebraicLaw::Identity { element: 0 },
                "left-identity" => vyre_spec::AlgebraicLaw::LeftIdentity { element: 0 },
                "right-identity" => vyre_spec::AlgebraicLaw::RightIdentity { element: 0 },
                "self-inverse" => vyre_spec::AlgebraicLaw::SelfInverse { result: 0 },
                "idempotent" => vyre_spec::AlgebraicLaw::Idempotent,
                "absorbing" => vyre_spec::AlgebraicLaw::Absorbing { element: 0 },
                "left-absorbing" => vyre_spec::AlgebraicLaw::LeftAbsorbing { element: 0 },
                "right-absorbing" => vyre_spec::AlgebraicLaw::RightAbsorbing { element: 0 },
                "involution" => vyre_spec::AlgebraicLaw::Involution,
                "de-morgan" => vyre_spec::AlgebraicLaw::DeMorgan {
                    inner_op: "and",
                    dual_op: "or",
                },
                "monotone" => vyre_spec::AlgebraicLaw::Monotone,
                "monotonic" => vyre_spec::AlgebraicLaw::Monotonic {
                    direction: vyre_spec::MonotonicDirection::NonDecreasing,
                },
                "bounded" => vyre_spec::AlgebraicLaw::Bounded {
                    lo: 0,
                    hi: u32::MAX,
                },
                "complement" => vyre_spec::AlgebraicLaw::Complement {
                    complement_op: "not",
                    universe: u32::MAX,
                },
                "distributive" => vyre_spec::AlgebraicLaw::DistributiveOver { over_op: "add" },
                "lattice-absorption" => {
                    vyre_spec::AlgebraicLaw::LatticeAbsorption { dual_op: "min" }
                }
                "inverse-of" => vyre_spec::AlgebraicLaw::InverseOf { op: "add" },
                "trichotomy" => vyre_spec::AlgebraicLaw::Trichotomy {
                    less_op: "lt",
                    equal_op: "eq",
                    greater_op: "gt",
                },
                "zero-product" => vyre_spec::AlgebraicLaw::ZeroProduct { holds: true },
                "categorical-identity" => vyre_spec::AlgebraicLaw::CategoricalIdentity,
                "categorical-associative" => vyre_spec::AlgebraicLaw::CategoricalAssociative,
                custom => vyre_spec::AlgebraicLaw::Custom {
                    name: custom,
                    description: "registered law",
                    arity: 1,
                    check: |_, _| true,
                },
            };
            guarded.push(vyre_spec::GuardedLaw::unconditional(law));
        }
        vyre_spec::TransformDecision::GuardedLaws(guarded)
    } else if let Some(reason) = opaque_reason {
        if reason.starts_with("no-transform:") || reason.starts_with("notransform:") {
            vyre_spec::TransformDecision::NoTransform {
                reason: reason.to_string(),
            }
        } else {
            vyre_spec::TransformDecision::Opaque {
                reason: reason.to_string(),
            }
        }
    } else {
        vyre_spec::TransformDecision::NotRecorded
    };

    vyre_spec::SemanticContractRecord {
        id: id.to_string(),
        signature: sig,
        effects: eff,
        aliasing: vyre_spec::AliasingContract::Disjoint,
        shape_index: vyre_spec::ShapeIndexContract::elementwise(),
        numerical: num,
        determinism: vyre_spec::DeterminismClass::Deterministic,
        range_preconditions: vyre_spec::RangeContract::unbounded(),
        resource_bounds: vyre_spec::ResourceBoundsContract::Unbounded,
        decision,
    }
}
