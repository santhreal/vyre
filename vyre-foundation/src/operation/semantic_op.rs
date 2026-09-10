//! Canonical unified semantic operation record.

use crate::dialect_lookup::Signature;
use crate::geometry::{GeometryConstraintConflict, GeometryRequirements};
use crate::ir::{BufferAccess, Program};
use crate::numeric::NumericContract;
use crate::operation::records::{
    AbsenceDecision, ConformanceProvider, ContractProvider, LoweringProvider, OperationFixtures,
    SemanticDescriptor,
};
use crate::operation::registry::OperationRegistry;
use crate::operation::semantics::{truncate_to_u64, OperationEffects, OperationTier};
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
    /// Recorded decision when the operation declares no unconditional law.
    pub absence: Option<AbsenceDecision>,
}

impl SemanticOperation {
    /// Build the canonical program and stamp its stable operation identity.
    #[must_use]
    pub fn program(self) -> Option<Program> {
        canonical_program(self.id, self.build)
    }

    /// Derive the effective neutral schedule constraints from the recorded
    /// decision and the canonical program.
    ///
    /// # Errors
    ///
    /// Returns a stable conflict when the recorded decision contradicts semantics.
    pub fn schedule_constraints(self) -> Result<GeometryRequirements, GeometryConstraintConflict> {
        composed_schedule_constraints(self.geometry_requirements, self.program().as_ref())
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
        local_capabilities(self.explicit_capabilities, self.program().as_ref())
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
        local_effects(self.explicit_effects, self.program().as_ref())
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

    /// Return the recorded law-absence decision, if one was recorded.
    #[must_use]
    pub const fn absence(self) -> Option<AbsenceDecision> {
        self.absence
    }

    /// Whether this operation records a transform decision: either declared
    /// laws or an explicit law-absence decision.
    #[must_use]
    pub fn has_transform_decision(self) -> bool {
        !self.laws.is_empty() || self.absence.is_some()
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
                    eff.hash_into(&mut hasher);
                }
                if let Some(caps) = self.direct_required_capabilities() {
                    caps.hash_into(&mut hasher);
                }
                truncate_to_u64(&hasher)
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
            absence: self.absence,
        }
    }

    /// Extract the lowering provider from this operation.
    #[must_use]
    pub fn lowering_provider(self) -> LoweringProvider {
        lowering_provider_of(self.id, self.build)
    }

    /// Extract the conformance case provider from this operation.
    #[must_use]
    pub fn conformance_provider(self) -> ConformanceProvider {
        conformance_provider_of(self.id, self.test_inputs, self.expected_output)
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
        build_contract_record(&ContractFacts {
            id: self.id,
            signature: self.signature,
            effects: self.direct_effects(),
            capabilities: self.direct_required_capabilities(),
            numeric: self.numeric,
            laws: self.laws,
            absence: self.absence,
            program: self.program(),
        })
    }
}

/// Build the canonical program for `id` and stamp its stable identity.
///
/// A registration and the semantic record derived from it must agree on what
/// the canonical program is, so both read it from here.
pub(super) fn canonical_program(id: &'static str, build: Option<fn() -> Program>) -> Option<Program> {
    build.map(|build| build().with_entry_op_id(id))
}

/// Compose the recorded schedule decision with what the program requires.
///
/// # Errors
///
/// Returns a stable conflict when the recorded decision contradicts semantics.
pub(super) fn composed_schedule_constraints(
    declared: GeometryRequirements,
    program: Option<&Program>,
) -> Result<GeometryRequirements, GeometryConstraintConflict> {
    match program {
        Some(program) => declared.compose(GeometryRequirements::from_program(program)?),
        None => Ok(declared),
    }
}

/// Local capability requirements: the explicit record, else a program scan.
pub(super) fn local_capabilities(
    explicit: Option<RequiredCapabilities>,
    program: Option<&Program>,
) -> Option<RequiredCapabilities> {
    explicit.or_else(|| program.map(scan_capabilities))
}

/// Local memory and synchronization effects: the explicit record, else what
/// the program does.
pub(super) fn local_effects(
    explicit: Option<OperationEffects>,
    program: Option<&Program>,
) -> Option<OperationEffects> {
    explicit.or_else(|| program.map(OperationEffects::from_program))
}

/// The lowering provider for one operation identity.
pub(super) fn lowering_provider_of(
    id: &'static str,
    build: Option<fn() -> Program>,
) -> LoweringProvider {
    LoweringProvider { id, build }
}

/// The conformance case provider for one operation identity.
pub(super) fn conformance_provider_of(
    id: &'static str,
    test_inputs: Option<OperationFixtures>,
    expected_output: Option<OperationFixtures>,
) -> ConformanceProvider {
    ConformanceProvider {
        id,
        test_inputs,
        expected_output,
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

/// Every fact a contract record is derived from.
///
/// A record built from a shorter argument list left five facets at a
/// declaration-order default, so `aliasing`, `shape_index`, `determinism`,
/// `range_preconditions` and `resource_bounds` stated the same value for every
/// operation in the catalog and carried no information. Each is derived here
/// from the canonical program, the resolved effects, the resolved capabilities
/// and the numeric contract.
pub(crate) struct ContractFacts<'a> {
    /// Stable operation identity.
    pub id: &'static str,
    /// Declared callable signature.
    pub signature: Option<&'a Signature>,
    /// Resolved effects: explicit when declared, program-derived otherwise.
    pub effects: Option<OperationEffects>,
    /// Resolved capabilities: explicit when declared, program-derived otherwise.
    pub capabilities: Option<RequiredCapabilities>,
    /// What the result is allowed to be.
    pub numeric: NumericContract,
    /// Declared law names.
    pub laws: &'static [&'static str],
    /// Recorded law-absence decision.
    pub absence: Option<AbsenceDecision>,
    /// Canonical program, when the registration builds one.
    pub program: Option<Program>,
}

/// Classify resolved effects into the closed memory-effect vocabulary.
fn derive_effects(effects: Option<OperationEffects>) -> vyre_spec::MemoryEffect {
    let Some(e) = effects else {
        return vyre_spec::MemoryEffect::Pure;
    };
    if e.atomics {
        vyre_spec::MemoryEffect::Atomic
    } else if e.synchronizes {
        vyre_spec::MemoryEffect::Synchronizing
    } else if e.writes {
        vyre_spec::MemoryEffect::Write
    } else if e.reads {
        vyre_spec::MemoryEffect::Read
    } else {
        vyre_spec::MemoryEffect::Pure
    }
}

/// Derive the aliasing contract from the declared buffer access modes.
///
/// A `ReadWrite` buffer is an in-place update, so its operand and its result
/// are required to be the same allocation. Two or more read-only inputs may
/// overlap without changing the result. Anything else requires disjointness.
fn derive_aliasing(program: Option<&Program>) -> vyre_spec::AliasingContract {
    let Some(program) = program else {
        return vyre_spec::AliasingContract::Disjoint;
    };
    let mut read_only = 0usize;
    for decl in program.buffers() {
        match decl.access {
            BufferAccess::ReadWrite => return vyre_spec::AliasingContract::MustAlias,
            BufferAccess::ReadOnly => read_only += 1,
            _ => {}
        }
    }
    if read_only >= 2 {
        vyre_spec::AliasingContract::ReadSharingOnly
    } else {
        vyre_spec::AliasingContract::Disjoint
    }
}

/// Derive the shape and index contract from declared element counts.
///
/// A storage buffer declares `count == 0` because its extent is a runtime
/// binding, which is exactly `Dynamic`. When every participating buffer states
/// a static extent the output-to-input element ratio separates elementwise,
/// contracting and expanding.
fn derive_shape_index(program: Option<&Program>) -> vyre_spec::ShapeIndexContract {
    let Some(program) = program else {
        return vyre_spec::ShapeIndexContract::agnostic();
    };
    let mut input_elements = 0u64;
    let mut output_elements = 0u64;
    let mut dynamic = false;
    let mut any_output = false;
    for decl in program.buffers() {
        if matches!(decl.access, BufferAccess::Workgroup) {
            continue;
        }
        let count = u64::from(decl.count);
        if count == 0 {
            dynamic = true;
        }
        let is_output = decl.is_output
            || decl.pipeline_live_out
            || matches!(
                decl.access,
                BufferAccess::WriteOnly | BufferAccess::ReadWrite
            );
        if is_output {
            any_output = true;
            output_elements = output_elements.saturating_add(count);
        } else {
            input_elements = input_elements.saturating_add(count);
        }
    }
    let relation = if dynamic || !any_output {
        vyre_spec::ShapeIndexRelation::Dynamic
    } else if output_elements == input_elements {
        vyre_spec::ShapeIndexRelation::Elementwise
    } else if output_elements < input_elements {
        vyre_spec::ShapeIndexRelation::Contracting
    } else {
        vyre_spec::ShapeIndexRelation::Expanding
    };
    vyre_spec::ShapeIndexContract {
        relation,
        rank_preserving: matches!(relation, vyre_spec::ShapeIndexRelation::Elementwise),
        dimension_invariants: Vec::new(),
    }
}

/// Derive the determinism class from rounding budget, atomics, and laws.
///
/// An atomic read-modify-write leaves the combine order to the schedule, so the
/// result is order-independent only when the operation records both
/// commutativity and associativity. This is where a law decision reaches a
/// facet other than itself.
fn derive_determinism(
    numeric: NumericContract,
    effects: Option<OperationEffects>,
    families: &[vyre_spec::LawFamily],
) -> vyre_spec::DeterminismClass {
    let atomics = effects.is_some_and(|e| e.atomics);
    let order_independent = families.contains(&vyre_spec::LawFamily::Commutative)
        && families.contains(&vyre_spec::LawFamily::Associative);
    if atomics && !order_independent {
        return vyre_spec::DeterminismClass::NonDeterministic;
    }
    match numeric.ulp_budget() {
        Some(0) | None => vyre_spec::DeterminismClass::Deterministic,
        Some(_) => vyre_spec::DeterminismClass::DeterministicModuloRounding,
    }
}

/// Derive the resource bounds contract from the resolved capability record.
fn derive_resource_bounds(
    capabilities: Option<RequiredCapabilities>,
) -> vyre_spec::ResourceBoundsContract {
    match capabilities.map(|caps| caps.static_storage_bytes) {
        Some(bytes) if bytes > 0 => vyre_spec::ResourceBoundsContract::StaticMemoryBytes(
            usize::try_from(bytes).unwrap_or(usize::MAX),
        ),
        _ => vyre_spec::ResourceBoundsContract::Unbounded,
    }
}

/// Derive the numerical behavior model from the numeric contract.
fn derive_numerical(numeric: NumericContract) -> vyre_spec::NumericBehavior {
    match numeric.ulp_budget() {
        Some(0) | None => vyre_spec::NumericBehavior::Exact,
        Some(ulps) => vyre_spec::NumericBehavior::IeeeFloatingPoint {
            ulp_budget: ulps,
            nan_behavior: vyre_spec::NanBehavior::CanonicalQuietNan,
            infinity_behavior: vyre_spec::InfinityBehavior::SignedInfinity,
        },
    }
}

/// Resolve declared law names into law records, the labels no family in the
/// closed vocabulary carries, and the families that carry executable evidence.
///
/// A name in the vocabulary always produces a record, so nothing a
/// registration declared is discarded. Whether that record is a proven law is
/// a separate question its proof method answers: a family whose statement needs
/// a payload a bare name cannot carry yields `ProofMethod::None`, which
/// [`vyre_spec::TransformDecision::absence_class`] reports as
/// [`vyre_spec::AbsenceClass::LawUnrecorded`].
///
/// A name no family carries records nothing at all. Mapping it onto a custom
/// law whose check returned true is what let a label assert a property nothing
/// examined.
fn resolve_laws(
    laws: &'static [&'static str],
) -> (
    Vec<vyre_spec::GuardedLaw>,
    Vec<vyre_spec::RejectedLawLabel>,
    Vec<vyre_spec::LawFamily>,
) {
    let mut declared = Vec::new();
    let mut rejected = Vec::new();
    let mut proven = Vec::new();
    for &name in laws {
        let Some(family) = vyre_spec::LawFamily::from_name(name) else {
            rejected.push(vyre_spec::RejectedLawLabel {
                law: name.to_string(),
                missing_payload: vyre_spec::RejectedLawLabel::UNKNOWN_LAW_NAME.to_string(),
            });
            continue;
        };
        let obligation = family.obligation();
        if obligation.evidence.is_executable() {
            proven.push(family);
        }
        declared.push(obligation.guarded_law(family.representative()));
    }
    (declared, rejected, proven)
}

/// State the recorded absence in the contract vocabulary, with the operation's
/// own declared shape as its reason.
///
/// The reason is derived, never authored. A registration used to carry a
/// sentence beside the decision, and those sentences named the domain rather
/// than the operation: one string served 74 registrations. What the shape
/// states is specific to the operation and cannot be shared by writing it
/// twice. The refutation or the missing witness that justifies the decision is
/// executed evidence and is recorded per operation by the conformance
/// disposition ledger.
fn absence_shape(program: Option<&Program>) -> String {
    let Some(program) = program else {
        return "the registration builds no program, so no witness can be derived".to_string();
    };
    let mut read_only = 0usize;
    let mut written = 0usize;
    for decl in program.buffers() {
        match decl.access {
            BufferAccess::ReadOnly => read_only += 1,
            BufferAccess::WriteOnly | BufferAccess::ReadWrite => written += 1,
            _ => {}
        }
    }
    format!(
        "the declared shape is {} buffer(s), {read_only} read-only and {written} written",
        program.buffers().len()
    )
}

/// The contract decision a recorded absence states.
fn absence_decision(
    decision: AbsenceDecision,
    program: Option<&Program>,
) -> vyre_spec::TransformDecision {
    let reason = absence_shape(program);
    match decision {
        AbsenceDecision::NoLegalRewrite => vyre_spec::TransformDecision::NoTransform { reason },
        AbsenceDecision::Uncharacterized => vyre_spec::TransformDecision::Opaque { reason },
    }
}

pub(crate) fn build_contract_record(
    facts: &ContractFacts<'_>,
) -> vyre_spec::SemanticContractRecord {
    let (declared, rejected, families) = resolve_laws(facts.laws);
    let decision = if declared.is_empty() {
        match facts.absence {
            Some(decision) => absence_decision(decision, facts.program.as_ref()),
            None => vyre_spec::TransformDecision::NotRecorded,
        }
    } else {
        vyre_spec::TransformDecision::GuardedLaws(declared)
    };

    vyre_spec::SemanticContractRecord {
        id: facts.id.to_string(),
        signature: signature_to_contract_sig(facts.signature),
        effects: derive_effects(facts.effects),
        aliasing: derive_aliasing(facts.program.as_ref()),
        shape_index: derive_shape_index(facts.program.as_ref()),
        numerical: derive_numerical(facts.numeric),
        determinism: derive_determinism(facts.numeric, facts.effects, &families),
        range_preconditions: vyre_spec::RangeContract::unbounded(),
        resource_bounds: derive_resource_bounds(facts.capabilities),
        decision,
        rejected_labels: rejected,
    }
}
