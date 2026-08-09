//! Canonical semantic operation registration and derived catalog views.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use crate::dialect_lookup::Signature;
use crate::ir::{BufferAccess, Program};
use crate::runtime::program_caps::{scan as scan_capabilities, RequiredCapabilities};

/// Deterministic fixture input cases. One case contains declaration-ordered buffers.
pub type OperationFixtures = fn() -> Vec<Vec<Vec<u8>>>;

/// Coarse semantic tier used by catalog and conformance consumers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OperationTier {
    /// Foundation IR or built-in operation.
    Foundation,
    /// Hardware-facing semantic intrinsic.
    Intrinsic,
    /// Reusable backend-neutral primitive.
    Primitive,
    /// Library composition over typed IR.
    Library,
    /// Runtime-owned semantic operation.
    Runtime,
    /// External extension operation.
    External,
}

/// Semantic memory and synchronization effects derived from an operation program.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct OperationEffects {
    /// The operation reads caller-visible storage.
    pub reads: bool,
    /// The operation writes caller-visible storage.
    pub writes: bool,
    /// The operation contains atomic memory effects.
    pub atomics: bool,
    /// The operation requires intra- or inter-workgroup synchronization.
    pub synchronizes: bool,
}

impl OperationEffects {
    /// Derive neutral effects from the canonical program declaration and statistics.
    #[must_use]
    pub fn from_program(program: &Program) -> Self {
        let mut effects = Self::default();
        for buffer in program.buffers() {
            match buffer.access() {
                BufferAccess::ReadOnly => effects.reads = true,
                BufferAccess::ReadWrite => {
                    effects.reads = true;
                    effects.writes = true;
                }
                BufferAccess::WriteOnly => effects.writes = true,
                _ => {
                    effects.reads = true;
                    effects.writes = true;
                }
            }
        }
        let stats = program.stats();
        effects.atomics = stats.atomic_op_count > 0;
        effects.synchronizes = stats.has_node_barrier() || stats.distributed_collectives();
        effects
    }
}

/// Numerical comparison policy owned by the semantic operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TolerancePolicy {
    /// Maximum accepted f32 drift measured in ULPs.
    pub f32_ulp: u32,
}

impl TolerancePolicy {
    /// Exact byte identity.
    pub const EXACT: Self = Self { f32_ulp: 0 };

    /// Construct an f32 ULP tolerance.
    #[must_use]
    pub const fn f32_ulp(maximum: u32) -> Self {
        Self { f32_ulp: maximum }
    }
}

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
    /// Numerical comparison policy.
    pub tolerance: TolerancePolicy,
}

impl OperationRegistration {
    /// Construct a neutral operation registration with exact comparison policy.
    #[must_use]
    pub const fn new(
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
            tolerance: TolerancePolicy::EXACT,
        }
    }

    /// Construct a library-composition registration.
    #[must_use]
    pub const fn library(
        id: &'static str,
        build: fn() -> Program,
        test_inputs: Option<OperationFixtures>,
        expected_output: Option<OperationFixtures>,
    ) -> Self {
        Self::new(
            id,
            OperationTier::Library,
            Some(build),
            test_inputs,
            expected_output,
        )
    }

    /// Construct a reusable primitive registration.
    #[must_use]
    pub const fn primitive(
        id: &'static str,
        build: fn() -> Program,
        test_inputs: Option<OperationFixtures>,
        expected_output: Option<OperationFixtures>,
    ) -> Self {
        Self::new(
            id,
            OperationTier::Primitive,
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

    /// Return the coarse category.
    #[must_use]
    pub const fn category(&self) -> Option<&'static str> {
        self.category
    }

    /// Return the permitted f32 drift in ULPs.
    #[must_use]
    pub const fn tolerance(&self) -> u32 {
        self.tolerance.f32_ulp
    }

    /// Attach the numerical tolerance policy.
    #[must_use]
    pub const fn with_tolerance(mut self, tolerance: TolerancePolicy) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Build the canonical program and stamp its stable operation identity.
    #[must_use]
    pub fn program(&self) -> Option<Program> {
        self.build.map(|build| build().with_entry_op_id(self.id))
    }

    /// Derive target-neutral capability requirements from the canonical program.
    #[must_use]
    pub fn required_capabilities(&self) -> Option<RequiredCapabilities> {
        self.program().map(|program| scan_capabilities(&program))
    }

    /// Derive target-neutral effects from the canonical program.
    #[must_use]
    pub fn effects(&self) -> Option<OperationEffects> {
        self.program()
            .map(|program| OperationEffects::from_program(&program))
    }
}

inventory::collect!(OperationRegistration);

/// Catalog validation failure.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum OperationRegistryError {
    /// Two linked registrations claimed one stable identity.
    #[error("duplicate operation registration `{id}`; keep exactly one semantic owner")]
    DuplicateId {
        /// Duplicated stable operation id.
        id: &'static str,
    },
    /// A registration used the reserved zero semantic version.
    #[error("operation `{id}` uses semantic version zero; use a positive schema version")]
    InvalidVersion {
        /// Invalid operation id.
        id: &'static str,
    },
    /// A registration supplied neither a neutral program nor an explicit signature.
    #[error("operation `{id}` supplies neither a neutral program nor an explicit signature")]
    MissingSemantics {
        /// Incomplete operation id.
        id: &'static str,
    },
}

/// Immutable validated view over every linked semantic operation registration.
pub struct OperationRegistry {
    ordered: Vec<&'static OperationRegistration>,
    by_id: BTreeMap<&'static str, &'static OperationRegistration>,
}

impl OperationRegistry {
    fn build() -> Result<Self, OperationRegistryError> {
        let mut ordered = inventory::iter::<OperationRegistration>
            .into_iter()
            .collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|entry| entry.id);
        let mut by_id = BTreeMap::new();
        for entry in &ordered {
            if entry.semantic_version == 0 {
                return Err(OperationRegistryError::InvalidVersion { id: entry.id });
            }
            if entry.build.is_none() && entry.signature.is_none() {
                return Err(OperationRegistryError::MissingSemantics { id: entry.id });
            }
            if by_id.insert(entry.id, *entry).is_some() {
                return Err(OperationRegistryError::DuplicateId { id: entry.id });
            }
        }
        Ok(Self { ordered, by_id })
    }

    /// Return the process-wide validated semantic operation registry.
    #[must_use]
    pub fn global() -> &'static Self {
        static REGISTRY: LazyLock<OperationRegistry> = LazyLock::new(|| {
            OperationRegistry::build()
                .unwrap_or_else(|error| panic!("invalid semantic operation registry: {error}"))
        });
        &REGISTRY
    }

    /// Resolve one stable operation identity.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&'static OperationRegistration> {
        self.by_id.get(id).copied()
    }

    /// Iterate registrations in stable operation-id order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &'static OperationRegistration> + '_ {
        self.ordered.iter().copied()
    }
}

/// Target-specific capability facet keyed by canonical semantic operation id.
pub struct TargetOperationFacet {
    /// Canonical semantic operation id.
    pub operation_id: &'static str,
    /// Registered target compiler/materializer id.
    pub target_id: &'static str,
    /// Target facet schema version.
    pub version: u32,
}

inventory::collect!(TargetOperationFacet);

/// Iterate linked target facets without allowing them to redefine semantics.
pub fn target_operation_facets() -> impl Iterator<Item = &'static TargetOperationFacet> {
    inventory::iter::<TargetOperationFacet>.into_iter()
}
