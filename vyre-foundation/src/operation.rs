//! Canonical semantic operation registration and derived catalog views.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::LazyLock;

use crate::dialect_lookup::Signature;
use crate::ir::{BufferAccess, Program};
use crate::program_caps::{scan as scan_capabilities, RequiredCapabilities};

/// Deterministic fixture input cases. One case contains declaration-ordered buffers.
pub type OperationFixtures = fn() -> Vec<Vec<Vec<u8>>>;
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
    /// Numerical comparison policy.
    pub tolerance: TolerancePolicy,
}

impl SemanticOperation {
    /// Build the canonical program and stamp its stable operation identity.
    #[must_use]
    pub fn program(self) -> Option<Program> {
        self.build.map(|build| build().with_entry_op_id(self.id))
    }

    /// Derive target-neutral capability requirements from the canonical program.
    #[must_use]
    pub fn required_capabilities(self) -> Option<RequiredCapabilities> {
        self.program().map(|program| scan_capabilities(&program))
    }

    /// Derive target-neutral effects from the canonical program.
    #[must_use]
    pub fn effects(self) -> Option<OperationEffects> {
        self.program()
            .map(|program| OperationEffects::from_program(&program))
    }

    /// Return the coarse category.
    #[must_use]
    pub const fn category(self) -> Option<&'static str> {
        self.category
    }

    /// Return the permitted f32 drift in ULPs.
    #[must_use]
    pub const fn tolerance(self) -> u32 {
        self.tolerance.f32_ulp
    }
}

/// Coarse semantic tier used by catalog and conformance consumers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OperationTier {
    /// Foundation IR or built-in operation.
    Foundation,
    /// Category C hardware intrinsic owned by `vyre-primitives`.
    Intrinsic,
    /// Category A library composition over typed IR, owned by `vyre-libs`.
    Library,
    /// External extension operation.
    External,
    /// Identifier does not match an accepted semantic namespace.
    Unknown,
}

impl OperationTier {
    /// Stable operation-matrix spelling.
    #[must_use]
    pub const fn matrix_value(self) -> &'static str {
        match self {
            Self::Foundation => "foundation_ir",
            Self::Intrinsic => "intrinsic",
            Self::Library => "libs",
            Self::External => "external",
            Self::Unknown => "unknown",
        }
    }
}

/// Classify one operation identity by its canonical namespace.
///
/// The namespace names the crate that owns the registration, so an id whose
/// prefix names no operation-owning crate classifies as [`OperationTier::Unknown`]
/// and [`OperationRegistry`] refuses it. Host-side runtime capabilities are
/// reached through the driver and runtime capability surfaces, not through an
/// operation id, so `core.`, `io.` and `mem.` are not accepted namespaces.
#[must_use]
pub fn classify_operation_id(id: &str) -> OperationTier {
    if id.starts_with("vyre-primitives::") {
        OperationTier::Intrinsic
    } else if id.starts_with("vyre-libs::") {
        OperationTier::Library
    } else if id
        .split_once("::")
        .is_some_and(|(crate_name, _)| !crate_name.is_empty() && !crate_name.starts_with("vyre-"))
    {
        OperationTier::External
    } else {
        OperationTier::Unknown
    }
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

    /// Construct a Category C registration owned by `vyre-primitives`.
    #[must_use]
    pub const fn primitive(
        id: &'static str,
        build: fn() -> Program,
        test_inputs: Option<OperationFixtures>,
        expected_output: Option<OperationFixtures>,
    ) -> Self {
        Self::new(
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
            tolerance: registration.tolerance,
        }
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
    /// Registration tier does not match its canonical namespace.
    #[error(
        "operation `{id}` declares tier {declared:?}, but its canonical namespace classifies as {classified:?}"
    )]
    InvalidTier {
        /// Invalid operation id.
        id: &'static str,
        /// Tier supplied by the registration.
        declared: OperationTier,
        /// Tier derived from the canonical namespace.
        classified: OperationTier,
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
            let classified = classify_operation_id(entry.id);
            if classified == OperationTier::Unknown || classified != entry.tier {
                return Err(OperationRegistryError::InvalidTier {
                    id: entry.id,
                    declared: entry.tier,
                    classified,
                });
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
    pub fn get(&self, id: &str) -> Option<SemanticOperation> {
        self.by_id.get(id).copied().map(SemanticOperation::from)
    }

    /// Iterate registrations in stable operation-id order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = SemanticOperation> + '_ {
        self.ordered.iter().copied().map(SemanticOperation::from)
    }
}

/// Validated target identity carried by target-owned facet registrations.
///
/// Linked target owners construct borrowed identities at declaration time.
/// Deserialized manifests retain owned identities without leaking storage.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TargetId(Cow<'static, str>);

impl TargetId {
    /// Construct a borrowed target identity from an owner-defined stable spelling.
    ///
    /// # Errors
    ///
    /// Empty or whitespace-padded identities are rejected.
    pub const fn new(id: &'static str) -> Result<Self, &'static str> {
        if id.is_empty() || has_surrounding_ascii_whitespace(id.as_bytes()) {
            return Err("target identity must be non-empty and contain no surrounding whitespace");
        }
        Ok(Self(Cow::Borrowed(id)))
    }

    /// Construct an owned target identity from persisted or caller-supplied data.
    ///
    /// # Errors
    ///
    /// Empty or whitespace-padded identities are rejected.
    pub fn from_owned(id: String) -> Result<Self, &'static str> {
        if id.is_empty() || has_surrounding_ascii_whitespace(id.as_bytes()) {
            return Err("target identity must be non-empty and contain no surrounding whitespace");
        }
        Ok(Self(Cow::Owned(id)))
    }

    /// Return the stable owner-defined spelling.
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.0.as_ref()
    }

    /// Construct a validated borrowed target identity for a compile-time constant.
    ///
    /// # Panics
    ///
    /// Panics when the identity is empty or has surrounding whitespace.
    #[must_use]
    pub const fn expect_valid(id: &'static str) -> Self {
        if id.is_empty() || has_surrounding_ascii_whitespace(id.as_bytes()) {
            panic!("target identity must be non-empty and contain no surrounding whitespace");
        }
        Self(Cow::Borrowed(id))
    }
}

impl serde::Serialize for TargetId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> serde::Deserialize<'de> for TargetId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let id = <String as serde::Deserialize>::deserialize(deserializer)?;
        Self::from_owned(id).map_err(serde::de::Error::custom)
    }
}

impl std::fmt::Display for TargetId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl PartialEq<&str> for TargetId {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

const fn has_surrounding_ascii_whitespace(bytes: &[u8]) -> bool {
    matches!(bytes.first(), Some(byte) if byte.is_ascii_whitespace())
        || matches!(bytes.last(), Some(byte) if byte.is_ascii_whitespace())
}

/// Derived target-specific capability keyed by canonical semantic operation id.
///
/// Concrete drivers submit one backend registration containing their validated
/// target identity, compiler, materializer, and supported-operation set. The
/// shared driver joins that record with [`OperationRegistry`] to produce this
/// read-only view without a second operation submission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetOperationFacet {
    /// Canonical semantic operation id.
    pub operation_id: &'static str,
    /// Validated target identity from the concrete driver's registration.
    pub target_id: TargetId,
    /// Target facet schema version.
    pub version: u32,
}
