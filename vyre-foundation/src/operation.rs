//! Canonical semantic operation registration and derived catalog views.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::panic::Location;
use std::sync::LazyLock;

use crate::dialect_lookup::Signature;
use crate::ir::{BufferAccess, Program};
use crate::program_caps::{scan as scan_capabilities, RequiredCapabilities};
use crate::visit::collect_call_op_ids;

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
    /// Optional target-neutral execution geometry requirements.
    pub geometry_requirements: Option<crate::geometry::GeometryRequirements>,
    /// Source file that owns the registration.
    pub source_file: &'static str,
    /// Optional explicit closed effects.
    pub explicit_effects: Option<OperationEffects>,
    /// Optional explicit closed capabilities.
    pub explicit_capabilities: Option<RequiredCapabilities>,
}

impl SemanticOperation {
    /// Build the canonical program and stamp its stable operation identity.
    #[must_use]
    pub fn program(self) -> Option<Program> {
        self.build.map(|build| build().with_entry_op_id(self.id))
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
        self.explicit_effects
            .or_else(|| self.program().map(|program| OperationEffects::from_program(&program)))
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

    /// Return the permitted f32 drift in ULPs.
    #[must_use]
    pub const fn tolerance(self) -> u32 {
        self.tolerance.f32_ulp
    }

    /// Effective composite semantic version including local schema version, transitive effects,
    /// capabilities, and the global call-graph closure identity.
    ///
    /// Any mutation that alters a direct or nested callee's effects, capabilities, or call graph
    /// deterministically alters this composite version, ensuring downstream consolidation and
    /// cache invalidation verdicts update soundly.
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
    /// Every tier, in declaration order.
    ///
    /// A consumer that has to enumerate the taxonomy reads this instead of
    /// restating the variants, so adding one reaches every reader. The enum is
    /// `non_exhaustive`, so only this crate can match it exhaustively:
    /// `the_roster_carries_every_tier` below does, and a new variant stops
    /// compiling there until it is listed here.
    pub const ALL: &'static [Self] = &[
        Self::Foundation,
        Self::Intrinsic,
        Self::Library,
        Self::External,
        Self::Unknown,
    ];

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

/// Which crate minted one operation identity.
///
/// The namespace is frozen when the identity is published. It records the crate
/// that minted the id, not the crate the definition lives in today: eighteen
/// composition domains moved to `vyre-libs` keeping their `vyre-primitives::`
/// ids, so 130 identities name a crate that no longer holds their code. Nothing
/// derives a tier or a placement fact from this prefix. Tier is declared by the
/// registration and cross-checked against the tree by `crate-structure` and by
/// the operation schema. Host-side runtime capabilities are reached through the
/// driver and runtime capability surfaces, so `core.`, `io.` and `mem.` name no
/// namespace and are refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IdNamespace<'a> {
    /// Minted by the named workspace crate.
    Workspace(&'a str),
    /// Minted by a consumer outside the workspace.
    External(&'a str),
    /// Names no crate.
    Unknown,
}

/// Read the minting namespace of one operation identity.
#[must_use]
pub fn operation_id_namespace(id: &str) -> IdNamespace<'_> {
    let Some((namespace, rest)) = id.split_once("::") else {
        return IdNamespace::Unknown;
    };
    if namespace.is_empty() || rest.is_empty() {
        IdNamespace::Unknown
    } else if namespace.starts_with("vyre-") {
        IdNamespace::Workspace(namespace)
    } else {
        IdNamespace::External(namespace)
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
    /// Conservative strongest applicable effects (all effects active).
    /// Pure computation with no side-effects or external storage access.
    pub const NONE: Self = Self {
        reads: false,
        writes: false,
        atomics: false,
        synchronizes: false,
    };

    /// Standard read-write storage access without atomics or barriers.
    pub const READ_WRITE: Self = Self {
        reads: true,
        writes: true,
        atomics: false,
        synchronizes: false,
    };

    /// Read-only storage access.
    pub const READ_ONLY: Self = Self {
        reads: true,
        writes: false,
        atomics: false,
        synchronizes: false,
    };

    /// Barrier synchronization effect.
    pub const SYNCHRONIZES: Self = Self {
        reads: false,
        writes: false,
        atomics: false,
        synchronizes: true,
    };

    /// Read-write storage access with synchronization barrier.
    pub const READ_WRITE_SYNCHRONIZES: Self = Self {
        reads: true,
        writes: true,
        atomics: false,
        synchronizes: true,
    };

    /// Conservative strongest applicable effects (all effects active).
    pub const ALL: Self = Self {
        reads: true,
        writes: true,
        atomics: true,
        synchronizes: true,
    };

    /// Merge another effect set into this one (field-wise OR).
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self {
            reads: self.reads || other.reads,
            writes: self.writes || other.writes,
            atomics: self.atomics || other.atomics,
            synchronizes: self.synchronizes || other.synchronizes,
        }
    }

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
    /// Optional target-neutral execution geometry requirements.
    pub geometry_requirements: Option<crate::geometry::GeometryRequirements>,
    /// Source file that owns the registration.
    pub source_file: &'static str,
    /// Optional explicit closed effects.
    pub explicit_effects: Option<OperationEffects>,
    /// Optional explicit closed capabilities.
    pub explicit_capabilities: Option<RequiredCapabilities>,
}

impl OperationRegistration {
    /// Construct a neutral operation registration with exact comparison policy.
    #[must_use]
    #[track_caller]
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
            geometry_requirements: None,
            source_file: Location::caller().file(),
            explicit_effects: None,
            explicit_capabilities: None,
        }
    }

    /// Construct a Category A composition registration.
    #[must_use]
    #[track_caller]
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

    /// Construct a Category C intrinsic hardware operation registration.
    #[must_use]
    #[track_caller]
    pub const fn intrinsic(
        id: &'static str,
        signature: Signature,
        build: Option<fn() -> Program>,
        test_inputs: Option<OperationFixtures>,
        expected_output: Option<OperationFixtures>,
    ) -> Self {
        Self::new(
            id,
            OperationTier::Intrinsic,
            build,
            test_inputs,
            expected_output,
        )
        .with_signature(signature)
        .with_category("hardware")
    }

    /// Construct a Category C registration owned by `vyre-primitives`.
    #[must_use]
    #[track_caller]
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
    pub const fn tolerance(&self) -> u32 {
        self.tolerance.f32_ulp
    }

    /// Attach the numerical tolerance policy.
    #[must_use]
    pub const fn with_tolerance(mut self, tolerance: TolerancePolicy) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Attach target-neutral execution geometry requirements.
    #[must_use]
    pub const fn with_geometry_requirements(
        mut self,
        requirements: crate::geometry::GeometryRequirements,
    ) -> Self {
        self.geometry_requirements = Some(requirements);
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

    /// Return the execution geometry requirements when declared.
    #[must_use]
    pub const fn geometry_requirements(&self) -> Option<crate::geometry::GeometryRequirements> {
        self.geometry_requirements
    }

    /// Build the canonical program and stamp its stable operation identity.
    #[must_use]
    pub fn program(&self) -> Option<Program> {
        self.build.map(|build| build().with_entry_op_id(self.id))
    }

    /// Direct (local) required capabilities without call-graph transitive propagation.
    #[must_use]
    pub fn direct_required_capabilities(&self) -> Option<RequiredCapabilities> {
        self.explicit_capabilities
            .or_else(|| self.program().map(|program| scan_capabilities(&program)))
    }

    /// Direct (local) memory and synchronization effects without call-graph transitive propagation.
    #[must_use]
    pub fn direct_effects(&self) -> Option<OperationEffects> {
        self.explicit_effects
            .or_else(|| self.program().map(|program| OperationEffects::from_program(&program)))
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
            geometry_requirements: registration.geometry_requirements,
            source_file: registration.source_file,
            explicit_effects: registration.explicit_effects,
            explicit_capabilities: registration.explicit_capabilities,
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
        /// Invalid operation id.
        id: &'static str,
    },
    /// A registration id names no crate.
    #[error(
        "operation `{id}` names no minting crate; an id is `<crate>::<path>` and the crate is the one that published the identity"
    )]
    UnknownNamespace {
        /// Invalid operation id.
        id: &'static str,
    },
    /// Registration tier does not match the kind of crate that minted the id.
    #[error("operation `{id}` declares tier {declared:?}, which no {origin} identity can carry")]
    InvalidTier {
        /// Invalid operation id.
        id: &'static str,
        /// Tier supplied by the registration.
        declared: OperationTier,
        /// Whether the minting crate is inside the workspace.
        origin: &'static str,
    },
}

fn validate_identity(entry: &OperationRegistration) -> Result<(), OperationRegistryError> {
    match operation_id_namespace(entry.id) {
        IdNamespace::Unknown => Err(OperationRegistryError::UnknownNamespace { id: entry.id }),
        IdNamespace::Workspace(_) => {
            if matches!(
                entry.tier,
                OperationTier::Intrinsic | OperationTier::Library | OperationTier::Foundation
            ) {
                Ok(())
            } else {
                Err(OperationRegistryError::InvalidTier {
                    id: entry.id,
                    declared: entry.tier,
                    origin: "workspace",
                })
            }
        }
        IdNamespace::External(_) => {
            if entry.tier == OperationTier::External {
                Ok(())
            } else {
                Err(OperationRegistryError::InvalidTier {
                    id: entry.id,
                    declared: entry.tier,
                    origin: "external",
                })
            }
        }
    }
}

/// Transitive call-graph closure over semantic operation registrations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallGraphClosure {
    /// Direct callees for each registered operation ID (sorted and deduplicated).
    pub direct_callees: BTreeMap<&'static str, Vec<&'static str>>,
    /// Direct (local) effects for each operation ID.
    pub direct_effects: BTreeMap<&'static str, OperationEffects>,
    /// Direct (local) capabilities for each operation ID.
    pub direct_capabilities: BTreeMap<&'static str, RequiredCapabilities>,
    /// Transitive effects solved to a fixed point for each operation ID.
    pub transitive_effects: BTreeMap<&'static str, OperationEffects>,
    /// Transitive capabilities solved to a fixed point for each operation ID.
    pub transitive_capabilities: BTreeMap<&'static str, RequiredCapabilities>,
    /// Operations participating in recursive cycles or unresolved callee chains without closed contracts.
    pub unclosed_or_cyclic: BTreeSet<&'static str>,
    /// Deterministic 64-bit fingerprint of the resolved call-graph closure.
    pub closure_identity: u64,
}

impl CallGraphClosure {
    /// Solve the call graph closure to a fixed point over a collection of registrations.
    ///
    /// Every canonical program is built at most once during this solve pass.
    #[must_use]
    pub fn solve_from_registrations<'a, I>(registrations: I) -> Self
    where
        I: IntoIterator<Item = &'a OperationRegistration>,
    {
        let reg_map: BTreeMap<&'static str, &OperationRegistration> = registrations
            .into_iter()
            .map(|reg| (reg.id, reg))
            .collect();

        let mut direct_callees: BTreeMap<&'static str, Vec<&'static str>> = BTreeMap::new();
        let mut direct_effects: BTreeMap<&'static str, OperationEffects> = BTreeMap::new();
        let mut direct_capabilities: BTreeMap<&'static str, RequiredCapabilities> = BTreeMap::new();
        let mut unclosed_or_cyclic: BTreeSet<&'static str> = BTreeSet::new();

        // 1. Build canonical program once for each registration and extract local facts.
        for (&id, &reg) in &reg_map {
            if let Some(build) = reg.build {
                let program = (build)().with_entry_op_id(id);
                let local_eff = reg
                    .explicit_effects
                    .unwrap_or_else(|| OperationEffects::from_program(&program));
                let local_caps = reg
                    .explicit_capabilities
                    .unwrap_or_else(|| scan_capabilities(&program));
                let raw_callees = collect_call_op_ids(&program);

                let mut callees: Vec<&'static str> = Vec::with_capacity(raw_callees.len());
                for raw_callee in raw_callees {
                    if let Some((&matched_id, _)) = reg_map.get_key_value(raw_callee.as_ref()) {
                        callees.push(matched_id);
                    } else {
                        let leaked: &'static str = Box::leak(raw_callee.to_string().into_boxed_str());
                        callees.push(leaked);
                    }
                }
                callees.sort_unstable();
                callees.dedup();

                direct_callees.insert(id, callees);
                direct_effects.insert(id, local_eff);
                direct_capabilities.insert(id, local_caps);
            } else {
                direct_callees.insert(id, Vec::new());
                if let (Some(eff), Some(caps)) = (reg.explicit_effects, reg.explicit_capabilities) {
                    direct_effects.insert(id, eff);
                    direct_capabilities.insert(id, caps);
                } else {
                    direct_effects.insert(id, reg.explicit_effects.unwrap_or(OperationEffects::ALL));
                    direct_capabilities.insert(
                        id,
                        reg.explicit_capabilities
                            .unwrap_or_else(RequiredCapabilities::all),
                    );
                    unclosed_or_cyclic.insert(id);
                }
            }
        }

        // 2. Identify unresolved callees (nodes calling an unregistered ID).
        for (&id, callees) in &direct_callees {
            for &callee in callees {
                if !reg_map.contains_key(callee) {
                    unclosed_or_cyclic.insert(id);
                }
            }
        }

        // 3. Detect recursive cycles via Tarjan's Strongly Connected Components algorithm.
        let sccs = compute_sccs(&direct_callees);
        for scc in sccs {
            let is_cycle = if scc.len() > 1 {
                true
            } else if scc.len() == 1 {
                let node = scc[0];
                direct_callees
                    .get(node)
                    .is_some_and(|callees| callees.contains(&node))
            } else {
                false
            };

            if is_cycle {
                for &node in &scc {
                    let reg = reg_map.get(node);
                    let has_contract = reg.is_some_and(|r| {
                        r.explicit_effects.is_some() && r.explicit_capabilities.is_some()
                    });
                    if !has_contract {
                        unclosed_or_cyclic.insert(node);
                        direct_effects.insert(node, OperationEffects::ALL);
                        direct_capabilities.insert(node, RequiredCapabilities::all());
                    }
                }
            }
        }

        // Ensure all unclosed/cyclic nodes default to strongest effects and capabilities.
        for &node in &unclosed_or_cyclic {
            let reg = reg_map.get(node);
            let has_contract = reg.is_some_and(|r| {
                r.explicit_effects.is_some() && r.explicit_capabilities.is_some()
            });
            if !has_contract {
                direct_effects.insert(node, OperationEffects::ALL);
                direct_capabilities.insert(node, RequiredCapabilities::all());
            }
        }

        // 4. Fixed-Point Propagation over the call graph edges.
        let mut transitive_effects = direct_effects.clone();
        let mut transitive_capabilities = direct_capabilities.clone();

        let mut changed = true;
        while changed {
            changed = false;
            for (&u, callees) in &direct_callees {
                for &v in callees {
                    let (v_eff, v_caps) = if reg_map.contains_key(v) {
                        let eff = transitive_effects
                            .get(v)
                            .copied()
                            .unwrap_or(OperationEffects::ALL);
                        let caps = transitive_capabilities
                            .get(v)
                            .copied()
                            .unwrap_or_else(RequiredCapabilities::all);
                        (eff, caps)
                    } else {
                        (OperationEffects::ALL, RequiredCapabilities::all())
                    };

                    let u_eff = transitive_effects.get_mut(u).unwrap();
                    let merged_eff = u_eff.union(v_eff);
                    if merged_eff != *u_eff {
                        *u_eff = merged_eff;
                        changed = true;
                    }

                    let u_caps = transitive_capabilities.get_mut(u).unwrap();
                    let merged_caps = u_caps.join(v_caps);
                    if merged_caps != *u_caps {
                        *u_caps = merged_caps;
                        changed = true;
                    }
                }
            }
        }

        // 5. Deterministic call-graph closure identity calculation.
        let closure_identity = compute_closure_identity(
            &reg_map,
            &direct_callees,
            &transitive_effects,
            &transitive_capabilities,
            &unclosed_or_cyclic,
        );

        Self {
            direct_callees,
            direct_effects,
            direct_capabilities,
            transitive_effects,
            transitive_capabilities,
            unclosed_or_cyclic,
            closure_identity,
        }
    }

    /// Return the resolved transitive effects for an operation.
    #[must_use]
    pub fn transitive_effects(&self, id: &str) -> Option<OperationEffects> {
        self.transitive_effects.get(id).copied()
    }

    /// Return the resolved transitive required capabilities for an operation.
    #[must_use]
    pub fn transitive_capabilities(&self, id: &str) -> Option<RequiredCapabilities> {
        self.transitive_capabilities.get(id).copied()
    }

    /// Return direct callees invoked by an operation via `Expr::Call`.
    #[must_use]
    pub fn callees(&self, id: &str) -> Option<&[&'static str]> {
        self.direct_callees
            .get(id)
            .map(|callees| callees.as_slice())
    }

    /// Return whether an operation participates in an unclosed recursive cycle or unresolved call.
    #[must_use]
    pub fn is_unclosed_or_cyclic(&self, id: &str) -> bool {
        self.unclosed_or_cyclic.contains(id)
    }

    /// Return the deterministic 64-bit closure identity.
    #[must_use]
    pub fn closure_identity(&self) -> u64 {
        self.closure_identity
    }

    /// Calculate effective composite version for an operation.
    #[must_use]
    pub fn composite_version(&self, id: &str, base_version: u32) -> Option<u64> {
        let eff = self.transitive_effects(id)?;
        let caps = self.transitive_capabilities(id)?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"vyre-foundation::call_graph_closure::composite_version::v1\n");
        hasher.update(id.as_bytes());
        hasher.update(&base_version.to_le_bytes());
        hasher.update(&[
            eff.reads as u8,
            eff.writes as u8,
            eff.atomics as u8,
            eff.synchronizes as u8,
        ]);
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
        hasher.update(&self.closure_identity.to_le_bytes());
        let hash_bytes = hasher.finalize();
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&hash_bytes.as_bytes()[..8]);
        Some(u64::from_le_bytes(bytes))
    }
}

fn compute_sccs(adj: &BTreeMap<&'static str, Vec<&'static str>>) -> Vec<Vec<&'static str>> {
    struct Tarjan<'a> {
        adj: &'a BTreeMap<&'static str, Vec<&'static str>>,
        index: usize,
        indices: BTreeMap<&'static str, usize>,
        lowlinks: BTreeMap<&'static str, usize>,
        on_stack: BTreeSet<&'static str>,
        stack: Vec<&'static str>,
        sccs: Vec<Vec<&'static str>>,
    }

    impl<'a> Tarjan<'a> {
        fn strongconnect(&mut self, v: &'static str) {
            self.indices.insert(v, self.index);
            self.lowlinks.insert(v, self.index);
            self.index += 1;
            self.stack.push(v);
            self.on_stack.insert(v);

            if let Some(neighbors) = self.adj.get(v) {
                for &w in neighbors {
                    if !self.indices.contains_key(w) {
                        if self.adj.contains_key(w) {
                            self.strongconnect(w);
                            let w_low = self.lowlinks[w];
                            let v_low = self.lowlinks.get_mut(v).unwrap();
                            *v_low = (*v_low).min(w_low);
                        }
                    } else if self.on_stack.contains(w) {
                        let w_idx = self.indices[w];
                        let v_low = self.lowlinks.get_mut(v).unwrap();
                        *v_low = (*v_low).min(w_idx);
                    }
                }
            }

            if self.lowlinks[v] == self.indices[v] {
                let mut scc = Vec::new();
                while let Some(w) = self.stack.pop() {
                    self.on_stack.remove(w);
                    scc.push(w);
                    if w == v {
                        break;
                    }
                }
                self.sccs.push(scc);
            }
        }
    }

    let mut tarjan = Tarjan {
        adj,
        index: 0,
        indices: BTreeMap::new(),
        lowlinks: BTreeMap::new(),
        on_stack: BTreeSet::new(),
        stack: Vec::new(),
        sccs: Vec::new(),
    };

    for &node in adj.keys() {
        if !tarjan.indices.contains_key(node) {
            tarjan.strongconnect(node);
        }
    }

    tarjan.sccs
}

fn compute_closure_identity(
    reg_map: &BTreeMap<&'static str, &OperationRegistration>,
    direct_callees: &BTreeMap<&'static str, Vec<&'static str>>,
    transitive_effects: &BTreeMap<&'static str, OperationEffects>,
    transitive_capabilities: &BTreeMap<&'static str, RequiredCapabilities>,
    unclosed_or_cyclic: &BTreeSet<&'static str>,
) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vyre-foundation::call_graph_closure::v1\n");
    for (&id, reg) in reg_map {
        hasher.update(id.as_bytes());
        hasher.update(&reg.semantic_version.to_le_bytes());
        if let Some(callees) = direct_callees.get(id) {
            for callee in callees {
                hasher.update(b"->");
                hasher.update(callee.as_bytes());
            }
        }
        if let Some(eff) = transitive_effects.get(id) {
            hasher.update(&[
                eff.reads as u8,
                eff.writes as u8,
                eff.atomics as u8,
                eff.synchronizes as u8,
            ]);
        }
        if let Some(caps) = transitive_capabilities.get(id) {
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
        hasher.update(&[unclosed_or_cyclic.contains(id) as u8]);
    }
    let hash_bytes = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&hash_bytes.as_bytes()[..8]);
    u64::from_le_bytes(bytes)
}

/// Immutable validated view over every linked semantic operation registration.
pub struct OperationRegistry {
    ordered: Vec<&'static OperationRegistration>,
    by_id: BTreeMap<&'static str, &'static OperationRegistration>,
    call_graph: CallGraphClosure,
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
            validate_identity(entry)?;
            if by_id.insert(entry.id, *entry).is_some() {
                return Err(OperationRegistryError::DuplicateId { id: entry.id });
            }
        }
        let call_graph = CallGraphClosure::solve_from_registrations(ordered.iter().copied());
        Ok(Self {
            ordered,
            by_id,
            call_graph,
        })
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

    /// Return the immutable precomputed call graph closure over all registered operations.
    #[must_use]
    pub fn call_graph_closure(&self) -> &CallGraphClosure {
        &self.call_graph
    }

    /// Return transitive effects for an operation.
    #[must_use]
    pub fn transitive_effects(&self, id: &str) -> Option<OperationEffects> {
        self.call_graph.transitive_effects(id)
    }

    /// Return transitive required capabilities for an operation.
    #[must_use]
    pub fn transitive_capabilities(&self, id: &str) -> Option<RequiredCapabilities> {
        self.call_graph.transitive_capabilities(id)
    }

    /// Return direct callees for an operation.
    #[must_use]
    pub fn callees(&self, id: &str) -> Option<&[&'static str]> {
        self.call_graph.callees(id)
    }

    /// Return deterministic call-graph closure identity.
    #[must_use]
    pub fn call_graph_closure_identity(&self) -> u64 {
        self.call_graph.closure_identity()
    }

    /// Return effective composite version for an operation.
    #[must_use]
    pub fn composite_version(&self, id: &str) -> Option<u64> {
        let entry = self.by_id.get(id)?;
        self.call_graph
            .composite_version(id, entry.semantic_version)
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetOperationFacet {
    /// Canonical semantic operation id.
    pub operation_id: &'static str,
    /// Validated target identity from the concrete driver's registration.
    pub target_id: TargetId,
    /// Target facet schema version.
    pub version: u32,
}

#[cfg(test)]
mod tests {
    use super::{
        operation_id_namespace, validate_identity, IdNamespace, OperationRegistration,
        OperationRegistryError, OperationTier,
    };
    use std::collections::BTreeSet;

    /// The tier roster carries every variant of the tier enum.
    #[test]
    fn the_roster_carries_every_tier() {
        let mut seen = BTreeSet::new();
        for tier in OperationTier::ALL {
            seen.insert(match tier {
                OperationTier::Foundation => 0,
                OperationTier::Intrinsic => 1,
                OperationTier::Library => 2,
                OperationTier::External => 3,
                OperationTier::Unknown => 4,
            });
        }
        assert_eq!(
            seen.len(),
            5,
            "OperationTier::ALL must list every variant the match above names"
        );
    }

    /// A namespace is a minting fact, never a placement one.
    #[test]
    fn the_namespace_never_answers_with_a_tier() {
        assert_eq!(
            operation_id_namespace("vyre-primitives::graph::toposort"),
            IdNamespace::Workspace("vyre-primitives")
        );
    }

    /// A workspace id cannot carry a tier only a consumer identity has.
    #[test]
    fn a_workspace_id_declaring_an_external_tier_is_rejected() {
        let entry = OperationRegistration::new(
            "vyre-libs::scan::literal_set",
            OperationTier::External,
            None,
            None,
            None,
        );
        assert_eq!(
            validate_identity(&entry),
            Err(OperationRegistryError::InvalidTier {
                id: "vyre-libs::scan::literal_set",
                declared: OperationTier::External,
                origin: "workspace",
            })
        );
    }

    /// A consumer id carries the external tier and no other.
    #[test]
    fn an_external_id_declaring_a_workspace_tier_is_rejected() {
        let entry = OperationRegistration::new(
            "community_pack::scan::signature",
            OperationTier::Library,
            None,
            None,
            None,
        );
        assert_eq!(
            validate_identity(&entry),
            Err(OperationRegistryError::InvalidTier {
                id: "community_pack::scan::signature",
                declared: OperationTier::Library,
                origin: "external",
            })
        );
        assert_eq!(
            validate_identity(&OperationRegistration::new(
                "community_pack::scan::signature",
                OperationTier::External,
                None,
                None,
                None,
            )),
            Ok(())
        );
    }

    /// Every tier a workspace crate can mint is accepted, and the two that name
    /// no minting crate are not.
    #[test]
    fn a_workspace_id_carries_every_workspace_tier() {
        for tier in [
            OperationTier::Foundation,
            OperationTier::Intrinsic,
            OperationTier::Library,
        ] {
            assert_eq!(
                validate_identity(&OperationRegistration::new(
                    "vyre-primitives::hardware::popcount_u32",
                    tier,
                    None,
                    None,
                    None,
                )),
                Ok(()),
                "{tier:?} is a tier a workspace crate mints"
            );
        }
        assert_eq!(
            validate_identity(&OperationRegistration::new(
                "vyre-primitives::hardware::popcount_u32",
                OperationTier::Unknown,
                None,
                None,
                None,
            )),
            Err(OperationRegistryError::InvalidTier {
                id: "vyre-primitives::hardware::popcount_u32",
                declared: OperationTier::Unknown,
                origin: "workspace",
            })
        );
    }

    /// An id that names no crate is refused before any tier question.
    #[test]
    fn an_id_naming_no_crate_is_refused_whatever_it_declares() {
        for id in ["not_a_namespace", "core.indirect_dispatch", "vyre-libs::"] {
            assert_eq!(
                validate_identity(&OperationRegistration::new(
                    id,
                    OperationTier::Library,
                    None,
                    None,
                    None,
                )),
                Err(OperationRegistryError::UnknownNamespace { id })
            );
        }
    }
}
