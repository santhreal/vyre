//! Canonical semantic operation registration and derived catalog views.

mod registry_error;

use self::registry_error::validate_identity;
pub use self::registry_error::OperationRegistryError;

mod semantics;
mod target_facet;

pub use self::semantics::{operation_id_namespace, IdNamespace, OperationEffects, OperationTier};
pub use self::target_facet::{TargetId, TargetOperationFacet};

use std::collections::{BTreeMap, BTreeSet};
use std::panic::Location;
use std::sync::LazyLock;

use crate::dialect_lookup::Signature;
use crate::ir::Program;
use crate::numeric::NumericContract;
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
    /// What the result is allowed to be.
    pub numeric: NumericContract,
    /// Recorded target-neutral schedule constraints.
    pub geometry_requirements: crate::geometry::GeometryRequirements,
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

    /// Derive the effective neutral schedule constraints from the recorded
    /// decision and the canonical program.
    ///
    /// # Errors
    ///
    /// Returns a stable conflict when the recorded decision contradicts semantics.
    pub fn schedule_constraints(
        self,
    ) -> Result<crate::geometry::GeometryRequirements, crate::geometry::GeometryConstraintConflict>
    {
        match self.program() {
            Some(program) => self.geometry_requirements.compose(
                crate::geometry::GeometryRequirements::from_program(&program)?,
            ),
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

    /// Return the permitted f32 drift in ULPs.
    ///
    /// A contract stated in ULPs answers with its count, an exact contract with
    /// zero, and a relative bound with the count it spans over the declared
    /// storage. An absolute bound has no ULP reading without a magnitude, so it
    /// answers `None` rather than a number a comparison would trust.
    #[must_use]
    pub fn ulp_budget(&self) -> Option<u32> {
        self.numeric.ulp_budget()
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
    pub geometry_requirements: crate::geometry::GeometryRequirements,
    /// Source file that owns the registration.
    pub source_file: &'static str,
    /// Optional explicit closed effects.
    pub explicit_effects: Option<OperationEffects>,
    /// Optional explicit closed capabilities.
    pub explicit_capabilities: Option<RequiredCapabilities>,
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
            geometry_requirements: crate::geometry::GeometryRequirements::agnostic(),
            source_file: Location::caller().file(),
            explicit_effects: None,
            explicit_capabilities: None,
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
    ///
    /// A contract stated in ULPs answers with its count, an exact contract with
    /// zero, and a relative bound with the count it spans over the declared
    /// storage. An absolute bound has no ULP reading without a magnitude, so it
    /// answers `None` rather than a number a comparison would trust.
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
    pub const fn with_geometry_requirements(
        mut self,
        requirements: crate::geometry::GeometryRequirements,
    ) -> Self {
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
    pub const fn declared_schedule_constraints(&self) -> crate::geometry::GeometryRequirements {
        self.geometry_requirements
    }

    /// Build the canonical program and stamp its stable operation identity.
    #[must_use]
    pub fn program(&self) -> Option<Program> {
        self.build.map(|build| build().with_entry_op_id(self.id))
    }

    /// Derive the effective neutral schedule constraints from the recorded
    /// decision and canonical program semantics.
    ///
    /// # Errors
    ///
    /// Returns a stable conflict when the recorded decision contradicts semantics.
    pub fn schedule_constraints(
        &self,
    ) -> Result<crate::geometry::GeometryRequirements, crate::geometry::GeometryConstraintConflict>
    {
        match self.program() {
            Some(program) => self.geometry_requirements.compose(
                crate::geometry::GeometryRequirements::from_program(&program)?,
            ),
            None => Ok(self.geometry_requirements),
        }
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
        self.explicit_effects.or_else(|| {
            self.program()
                .map(|program| OperationEffects::from_program(&program))
        })
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
            numeric: registration.numeric,
            geometry_requirements: registration.geometry_requirements,
            source_file: registration.source_file,
            explicit_effects: registration.explicit_effects,
            explicit_capabilities: registration.explicit_capabilities,
        }
    }
}

inventory::collect!(OperationRegistration);

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
    ///
    /// # Panics
    ///
    /// Panics if internal call graph propagation encounters inconsistent node registration state.
    #[must_use]
    pub fn solve_from_registrations<'a, I>(registrations: I) -> Self
    where
        I: IntoIterator<Item = &'a OperationRegistration>,
    {
        let reg_map: BTreeMap<&'static str, &OperationRegistration> =
            registrations.into_iter().map(|reg| (reg.id, reg)).collect();

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
                        let leaked: &'static str =
                            Box::leak(raw_callee.to_string().into_boxed_str());
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
                    direct_effects
                        .insert(id, reg.explicit_effects.unwrap_or(OperationEffects::ALL));
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
            let has_contract = reg
                .is_some_and(|r| r.explicit_effects.is_some() && r.explicit_capabilities.is_some());
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
                            .expect("Fix: registered callee must have transitive effect state");
                        let caps = transitive_capabilities
                            .get(v)
                            .copied()
                            .expect("Fix: registered callee must have transitive capability state");
                        (eff, caps)
                    } else {
                        (OperationEffects::ALL, RequiredCapabilities::all())
                    };

                    let u_eff = transitive_effects
                        .get_mut(u)
                        .expect("Fix: caller node in direct_callees must be registered in transitive_effects");
                    let merged_eff = u_eff.union(v_eff);
                    if merged_eff != *u_eff {
                        *u_eff = merged_eff;
                        changed = true;
                    }

                    let u_caps = transitive_capabilities
                        .get_mut(u)
                        .expect("Fix: caller node in direct_callees must be registered in transitive_capabilities");
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

/// Compute strongly connected components for the call graph adjacency map.
///
/// # Panics
///
/// Panics if Tarjan traversal state invariants are violated.
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
        /// Recursively traverse and find strongly connected components from node `v`.
        ///
        /// # Panics
        ///
        /// Panics if internal Tarjan state invariants are violated (missing index/lowlink entries).
        fn strongconnect(&mut self, v: &'static str) {
            self.indices.insert(v, self.index);
            self.lowlinks.insert(v, self.index);
            self.index += 1;
            self.stack.push(v);
            self.on_stack.insert(v);

            let neighbors = self
                .adj
                .get(v)
                .expect("Fix: Tarjan traversal node must have an adjacency entry");
            for &w in neighbors {
                if !self.indices.contains_key(w) {
                    if self.adj.contains_key(w) {
                        self.strongconnect(w);
                        let w_low = self
                            .lowlinks
                            .get(w)
                            .copied()
                            .expect("Fix: Tarjan visited child node must have a lowlink entry");
                        let v_low = self
                            .lowlinks
                            .get_mut(v)
                            .expect("Fix: Tarjan active node must have a lowlink entry");
                        *v_low = (*v_low).min(w_low);
                    }
                } else if self.on_stack.contains(w) {
                    let w_idx = self
                        .indices
                        .get(w)
                        .copied()
                        .expect("Fix: Tarjan on-stack node must have an index entry");
                    let v_low = self
                        .lowlinks
                        .get_mut(v)
                        .expect("Fix: Tarjan active node must have a lowlink entry");
                    *v_low = (*v_low).min(w_idx);
                }
            }
            let v_low = self
                .lowlinks
                .get(v)
                .copied()
                .expect("Fix: Tarjan active node must have a lowlink entry");
            let v_idx = self
                .indices
                .get(v)
                .copied()
                .expect("Fix: Tarjan active node must have an index entry");
            if v_low == v_idx {
                let mut scc = Vec::new();
                loop {
                    let w = self
                        .stack
                        .pop()
                        .expect("Fix: Tarjan component stack must contain its root");
                    assert!(
                        self.on_stack.remove(w),
                        "Fix: Tarjan component member must be marked on-stack"
                    );
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
    ///
    /// # Panics
    ///
    /// Panics if the static operation inventory fails validation or contains duplicate IDs.
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

#[cfg(test)]
mod tests;
