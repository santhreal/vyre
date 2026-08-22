//! Backend-neutral target-payload admission shared by every concrete driver.
//!
//! Materializing a target payload is two neutral checks bracketing one
//! backend-specific step: admit the payload against the artifact it claims to
//! implement, decode the dialect image, then project the artifact's resources
//! onto the instance. Only the middle step is target-specific.
//!
//! Copying the neutral halves per backend is what let them drift. Before this
//! module the same admission checks were written four times, and they had
//! stopped agreeing: two backends rejected a module whose entry point was not
//! `main` and two accepted it, and one spelled several shared failures with
//! different text than the other three. A payload rejected by one backend was
//! accepted by another for reasons nobody chose.

use std::collections::{BTreeMap, BTreeSet};

use vyre_foundation::ir::Program;
use vyre_megakernel::{
    Artifact, ArtifactNodeId, ArtifactValueId, Digest, TargetModuleBundle, TargetPayload,
};

use crate::{
    BackendError, BindingPlan, BindingRole, BindingSet, Completion, DeviceIdentity, DispatchConfig,
};

pub use crate::materialize_admission::*;
pub use crate::materialize_instance::*;

/// Build the shared "recompile the payload" rejection.
#[must_use]
pub fn invalid_module(reason: &str) -> BackendError {
    BackendError::InvalidProgram {
        fix: format!("Fix: {reason}. Recompile the target payload from the neutral artifact."),
    }
}

/// Build the shared payload-decode failure for `backend`.
#[must_use]
pub fn compile_error(backend: &str, error: impl std::fmt::Display) -> BackendError {
    BackendError::KernelCompileFailed {
        backend: backend.to_string(),
        compiler_message: format!(
            "{error}. Fix: rebuild the target payload from the neutral artifact."
        ),
    }
}

/// Rejection text the shared submission path asks its caller for.
///
/// The submission path itself is one decision per backend, but the wording of
/// each rejection is observable and the backends do not agree on it. Passing
/// the text in keeps every message byte-identical to what the backend shipped
/// while the control flow around it has one owner.
#[derive(Clone, Copy, Debug)]
pub struct InstanceMessages {
    /// Bindings name an artifact this instance does not implement.
    pub foreign_artifact: fn() -> BackendError,
    /// A Program buffer has no canonical artifact value.
    pub unmapped_buffer: fn(&str) -> BackendError,
    /// Execution left a declared output value unproduced.
    pub missing_output_value: fn(ArtifactValueId) -> BackendError,
    /// Execution left a retained value unpreserved.
    pub missing_retained_value: fn(ArtifactValueId) -> BackendError,
    /// A completion was taken from its submission twice.
    pub completion_consumed: fn() -> BackendError,
}

/// The rejection text that names nothing target-specific.
///
/// Every backend ships these strings. A backend whose wording differs supplies
/// its own record rather than moving anyone else's text.
///
/// Two of the five name their corrective action directly instead of routing
/// through [`invalid_module`], because recompiling the target payload does not
/// fix them. A foreign artifact is a binding mistake at the call site, and a
/// twice-consumed completion is a caller mistake; both payloads are fine.
pub const NEUTRAL_MESSAGES: InstanceMessages = InstanceMessages {
    foreign_artifact: || BackendError::InvalidProgram {
        fix: "Fix: bind resources against the exact artifact digest owned by this instance."
            .to_string(),
    },
    unmapped_buffer: |name| {
        invalid_module(&format!(
            "Program buffer `{name}` is absent from the canonical artifact ABI"
        ))
    },
    missing_output_value: |value| {
        invalid_module(&format!(
            "selected execution did not produce canonical output value {}",
            value.0
        ))
    },
    missing_retained_value: |value| {
        invalid_module(&format!(
            "selected execution did not preserve retained value {}",
            value.0
        ))
    },
    completion_consumed: || BackendError::InvalidProgram {
        fix: "Fix: consume each Submission completion exactly once.".to_string(),
    },
};

/// Rejection for a declared input whose canonical value was never bound.
///
/// This is the wording two backends already shipped byte-for-byte. It sits
/// beside [`NEUTRAL_MESSAGES`] rather than inside it because
/// [`InstanceCore::gather_inputs`] takes the rejection as an argument, so a
/// backend whose text differs passes its own function and moves nobody else's.
#[must_use]
pub fn unbound_input(value: ArtifactValueId, name: &str) -> BackendError {
    BackendError::InvalidProgram {
        fix: format!(
            "Fix: bind canonical artifact value {} for Program buffer `{name}` before submission.",
            value.0
        ),
    }
}

/// Rejection for a declared resident buffer whose canonical value carries no
/// resource.
///
/// A resident launch is handed device memory the caller already filled, so the
/// refusal names the buffer and the value rather than the submission: this is
/// the wording two backends shipped byte-for-byte beside their own copy of the
/// binding walk.
#[must_use]
pub fn unbound_resident_buffer(value: ArtifactValueId, name: &str) -> BackendError {
    BackendError::InvalidProgram {
        fix: format!(
            "Fix: bind canonical artifact value {} for resident Program buffer `{name}`.",
            value.0
        ),
    }
}

/// The Program buffer names a resident launch takes a resource for, in binding
/// order.
///
/// A shared binding is filled by the launch itself rather than by the caller,
/// so it takes no resource. Two backends read the same order off the same plan
/// through their own copy of this filter.
pub fn resident_buffer_names<'plan>(
    plan: &'plan BindingPlan,
    program: &'plan Program,
) -> impl Iterator<Item = &'plan str> {
    plan.bindings
        .iter()
        .filter(|binding| binding.role != BindingRole::Shared)
        .map(|binding| program.buffers()[binding.buffer_index].name())
}

/// Rejection for a declared output slot the target module never produced.
///
/// `module_label` names the dialect and path in the backend's own words, which
/// is the only part the four backends disagreed on. The sentence around it was
/// written six times, and one of the six had already lost the recompile
/// instruction the other five give, so the same defect read as two different
/// classes of failure depending on which backend hit it.
#[must_use]
pub fn omitted_output(module_label: &str, output_index: usize, name: &str) -> BackendError {
    invalid_module(&format!(
        "{module_label} omitted output {output_index} for Program buffer `{name}`"
    ))
}

/// One executable module of a materialized instance, as the shared execution
/// loop reads it.
///
/// A backend's module record also holds its native pipeline and whatever
/// binding metadata its dialect needs; those stay private to the backend
/// because the shared loop never touches them. What it does need is the
/// canonical Program and the payload-declared dispatch config, which every
/// backend records under the same two names.
pub trait ExecutableModule {
    /// Canonical Program this module implements.
    fn program(&self) -> &Program;
    /// Dispatch configuration the payload entry declared.
    fn config(&self) -> &DispatchConfig;
}

/// Answer the two [`ExecutableModule`] methods from fields named `program` and
/// `config`.
///
/// The trait's whole purpose is to let the shared execution loop read those two
/// out of a record it does not otherwise know, and every backend stores them
/// under those names, so every backend wrote the same eight lines. A backend
/// that stores them elsewhere writes the impl by hand instead.
#[macro_export]
macro_rules! executable_module {
    () => {
        fn program(&self) -> &vyre_foundation::ir::Program {
            &self.program
        }

        fn config(&self) -> &$crate::DispatchConfig {
            &self.config
        }
    };
}

/// Identity and artifact ABI projection every materialized instance keeps.
///
/// The four backends held these six fields under four names, filled them with
/// the same eight lines, and reimplemented the same lookups over them.
pub struct InstanceCore {
    /// Neutral artifact identity this instance implements.
    pub artifact: Digest,
    /// Exact payload identity materialized into this instance.
    pub payload: Digest,
    /// Device generation that owns every native handle.
    pub device: DeviceIdentity,
    /// Every artifact resource by name.
    pub values: BTreeMap<String, ArtifactValueId>,
    /// Resources the artifact reports as outputs.
    pub outputs: BTreeSet<ArtifactValueId>,
    /// Resources the artifact retains across dispatches.
    pub retained: BTreeSet<ArtifactValueId>,
    /// Rejection text this backend ships.
    pub messages: InstanceMessages,
    /// Input value identities per module entry in target execution order.
    pub module_inputs: Vec<Vec<ArtifactValueId>>,
    /// Output value identities per module entry in target execution order.
    pub module_outputs: Vec<Vec<ArtifactValueId>>,
    /// Canonical artifact identities keyed by exact Program buffer name per module.
    pub module_named_resources: Vec<BTreeMap<String, ArtifactValueId>>,
    /// Exact target descriptor slots keyed by Program buffer name per module.
    pub module_buffer_slots: Vec<BTreeMap<String, (u32, u32)>>,
    /// Canonical artifact identities keyed by exact target `(group, slot)` per module.
    pub module_resources: Vec<BTreeMap<(u32, u32), ArtifactValueId>>,
    /// Transitive retained values that feed each successor or final public output.
    pub retained_predecessors: BTreeMap<ArtifactValueId, Vec<ArtifactValueId>>,
}

/// Whether two artifact identities are the same buffer at two points of its
/// retained chain.
///
/// A `ReadWrite` buffer is wired to the resource it reads and to the renamed
/// resource it writes, and the writer records the reader as its
/// `retained_predecessor`. Either identity therefore names the same buffer, and
/// the map holds transitive priors, so one containment test in each direction
/// covers a chain of any depth.
fn retained_chain_relates(
    retained_predecessors: &BTreeMap<ArtifactValueId, Vec<ArtifactValueId>>,
    left: ArtifactValueId,
    right: ArtifactValueId,
) -> bool {
    retained_predecessors
        .get(&left)
        .is_some_and(|priors| priors.contains(&right))
        || retained_predecessors
            .get(&right)
            .is_some_and(|priors| priors.contains(&left))
}

impl InstanceCore {
    /// Record what an instance materialized from `artifact` and `payload` keeps.
    ///
    /// # Errors
    ///
    /// Returns an invalid-module rejection when payload entries cannot be
    /// projected into the exact `(stage, group)` execution order of the target
    /// module bundle.
    pub fn new(
        artifact: &Artifact,
        payload: &TargetPayload,
        device: DeviceIdentity,
        messages: InstanceMessages,
    ) -> Result<Self, BackendError> {
        let bundle = TargetModuleBundle::from_bytes(payload.bytes()).map_err(|error| {
            invalid_module(&format!(
                "target module bundle cannot project Program buffer identities: {error}"
            ))
        })?;
        let mut module_buffer_slots = Vec::with_capacity(bundle.modules.len());
        for module in &bundle.modules {
            let mut slots = BTreeMap::new();
            for slot in &module.descriptor.bindings.slots {
                let Some(identity) = module.binding_slot(&slot.name) else {
                    continue;
                };
                if slots.insert(slot.name.clone(), identity).is_some() {
                    return Err(invalid_module(
                        "target module descriptor names one Program buffer twice",
                    ));
                }
            }
            module_buffer_slots.push(slots);
        }
        Self::new_with_module_slots(artifact, payload, device, messages, module_buffer_slots)
    }

    pub(crate) fn new_with_module_slots(
        artifact: &Artifact,
        payload: &TargetPayload,
        device: DeviceIdentity,
        messages: InstanceMessages,
        module_buffer_slots: Vec<BTreeMap<String, (u32, u32)>>,
    ) -> Result<Self, BackendError> {
        artifact.canonical_value_by_name().map_err(|collision| {
            invalid_module(&format!("resource name collision: {collision}"))
        })?;
        let resources = project_resources(artifact);
        let mut direct_predecessors: BTreeMap<ArtifactValueId, ArtifactValueId> = BTreeMap::new();
        for resource in artifact.resources() {
            if let Some(pred) = resource.retained_predecessor {
                direct_predecessors.insert(resource.value, pred);
            }
        }
        let mut retained_predecessors: BTreeMap<ArtifactValueId, Vec<ArtifactValueId>> =
            BTreeMap::new();
        for &succ in direct_predecessors.keys() {
            let mut priors = Vec::new();
            let mut curr = succ;
            let mut visited = BTreeSet::new();
            while let Some(&prev) = direct_predecessors.get(&curr) {
                if !visited.insert(prev) {
                    return Err(invalid_module(&format!(
                        "retained predecessor lineage for resource {} contains a cycle at resource {}",
                        succ.0, prev.0
                    )));
                }
                priors.push(prev);
                curr = prev;
            }
            if !priors.is_empty() {
                retained_predecessors.insert(succ, priors);
            }
        }

        let mut entries_by_node: BTreeMap<ArtifactNodeId, _> = BTreeMap::new();
        for entry in payload.entries() {
            if entries_by_node.insert(entry.node, entry).is_some() {
                return Err(invalid_module(
                    "target payload entries must name distinct canonical nodes",
                ));
            }
        }
        let mut fusion = artifact.fusion().iter().collect::<Vec<_>>();
        fusion.sort_by_key(|record| (record.stage, record.id));
        let mut ordered_entries = Vec::with_capacity(fusion.len());
        let mut module_named_resources = Vec::with_capacity(fusion.len());
        for record in fusion {
            let node = *record.members.first().ok_or_else(|| {
                invalid_module("the selected plan lists a fusion group with no member node")
            })?;
            let entry = entries_by_node.remove(&node).ok_or_else(|| {
                invalid_module(
                    "target payload entry identity must match the first node of its fusion group",
                )
            })?;
            // One Program buffer name carries up to two artifact identities: a
            // `ReadWrite` buffer reads the resource it was wired to and writes the
            // renamed successor, and those two are linked by
            // `retained_predecessor`. Rejecting the pair rejected every fixpoint
            // node in the tree, so the collision that matters is two identities
            // under one name that the retained chain does not relate at all.
            // Resolution keeps the write identity because
            // `value_for_module_canonical` walks the chain in either direction.
            let mut named_resources = BTreeMap::new();
            for member in &record.members {
                let abi_entry = artifact
                    .abi()
                    .entries
                    .iter()
                    .find(|candidate| candidate.node == *member)
                    .ok_or_else(|| {
                        invalid_module(
                            "a selected fusion-group member has no named artifact ABI entry",
                        )
                    })?;
                for (direction, bindings) in [
                    ("input", &abi_entry.input_bindings),
                    ("output", &abi_entry.output_bindings),
                ] {
                    for binding in bindings {
                        if let Some(existing) =
                            named_resources.insert(binding.buffer.clone(), binding.value)
                        {
                            if existing != binding.value
                                && !retained_chain_relates(
                                    &retained_predecessors,
                                    existing,
                                    binding.value,
                                )
                            {
                                return Err(invalid_module(&format!(
                                    "fusion group {} maps Program buffer `{}` to unrelated {direction} resources {} and {}",
                                    record.id.0, binding.buffer, existing.0, binding.value.0
                                )));
                            }
                        }
                    }
                }
            }
            ordered_entries.push(entry);
            module_named_resources.push(named_resources);
        }

        let mut module_inputs = Vec::with_capacity(ordered_entries.len());
        let mut module_outputs = Vec::with_capacity(ordered_entries.len());
        let mut module_resources = Vec::with_capacity(ordered_entries.len());
        for entry in ordered_entries {
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();
            let mut resources = BTreeMap::new();
            for binding in &entry.resource_bindings {
                resources.insert((binding.group, binding.slot), binding.resource);
                match binding.access {
                    vyre_megakernel::TargetResourceAccess::ReadOnly => {
                        inputs.push(binding.resource);
                    }
                    vyre_megakernel::TargetResourceAccess::WriteOnly => {
                        outputs.push(binding.resource);
                    }
                    vyre_megakernel::TargetResourceAccess::ReadWrite => {
                        // A read-write resource is read before it is written, so
                        // the module needs an identity carrying its current bytes
                        // whatever the resource's lifetime is. An `Output`
                        // lifetime says where the bytes end up, not that nothing
                        // reads them: `vyre-libs::security::aliases_dataflow`
                        // binds `out` as one of its witnesses and reads it in the
                        // same pass that writes it, and naming no input identity
                        // for it failed the backend dispatch before the first
                        // case ran.
                        //
                        // A retained buffer is one allocation the caller binds
                        // once, under the identity at the head of its chain; the
                        // renamed successors are versions of that same
                        // allocation. Reading the successor here made a module
                        // demand an identity no caller can bind, so the input
                        // projection names the root and the output projection
                        // names the version this module writes.
                        inputs.push(
                            retained_predecessors
                                .get(&binding.resource)
                                .and_then(|priors| priors.last())
                                .copied()
                                .unwrap_or(binding.resource),
                        );
                        outputs.push(binding.resource);
                    }
                }
            }
            module_inputs.push(inputs);
            module_outputs.push(outputs);
            module_resources.push(resources);
        }

        Ok(Self {
            artifact: artifact.digest(),
            payload: payload.digest(),
            device,
            values: resources.values,
            outputs: resources.outputs,
            retained: resources.retained,
            messages,
            module_inputs,
            module_outputs,
            module_named_resources,
            module_buffer_slots,
            module_resources,
            retained_predecessors,
        })
    }

    /// Reject bindings that name a different artifact than this instance.
    ///
    /// # Errors
    ///
    /// Returns the backend's `foreign_artifact` rejection when the digests differ.
    pub fn accept(&self, bindings: &BindingSet) -> Result<(), BackendError> {
        if bindings.artifact() == self.artifact {
            return Ok(());
        }
        Err((self.messages.foreign_artifact)())
    }

    /// Resolve the canonical artifact value a Program buffer projects onto.
    ///
    /// # Errors
    ///
    /// Returns the backend's `unmapped_buffer` rejection when the artifact ABI
    /// declares no resource under `name`.
    pub fn value_for_buffer(&self, name: &str) -> Result<ArtifactValueId, BackendError> {
        self.values
            .get(name)
            .copied()
            .ok_or_else(|| (self.messages.unmapped_buffer)(name))
    }

    /// Resolve one Program binding by authenticated identity.
    ///
    /// Active entry-boundary buffers resolve through the named neutral ABI.
    /// A split segment retains the source Program's complete buffer table, so
    /// an inactive declaration may be absent from that segment's entry ABI; it
    /// resolves through the exact target `(group, slot)` metadata instead.
    ///
    /// # Errors
    ///
    /// Returns `unmapped_buffer` when neither identity projection contains the
    /// binding or when the requested direction excludes its canonical value.
    pub fn value_for_module_binding(
        &self,
        module_values: &[Vec<ArtifactValueId>],
        module_index: usize,
        binding: &crate::binding::Binding,
    ) -> Result<ArtifactValueId, BackendError> {
        if let Some(canonical) = self
            .module_named_resources
            .get(module_index)
            .and_then(|resources| resources.get(binding.name.as_ref()))
            .copied()
        {
            return self.value_for_module_canonical(
                module_values,
                module_index,
                canonical,
                &binding.name,
            );
        }
        let (group, slot) = self
            .module_buffer_slots
            .get(module_index)
            .and_then(|slots| slots.get(binding.name.as_ref()))
            .copied()
            .ok_or_else(|| (self.messages.unmapped_buffer)(&binding.name))?;
        self.value_for_module_slot(module_values, module_index, group, slot, &binding.name)
    }

    /// Resolve one exact target descriptor slot through an input or output projection.
    ///
    /// # Errors
    ///
    /// Returns `unmapped_buffer` when the module, slot, or directional
    /// projection does not contain the target binding.
    pub fn value_for_module_slot(
        &self,
        module_values: &[Vec<ArtifactValueId>],
        module_index: usize,
        group: u32,
        slot: u32,
        name: &str,
    ) -> Result<ArtifactValueId, BackendError> {
        let resources = self.module_resources.get(module_index).ok_or_else(|| {
            invalid_module(&format!(
                "target module {module_index} is absent while resolving binding `{name}`"
            ))
        })?;
        let canonical = resources.get(&(group, slot)).copied().ok_or_else(|| {
            invalid_module(&format!(
                "target binding `{name}` at group {group}, slot {slot} has no canonical resource identity"
            ))
        })?;
        self.value_for_module_canonical(module_values, module_index, canonical, name)
    }

    fn value_for_module_canonical(
        &self,
        module_values: &[Vec<ArtifactValueId>],
        module_index: usize,
        canonical: ArtifactValueId,
        name: &str,
    ) -> Result<ArtifactValueId, BackendError> {
        let values = module_values.get(module_index).ok_or_else(|| {
            invalid_module(&format!(
                "target module {module_index} has no directional resource projection for binding `{name}`"
            ))
        })?;
        if values.contains(&canonical) {
            return Ok(canonical);
        }
        values
            .iter()
            .find(|&&value| {
                self.retained_predecessors
                    .get(&canonical)
                    .is_some_and(|priors| priors.contains(&value))
                    || self
                        .retained_predecessors
                        .get(&value)
                        .is_some_and(|priors| priors.contains(&canonical))
            })
            .copied()
            .ok_or_else(|| {
                let input = self
                    .module_inputs
                    .get(module_index)
                    .is_some_and(|values| {
                        values.contains(&canonical)
                            || self
                                .retained_predecessors
                                .get(&canonical)
                                .is_some_and(|priors| priors.iter().any(|p| values.contains(p)))
                            || values.iter().any(|v| {
                                self.retained_predecessors
                                    .get(v)
                                    .is_some_and(|priors| priors.contains(&canonical))
                            })
                    });
                let output = self
                    .module_outputs
                    .get(module_index)
                    .is_some_and(|values| {
                        values.contains(&canonical)
                            || self
                                .retained_predecessors
                                .get(&canonical)
                                .is_some_and(|priors| priors.iter().any(|p| values.contains(p)))
                            || values.iter().any(|v| {
                                self.retained_predecessors
                                    .get(v)
                                    .is_some_and(|priors| priors.contains(&canonical))
                            })
                    });
                invalid_module(&format!(
                    "canonical artifact value {} for target binding `{name}` is absent from this module's requested directional resource projection (input={input}, output={output})",
                    canonical.0
                ))
            })
    }

    /// Borrow bound host bytes into the input order the binding plan declares.
    ///
    /// # Errors
    ///
    /// Returns `unmapped_buffer` for a buffer outside the artifact ABI, and
    /// `unbound` for a declared input whose value was never bound.
    pub fn gather_inputs_for_module<'state>(
        &self,
        module_index: usize,
        plan: &BindingPlan,
        program: &Program,
        state: &'state BTreeMap<ArtifactValueId, Vec<u8>>,
        unbound: fn(ArtifactValueId, &str) -> BackendError,
    ) -> Result<Vec<&'state [u8]>, BackendError> {
        let input_count = plan
            .bindings
            .iter()
            .filter_map(|binding| binding.input_index)
            .max()
            .map_or(0, |index| index + 1);
        let mut inputs = vec![&[][..]; input_count];
        for binding in &plan.bindings {
            let Some(input_index) = binding.input_index else {
                continue;
            };
            let buffer = &program.buffers()[binding.buffer_index];
            let value =
                self.value_for_module_binding(&self.module_inputs, module_index, binding)?;
            inputs[input_index] = state
                .get(&value)
                .map(Vec::as_slice)
                .ok_or_else(|| unbound(value, buffer.name()))?;
        }
        Ok(inputs)
    }

    /// Borrow bound host bytes into the input order the binding plan declares.
    ///
    /// # Errors
    ///
    /// Returns `unmapped_buffer` for a buffer outside the artifact ABI, and
    /// `unbound` for a declared input whose value was never bound.
    pub fn gather_inputs<'state>(
        &self,
        plan: &BindingPlan,
        program: &Program,
        state: &'state BTreeMap<ArtifactValueId, Vec<u8>>,
        unbound: fn(ArtifactValueId, &str) -> BackendError,
    ) -> Result<Vec<&'state [u8]>, BackendError> {
        self.gather_inputs_for_module(0, plan, program, state, unbound)
    }

    /// Move dispatch results onto the canonical values they implement for a module.
    ///
    /// # Errors
    ///
    /// Returns `unmapped_buffer` for a buffer outside the artifact ABI, and
    /// `missing` when the dispatch produced no bytes for a declared output
    /// index.
    pub fn absorb_outputs_for_module(
        &self,
        module_index: usize,
        plan: &BindingPlan,
        program: &Program,
        produced: Vec<Vec<u8>>,
        state: &mut BTreeMap<ArtifactValueId, Vec<u8>>,
        missing: impl Fn(usize, &str) -> BackendError,
    ) -> Result<(), BackendError> {
        let mut produced: Vec<Option<Vec<u8>>> = produced.into_iter().map(Some).collect();
        for binding in &plan.bindings {
            let Some(output_index) = binding.output_index else {
                continue;
            };
            let buffer = &program.buffers()[binding.buffer_index];
            let value =
                self.value_for_module_binding(&self.module_outputs, module_index, binding)?;
            let bytes = produced
                .get_mut(output_index)
                .and_then(Option::take)
                .ok_or_else(|| missing(output_index, buffer.name()))?;
            if let Some(priors) = self.retained_predecessors.get(&value) {
                for prior in priors {
                    state.insert(*prior, bytes.clone());
                }
            }
            state.insert(value, bytes);
        }
        Ok(())
    }

    /// Move dispatch results onto the canonical values they implement.
    ///
    /// `produced` is consumed. It was borrowed and each buffer cloned, which
    /// copied every output byte a dispatch returned, on every dispatch, into a
    /// map whose only reader then cloned it again on the way out. The caller
    /// owns the dispatch result and drops it here, so there is nothing for the
    /// copy to protect.
    ///
    /// A binding's output slot is assigned per buffer, so two bindings cannot
    /// name one slot. If one ever does, the second finds the slot already taken
    /// and is refused rather than served the bytes of the value that took it.
    ///
    /// # Errors
    ///
    /// Returns `unmapped_buffer` for a buffer outside the artifact ABI, and
    /// `missing` when the dispatch produced no bytes for a declared output
    /// index.
    pub fn absorb_outputs(
        &self,
        plan: &BindingPlan,
        program: &Program,
        produced: Vec<Vec<u8>>,
        state: &mut BTreeMap<ArtifactValueId, Vec<u8>>,
        missing: impl Fn(usize, &str) -> BackendError,
    ) -> Result<(), BackendError> {
        self.absorb_outputs_for_module(0, plan, program, produced, state, missing)
    }

    /// Collect `values` out of executed state, rejecting any that is absent.
    ///
    /// # Errors
    ///
    /// Returns `missing` for the first value execution did not leave in `state`.
    pub fn project(
        &self,
        values: &BTreeSet<ArtifactValueId>,
        state: &BTreeMap<ArtifactValueId, Vec<u8>>,
        missing: fn(ArtifactValueId) -> BackendError,
    ) -> Result<BTreeMap<ArtifactValueId, Vec<u8>>, BackendError> {
        values
            .iter()
            .map(|value| {
                state
                    .get(value)
                    .cloned()
                    .map(|bytes| (*value, bytes))
                    .ok_or_else(|| missing(*value))
            })
            .collect()
    }

    /// Build the completion for one execution's final state.
    ///
    /// # Errors
    ///
    /// Returns `missing_output_value` or `missing_retained_value` when execution
    /// did not leave a declared value behind.
    pub fn completion(
        &self,
        state: &BTreeMap<ArtifactValueId, Vec<u8>>,
        device_ns: Option<u64>,
    ) -> Result<Completion, BackendError> {
        Ok(Completion {
            artifact: self.artifact,
            outputs: self.project(&self.outputs, state, self.messages.missing_output_value)?,
            retained: self.project(&self.retained, state, self.messages.missing_retained_value)?,
            device_ns,
        })
    }
}

#[path = "materialize_test_fixtures.rs"]
#[cfg(test)]
mod materialize_test_fixtures;

#[path = "materialize_tests.rs"]
#[cfg(test)]
mod materialize_tests;

#[path = "materialize_resident_tests.rs"]
#[cfg(test)]
mod materialize_resident_tests;
