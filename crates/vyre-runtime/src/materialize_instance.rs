//! Materialized instance execution and resident submission paths.

use std::collections::BTreeMap;

use vyre_compiler::ArtifactValueId;
use vyre_foundation::ir::Program;

use crate::{
    BackendError, BindingPlan, BindingRole, BindingSet, BoundResource, Completion, DispatchConfig,
    OutputBuffers, Resource, Submission, TimedDispatchResult,
};

use crate::materialize::{
    omitted_output, unbound_input, unbound_resident_buffer, ExecutableModule, InstanceCore,
    InstanceMessages,
};

/// A binding set whose every value is bound to a device-resident resource.
///
/// The view borrows the caller's bindings, so routing a resident submission
/// builds no second map and clones no handle until a launch orders them.
#[derive(Clone, Copy, Debug)]
pub struct ResidentBindings<'a> {
    resources: &'a BTreeMap<ArtifactValueId, BoundResource>,
}

impl<'a> ResidentBindings<'a> {
    /// Borrow `bindings` as resident handles, or `None` when any value is
    /// bound to host bytes.
    #[must_use]
    pub fn new(bindings: &'a BindingSet) -> Option<Self> {
        let resources = bindings.resources();
        resources
            .values()
            .all(|resource| matches!(resource, BoundResource::Resident(_)))
            .then_some(Self { resources })
    }

    /// The resident handle bound to `value`.
    #[must_use]
    pub fn get(self, value: &ArtifactValueId) -> Option<&'a Resource> {
        match self.resources.get(value)? {
            BoundResource::Resident(resource) => Some(resource),
            BoundResource::Host(_) => None,
        }
    }

    /// Every bound value and its resident handle, in value order.
    pub fn iter(self) -> impl Iterator<Item = (&'a ArtifactValueId, &'a Resource)> {
        self.resources
            .iter()
            .filter_map(|(value, resource)| match resource {
                BoundResource::Resident(resource) => Some((value, resource)),
                BoundResource::Host(_) => None,
            })
    }
}

/// Move the host bytes out of `bindings`, dropping any resident binding.
///
/// Callers route a set to the host path only after checking it binds no
/// resident value.
pub(crate) fn host_state(bindings: BindingSet) -> BTreeMap<ArtifactValueId, Vec<u8>> {
    bindings
        .into_resources()
        .into_iter()
        .filter_map(|(value, resource)| match resource {
            BoundResource::Host(bytes) => Some((value, bytes)),
            BoundResource::Resident(_) => None,
        })
        .collect()
}

/// Whether `bindings` binds any host value and any resident value.
fn binding_kinds(bindings: &BindingSet) -> (bool, bool) {
    bindings.resources().values().fold(
        (false, false),
        |(host, resident), resource| match resource {
            BoundResource::Host(_) => (true, resident),
            BoundResource::Resident(_) => (host, true),
        },
    )
}

/// Take the bound host bytes on a backend with no resident submission path.
///
/// `feature` names the rejected capability in the backend's own words; the
/// walk and the rejection class are the same wherever it is refused.
///
/// # Errors
///
/// Returns `BackendError::UnsupportedFeature` when any value is bound to a
/// device-resident resource.
pub fn host_only_bindings(
    bindings: BindingSet,
    feature: &str,
    backend: &str,
) -> Result<BTreeMap<ArtifactValueId, Vec<u8>>, BackendError> {
    if binding_kinds(&bindings).1 {
        return Err(BackendError::UnsupportedFeature {
            name: feature.to_string(),
            backend: backend.to_string(),
        });
    }
    Ok(host_state(bindings))
}

impl InstanceCore {
    /// Wrap an already-finished execution as a submission.
    #[must_use]
    pub fn ready(&self, result: Result<Completion, BackendError>) -> Box<dyn Submission> {
        Box::new(ReadySubmission {
            result: Some(result),
            consumed: self.messages.completion_consumed,
        })
    }

    /// Route one submission to the host or the resident execution path.
    ///
    /// A backend with both paths refuses a binding set that spans them: a
    /// single dispatch cannot read half its resources from caller memory and
    /// half from device memory without deciding which side wins for a value
    /// bound twice, and no backend makes that decision. `mixed` supplies the
    /// refusal in the backend's own words.
    ///
    /// # Errors
    ///
    /// Returns the backend's `foreign_artifact` rejection when the bindings
    /// name another artifact, and `mixed` when they span both paths. A failure
    /// inside either closure is carried by the returned submission instead,
    /// because the execution it describes did begin.
    pub fn route_submission(
        &self,
        bindings: BindingSet,
        mixed: fn() -> BackendError,
        host: impl FnOnce(BTreeMap<ArtifactValueId, Vec<u8>>) -> Result<Completion, BackendError>,
        resident: impl FnOnce(ResidentBindings<'_>) -> Result<Completion, BackendError>,
    ) -> Result<Box<dyn Submission>, BackendError> {
        self.accept(&bindings)?;
        let result = match binding_kinds(&bindings) {
            (true, true) => return Err(mixed()),
            (_, false) => host(host_state(bindings)),
            (false, true) => resident(ResidentBindings {
                resources: bindings.resources(),
            }),
        };
        Ok(self.ready(result))
    }

    /// Accept every binding set of a resident batch and resolve each one's
    /// resident resources.
    ///
    /// A batched submission is resident by definition: it exists so several
    /// items reach the device in one submission, and a host-bound value has
    /// nothing to submit alongside them.
    ///
    /// # Errors
    ///
    /// Returns the backend's `foreign_artifact` rejection when any set names
    /// another artifact, and `mixed` when any set binds a host value.
    pub fn resident_batch_resources<'a>(
        &self,
        batches: &'a [BindingSet],
        mixed: fn() -> BackendError,
    ) -> Result<Vec<ResidentBindings<'a>>, BackendError> {
        let mut items = Vec::new();
        reserve_batch_items(&mut items, batches.len(), "resident batch binding set")?;
        for bindings in batches {
            self.accept(bindings)?;
            items.push(ResidentBindings::new(bindings).ok_or_else(mixed)?);
        }
        Ok(items)
    }

    /// Route one submission on a backend that has no resident execution path.
    ///
    /// The refusal and host state come from the same calls wherever there is
    /// only one path to route to, so the backends that have one wrote the same
    /// three lines. `feature` names the refused capability in
    /// the backend's own words; the backend itself comes from the recorded
    /// device generation rather than a constant restated at the call site, so a
    /// materializer cannot report a resident refusal against a backend it did
    /// not acquire.
    ///
    /// # Errors
    ///
    /// Returns the backend's `foreign_artifact` rejection when the bindings name
    /// another artifact, and `BackendError::UnsupportedFeature` when any value
    /// is bound to a device-resident resource. A failure inside `host` is
    /// carried by the returned submission instead, because the execution it
    /// describes did begin.
    pub fn submit_host_only(
        &self,
        bindings: BindingSet,
        feature: &str,
        host: impl FnOnce(BTreeMap<ArtifactValueId, Vec<u8>>) -> Result<Completion, BackendError>,
    ) -> Result<Box<dyn Submission>, BackendError> {
        self.accept(&bindings)?;
        let state = host_only_bindings(bindings, feature, &self.device.backend)?;
        Ok(self.ready(host(state)))
    }

    /// Resolve one Program buffer name by authenticated identity for a resident launch.
    ///
    /// Active entry-boundary buffers resolve through the named neutral ABI or
    /// module named resources. Inactive or segment-internal declarations resolve
    /// through the exact target `(group, slot)` descriptor metadata.
    ///
    /// # Errors
    ///
    /// Returns `unmapped_buffer` when neither projection contains the name.
    pub fn value_for_resident_name(
        &self,
        module_index: usize,
        name: &str,
    ) -> Result<ArtifactValueId, BackendError> {
        if let Some(canonical) = self
            .module_named_resources
            .get(module_index)
            .and_then(|resources| resources.get(name))
            .copied()
        {
            return Ok(canonical);
        }
        if let Some(&(group, slot)) = self
            .module_buffer_slots
            .get(module_index)
            .and_then(|slots| slots.get(name))
        {
            if let Some(canonical) = self
                .module_resources
                .get(module_index)
                .and_then(|resources| resources.get(&(group, slot)))
                .copied()
            {
                return Ok(canonical);
            }
        }
        self.value_for_buffer(name)
    }

    /// Lookup one bound resident resource, honoring transitive retained lineage.
    #[must_use]
    pub fn lookup_resident_resource<'a>(
        &self,
        value: ArtifactValueId,
        resources: ResidentBindings<'a>,
    ) -> Option<&'a Resource> {
        if let Some(resource) = resources.get(&value) {
            return Some(resource);
        }
        if let Some(priors) = self.retained_predecessors.get(&value) {
            for prior in priors {
                if let Some(resource) = resources.get(prior) {
                    return Some(resource);
                }
            }
        }
        for (bound_val, resource) in resources.iter() {
            if let Some(priors) = self.retained_predecessors.get(bound_val) {
                if priors.contains(&value) {
                    return Some(resource);
                }
            }
        }
        None
    }

    /// Append resident handles to `ordered` in the order `names` declares.
    ///
    /// A resident launch takes an ordered handle list, and the order is the
    /// backend's: one reads it off the binding plan's non-shared roles, another
    /// off the persistent resource names its pipeline reports. The lookup is
    /// the same per name: project the buffer name onto its canonical artifact
    /// value, find the resource bound to that value, and fail when nothing is.
    /// Appending lets a batch lay every item's handles out in one buffer.
    ///
    /// # Errors
    ///
    /// Returns `unmapped_buffer` for a name outside the artifact ABI, and
    /// `unbound` for a declared resident name whose canonical value carries no
    /// resource.
    pub fn ordered_resident_resources<'names>(
        &self,
        module_index: usize,
        names: impl IntoIterator<Item = &'names str>,
        resources: ResidentBindings<'_>,
        unbound: impl Fn(ArtifactValueId, &str) -> BackendError,
        ordered: &mut Vec<Resource>,
    ) -> Result<(), BackendError> {
        for name in names {
            let value = self.value_for_resident_name(module_index, name)?;
            let resource = self
                .lookup_resident_resource(value, resources)
                .ok_or_else(|| unbound(value, name))?;
            ordered.push(resource.clone());
        }
        Ok(())
    }

    /// Append resident handles for one module, in binding-plan order, to
    /// `ordered`.
    ///
    /// # Errors
    ///
    /// Returns `unmapped_buffer` for a binding outside the artifact ABI, and
    /// `unbound` for a declared resident binding whose canonical value carries no
    /// resource.
    pub fn ordered_resident_resources_for_module(
        &self,
        module_index: usize,
        plan: &BindingPlan,
        program: &Program,
        resources: ResidentBindings<'_>,
        unbound: impl Fn(ArtifactValueId, &str) -> BackendError,
        ordered: &mut Vec<Resource>,
    ) -> Result<(), BackendError> {
        let names = plan
            .bindings
            .iter()
            .filter(|binding| binding.role != BindingRole::Shared)
            .map(|binding| program.buffers()[binding.buffer_index].name());
        self.ordered_resident_resources(module_index, names, resources, unbound, ordered)
    }

    /// Dispatch every module in order and complete the accumulated state.
    ///
    /// Device time sums across modules and is reported only when at least one
    /// module carried a device timer, so a backend that times some modules and
    /// not others reports the partial sum rather than a total that silently
    /// omits the untimed work. The accumulation saturates because a wrapped sum
    /// would report a fast execution.
    ///
    /// `dispatch` gathers the module's inputs out of the state read so far and
    /// runs it; the outputs it returns are absorbed back onto the canonical
    /// values before the next module reads them.
    ///
    /// # Errors
    ///
    /// Returns whatever `dispatch` reports, `omitted` for a declared output the
    /// module did not produce, and the backend's completion rejections when
    /// execution left a declared value behind.
    pub fn execute_modules<M: ExecutableModule>(
        &self,
        modules: &[M],
        mut state: BTreeMap<ArtifactValueId, Vec<u8>>,
        omitted: impl Fn(usize, &str) -> BackendError,
        mut dispatch: impl FnMut(
            usize,
            &M,
            &BindingPlan,
            &DispatchConfig,
            &BTreeMap<ArtifactValueId, Vec<u8>>,
        ) -> Result<TimedDispatchResult, BackendError>,
    ) -> Result<Completion, BackendError> {
        let mut device_ns = 0_u64;
        let mut has_device_timing = false;
        for (module_index, module) in modules.iter().enumerate() {
            let plan = module.plan();
            let dispatched = dispatch(module_index, module, plan, module.config(), &state)?;
            if let Some(ns) = dispatched.device_ns {
                device_ns = device_ns.saturating_add(ns);
                has_device_timing = true;
            }
            self.absorb_outputs_for_module(
                module_index,
                plan,
                module.program(),
                dispatched.outputs,
                &mut state,
                &omitted,
            )?;
        }
        self.completion(state, has_device_timing.then_some(device_ns))
    }
}

/// The execution paths of a materialized instance, defaulted over
/// [`InstanceCore`].
///
/// Every backend wrote the same two bodies around the same [`InstanceCore`]
/// calls: gather the module's inputs and launch it once per module of the
/// selected plan, then complete the accumulated state. Only the launch is
/// target-specific, so only the launch is required here. The identity methods
/// of [`crate::ArtifactInstance`] come from [`macro@crate::artifact_instance_identity`] and
/// `submit` routes through [`Self::submit_host_only`] or
/// [`ResidentInstance::submit_routed`], which leaves a backend the launch, its
/// rejection text, and nothing else.
pub trait MaterializedInstance {
    /// The backend's own module record, holding its native handle beside the
    /// canonical Program and dispatch config the shared loop reads.
    type Module: ExecutableModule;

    /// Identity and artifact ABI recorded when this instance materialized.
    fn core(&self) -> &InstanceCore;

    /// Every module of the compiler-selected plan, in dispatch order.
    fn modules(&self) -> &[Self::Module];

    /// Label naming this backend's target module in an omitted-output
    /// rejection.
    ///
    /// The sentence is [`omitted_output`]; the label is the only part the
    /// backends disagreed on, and every one of them had wrapped that one string
    /// in its own free function to hand the loop a rejection.
    fn module_label(&self) -> &'static str;

    /// Launch one module over its borrowed input bytes.
    ///
    /// `inputs` is in the order [`Self::gather`] produced and `config` is the
    /// module's admitted dispatch config.
    ///
    /// # Errors
    ///
    /// Returns whatever the launch reports.
    fn dispatch(
        &self,
        module: &Self::Module,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<TimedDispatchResult, BackendError>;

    /// Borrow one module's inputs out of the bound state.
    ///
    /// The default is the input order the binding plan declares, which is what
    /// a launch that binds from the Program reads. A backend whose target
    /// module declares its own binding order overrides this and walks that
    /// order instead.
    ///
    /// A returned slice may borrow from the instance or its module record
    /// rather than from `state`: a target module whose emitted binding order
    /// carries no gap stages a slot the same module writes, and nothing has
    /// bound that value yet.
    ///
    /// # Errors
    ///
    /// Returns `unmapped_buffer` for a buffer outside the artifact ABI, and the
    /// unbound-input rejection for a declared input whose value is not bound.
    fn gather<'a>(
        &'a self,
        module_index: usize,
        module: &'a Self::Module,
        plan: &BindingPlan,
        state: &'a BTreeMap<ArtifactValueId, Vec<u8>>,
    ) -> Result<Vec<&'a [u8]>, BackendError> {
        self.core().gather_inputs_for_module(
            module_index,
            plan,
            module.program(),
            state,
            unbound_input,
        )
    }

    /// Dispatch every module over caller-owned bytes and complete the state.
    ///
    /// # Errors
    ///
    /// Returns whatever [`Self::gather`] and [`Self::dispatch`] report, the
    /// omitted-output rejection for a declared output no module produced, and
    /// the instance's completion rejections when execution left a declared
    /// value behind.
    fn execute_host(
        &self,
        state: BTreeMap<ArtifactValueId, Vec<u8>>,
    ) -> Result<Completion, BackendError> {
        let label = self.module_label();
        self.core().execute_modules(
            self.modules(),
            state,
            |output_index, name| omitted_output(label, output_index, name),
            |module_index, module, plan, config, state| {
                let inputs = self.gather(module_index, module, plan, state)?;
                self.dispatch(module, &inputs, config)
            },
        )
    }

    /// Route one submission on an instance with no resident execution path.
    ///
    /// `feature` names the refused capability in the backend's own words.
    ///
    /// # Errors
    ///
    /// Returns the instance's `foreign_artifact` rejection when the bindings
    /// name another artifact, and `BackendError::UnsupportedFeature` when any
    /// value is bound to a device-resident resource.
    fn submit_host_only(
        &self,
        bindings: BindingSet,
        feature: &str,
    ) -> Result<Box<dyn Submission>, BackendError> {
        self.core()
            .submit_host_only(bindings, feature, |state| self.execute_host(state))
    }
}

/// Reserve exactly `len` slots for a resident batch, naming what ran out.
///
/// A batch is sized by the caller, so an allocation that cannot be served is a
/// rejected submission and not a panic.
///
/// # Errors
///
/// Returns [`BackendError::InvalidProgram`] carrying `what` and the allocator's
/// reason.
fn reserve_batch_items<T>(items: &mut Vec<T>, len: usize, what: &str) -> Result<(), BackendError> {
    items.try_reserve_exact(len).map_err(|error| {
        BackendError::InvalidProgram {
            fix: format!(
                "Fix: failed to reserve {len} {what} slot(s): {error}. Submit a smaller resident batch."
            ),
        }
    })
}

/// The resident execution path of a materialized instance whose backend has
/// one.
///
/// A resident launch is one module over device memory the caller already
/// filled, and an artifact contributes as many of them as its selected plan
/// recorded. Three backends wrote the same seven steps around a single launch
/// and differed on two: where the handle order comes from, and whether an
/// unproduced resident value is worded the way the host path words it.
pub trait ResidentInstance: MaterializedInstance {
    /// Label naming this backend's resident target module in an omitted-output
    /// rejection.
    fn resident_module_label(&self) -> &'static str;

    /// Completion rejections for the resident path.
    ///
    /// The host path's wording unless the backend words a resident value
    /// differently.
    fn resident_messages(&self) -> &InstanceMessages {
        &self.core().messages
    }

    /// Append the caller's resources to `ordered` in the order the resident
    /// launch reads.
    ///
    /// The default is the binding plan's non-shared roles, which is the order a
    /// launch that takes its handles from the Program reads. A backend whose
    /// target module declares its own resident order overrides this and names
    /// that order instead.
    ///
    /// # Errors
    ///
    /// Returns `unmapped_buffer` for a name outside the artifact ABI, and the
    /// unbound-resident rejection for a declared name carrying no resource.
    fn ordered_resident(
        &self,
        module_index: usize,
        module: &Self::Module,
        plan: &BindingPlan,
        resources: ResidentBindings<'_>,
        ordered: &mut Vec<Resource>,
    ) -> Result<(), BackendError> {
        self.core().ordered_resident_resources_for_module(
            module_index,
            plan,
            module.program(),
            resources,
            unbound_resident_buffer,
            ordered,
        )
    }

    /// Launch one resident module over `ordered`.
    ///
    /// `config` is the module's admitted dispatch config, which carries the
    /// frozen launch.
    ///
    /// # Errors
    ///
    /// Returns whatever the launch reports.
    fn launch_resident(
        &self,
        module: &Self::Module,
        ordered: &[Resource],
        config: &DispatchConfig,
    ) -> Result<TimedDispatchResult, BackendError>;

    /// Launch every module of the selected plan over caller-owned resident
    /// resources, in the recorded order.
    ///
    /// A multi-entry artifact records which entry point produces each value a
    /// later one reads, and the allocation plan states where those values live, so
    /// a resident submission needs no host round trip between entry points: the
    /// caller binds every canonical value once and each launch reads the
    /// handles its own module declares. Nothing here decides an order or a
    /// resource. The order is the artifact's recorded plan order, which decode
    /// refuses unless it follows the dependency DAG, and every handle comes out
    /// of `resources`.
    ///
    /// Device time sums across modules and is reported only when at least one
    /// module carried a device timer, so a backend that times some launches and
    /// not others reports the partial sum rather than a total that omits the
    /// untimed work.
    ///
    /// # Errors
    ///
    /// Returns whatever [`Self::ordered_resident`] and [`Self::launch_resident`]
    /// report, and the resident completion rejections when a launch left a
    /// declared value behind.
    fn execute_resident(
        &self,
        resources: ResidentBindings<'_>,
    ) -> Result<Completion, BackendError> {
        let core = self.core();
        let label = self.resident_module_label();
        let mut state = BTreeMap::new();
        let mut device_ns = 0_u64;
        let mut has_device_timing = false;
        let mut ordered = Vec::new();
        for (module_index, module) in self.modules().iter().enumerate() {
            let plan = module.plan();
            ordered.clear();
            self.ordered_resident(module_index, module, plan, resources, &mut ordered)?;
            let dispatched = self.launch_resident(module, &ordered, module.config())?;
            if let Some(ns) = dispatched.device_ns {
                device_ns = device_ns.saturating_add(ns);
                has_device_timing = true;
            }
            core.absorb_outputs_for_module(
                module_index,
                plan,
                module.program(),
                dispatched.outputs,
                &mut state,
                |output_index, name| omitted_output(label, output_index, name),
            )?;
        }
        core.completion_with(
            state,
            has_device_timing.then_some(device_ns),
            self.resident_messages(),
        )
    }

    /// Launch one resident module over every item of a batch.
    ///
    /// A backend that can submit the whole batch before awaiting any of it
    /// overrides this, so item N is already enqueued when item N-1's work
    /// finishes and the device timer of each item covers the item's work rather
    /// than the host latency of submitting it. The default is one
    /// submit-and-wait per item, which is what the batch replaces.
    ///
    /// A batched item has no host window of its own, because the whole batch is
    /// submitted before any item is awaited, so each item reports its outputs
    /// and its device duration and no wall time.
    ///
    /// # Errors
    ///
    /// Returns whatever the launches report.
    fn launch_resident_batch(
        &self,
        module: &Self::Module,
        batches: &[&[Resource]],
        config: &DispatchConfig,
    ) -> Result<Vec<(OutputBuffers, Option<u64>)>, BackendError> {
        let mut dispatched = Vec::new();
        reserve_batch_items(&mut dispatched, batches.len(), "resident batch launch")?;
        for ordered in batches {
            let timed = self.launch_resident(module, ordered, config)?;
            dispatched.push((timed.outputs, timed.device_ns));
        }
        Ok(dispatched)
    }

    /// Execute every item of a resident batch and report one completion each.
    ///
    /// A single-module plan submits the whole batch through
    /// [`Self::launch_resident_batch`], because every item runs the same module
    /// over its own resources and the launch order across items is the caller's
    /// submission order. A multi-module plan runs item by item through
    /// [`Self::execute_resident`]: its modules alternate within one item, so a
    /// per-module batch would reorder the recorded plan.
    ///
    /// # Errors
    ///
    /// Returns whatever [`Self::ordered_resident`] and
    /// [`Self::launch_resident_batch`] report, and the resident completion
    /// rejections when a launch left a declared value behind.
    fn execute_resident_batch(
        &self,
        items: &[ResidentBindings<'_>],
    ) -> Result<Vec<Completion>, BackendError> {
        let mut completions = Vec::new();
        reserve_batch_items(&mut completions, items.len(), "resident batch completion")?;
        let modules = self.modules();
        let [module] = modules else {
            for resources in items {
                completions.push(self.execute_resident(*resources)?);
            }
            return Ok(completions);
        };
        if items.is_empty() {
            return Ok(completions);
        }
        let core = self.core();
        let label = self.resident_module_label();
        let plan = module.plan();
        // Every item orders the same module's declared names, so all items
        // share one stride and one handle buffer.
        let mut ordered = Vec::new();
        self.ordered_resident(0, module, plan, items[0], &mut ordered)?;
        let stride = ordered.len();
        reserve_batch_items(
            &mut ordered,
            stride.saturating_mul(items.len() - 1),
            "resident batch handles",
        )?;
        for resources in &items[1..] {
            self.ordered_resident(0, module, plan, *resources, &mut ordered)?;
        }
        let mut views = Vec::new();
        reserve_batch_items(&mut views, items.len(), "resident batch handle views")?;
        if stride == 0 {
            views.resize(items.len(), &[][..]);
        } else {
            views.extend(ordered.chunks_exact(stride));
        }
        let dispatched = self.launch_resident_batch(module, &views, module.config())?;
        for (outputs, device_ns) in dispatched {
            let mut state = BTreeMap::new();
            core.absorb_outputs_for_module(
                0,
                plan,
                module.program(),
                outputs,
                &mut state,
                |output_index, name| omitted_output(label, output_index, name),
            )?;
            completions.push(core.completion_with(state, device_ns, self.resident_messages())?);
        }
        Ok(completions)
    }

    /// Route one submission to the host or the resident path.
    ///
    /// `mixed` supplies the refusal for a binding set spanning both, in the
    /// backend's own words.
    ///
    /// # Errors
    ///
    /// Returns the instance's `foreign_artifact` rejection when the bindings
    /// name another artifact, and `mixed` when they span both paths.
    fn submit_routed(
        &self,
        bindings: BindingSet,
        mixed: fn() -> BackendError,
    ) -> Result<Box<dyn Submission>, BackendError> {
        self.core().route_submission(
            bindings,
            mixed,
            |state| self.execute_host(state),
            |resources| self.execute_resident(resources),
        )
    }

    /// Validate every binding set of a resident batch and execute the batch
    /// through [`Self::execute_resident_batch`].
    ///
    /// `mixed` supplies the refusal for a set that binds a host value, in the
    /// backend's own words.
    ///
    /// # Errors
    ///
    /// Returns the instance's `foreign_artifact` rejection when any set names
    /// another artifact, `mixed` when any set binds a host value, and whatever
    /// [`Self::execute_resident_batch`] reports.
    fn submit_resident_batch_routed(
        &self,
        batches: &[BindingSet],
        mixed: fn() -> BackendError,
    ) -> Result<Vec<Completion>, BackendError> {
        let items = self.core().resident_batch_resources(batches, mixed)?;
        self.execute_resident_batch(&items)
    }
}

/// A submission whose execution already finished when it was created.
pub(crate) struct ReadySubmission {
    pub(crate) result: Option<Result<Completion, BackendError>>,
    pub(crate) consumed: fn() -> BackendError,
}

impl Submission for ReadySubmission {
    fn is_ready(&self) -> bool {
        true
    }

    fn wait(mut self: Box<Self>) -> Result<Completion, BackendError> {
        self.result.take().ok_or_else(self.consumed)?
    }
}
