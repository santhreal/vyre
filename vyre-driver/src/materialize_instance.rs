//! Materialized instance execution and resident submission paths.

use std::collections::BTreeMap;

use vyre_foundation::ir::Program;
use vyre_megakernel::ArtifactValueId;

use crate::{
    BackendError, BindingPlan, BindingRole, BindingSet, BoundResource, Completion, DispatchConfig,
    Resource, Submission, TimedDispatchResult,
};

use crate::materialize::{
    omitted_output, unbound_input, unbound_resident_buffer, ExecutableModule, InstanceCore,
    InstanceMessages,
};

/// Bound resources split by where their bytes live.
#[derive(Debug, Default)]
pub struct BoundState {
    /// Caller-owned bytes, keyed by canonical value.
    pub host: BTreeMap<ArtifactValueId, Vec<u8>>,
    /// Device-resident handles, keyed by canonical value.
    pub resident: BTreeMap<ArtifactValueId, Resource>,
}

/// Split a binding set into its host and resident halves.
///
/// Every backend walked the same map and sorted it the same way; what they do
/// with a resident binding is where they differ.
#[must_use]
pub fn partition_bindings(bindings: &BindingSet) -> BoundState {
    let mut bound = BoundState::default();
    for (value, resource) in bindings.resources() {
        match resource {
            BoundResource::Host(bytes) => {
                bound.host.insert(*value, bytes.clone());
            }
            BoundResource::Resident(resource) => {
                bound.resident.insert(*value, resource.clone());
            }
        }
    }
    bound
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
    bindings: &BindingSet,
    feature: &str,
    backend: &str,
) -> Result<BTreeMap<ArtifactValueId, Vec<u8>>, BackendError> {
    let bound = partition_bindings(bindings);
    if bound.resident.is_empty() {
        return Ok(bound.host);
    }
    Err(BackendError::UnsupportedFeature {
        name: feature.to_string(),
        backend: backend.to_string(),
    })
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
        bindings: &BindingSet,
        mixed: fn() -> BackendError,
        host: impl FnOnce(BTreeMap<ArtifactValueId, Vec<u8>>) -> Result<Completion, BackendError>,
        resident: impl FnOnce(&BTreeMap<ArtifactValueId, Resource>) -> Result<Completion, BackendError>,
    ) -> Result<Box<dyn Submission>, BackendError> {
        self.accept(bindings)?;
        let bound = partition_bindings(bindings);
        if !bound.host.is_empty() && !bound.resident.is_empty() {
            return Err(mixed());
        }
        let result = if bound.resident.is_empty() {
            host(bound.host)
        } else {
            resident(&bound.resident)
        };
        Ok(self.ready(result))
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
        bindings: &BindingSet,
        feature: &str,
        host: impl FnOnce(BTreeMap<ArtifactValueId, Vec<u8>>) -> Result<Completion, BackendError>,
    ) -> Result<Box<dyn Submission>, BackendError> {
        self.accept(bindings)?;
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
        resources: &'a BTreeMap<ArtifactValueId, Resource>,
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
        for (bound_val, resource) in resources {
            if let Some(priors) = self.retained_predecessors.get(bound_val) {
                if priors.contains(&value) {
                    return Some(resource);
                }
            }
        }
        None
    }

    /// Resolve resident handles into the order `names` declares.
    ///
    /// A resident launch takes an ordered handle list, and the order is the
    /// backend's: one reads it off the binding plan's non-shared roles, another
    /// off the persistent resource names its pipeline reports. What neither
    /// owns is the lookup, which is the same three steps per name and was
    /// written once per backend: project the buffer name onto its canonical
    /// artifact value, find the resource bound to that value, and refuse when
    /// nothing is.
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
        resources: &BTreeMap<ArtifactValueId, Resource>,
        unbound: impl Fn(ArtifactValueId, &str) -> BackendError,
    ) -> Result<Vec<Resource>, BackendError> {
        let names = names.into_iter();
        let mut ordered = Vec::with_capacity(names.size_hint().0);
        for name in names {
            let value = self.value_for_resident_name(module_index, name)?;
            let resource = self
                .lookup_resident_resource(value, resources)
                .ok_or_else(|| unbound(value, name))?;
            ordered.push(resource.clone());
        }
        Ok(ordered)
    }

    /// Resolve resident handles for one module by authenticated identity.
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
        resources: &BTreeMap<ArtifactValueId, Resource>,
        unbound: impl Fn(ArtifactValueId, &str) -> BackendError,
    ) -> Result<Vec<Resource>, BackendError> {
        let non_shared = plan
            .bindings
            .iter()
            .filter(|binding| binding.role != BindingRole::Shared);
        let mut ordered = Vec::new();
        for binding in non_shared {
            let buffer = &program.buffers()[binding.buffer_index];
            let name = buffer.name();
            let value = self.value_for_resident_name(module_index, name)?;
            let resource = self
                .lookup_resident_resource(value, resources)
                .ok_or_else(|| unbound(value, name))?;
            ordered.push(resource.clone());
        }
        Ok(ordered)
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
            let plan = BindingPlan::build(module.program())?;
            let dispatched = dispatch(module_index, module, &plan, module.config(), &state)?;
            if let Some(ns) = dispatched.device_ns {
                device_ns = device_ns.saturating_add(ns);
                has_device_timing = true;
            }
            self.absorb_outputs_for_module(
                module_index,
                &plan,
                module.program(),
                dispatched.outputs,
                &mut state,
                &omitted,
            )?;
        }
        self.completion(&state, has_device_timing.then_some(device_ns))
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
    /// # Errors
    ///
    /// Returns `unmapped_buffer` for a buffer outside the artifact ABI, and the
    /// unbound-input rejection for a declared input whose value is not bound.
    fn gather<'state>(
        &self,
        module_index: usize,
        module: &Self::Module,
        plan: &BindingPlan,
        state: &'state BTreeMap<ArtifactValueId, Vec<u8>>,
    ) -> Result<Vec<&'state [u8]>, BackendError> {
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
        bindings: &BindingSet,
        feature: &str,
    ) -> Result<Box<dyn Submission>, BackendError> {
        self.core()
            .submit_host_only(bindings, feature, |state| self.execute_host(state))
    }
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

    /// Resolve the caller's resources into the order the resident launch reads.
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
        resources: &BTreeMap<ArtifactValueId, Resource>,
    ) -> Result<Vec<Resource>, BackendError> {
        self.core().ordered_resident_resources_for_module(
            module_index,
            plan,
            module.program(),
            resources,
            unbound_resident_buffer,
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
        resources: &BTreeMap<ArtifactValueId, Resource>,
    ) -> Result<Completion, BackendError> {
        let core = self.core();
        let label = self.resident_module_label();
        let mut state = BTreeMap::new();
        let mut device_ns = 0_u64;
        let mut has_device_timing = false;
        for (module_index, module) in self.modules().iter().enumerate() {
            let plan = BindingPlan::build(module.program())?;
            let ordered = self.ordered_resident(module_index, module, &plan, resources)?;
            let dispatched = self.launch_resident(module, &ordered, module.config())?;
            if let Some(ns) = dispatched.device_ns {
                device_ns = device_ns.saturating_add(ns);
                has_device_timing = true;
            }
            core.absorb_outputs_for_module(
                module_index,
                &plan,
                module.program(),
                dispatched.outputs,
                &mut state,
                |output_index, name| omitted_output(label, output_index, name),
            )?;
        }
        core.completion_with(
            &state,
            has_device_timing.then_some(device_ns),
            self.resident_messages(),
        )
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
        bindings: &BindingSet,
        mixed: fn() -> BackendError,
    ) -> Result<Box<dyn Submission>, BackendError> {
        self.core().route_submission(
            bindings,
            mixed,
            |state| self.execute_host(state),
            |resources| self.execute_resident(resources),
        )
    }
}

/// Answer the three [`crate::ArtifactInstance`] identity methods from an
/// [`InstanceCore`] field named `core`.
///
/// Every backend forwards these three the same way, because the identity of a
/// materialized instance is exactly what its core recorded at materialization.
/// Only `submit` is a per-backend decision, so this expands to associated items
/// inside the backend's own `impl ArtifactInstance` block rather than to a whole
/// impl:
///
/// ```ignore
/// impl ArtifactInstance for TargetArtifactInstance {
///     vyre_driver::artifact_instance_identity!();
///
///     fn submit(&self, bindings: BindingSet) -> Result<Box<dyn Submission>, BackendError> {
///         // the one method that differs
///     }
/// }
/// ```
#[macro_export]
macro_rules! artifact_instance_identity {
    () => {
        fn artifact(&self) -> ::vyre_megakernel::Digest {
            self.core.artifact
        }

        fn payload(&self) -> ::vyre_megakernel::Digest {
            self.core.payload
        }

        fn device(&self) -> &$crate::DeviceIdentity {
            &self.core.device
        }
    };
}

/// Answer [`crate::ArtifactInstance::emitted_resources`] with one unreported
/// record per module.
///
/// A backend whose API exposes no register, spill or static shared figure for a
/// loaded entry point answers this way. The record count still has to match the
/// module count, because the compiler pairs the records with payload entries by
/// position, and zero in every field is what leaves the analytic estimate in
/// force for that entry.
#[macro_export]
macro_rules! artifact_instance_unreported_resources {
    () => {
        fn emitted_resources(
            &self,
        ) -> ::std::result::Result<
            ::std::vec::Vec<::vyre_megakernel::EmittedResources>,
            $crate::BackendError,
        > {
            ::std::result::Result::Ok(::std::vec![
                ::vyre_megakernel::EmittedResources::default();
                $crate::materialize::MaterializedInstance::modules(self).len()
            ])
        }
    };
}

/// Answer [`crate::ArtifactMaterializer::device`] from a
/// [`crate::materialize::MaterializerDevice`]
/// field named `descriptor`, and optionally forward the four resident-resource
/// methods to a [`crate::VyreBackend`] field.
///
/// The device accessor is the same line in every backend, because the device a
/// materializer reports is exactly the descriptor it was acquired with. The
/// resident four are the same four bodies wherever the backend has a resident
/// path at all: the materializer owns no allocator of its own, so each one
/// names the backend field and forwards unchanged. A backend without a resident
/// path passes no field and keeps the trait's refusals.
///
/// ```ignore
/// impl ArtifactMaterializer for TargetMaterializer {
///     vyre_driver::materializer_passthrough!(resident);
///
///     fn materialize(&self, artifact: &Artifact, payload: &TargetPayload)
///         -> Result<Box<dyn ArtifactInstance>, BackendError> { /* per backend */ }
/// }
/// ```
#[macro_export]
macro_rules! materializer_passthrough {
    () => {
        fn device(&self) -> &dyn $crate::Device {
            &self.descriptor
        }
    };
    ($backend:ident) => {
        $crate::materializer_passthrough!();

        fn allocate_resident(
            &self,
            byte_len: usize,
        ) -> ::std::result::Result<$crate::Resource, $crate::BackendError> {
            $crate::VyreBackend::allocate_resident(&self.$backend, byte_len)
        }

        fn upload_resident(
            &self,
            resource: &$crate::Resource,
            bytes: &[u8],
        ) -> ::std::result::Result<(), $crate::BackendError> {
            $crate::VyreBackend::upload_resident(&self.$backend, resource, bytes)
        }

        fn upload_resident_at(
            &self,
            resource: &$crate::Resource,
            offset_bytes: usize,
            bytes: &[u8],
        ) -> ::std::result::Result<(), $crate::BackendError> {
            $crate::VyreBackend::upload_resident_at(&self.$backend, resource, offset_bytes, bytes)
        }

        fn free_resident(
            &self,
            resource: $crate::Resource,
        ) -> ::std::result::Result<(), $crate::BackendError> {
            $crate::VyreBackend::free_resident(&self.$backend, resource)
        }
    };
}

/// Answer [`ResidentInstance::launch_resident`] from a module field named
/// `pipeline` that implements [`crate::CompiledPipeline`].
///
/// A resident launch through a compiled pipeline is one call taking the ordered
/// handles and the config the shared path already built, so the backends whose
/// module holds a pipeline wrote the same body. A backend whose module holds
/// something else writes the method by hand.
///
/// ```ignore
/// impl ResidentInstance for TargetArtifactInstance {
///     vyre_driver::resident_pipeline_launch!();
///
///     fn resident_module_label(&self) -> &'static str { /* per backend */ }
/// }
/// ```
#[macro_export]
macro_rules! resident_pipeline_launch {
    () => {
        fn launch_resident(
            &self,
            module: &Self::Module,
            ordered: &[$crate::Resource],
            config: &$crate::DispatchConfig,
        ) -> ::std::result::Result<$crate::TimedDispatchResult, $crate::BackendError> {
            $crate::CompiledPipeline::dispatch_persistent_handles_timed(
                module.pipeline.as_ref(),
                ordered,
                config,
            )
        }
    };
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
