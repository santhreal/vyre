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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use vyre_foundation::ir::Program;
use vyre_megakernel::{
    Artifact, ArtifactValueId, Digest, FusionRecord, ResourceLifetime, TargetModuleBundle,
    TargetModuleImage, TargetPayload, TargetPayloadFormat, TargetProfile,
};

use crate::{
    BackendError, BindingPlan, BindingRole, BindingSet, BoundResource, Completion, Device,
    DeviceIdentity, DispatchConfig, ResidentOwner, Resource, Submission, TimedDispatchResult,
};

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

/// What the acquired materializer accepts, as declared by its device.
#[derive(Clone, Copy, Debug)]
pub struct MaterializerTarget<'a> {
    /// Stable identity of the acquiring backend, used in rejection text.
    pub backend_id: &'a str,
    /// Payload format the materializer was acquired for.
    pub format: &'a TargetPayloadFormat,
    /// Device profile the materializer was acquired for.
    pub profile: &'a TargetProfile,
}

/// One target module whose identity matches the compiler-selected plan.
#[derive(Debug)]
pub struct AdmittedModule {
    /// The target-native module image, identity already verified.
    pub image: TargetModuleImage,
    /// Canonical Program decoded from the module wire.
    pub program: Arc<Program>,
    /// Dispatch configuration carried by the payload entry.
    pub config: DispatchConfig,
}

/// Admit a target payload against the artifact it claims to implement.
///
/// Every check here is a property of the neutral artifact and the payload
/// envelope, so it holds identically for every backend. The returned modules
/// are paired with their decoded Program and dispatch config, identity already
/// verified; the caller decodes `image.bytes` in its own dialect.
///
/// # Errors
///
/// Returns `BackendError::UnsupportedFeature` when the payload format is not
/// the one the materializer was acquired for, and `BackendError::InvalidProgram`
/// when the payload is not authenticated for this artifact, its profile
/// disagrees, or its module and entry counts do not match the compiler-selected
/// fusion plan.
pub fn admit(
    artifact: &Artifact,
    payload: &TargetPayload,
    target: MaterializerTarget<'_>,
) -> Result<Vec<AdmittedModule>, BackendError> {
    if payload.neutral_artifact() != artifact.digest() {
        return Err(invalid_module(
            "target payload is not authenticated for the supplied neutral artifact",
        ));
    }
    if payload.format() != target.format {
        return Err(BackendError::UnsupportedFeature {
            name: format!("target payload format `{}`", payload.format().identity()),
            backend: target.backend_id.to_string(),
        });
    }
    if payload.profile() != target.profile {
        return Err(invalid_module(
            "target payload profile does not match the acquired materializer profile",
        ));
    }

    let bundle = TargetModuleBundle::from_bytes(payload.bytes())
        .map_err(|error| compile_error(target.backend_id, error))?;
    let selected = artifact.fusion();
    if bundle.modules.len() != selected.len() {
        return Err(invalid_module(
            "target module count must equal the compiler-selected fusion-group count",
        ));
    }
    if payload.entries().len() != selected.len() {
        return Err(invalid_module(
            "target entry count must equal the compiler-selected fusion-group count",
        ));
    }
    // Three per-group lists arrive in three orders: the bundle canonically sorts its
    // modules by (stage, group), the fusion records are in the artifact's own plan
    // order, and the entries are in the order the target compiler emitted them.
    // Zipping them made every pairing an unstated assumption that all three orders
    // agree, and the only check that would have caught a disagreement between an
    // entry and its module compared entry names, which are all `main`. Each module
    // now names the record and entry it belongs to: its group id, and the first
    // member node of that group.
    let mut records = BTreeMap::new();
    for record in selected {
        if records.insert(record.id, record).is_some() {
            return Err(invalid_module(
                "the selected plan lists one fusion group twice",
            ));
        }
    }
    let mut entries_by_node = BTreeMap::new();
    for entry in payload.entries() {
        if entries_by_node.insert(entry.node, entry).is_some() {
            return Err(invalid_module(
                "target entries must name distinct canonical nodes",
            ));
        }
    }

    let mut admitted = Vec::with_capacity(selected.len());
    for image in bundle.modules {
        let record = records.remove(&image.group).ok_or_else(|| {
            invalid_module("target module names a fusion group the selected plan does not list")
        })?;
        admit_module_identity(&image, record)?;
        let entry_node = *record.members.first().ok_or_else(|| {
            invalid_module("the selected plan lists a fusion group with no member node")
        })?;
        let entry = entries_by_node.remove(&entry_node).ok_or_else(|| {
            invalid_module(
                "target entry node identity must match the first member node of a selected fusion group",
            )
        })?;
        if image.entry_point != "main" {
            return Err(invalid_module("target module entry point must be `main`"));
        }
        if entry.name != image.entry_point {
            return Err(invalid_module(
                "target entry metadata must name the emitted target entry point",
            ));
        }
        let program =
            Arc::new(Program::from_wire(&image.program).map_err(|error| {
                invalid_module(&format!("selected Program is malformed: {error}"))
            })?);
        let mut config = DispatchConfig::default();
        config.grid_override = Some(entry.grid_size);
        config.dispatch_grid = Some(entry.grid_size);
        admitted.push(AdmittedModule {
            image,
            program,
            config,
        });
    }
    Ok(admitted)
}

/// Reject a module whose identity disagrees with the neutral selected plan.
fn admit_module_identity(
    image: &TargetModuleImage,
    record: &FusionRecord,
) -> Result<(), BackendError> {
    if image.group != record.id || image.stage != record.stage || image.nodes != record.members {
        return Err(invalid_module(
            "target module group/stage/node identity must match the neutral selected plan",
        ));
    }
    Ok(())
}

/// Artifact resources sorted by lifetime, as an instance records them.
pub struct ResourceProjection {
    /// Every artifact resource by name.
    pub values: BTreeMap<String, ArtifactValueId>,
    /// Resources the artifact reports as outputs.
    pub outputs: BTreeSet<ArtifactValueId>,
    /// Resources the artifact retains across dispatches.
    pub retained: BTreeSet<ArtifactValueId>,
}

/// Project an artifact's resources onto the three sets every instance keeps.
///
/// One pass over the resource records; the per-backend copies walked them
/// three times to build the same three collections.
#[must_use]
pub fn project_resources(artifact: &Artifact) -> ResourceProjection {
    let mut projection = ResourceProjection {
        values: BTreeMap::new(),
        outputs: BTreeSet::new(),
        retained: BTreeSet::new(),
    };
    for resource in artifact.resources() {
        projection
            .values
            .insert(resource.name.clone(), resource.value);
        match resource.lifetime {
            ResourceLifetime::Output => {
                projection.outputs.insert(resource.value);
            }
            ResourceLifetime::Retained => {
                projection.retained.insert(resource.value);
            }
            _ => {}
        }
    }
    projection
}

/// Device descriptor a concrete materializer reports for its acquired generation.
///
/// Every backend recorded the same three fields and answered the same three
/// accessors from them. Only revocation differed: one backend invalidates its
/// generation from a device-loss callback, the others stay healthy for the life
/// of the materializer.
pub struct MaterializerDevice {
    identity: DeviceIdentity,
    format: TargetPayloadFormat,
    profile: TargetProfile,
    revoked: Option<Arc<AtomicBool>>,
}

/// What a backend resolves before it can name the device it admits artifacts for.
///
/// Every backend spelled the same three resolutions ahead of its descriptor:
/// build the payload format from an extension and a version, mint a resident
/// owner generation, and read the compilation profile. Four copies of that
/// sequence are four places for the format rejection to lose the backend name.
pub struct DeviceSpec<'a> {
    /// Stable registered backend identifier.
    pub backend: &'static str,
    /// Backend-local physical or logical device identifier.
    pub device: String,
    /// Payload file extension this backend admits.
    pub format_extension: &'a str,
    /// Payload format version this backend admits.
    pub format_version: u16,
    /// Immutable compilation profile of the acquired device.
    pub profile: TargetProfile,
}

impl MaterializerDevice {
    /// Resolve a device that stays healthy for the life of the materializer.
    ///
    /// # Errors
    ///
    /// Returns [`compile_error`] naming `spec.backend` when the extension and
    /// version do not form a payload format, and whatever
    /// [`ResidentOwner::new`] rejects when no generation can be minted.
    pub fn acquire(spec: DeviceSpec<'_>) -> Result<Self, BackendError> {
        Self::resolve(spec, None)
    }

    /// Resolve a device whose generation is invalidated when `revoked` is set.
    ///
    /// # Errors
    ///
    /// Rejects exactly what [`MaterializerDevice::acquire`] rejects.
    pub fn acquire_revocable(
        spec: DeviceSpec<'_>,
        revoked: Arc<AtomicBool>,
    ) -> Result<Self, BackendError> {
        Self::resolve(spec, Some(revoked))
    }

    /// Build the descriptor both acquisition paths report.
    fn resolve(
        spec: DeviceSpec<'_>,
        revoked: Option<Arc<AtomicBool>>,
    ) -> Result<Self, BackendError> {
        let format = TargetPayloadFormat::new(spec.format_extension, spec.format_version)
            .map_err(|error| compile_error(spec.backend, error))?;
        let generation = ResidentOwner::new()?.get();
        Ok(Self {
            identity: DeviceIdentity {
                backend: spec.backend,
                device: spec.device,
                generation,
            },
            format,
            profile: spec.profile,
            revoked,
        })
    }

    /// Describe what `backend_id` accepts, for [`admit`].
    #[must_use]
    pub fn target<'a>(&'a self, backend_id: &'a str) -> MaterializerTarget<'a> {
        MaterializerTarget {
            backend_id,
            format: &self.format,
            profile: &self.profile,
        }
    }

    /// Record what an instance materialized on this device generation keeps.
    #[must_use]
    pub fn instance(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
        messages: InstanceMessages,
    ) -> InstanceCore {
        InstanceCore::new(artifact, payload, self.identity.clone(), messages)
    }

    /// Admit a payload and decode each admitted module in this backend's dialect.
    ///
    /// Every backend wrote the same four steps around the one that is its own:
    /// call [`admit`], size a vector to the admitted count, push one decoded
    /// module per admitted module, and stop at the first rejection. Only
    /// `decode` is target-specific, and it is the only thing a backend has to
    /// supply. Restating the loop per backend is how one of them came to
    /// allocate without the admitted capacity and another to keep going past a
    /// module it had already rejected.
    ///
    /// # Errors
    ///
    /// Returns whatever [`admit`] rejects, and whatever `decode` rejects for
    /// the first module it refuses.
    pub fn admit_modules<M>(
        &self,
        backend_id: &str,
        artifact: &Artifact,
        payload: &TargetPayload,
        mut decode: impl FnMut(AdmittedModule) -> Result<M, BackendError>,
    ) -> Result<Vec<M>, BackendError> {
        let admitted = admit(artifact, payload, self.target(backend_id))?;
        let mut modules = Vec::with_capacity(admitted.len());
        for module in admitted {
            modules.push(decode(module)?);
        }
        Ok(modules)
    }
}

impl Device for MaterializerDevice {
    fn identity(&self) -> &DeviceIdentity {
        &self.identity
    }

    fn target_format(&self) -> &TargetPayloadFormat {
        &self.format
    }

    fn target_profile(&self) -> &TargetProfile {
        &self.profile
    }

    fn is_healthy(&self) -> bool {
        self.revoked
            .as_ref()
            .is_none_or(|revoked| !revoked.load(Ordering::Acquire))
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
    /// Input value identities per module entry in binding plan order.
    pub module_inputs: Vec<Vec<ArtifactValueId>>,
    /// Output value identities per module entry in binding plan order.
    pub module_outputs: Vec<Vec<ArtifactValueId>>,
    /// Transitive prior retained values that feed each successor retained value.
    pub retained_predecessors: BTreeMap<ArtifactValueId, Vec<ArtifactValueId>>,
}

impl InstanceCore {
    /// Record what an instance materialized from `artifact` and `payload` keeps.
    #[must_use]
    pub fn new(
        artifact: &Artifact,
        payload: &TargetPayload,
        device: DeviceIdentity,
        messages: InstanceMessages,
    ) -> Self {
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
                    break;
                }
                if resources.retained.contains(&prev) {
                    priors.push(prev);
                }
                curr = prev;
            }
            if !priors.is_empty() {
                retained_predecessors.insert(succ, priors);
            }
        }

        let mut module_inputs = Vec::with_capacity(payload.entries().len());
        let mut module_outputs = Vec::with_capacity(payload.entries().len());
        for entry in payload.entries() {
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();
            for binding in &entry.resource_bindings {
                match binding.access {
                    vyre_megakernel::TargetResourceAccess::ReadOnly => {
                        inputs.push(binding.resource);
                    }
                    vyre_megakernel::TargetResourceAccess::WriteOnly => {
                        outputs.push(binding.resource);
                    }
                    vyre_megakernel::TargetResourceAccess::ReadWrite => {
                        outputs.push(binding.resource);
                        let input_val = direct_predecessors
                            .get(&binding.resource)
                            .copied()
                            .unwrap_or(binding.resource);
                        inputs.push(input_val);
                    }
                }
            }
            module_inputs.push(inputs);
            module_outputs.push(outputs);
        }

        Self {
            artifact: artifact.digest(),
            payload: payload.digest(),
            device,
            values: resources.values,
            outputs: resources.outputs,
            retained: resources.retained,
            messages,
            module_inputs,
            module_outputs,
            retained_predecessors,
        }
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
            let value = self
                .module_inputs
                .get(module_index)
                .and_then(|in_vals| in_vals.get(input_index))
                .copied()
                .map(Ok)
                .unwrap_or_else(|| self.value_for_buffer(buffer.name()))?;
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
            let value = self
                .module_outputs
                .get(module_index)
                .and_then(|out_vals| out_vals.get(output_index))
                .copied()
                .map(Ok)
                .unwrap_or_else(|| self.value_for_buffer(buffer.name()))?;
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
        host: impl FnOnce(
            BTreeMap<ArtifactValueId, Vec<u8>>,
            Option<[u32; 3]>,
        ) -> Result<Completion, BackendError>,
        resident: impl FnOnce(
            &BTreeMap<ArtifactValueId, Resource>,
            Option<[u32; 3]>,
        ) -> Result<Completion, BackendError>,
    ) -> Result<Box<dyn Submission>, BackendError> {
        self.accept(bindings)?;
        let invocation_grid = bindings.invocation_grid();
        let bound = partition_bindings(bindings);
        if !bound.host.is_empty() && !bound.resident.is_empty() {
            return Err(mixed());
        }
        let result = if bound.resident.is_empty() {
            host(bound.host, invocation_grid)
        } else {
            resident(&bound.resident, invocation_grid)
        };
        Ok(self.ready(result))
    }

    /// Route one submission on a backend that has no resident execution path.
    ///
    /// The refusal, the grid, and the host state come from the same three calls
    /// wherever there is only one path to route to, so the backends that have
    /// one wrote the same four lines. `feature` names the refused capability in
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
        host: impl FnOnce(
            BTreeMap<ArtifactValueId, Vec<u8>>,
            Option<[u32; 3]>,
        ) -> Result<Completion, BackendError>,
    ) -> Result<Box<dyn Submission>, BackendError> {
        self.accept(bindings)?;
        let invocation_grid = bindings.invocation_grid();
        let state = host_only_bindings(bindings, feature, &self.device.backend)?;
        Ok(self.ready(host(state, invocation_grid)))
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
        names: impl IntoIterator<Item = &'names str>,
        resources: &BTreeMap<ArtifactValueId, Resource>,
        unbound: impl Fn(ArtifactValueId, &str) -> BackendError,
    ) -> Result<Vec<Resource>, BackendError> {
        let names = names.into_iter();
        let mut ordered = Vec::with_capacity(names.size_hint().0);
        for name in names {
            let value = self.value_for_buffer(name)?;
            let resource = resources.get(&value).ok_or_else(|| unbound(value, name))?;
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
        invocation_grid: Option<[u32; 3]>,
        omitted: impl Fn(usize, &str) -> BackendError,
        mut dispatch: impl FnMut(
            &M,
            &BindingPlan,
            &DispatchConfig,
            &BTreeMap<ArtifactValueId, Vec<u8>>,
        ) -> Result<TimedDispatchResult, BackendError>,
    ) -> Result<Completion, BackendError> {
        let mut device_ns = 0_u64;
        let mut has_device_timing = false;
        for (module_index, module) in modules.iter().enumerate() {
            let mut config = module.config().clone();
            override_grid(&mut config, invocation_grid);
            let plan = BindingPlan::build(module.program())?;
            let dispatched = dispatch(module, &plan, &config, &state)?;
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

    /// Take the single module a resident submission can run.
    ///
    /// Resident bindings are ordered handles for one launch, so a multi-module
    /// artifact has no place to put the intermediate values its later modules
    /// would read. `feature` names the refused capability in the backend's own
    /// words; the backend itself comes from the device generation.
    ///
    /// # Errors
    ///
    /// Returns `BackendError::UnsupportedFeature` unless `modules` holds
    /// exactly one module.
    pub fn single_resident_module<'modules, M>(
        &self,
        modules: &'modules [M],
        feature: &str,
    ) -> Result<&'modules M, BackendError> {
        match modules {
            [module] => Ok(module),
            _ => Err(BackendError::UnsupportedFeature {
                name: feature.to_string(),
                backend: self.device.backend.to_string(),
            }),
        }
    }

    /// Complete a resident dispatch, which starts from empty state.
    ///
    /// A resident launch reads its inputs from device memory the caller already
    /// filled, so nothing is carried in and every completed value comes from
    /// this dispatch. `messages` supplies the completion rejections, which is
    /// not always `self.messages`: a backend may word an unproduced resident
    /// value differently than a host one.
    ///
    /// # Errors
    ///
    /// Returns `omitted` for a declared output the module did not produce, and
    /// the `messages` rejections when a declared value is absent afterwards.
    pub fn resident_completion(
        &self,
        plan: &BindingPlan,
        program: &Program,
        dispatched: TimedDispatchResult,
        omitted: impl Fn(usize, &str) -> BackendError,
        messages: &InstanceMessages,
    ) -> Result<Completion, BackendError> {
        let device_ns = dispatched.device_ns;
        let mut state = BTreeMap::new();
        self.absorb_outputs(plan, program, dispatched.outputs, &mut state, omitted)?;
        Ok(Completion {
            artifact: self.artifact,
            outputs: self.project(&self.outputs, &state, messages.missing_output_value)?,
            retained: self.project(&self.retained, &state, messages.missing_retained_value)?,
            device_ns,
        })
    }
}

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

/// Apply a submission-time invocation grid to a payload dispatch config.
pub fn override_grid(config: &mut DispatchConfig, grid: Option<[u32; 3]>) {
    if let Some(grid) = grid {
        config.grid_override = Some(grid);
        config.dispatch_grid = Some(grid);
    }
}

/// The execution paths of a materialized instance, defaulted over
/// [`InstanceCore`].
///
/// Every backend wrote the same two bodies around the same [`InstanceCore`]
/// calls: gather the module's inputs and launch it once per module of the
/// selected plan, then complete the accumulated state. Only the launch is
/// target-specific, so only the launch is required here. The identity methods
/// of [`crate::ArtifactInstance`] come from [`artifact_instance_identity`] and
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
    /// module's dispatch config with the submission grid already applied.
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
        module: &Self::Module,
        plan: &BindingPlan,
        state: &'state BTreeMap<ArtifactValueId, Vec<u8>>,
    ) -> Result<Vec<&'state [u8]>, BackendError> {
        let module_index = self
            .modules()
            .iter()
            .position(|m| std::ptr::eq(m, module))
            .unwrap_or(0);
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
        invocation_grid: Option<[u32; 3]>,
    ) -> Result<Completion, BackendError> {
        let label = self.module_label();
        self.core().execute_modules(
            self.modules(),
            state,
            invocation_grid,
            |output_index, name| omitted_output(label, output_index, name),
            |module, plan, config, state| {
                let inputs = self.gather(module, plan, state)?;
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
            .submit_host_only(bindings, feature, |state, invocation_grid| {
                self.execute_host(state, invocation_grid)
            })
    }
}

/// The resident execution path of a materialized instance whose backend has
/// one.
///
/// A resident launch is one module over device memory the caller already
/// filled. Three backends wrote the same seven steps around it and differed on
/// two: where the handle order comes from, and whether an unproduced resident
/// value is worded the way the host path words it.
pub trait ResidentInstance: MaterializedInstance {
    /// Refused capability when a resident submission names more than one module.
    fn multi_module_feature(&self) -> &str;

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
        module: &Self::Module,
        plan: &BindingPlan,
        resources: &BTreeMap<ArtifactValueId, Resource>,
    ) -> Result<Vec<Resource>, BackendError> {
        self.core().ordered_resident_resources(
            resident_buffer_names(plan, module.program()),
            resources,
            unbound_resident_buffer,
        )
    }

    /// Launch the single resident module over `ordered`.
    ///
    /// `config` is the module's dispatch config with the submission grid
    /// already applied.
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

    /// Launch the single module over caller-owned resident resources.
    ///
    /// # Errors
    ///
    /// Returns `BackendError::UnsupportedFeature` unless the plan selected one
    /// module, whatever [`Self::ordered_resident`] and [`Self::launch_resident`]
    /// report, and the resident completion rejections when the launch left a
    /// declared value behind.
    fn execute_resident(
        &self,
        resources: &BTreeMap<ArtifactValueId, Resource>,
        invocation_grid: Option<[u32; 3]>,
    ) -> Result<Completion, BackendError> {
        let core = self.core();
        let module = core.single_resident_module(self.modules(), self.multi_module_feature())?;
        let plan = BindingPlan::build(module.program())?;
        let ordered = self.ordered_resident(module, &plan, resources)?;
        let mut config = module.config().clone();
        override_grid(&mut config, invocation_grid);
        let dispatched = self.launch_resident(module, &ordered, &config)?;
        let label = self.resident_module_label();
        core.resident_completion(
            &plan,
            module.program(),
            dispatched,
            |output_index, name| omitted_output(label, output_index, name),
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
            |state, invocation_grid| self.execute_host(state, invocation_grid),
            |resources, invocation_grid| self.execute_resident(resources, invocation_grid),
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

/// Answer [`crate::ArtifactMaterializer::device`] from a [`MaterializerDevice`]
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
///     fn multi_module_feature(&self) -> &str { /* per backend */ }
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
struct ReadySubmission {
    result: Option<Result<Completion, BackendError>>,
    consumed: fn() -> BackendError,
}

impl Submission for ReadySubmission {
    fn is_ready(&self) -> bool {
        true
    }

    fn wait(mut self: Box<Self>) -> Result<Completion, BackendError> {
        self.result.take().ok_or_else(self.consumed)?
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use vyre_foundation::ir::{
        BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program,
        ProgramGraph, ShapeDim, ValueContract, ValueLifetime,
    };
    use vyre_megakernel::{
        compile, ArtifactNodeId, ArtifactValueId, CompileRequest, DeviceFacts, Digest,
        ExternalFacts, SearchBudget, TargetEntryPoint, TargetPayload, TargetPayloadFormat,
        TargetProfile, TargetResourceAccess, TargetResourceBinding, TargetResourceMemory,
    };

    fn test_format() -> TargetPayloadFormat {
        TargetPayloadFormat::new("test.target-binary", 1).unwrap()
    }

    fn test_profile() -> TargetProfile {
        TargetProfile::new("test.target-binary", 1, [32, 1, 1], 32, 1024, 0).unwrap()
    }

    fn test_device() -> DeviceIdentity {
        DeviceIdentity {
            backend: "test",
            device: "test-device".into(),
            generation: 1,
        }
    }
    fn contract(access: BufferAccess, lifetime: ValueLifetime) -> ValueContract {
        ValueContract {
            dtype: DataType::U32,
            shape: vec![ShapeDim::Known(32)],
            access,
            lifetime,
        }
    }

    #[test]
    fn sparse_and_reordered_module_binding_identities() {
        let mut graph = ProgramGraph::new();
        let val_x = graph
            .add_external_value(
                "x",
                contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
            )
            .unwrap();
        let val_y = graph
            .add_external_value(
                "y",
                contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
            )
            .unwrap();

        let program0 = Program::wrapped(
            vec![
                BufferDecl::storage("in_b", 0, BufferAccess::ReadOnly, DataType::U32),
                BufferDecl::storage("in_a", 1, BufferAccess::ReadOnly, DataType::U32),
                BufferDecl::storage("out_0", 2, BufferAccess::WriteOnly, DataType::U32),
            ],
            [32, 1, 1],
            vec![Node::store(
                "out_0",
                Expr::u32(0),
                Expr::add(
                    Expr::load("in_b", Expr::u32(0)),
                    Expr::load("in_a", Expr::u32(0)),
                ),
            )],
        );

        let (_, outputs) = graph
            .add_node(
                "node0",
                program0.clone(),
                vec![
                    GraphInput {
                        buffer: "in_b".into(),
                        value: val_y,
                        contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
                    },
                    GraphInput {
                        buffer: "in_a".into(),
                        value: val_x,
                        contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
                    },
                ],
                vec![GraphOutput {
                    buffer: "out_0".into(),
                    name: "res".into(),
                    contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output),
                    retained_successor_of: None,
                }],
            )
            .unwrap();
        let res_id = outputs[0];

        let req = CompileRequest::new(
            graph,
            ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
            DeviceFacts::unknown(),
            SearchBudget::new(128, 1_000_000, 8, 0, 1_000_000_000),
            1_000_000,
        )
        .validate()
        .unwrap();

        let artifact = compile(&req).expect("compilation must succeed");

        let bindings = vec![
            TargetResourceBinding {
                resource: ArtifactValueId(val_y.0),
                group: 0,
                slot: 0,
                memory: TargetResourceMemory::Global,
                access: TargetResourceAccess::ReadOnly,
            },
            TargetResourceBinding {
                resource: ArtifactValueId(val_x.0),
                group: 0,
                slot: 1,
                memory: TargetResourceMemory::Global,
                access: TargetResourceAccess::ReadOnly,
            },
            TargetResourceBinding {
                resource: ArtifactValueId(res_id.0),
                group: 0,
                slot: 2,
                memory: TargetResourceMemory::Global,
                access: TargetResourceAccess::WriteOnly,
            },
        ];

        let payload = TargetPayload::new(
            &artifact,
            test_format(),
            test_profile(),
            vec![TargetEntryPoint {
                name: "entry0".into(),
                node: ArtifactNodeId(0),
                workgroup_size: [32, 1, 1],
                grid_size: [1, 1, 1],
                dynamic_shared_bytes: 0,
                resource_bindings: bindings,
            }],
            vec![1, 2, 3],
        )
        .unwrap();

        let core = InstanceCore::new(&artifact, &payload, test_device(), NEUTRAL_MESSAGES);
        let plan0 = BindingPlan::build(&program0).unwrap();

        let mut state = BTreeMap::new();
        state.insert(ArtifactValueId(val_y.0), vec![2, 0, 0, 0]);
        state.insert(ArtifactValueId(val_x.0), vec![1, 0, 0, 0]);

        let gathered = core
            .gather_inputs_for_module(0, &plan0, &program0, &state, unbound_input)
            .unwrap();

        assert_eq!(gathered[0], &[2, 0, 0, 0]);
        assert_eq!(gathered[1], &[1, 0, 0, 0]);

        core.absorb_outputs_for_module(
            0,
            &plan0,
            &program0,
            vec![vec![3, 0, 0, 0]],
            &mut state,
            |idx, name| BackendError::InvalidProgram {
                fix: format!("missing output {idx} {name}"),
            },
        )
        .unwrap();

        assert_eq!(
            state.get(&ArtifactValueId(res_id.0)).unwrap(),
            &[3, 0, 0, 0]
        );

        let completion = core.completion(&state, None).unwrap();
        assert_eq!(
            completion.outputs.get(&ArtifactValueId(res_id.0)).unwrap(),
            &[3, 0, 0, 0]
        );
    }

    #[test]
    fn transitive_retained_predecessor_lineage_preservation() {
        let mut graph = ProgramGraph::new();
        let state_init = graph
            .add_external_value(
                "state_init",
                contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
            )
            .unwrap();

        let seg0_prog = Program::wrapped(
            vec![
                BufferDecl::storage("in_s", 0, BufferAccess::ReadWrite, DataType::U32),
                BufferDecl::storage("out_s", 1, BufferAccess::ReadWrite, DataType::U32),
            ],
            [32, 1, 1],
            vec![Node::store(
                "out_s",
                Expr::u32(0),
                Expr::add(Expr::load("in_s", Expr::u32(0)), Expr::u32(1)),
            )],
        );

        let (node0, seg0_outputs) = graph
            .add_node(
                "seg0",
                seg0_prog.clone(),
                vec![GraphInput {
                    buffer: "in_s".into(),
                    value: state_init,
                    contract: contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
                }],
                vec![GraphOutput {
                    buffer: "out_s".into(),
                    name: "state_mid".into(),
                    contract: contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
                    retained_successor_of: Some(state_init),
                }],
            )
            .unwrap();
        let state_mid = seg0_outputs[0];

        let seg1_prog = Program::wrapped(
            vec![
                BufferDecl::storage("in_s", 0, BufferAccess::ReadWrite, DataType::U32),
                BufferDecl::storage("out_s", 1, BufferAccess::ReadWrite, DataType::U32),
                BufferDecl::storage("out_res", 2, BufferAccess::WriteOnly, DataType::U32),
            ],
            [32, 1, 1],
            vec![
                Node::store(
                    "out_s",
                    Expr::u32(0),
                    Expr::add(Expr::load("in_s", Expr::u32(0)), Expr::u32(2)),
                ),
                Node::store("out_res", Expr::u32(0), Expr::load("in_s", Expr::u32(0))),
            ],
        );

        let (node1, seg1_outputs) = graph
            .add_node(
                "seg1",
                seg1_prog.clone(),
                vec![GraphInput {
                    buffer: "in_s".into(),
                    value: state_mid,
                    contract: contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
                }],
                vec![
                    GraphOutput {
                        buffer: "out_s".into(),
                        name: "state_final".into(),
                        contract: contract(BufferAccess::ReadWrite, ValueLifetime::Retained),
                        retained_successor_of: Some(state_mid),
                    },
                    GraphOutput {
                        buffer: "out_res".into(),
                        name: "res".into(),
                        contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output),
                        retained_successor_of: None,
                    },
                ],
            )
            .unwrap();
        let state_final = seg1_outputs[0];
        let out_id = seg1_outputs[1];

        let req = CompileRequest::new(
            graph,
            ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
            DeviceFacts::unknown(),
            SearchBudget::new(128, 1_000_000, 8, 0, 1_000_000_000),
            1_000_000,
        )
        .validate()
        .unwrap();

        let artifact = compile(&req).expect("compilation must succeed");

        let payload = TargetPayload::new(
            &artifact,
            test_format(),
            test_profile(),
            vec![
                TargetEntryPoint {
                    name: "seg0_entry".into(),
                    node: ArtifactNodeId(node0.0),
                    workgroup_size: [32, 1, 1],
                    grid_size: [1, 1, 1],
                    dynamic_shared_bytes: 0,
                    resource_bindings: vec![
                        TargetResourceBinding {
                            resource: ArtifactValueId(state_init.0),
                            group: 0,
                            slot: 0,
                            memory: TargetResourceMemory::Global,
                            access: TargetResourceAccess::ReadWrite,
                        },
                        TargetResourceBinding {
                            resource: ArtifactValueId(state_mid.0),
                            group: 0,
                            slot: 1,
                            memory: TargetResourceMemory::Global,
                            access: TargetResourceAccess::ReadWrite,
                        },
                    ],
                },
                TargetEntryPoint {
                    name: "seg1_entry".into(),
                    node: ArtifactNodeId(node1.0),
                    workgroup_size: [32, 1, 1],
                    grid_size: [1, 1, 1],
                    dynamic_shared_bytes: 0,
                    resource_bindings: vec![
                        TargetResourceBinding {
                            resource: ArtifactValueId(state_mid.0),
                            group: 0,
                            slot: 0,
                            memory: TargetResourceMemory::Global,
                            access: TargetResourceAccess::ReadWrite,
                        },
                        TargetResourceBinding {
                            resource: ArtifactValueId(state_final.0),
                            group: 0,
                            slot: 1,
                            memory: TargetResourceMemory::Global,
                            access: TargetResourceAccess::ReadWrite,
                        },
                        TargetResourceBinding {
                            resource: ArtifactValueId(out_id.0),
                            group: 0,
                            slot: 2,
                            memory: TargetResourceMemory::Global,
                            access: TargetResourceAccess::WriteOnly,
                        },
                    ],
                },
            ],
            vec![1, 2, 3],
        )
        .unwrap();

        let core = InstanceCore::new(&artifact, &payload, test_device(), NEUTRAL_MESSAGES);
        let seg0_plan = BindingPlan::build(&seg0_prog).unwrap();
        let seg1_plan = BindingPlan::build(&seg1_prog).unwrap();

        let mut state = BTreeMap::new();
        state.insert(ArtifactValueId(state_init.0), vec![0, 0, 0, 0]);

        let gathered0 = core
            .gather_inputs_for_module(0, &seg0_plan, &seg0_prog, &state, unbound_input)
            .unwrap();
        assert_eq!(gathered0[0], &[0, 0, 0, 0]);

        core.absorb_outputs_for_module(
            0,
            &seg0_plan,
            &seg0_prog,
            vec![vec![0, 0, 0, 0], vec![42, 0, 0, 0]],
            &mut state,
            |idx, name| BackendError::InvalidProgram {
                fix: format!("missing output {idx} {name}"),
            },
        )
        .unwrap();

        assert_eq!(
            state.get(&ArtifactValueId(state_mid.0)).unwrap(),
            &[42, 0, 0, 0]
        );
        assert_eq!(
            state.get(&ArtifactValueId(state_init.0)).unwrap(),
            &[42, 0, 0, 0]
        );

        let gathered1 = core
            .gather_inputs_for_module(1, &seg1_plan, &seg1_prog, &state, unbound_input)
            .unwrap();
        assert_eq!(gathered1[0], &[42, 0, 0, 0]);

        core.absorb_outputs_for_module(
            1,
            &seg1_plan,
            &seg1_prog,
            vec![vec![42, 0, 0, 0], vec![99, 0, 0, 0], vec![1, 2, 3, 4]],
            &mut state,
            |idx, name| BackendError::InvalidProgram {
                fix: format!("missing output {idx} {name}"),
            },
        )
        .unwrap();

        assert_eq!(
            state.get(&ArtifactValueId(state_final.0)).unwrap(),
            &[99, 0, 0, 0]
        );
        assert_eq!(
            state.get(&ArtifactValueId(state_mid.0)).unwrap(),
            &[99, 0, 0, 0]
        );
        assert_eq!(
            state.get(&ArtifactValueId(state_init.0)).unwrap(),
            &[99, 0, 0, 0]
        );
        assert_eq!(
            state.get(&ArtifactValueId(out_id.0)).unwrap(),
            &[1, 2, 3, 4]
        );

        let completion = core.completion(&state, Some(1000)).unwrap();
        assert_eq!(
            completion
                .retained
                .get(&ArtifactValueId(state_init.0))
                .unwrap(),
            &[99, 0, 0, 0]
        );
        assert_eq!(
            completion
                .retained
                .get(&ArtifactValueId(state_mid.0))
                .unwrap(),
            &[99, 0, 0, 0]
        );
        assert_eq!(
            completion
                .retained
                .get(&ArtifactValueId(state_final.0))
                .unwrap(),
            &[99, 0, 0, 0]
        );
        assert_eq!(
            completion.outputs.get(&ArtifactValueId(out_id.0)).unwrap(),
            &[1, 2, 3, 4]
        );
    }

    #[test]
    fn fused_module_later_node_inputs_and_outputs_resolve() {
        let mut graph = ProgramGraph::new();
        let val_a = graph
            .add_external_value(
                "a",
                contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
            )
            .unwrap();
        let val_b = graph
            .add_external_value(
                "b",
                contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
            )
            .unwrap();

        let prog0 = Program::wrapped(
            vec![
                BufferDecl::storage("node0_in", 0, BufferAccess::ReadOnly, DataType::U32),
                BufferDecl::storage("node0_out", 1, BufferAccess::WriteOnly, DataType::U32),
            ],
            [32, 1, 1],
            vec![Node::store(
                "node0_out",
                Expr::u32(0),
                Expr::load("node0_in", Expr::u32(0)),
            )],
        );

        let prog1 = Program::wrapped(
            vec![
                BufferDecl::storage("node1_in", 0, BufferAccess::ReadOnly, DataType::U32),
                BufferDecl::storage("node1_out", 1, BufferAccess::WriteOnly, DataType::U32),
            ],
            [32, 1, 1],
            vec![Node::store(
                "node1_out",
                Expr::u32(0),
                Expr::load("node1_in", Expr::u32(0)),
            )],
        );

        let (node0, outputs0) = graph
            .add_node(
                "node0",
                prog0,
                vec![GraphInput {
                    buffer: "node0_in".into(),
                    value: val_a,
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
                }],
                vec![GraphOutput {
                    buffer: "node0_out".into(),
                    name: "out0".into(),
                    contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output),
                    retained_successor_of: None,
                }],
            )
            .unwrap();
        let out0 = outputs0[0];

        let (_node1, outputs1) = graph
            .add_node(
                "node1",
                prog1,
                vec![GraphInput {
                    buffer: "node1_in".into(),
                    value: val_b,
                    contract: contract(BufferAccess::ReadOnly, ValueLifetime::Invocation),
                }],
                vec![GraphOutput {
                    buffer: "node1_out".into(),
                    name: "out1".into(),
                    contract: contract(BufferAccess::WriteOnly, ValueLifetime::Output),
                    retained_successor_of: None,
                }],
            )
            .unwrap();
        let out1 = outputs1[0];

        let req = CompileRequest::new(
            graph,
            ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
            DeviceFacts::unknown(),
            SearchBudget::new(128, 1_000_000, 8, 0, 1_000_000_000),
            1_000_000,
        )
        .validate()
        .unwrap();

        let artifact = compile(&req).expect("compilation must succeed");

        let fused_prog = Program::wrapped(
            vec![
                BufferDecl::storage("node0_in", 0, BufferAccess::ReadOnly, DataType::U32),
                BufferDecl::storage("node0_out", 1, BufferAccess::WriteOnly, DataType::U32),
                BufferDecl::storage("node1_in", 2, BufferAccess::ReadOnly, DataType::U32),
                BufferDecl::storage("node1_out", 3, BufferAccess::WriteOnly, DataType::U32),
            ],
            [32, 1, 1],
            vec![
                Node::store(
                    "node0_out",
                    Expr::u32(0),
                    Expr::load("node0_in", Expr::u32(0)),
                ),
                Node::store(
                    "node1_out",
                    Expr::u32(0),
                    Expr::load("node1_in", Expr::u32(0)),
                ),
            ],
        );

        let payload = TargetPayload::new(
            &artifact,
            test_format(),
            test_profile(),
            vec![TargetEntryPoint {
                name: "fused_entry".into(),
                node: ArtifactNodeId(node0.0),
                workgroup_size: [32, 1, 1],
                grid_size: [1, 1, 1],
                dynamic_shared_bytes: 0,
                resource_bindings: vec![
                    TargetResourceBinding {
                        resource: ArtifactValueId(val_a.0),
                        group: 0,
                        slot: 0,
                        memory: TargetResourceMemory::Global,
                        access: TargetResourceAccess::ReadOnly,
                    },
                    TargetResourceBinding {
                        resource: ArtifactValueId(out0.0),
                        group: 0,
                        slot: 1,
                        memory: TargetResourceMemory::Global,
                        access: TargetResourceAccess::WriteOnly,
                    },
                    TargetResourceBinding {
                        resource: ArtifactValueId(val_b.0),
                        group: 0,
                        slot: 2,
                        memory: TargetResourceMemory::Global,
                        access: TargetResourceAccess::ReadOnly,
                    },
                    TargetResourceBinding {
                        resource: ArtifactValueId(out1.0),
                        group: 0,
                        slot: 3,
                        memory: TargetResourceMemory::Global,
                        access: TargetResourceAccess::WriteOnly,
                    },
                ],
            }],
            vec![1, 2, 3],
        )
        .unwrap();

        let core = InstanceCore::new(&artifact, &payload, test_device(), NEUTRAL_MESSAGES);
        let plan = BindingPlan::build(&fused_prog).unwrap();

        let mut state = BTreeMap::new();
        state.insert(ArtifactValueId(val_a.0), vec![10, 0, 0, 0]);
        state.insert(ArtifactValueId(val_b.0), vec![30, 0, 0, 0]);

        let gathered = core
            .gather_inputs_for_module(0, &plan, &fused_prog, &state, unbound_input)
            .unwrap();

        assert_eq!(gathered[0], &[10, 0, 0, 0]);
        assert_eq!(gathered[1], &[30, 0, 0, 0]);

        core.absorb_outputs_for_module(
            0,
            &plan,
            &fused_prog,
            vec![vec![20, 0, 0, 0], vec![40, 0, 0, 0]],
            &mut state,
            |idx, name| BackendError::InvalidProgram {
                fix: format!("missing output {idx} {name}"),
            },
        )
        .unwrap();

        assert_eq!(state.get(&ArtifactValueId(out0.0)).unwrap(), &[20, 0, 0, 0]);
        assert_eq!(state.get(&ArtifactValueId(out1.0)).unwrap(), &[40, 0, 0, 0]);

        let completion = core.completion(&state, None).unwrap();
        assert_eq!(
            completion.outputs.get(&ArtifactValueId(out0.0)).unwrap(),
            &[20, 0, 0, 0]
        );
        assert_eq!(
            completion.outputs.get(&ArtifactValueId(out1.0)).unwrap(),
            &[40, 0, 0, 0]
        );
    }
}
