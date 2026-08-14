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
    BackendError, BindingPlan, BindingSet, BoundResource, Completion, Device, DeviceIdentity,
    DispatchConfig, Resource, Submission,
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

    let mut admitted = Vec::with_capacity(selected.len());
    for ((image, record), entry) in bundle
        .modules
        .into_iter()
        .zip(selected)
        .zip(payload.entries())
    {
        admit_module_identity(&image, record)?;
        if image.entry_point != "main" {
            return Err(invalid_module("target module entry point must be `main`"));
        }
        if entry.name != image.entry_point {
            return Err(invalid_module(
                "target entry metadata must name the emitted target entry point",
            ));
        }
        let program = Arc::new(Program::from_wire(&image.program).map_err(|error| {
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

impl MaterializerDevice {
    /// Describe a device that stays healthy for the life of the materializer.
    #[must_use]
    pub fn new(
        identity: DeviceIdentity,
        format: TargetPayloadFormat,
        profile: TargetProfile,
    ) -> Self {
        Self {
            identity,
            format,
            profile,
            revoked: None,
        }
    }

    /// Describe a device whose generation is invalidated when `revoked` is set.
    #[must_use]
    pub fn revocable(
        identity: DeviceIdentity,
        format: TargetPayloadFormat,
        profile: TargetProfile,
        revoked: Arc<AtomicBool>,
    ) -> Self {
        Self {
            identity,
            format,
            profile,
            revoked: Some(revoked),
        }
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
/// Three of the four backends already shipped exactly these strings, so they
/// are one code path now. A backend whose wording differs supplies its own
/// record rather than moving anyone else's text.
pub const NEUTRAL_MESSAGES: InstanceMessages = InstanceMessages {
    foreign_artifact: || invalid_module("bindings name a different neutral artifact"),
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
    completion_consumed: || {
        invalid_module("each Submission completion may be consumed only once")
    },
};

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
        Self {
            artifact: artifact.digest(),
            payload: payload.digest(),
            device,
            values: resources.values,
            outputs: resources.outputs,
            retained: resources.retained,
            messages,
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
    pub fn gather_inputs<'state>(
        &self,
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
            let value = self.value_for_buffer(buffer.name())?;
            inputs[input_index] = state
                .get(&value)
                .map(Vec::as_slice)
                .ok_or_else(|| unbound(value, buffer.name()))?;
        }
        Ok(inputs)
    }

    /// Write dispatch results back onto the canonical values they implement.
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
        produced: &[Vec<u8>],
        state: &mut BTreeMap<ArtifactValueId, Vec<u8>>,
        missing: fn(usize, &str) -> BackendError,
    ) -> Result<(), BackendError> {
        for binding in &plan.bindings {
            let Some(output_index) = binding.output_index else {
                continue;
            };
            let buffer = &program.buffers()[binding.buffer_index];
            let value = self.value_for_buffer(buffer.name())?;
            let bytes = produced
                .get(output_index)
                .ok_or_else(|| missing(output_index, buffer.name()))?;
            state.insert(value, bytes.clone());
        }
        Ok(())
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
