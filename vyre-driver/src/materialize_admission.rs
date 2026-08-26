//! Backend-neutral target-payload admission shared by every concrete driver.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use vyre_foundation::ir::Program;
use vyre_megakernel::{
    Artifact, ArtifactValueId, FusionRecord, ResourceLifetime, TargetModuleBundle,
    TargetModuleImage, TargetPayload, TargetPayloadFormat, TargetProfile, TargetResourceBinding,
};

use crate::{BackendError, Device, DeviceIdentity, DispatchConfig, ResidentOwner};

use crate::materialize::{compile_error, invalid_module, InstanceCore, InstanceMessages};

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
    /// Canonical directional resource metadata carried by the payload entry.
    pub resource_bindings: Vec<TargetResourceBinding>,
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
    let mut groups = BTreeSet::new();
    for record in selected {
        if !groups.insert(record.id) {
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

    // The selected plan's order is the recorded dependency order: the artifact
    // refuses a plan whose groups precede a group they depend on. Walking it,
    // rather than the bundle's own module order, is what makes the recorded DAG
    // the submission order for every backend.
    let mut images = BTreeMap::new();
    for image in bundle.modules {
        if images.insert(image.group, image).is_some() {
            return Err(invalid_module("target bundle names one fusion group twice"));
        }
    }
    let mut admitted = Vec::with_capacity(selected.len());
    for record in selected {
        let image = images.remove(&record.id).ok_or_else(|| {
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
        // The neutral record is the selected schedule projected onto one launch,
        // and the envelope already refused a payload that states another shape.
        // Reading it here rather than the payload keeps one authority for the
        // submitted geometry.
        let geometry = artifact
            .geometry()
            .iter()
            .find(|geometry| geometry.node == entry_node)
            .ok_or_else(|| {
                invalid_module("admitted node has no selected launch geometry in the artifact")
            })?;
        let mut config = DispatchConfig::default();
        config.launch = Some(crate::launch_directive::LaunchDirective::from_record(
            geometry,
            target.backend_id,
        )?);
        admitted.push(AdmittedModule {
            image,
            program,
            config,
            resource_bindings: entry.resource_bindings.clone(),
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
    if let Ok(by_name) = artifact.canonical_value_by_name() {
        for (name, value) in by_name {
            projection.values.insert(name.to_string(), value);
        }
    } else {
        for resource in artifact.resources() {
            projection
                .values
                .insert(resource.name.clone(), resource.value);
        }
    }
    for resource in artifact.resources() {
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
    /// Returns [`fn@compile_error`] naming `spec.backend` when the extension and
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
    ///
    /// # Errors
    ///
    /// Returns the module-order projection failures from [`InstanceCore::new`].
    pub fn instance(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
        messages: InstanceMessages,
    ) -> Result<InstanceCore, BackendError> {
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
