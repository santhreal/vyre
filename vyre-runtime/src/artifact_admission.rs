use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, RwLock};

use thiserror::Error;
use vyre_foundation::ir::{BufferAccess, BufferDecl, GraphValueId, Program};
use vyre_megakernel::{
    Artifact, ArtifactEnvelope, CompileError, Diagnostic, FinalistEvaluator, ResourceAbiRecord,
    TargetCompileError, TargetCompiler, TargetPayload, TargetPayloadFormat,
    ValidatedCompileRequest,
};

use crate::pipeline_cache::{PipelineCacheStore, PipelineFingerprint};
use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BackendRegistration, BindingSet,
    BoundResource, Completion, DeviceIdentity, Resource, Submission,
};
use vyre_megakernel::{AbiAccess, ArtifactValueId, Digest, ResourceLifetime, ResourceRecord};

/// Failure to authenticate an artifact envelope or select its exact required payload.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("artifact admission rejected: {source}")]
pub struct ArtifactAdmissionError {
    #[source]
    source: CompileError,
}

impl ArtifactAdmissionError {
    /// Canonical structured diagnostic produced while decoding or selecting the payload.
    #[must_use]
    pub const fn diagnostic(&self) -> &Diagnostic {
        &self.source.diagnostic
    }

    /// Recover the canonical error without flattening its diagnostic context.
    #[must_use]
    pub fn into_compile_error(self) -> CompileError {
        self.source
    }
}

impl From<CompileError> for ArtifactAdmissionError {
    fn from(source: CompileError) -> Self {
        Self { source }
    }
}

/// Authenticated canonical envelope with one caller-selected exact payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmittedArtifact {
    envelope: ArtifactEnvelope,
    target_payload_index: usize,
}

impl AdmittedArtifact {
    /// Borrow the authenticated canonical envelope.
    #[must_use]
    pub const fn envelope(&self) -> &ArtifactEnvelope {
        &self.envelope
    }

    /// Borrow the canonical backend-neutral artifact.
    #[must_use]
    pub const fn neutral(&self) -> &Artifact {
        self.envelope.neutral()
    }

    /// Borrow the exact target payload selected during admission.
    #[must_use]
    pub fn target_payload(&self) -> &TargetPayload {
        &self.envelope.target_payloads()[self.target_payload_index]
    }

    /// Consume the admission result and recover its owned canonical envelope.
    #[must_use]
    pub fn into_envelope(self) -> ArtifactEnvelope {
        self.envelope
    }
}

/// Decode and authenticate canonical envelope bytes, then require one exact payload format.
pub fn admit_artifact(
    envelope_bytes: &[u8],
    required_format: &TargetPayloadFormat,
) -> Result<AdmittedArtifact, ArtifactAdmissionError> {
    let envelope = ArtifactEnvelope::from_bytes(envelope_bytes)?;
    admit_envelope(envelope, required_format)
}

/// Authenticate an already-decoded canonical envelope and require one exact payload format.
///
/// Prefer this when a producer such as AOT packaging has already decoded the envelope
/// and only the exact target-format selection remains.
pub fn admit_envelope(
    envelope: ArtifactEnvelope,
    required_format: &TargetPayloadFormat,
) -> Result<AdmittedArtifact, ArtifactAdmissionError> {
    let target_payload_index = envelope.require_target_payload_index(required_format)?;
    Ok(AdmittedArtifact {
        envelope,
        target_payload_index,
    })
}

/// Load verified cache payload bytes and admit them as a canonical envelope.
///
/// `DiskCache` / `PipelineCacheStore` are format-agnostic blob stores. AOT
/// writes `ArtifactEnvelope` bytes as the payload (plus the store's
/// own BLAKE3 footer, stripped by [`PipelineCacheStore::get`]). Callers that
/// treat a cache hit as executable MUST run this helper (or
/// [`admit_artifact`] on the payload) before dispatch. A miss is `Ok(None)`.
///
/// # Errors
///
/// Returns [`ArtifactAdmissionError`] when payload bytes are present but are
/// not an authentic envelope with the required target format.
pub fn admit_cached_artifact(
    store: &dyn PipelineCacheStore,
    fingerprint: &PipelineFingerprint,
    required_format: &TargetPayloadFormat,
) -> Result<Option<AdmittedArtifact>, ArtifactAdmissionError> {
    let Some(payload) = store.get(fingerprint) else {
        return Ok(None);
    };
    admit_artifact(&payload, required_format).map(Some)
}

/// Runtime materialization or submission failure with structured admission preserved.
#[derive(Debug, Error)]
pub enum ArtifactSessionError {
    /// Canonical envelope or target-format admission failed.
    #[error(transparent)]
    Admission(#[from] ArtifactAdmissionError),
    /// Neutral compilation or target payload construction failed.
    #[error(transparent)]
    Compile(#[from] CompileError),
    /// The registered target compiler rejected the selected artifact.
    #[error(transparent)]
    Target(#[from] TargetCompileError),
    /// Registered device materialization or submission failed.
    #[error(transparent)]
    Backend(#[from] BackendError),
    /// Runtime lifecycle state was poisoned by a panic while locked.
    #[error("artifact session state is poisoned: {0}. Fix: discard and rebuild the session")]
    State(String),
}

struct MaterializedArtifact {
    admitted: AdmittedArtifact,
    materializer: Arc<dyn ArtifactMaterializer>,
    instance: Box<dyn ArtifactInstance>,
}

/// Authenticated immutable artifact materialized on one registered device generation.
pub struct ArtifactSession {
    registration: &'static BackendRegistration,
    state: RwLock<MaterializedArtifact>,
}

impl ArtifactSession {
    /// Compile one validated request, attach the registered target payload, and
    /// materialize the authenticated artifact.
    pub fn compile(
        registration: &'static BackendRegistration,
        request: &ValidatedCompileRequest,
    ) -> Result<Self, ArtifactSessionError> {
        let materializer = Arc::from(registration.materializer()?);
        Self::compile_with_materializer(registration, request, materializer)
    }
    /// Compile and materialize through one caller-owned materializer generation.
    pub fn compile_with_materializer(
        registration: &'static BackendRegistration,
        request: &ValidatedCompileRequest,
        materializer: Arc<dyn ArtifactMaterializer>,
    ) -> Result<Self, ArtifactSessionError> {
        let compiler = registration.target_compiler()?;
        let artifact = vyre_megakernel::compile_measured(
            request,
            &DeviceFinalists {
                compiler: compiler.as_ref(),
                materializer: materializer.as_ref(),
                representative_inputs: request.representative_inputs(),
            },
        )?;
        let envelope = vyre_megakernel::attach_target(artifact, compiler.as_ref())?;
        Self::from_envelope_with_materializer(registration, envelope, materializer)
    }

    /// Admit one already-decoded canonical envelope and materialize its exact target bytes.
    pub fn from_envelope(
        registration: &'static BackendRegistration,
        envelope: ArtifactEnvelope,
    ) -> Result<Self, ArtifactSessionError> {
        let materializer = Arc::from(registration.materializer()?);
        Self::from_envelope_with_materializer(registration, envelope, materializer)
    }

    /// Admit and materialize through one caller-owned materializer generation.
    pub fn from_envelope_with_materializer(
        registration: &'static BackendRegistration,
        envelope: ArtifactEnvelope,
        materializer: Arc<dyn ArtifactMaterializer>,
    ) -> Result<Self, ArtifactSessionError> {
        let admitted = admit_envelope(envelope, materializer.device().target_format())?;
        let instance = materializer.materialize(admitted.neutral(), admitted.target_payload())?;
        validate_instance(&admitted, materializer.as_ref(), instance.as_ref())?;
        Ok(Self {
            registration,
            state: RwLock::new(MaterializedArtifact {
                admitted,
                materializer,
                instance,
            }),
        })
    }

    /// Authenticate canonical envelope bytes and materialize the exact device format.
    pub fn from_bytes(
        registration: &'static BackendRegistration,
        envelope_bytes: &[u8],
    ) -> Result<Self, ArtifactSessionError> {
        let envelope =
            ArtifactEnvelope::from_bytes(envelope_bytes).map_err(ArtifactAdmissionError::from)?;
        Self::from_envelope(registration, envelope)
    }

    /// Neutral artifact identity shared by every session and device generation.
    pub fn artifact(&self) -> Result<Digest, ArtifactSessionError> {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        Ok(state.admitted.neutral().digest())
    }
    /// Exact authenticated target payload identity materialized by this session.
    pub fn payload(&self) -> Result<Digest, ArtifactSessionError> {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        Ok(state.admitted.target_payload().digest())
    }

    /// Current immutable device generation identity.
    pub fn device(&self) -> Result<DeviceIdentity, ArtifactSessionError> {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        Ok(state.instance.device().clone())
    }

    /// Build an empty typed binding set for this exact artifact.
    pub fn bindings(&self) -> Result<BindingSet, ArtifactSessionError> {
        Ok(BindingSet::new(self.artifact()?))
    }

    /// Submit typed bindings without exposing the materialized native instance.
    pub fn submit(
        &self,
        bindings: BindingSet,
    ) -> Result<Box<dyn Submission>, ArtifactSessionError> {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        Ok(state.instance.submit(bindings)?)
    }

    /// Submit and wait for typed completion/readback.
    pub fn submit_and_wait(
        &self,
        bindings: BindingSet,
    ) -> Result<Completion, ArtifactSessionError> {
        Ok(self.submit(bindings)?.wait()?)
    }

    /// Reacquire the registered device and rematerialize authenticated target bytes.
    ///
    /// This path never invokes the target compiler, semantic optimizer, or lowering.
    pub fn rematerialize(&self) -> Result<DeviceIdentity, ArtifactSessionError> {
        let mut state = self
            .state
            .write()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        let materializer: Arc<dyn ArtifactMaterializer> =
            Arc::from(self.registration.materializer()?);
        let admitted = admit_envelope(
            state.admitted.envelope().clone(),
            materializer.device().target_format(),
        )?;
        let instance = materializer.materialize(admitted.neutral(), admitted.target_payload())?;
        validate_instance(&admitted, materializer.as_ref(), instance.as_ref())?;
        let identity = instance.device().clone();
        *state = MaterializedArtifact {
            admitted,
            materializer,
            instance,
        };
        Ok(identity)
    }

    /// Resolve one canonical artifact ABI value by its stable resource name.
    pub fn resource(&self, name: &str) -> Result<ArtifactValueId, ArtifactSessionError> {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        state
            .admitted
            .neutral()
            .resources()
            .iter()
            .find(|resource| resource.name == name)
            .map(|resource| resource.value)
            .ok_or_else(|| {
                BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: artifact ABI does not declare required runtime resource `{name}`."
                    ),
                }
                .into()
            })
    }
    /// Allocate one resident resource from this session's materializer generation.
    pub fn allocate_resident(&self, byte_len: usize) -> Result<Resource, ArtifactSessionError> {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        Ok(state.materializer.allocate_resident(byte_len)?)
    }

    /// Upload bytes into one resource owned by this session's materializer.
    pub fn upload_resident(
        &self,
        resource: &Resource,
        bytes: &[u8],
    ) -> Result<(), ArtifactSessionError> {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        Ok(state.materializer.upload_resident(resource, bytes)?)
    }

    /// Upload bytes at one offset into a resource owned by this session's materializer.
    pub fn upload_resident_at(
        &self,
        resource: &Resource,
        offset_bytes: usize,
        bytes: &[u8],
    ) -> Result<(), ArtifactSessionError> {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        Ok(state
            .materializer
            .upload_resident_at(resource, offset_bytes, bytes)?)
    }

    /// Release one resource owned by this session's materializer.
    pub fn free_resident(&self, resource: Resource) -> Result<(), ArtifactSessionError> {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        Ok(state.materializer.free_resident(resource)?)
    }

    /// Bind backend-resident resources by canonical value identity.
    pub fn resident_bindings(
        &self,
        resources: &[Resource],
    ) -> Result<BindingSet, ArtifactSessionError> {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        let canonical_resources = state.admitted.neutral().resources();
        let target_entries = state.admitted.target_payload().entries();
        if target_entries.len() != 1 {
            return Err(BackendError::UnsupportedFeature {
                name: "resident bindings for multi-entry artifacts".to_string(),
                backend: state.instance.device().backend.to_string(),
            }
            .into());
        }
        if canonical_resources.len() != resources.len() {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: target entry requires {} resident resource(s), but the caller supplied {}.",
                    canonical_resources.len(),
                    resources.len()
                ),
            }
            .into());
        }
        let mut typed = BindingSet::new(state.admitted.neutral().digest());
        for (record, resource) in canonical_resources.iter().zip(resources) {
            let bound = BoundResource::Resident(resource.clone());
            if let Some(existing) = typed.resources().get(&record.value) {
                if existing != &bound {
                    return Err(BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: conflicting resident resources supplied for canonical value {}.",
                            record.value.0
                        ),
                    }
                    .into());
                }
            }
            typed.insert(record.value, bound);
        }
        for entry in target_entries {
            for target_binding in &entry.resource_bindings {
                if !typed.resources().contains_key(&target_binding.resource) {
                    return Err(BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: target entry `{}` requires resident resource for canonical value {} at group {}, slot {}.",
                            entry.name,
                            target_binding.resource.0,
                            target_binding.group,
                            target_binding.slot
                        ),
                    }
                    .into());
                }
            }
        }
        Ok(typed)
    }

    /// Bind resident resources matching the declared non-shared buffer order of `program`.
    pub fn program_resident_bindings(
        &self,
        program: &Program,
        resources: &[Resource],
    ) -> Result<BindingSet, ArtifactSessionError> {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        let target_entries = state.admitted.target_payload().entries();
        if target_entries.len() != 1 {
            return Err(BackendError::UnsupportedFeature {
                name: "resident bindings for multi-entry artifacts".to_string(),
                backend: state.instance.device().backend.to_string(),
            }
            .into());
        }
        let non_shared_buffers: Vec<&BufferDecl> = program
            .buffers()
            .iter()
            .filter(|decl| decl.access != BufferAccess::Workgroup)
            .collect();
        if non_shared_buffers.len() != resources.len() {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: target entry requires {} resident resource(s), but the caller supplied {}.",
                    non_shared_buffers.len(),
                    resources.len()
                ),
            }
            .into());
        }
        let canonical_by_name = state
            .admitted
            .neutral()
            .canonical_value_by_name()
            .map_err(|collision| {
                ArtifactSessionError::from(BackendError::InvalidProgram {
                    fix: collision.to_string(),
                })
            })?;
        let mut typed = BindingSet::new(state.admitted.neutral().digest());
        for (buffer_decl, resource) in non_shared_buffers.into_iter().zip(resources) {
            let value_id = canonical_by_name
                .get(buffer_decl.name.as_ref())
                .copied()
                .ok_or_else(|| {
                    ArtifactSessionError::from(BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: artifact resources must carry canonical value for Program buffer `{}`.",
                            buffer_decl.name
                        ),
                    })
                })?;
            let bound = BoundResource::Resident(resource.clone());
            if let Some(existing) = typed.resources().get(&value_id) {
                if existing != &bound {
                    return Err(BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: conflicting resident resources supplied for Program buffer `{}` (canonical value {}).",
                            buffer_decl.name, value_id.0
                        ),
                    }
                    .into());
                }
            }
            typed.insert(value_id, bound);
        }
        for entry in target_entries {
            for target_binding in &entry.resource_bindings {
                if !typed.resources().contains_key(&target_binding.resource) {
                    return Err(BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: target entry `{}` requires resident resource for canonical value {} at group {}, slot {}.",
                            entry.name,
                            target_binding.resource.0,
                            target_binding.group,
                            target_binding.slot
                        ),
                    }
                    .into());
                }
            }
        }
        Ok(typed)
    }

    /// Bind resident resources by exact resource name.
    pub fn resident_bindings_by_name<'a, I>(
        &self,
        resources: I,
    ) -> Result<BindingSet, ArtifactSessionError>
    where
        I: IntoIterator<Item = (&'a str, &'a Resource)>,
    {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        let target_entries = state.admitted.target_payload().entries();
        if target_entries.len() != 1 {
            return Err(BackendError::UnsupportedFeature {
                name: "resident bindings for multi-entry artifacts".to_string(),
                backend: state.instance.device().backend.to_string(),
            }
            .into());
        }
        let canonical_by_name = state
            .admitted
            .neutral()
            .canonical_value_by_name()
            .map_err(|collision| {
                ArtifactSessionError::from(BackendError::InvalidProgram {
                    fix: collision.to_string(),
                })
            })?;
        let mut typed = BindingSet::new(state.admitted.neutral().digest());
        for (name, resource) in resources {
            let value_id = canonical_by_name.get(name).copied().ok_or_else(|| {
                ArtifactSessionError::from(BackendError::InvalidProgram {
                    fix: format!("Fix: artifact resources do not carry a canonical value named `{name}`."),
                })
            })?;
            let bound = BoundResource::Resident(resource.clone());
            if let Some(existing) = typed.resources().get(&value_id) {
                if existing != &bound {
                    return Err(BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: conflicting resident resources supplied for resource `{name}` (canonical value {}).",
                            value_id.0
                        ),
                    }
                    .into());
                }
            }
            typed.insert(value_id, bound);
        }
        for entry in target_entries {
            for target_binding in &entry.resource_bindings {
                if !typed.resources().contains_key(&target_binding.resource) {
                    return Err(BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: target entry `{}` requires resident resource for canonical value {} at group {}, slot {}.",
                            entry.name,
                            target_binding.resource.0,
                            target_binding.group,
                            target_binding.slot
                        ),
                    }
                    .into());
                }
            }
        }
        Ok(typed)
    }

    /// Bind resident resources by exact canonical value identity.
    pub fn resident_bindings_by_value<I>(
        &self,
        resources: I,
    ) -> Result<BindingSet, ArtifactSessionError>
    where
        I: IntoIterator<Item = (ArtifactValueId, Resource)>,
    {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        let target_entries = state.admitted.target_payload().entries();
        if target_entries.len() != 1 {
            return Err(BackendError::UnsupportedFeature {
                name: "resident bindings for multi-entry artifacts".to_string(),
                backend: state.instance.device().backend.to_string(),
            }
            .into());
        }
        let valid_values: BTreeSet<ArtifactValueId> = state
            .admitted
            .neutral()
            .resources()
            .iter()
            .map(|r| r.value)
            .collect();
        let mut typed = BindingSet::new(state.admitted.neutral().digest());
        for (value_id, resource) in resources {
            if !valid_values.contains(&value_id) {
                return Err(BackendError::InvalidProgram {
                    fix: format!("Fix: artifact has no canonical value {}.", value_id.0),
                }
                .into());
            }
            let bound = BoundResource::Resident(resource);
            if let Some(existing) = typed.resources().get(&value_id) {
                if existing != &bound {
                    return Err(BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: conflicting resident resources supplied for canonical value {}.",
                            value_id.0
                        ),
                    }
                    .into());
                }
            }
            typed.insert(value_id, bound);
        }
        for entry in target_entries {
            for target_binding in &entry.resource_bindings {
                if !typed.resources().contains_key(&target_binding.resource) {
                    return Err(BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: target entry `{}` requires resident resource for canonical value {} at group {}, slot {}.",
                            entry.name,
                            target_binding.resource.0,
                            target_binding.group,
                            target_binding.slot
                        ),
                    }
                    .into());
                }
            }
        }
        Ok(typed)
    }

    /// Bind host inputs in canonical ABI slot order.
    pub fn host_bindings(&self, inputs: &[&[u8]]) -> Result<BindingSet, ArtifactSessionError> {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        let artifact = state.admitted.neutral();
        let resources = host_input_resources(artifact)?;
        if resources.len() != inputs.len() {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: artifact ABI requires {} host input buffer(s), but the caller supplied {}.",
                    resources.len(),
                    inputs.len()
                ),
            }
            .into());
        }
        let mut bindings = BindingSet::new(state.admitted.neutral().digest());
        for ((resource, _), bytes) in resources.into_iter().zip(inputs) {
            bindings.insert(resource.value, BoundResource::Host(bytes.to_vec()));
        }
        Ok(bindings)
    }

    /// Submit host inputs in canonical ABI order and wait for typed completion.
    pub fn submit_host_inputs(&self, inputs: &[&[u8]]) -> Result<Completion, ArtifactSessionError> {
        self.submit_and_wait(self.host_bindings(inputs)?)
    }

    /// Project writable completion values in canonical ABI slot order.
    ///
    /// Slot order is graph value order. A caller holding the Program the graph was
    /// lifted from reads [`Self::program_outputs`] instead, because that order is
    /// the buffer declaration order the Program author bound.
    pub fn ordered_outputs(
        &self,
        completion: &Completion,
    ) -> Result<Vec<Vec<u8>>, ArtifactSessionError> {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        let mut resources = state
            .admitted
            .neutral()
            .abi()
            .resources
            .iter()
            .filter(|resource| {
                matches!(resource.access, AbiAccess::ReadWrite | AbiAccess::WriteOnly)
            })
            .collect::<Vec<_>>();
        resources.sort_unstable_by_key(|resource| resource.slot);
        resources
            .into_iter()
            .map(|resource| {
                completion
                    .outputs
                    .get(&resource.value)
                    .or_else(|| completion.retained.get(&resource.value))
                    .cloned()
                    .ok_or_else(|| {
                        BackendError::InvalidProgram {
                            fix: format!(
                                "Fix: materializer completion must project writable artifact value {}.",
                                resource.value.0
                            ),
                        }
                        .into()
                    })
            })
            .collect()
    }

    /// Project writable completion values in Program buffer declaration order.
    ///
    /// [`Self::ordered_outputs`] projects canonical ABI slot order, which numbers
    /// graph values. A graph lifted from one Program mints an external value for
    /// every retained read-write buffer before the node that produces the declared
    /// outputs, so slot order is retained-then-output and cannot express a Program
    /// that declares an output buffer before a retained one. A caller that authored
    /// the Program binds its inputs and reads its outputs in declaration order, the
    /// order `Program::output_buffer_indices` reports, so this projects onto that
    /// order through the canonical resource names.
    ///
    /// # Errors
    ///
    /// Returns an error when the artifact does not carry one canonical resource per
    /// declared writable buffer, or when the completion omits one of those values.
    pub fn program_outputs(
        &self,
        program: &Program,
        completion: &Completion,
    ) -> Result<Vec<Vec<u8>>, ArtifactSessionError> {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        let canonical =
            state
                .admitted
                .neutral()
                .canonical_value_by_name()
                .map_err(|collision| {
                    ArtifactSessionError::from(BackendError::InvalidProgram {
                        fix: collision.to_string(),
                    })
                })?;
        let buffers = program.buffers();
        program
            .output_buffer_indices()
            .iter()
            .map(|index| {
                let name = buffers
                    .get(*index as usize)
                    .ok_or_else(|| {
                        ArtifactSessionError::from(BackendError::InvalidProgram {
                            fix: format!(
                                "Fix: Program declares writable buffer index {index}, which is outside its buffer list."
                            ),
                        })
                    })?
                    .name();
                let value = canonical.get(name).copied().ok_or_else(|| {
                    ArtifactSessionError::from(BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: artifact resources must carry canonical value `{name}` for the declared writable buffer."
                        ),
                    })
                })?;
                completion
                    .outputs
                    .get(&value)
                    .or_else(|| completion.retained.get(&value))
                    .cloned()
                    .ok_or_else(|| {
                        ArtifactSessionError::from(BackendError::InvalidProgram {
                            fix: format!(
                                "Fix: materializer completion must project writable artifact value {} (`{name}`).",
                                value.0
                            ),
                        })
                    })
            })
            .collect()
    }

    fn retained_values(&self) -> Result<BTreeSet<ArtifactValueId>, ArtifactSessionError> {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        Ok(state
            .admitted
            .neutral()
            .resources()
            .iter()
            .filter(|resource| resource.lifetime == ResourceLifetime::Retained)
            .map(|resource| resource.value)
            .collect())
    }
}

/// Runtime-owned retained binding policy over one immutable [`ArtifactSession`].
pub struct RetainedArtifactSession {
    session: ArtifactSession,
    retained_values: BTreeSet<ArtifactValueId>,
    retained: Mutex<BTreeMap<ArtifactValueId, Vec<u8>>>,
}

impl RetainedArtifactSession {
    /// Create retained policy state and require every retained ABI value initially.
    pub fn new(
        session: ArtifactSession,
        initial: BTreeMap<ArtifactValueId, Vec<u8>>,
    ) -> Result<Self, ArtifactSessionError> {
        let retained_values = session.retained_values()?;
        if initial.keys().copied().collect::<BTreeSet<_>>() != retained_values {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: initialize exactly every retained artifact value before creating a retained session.".to_string(),
            }
            .into());
        }
        Ok(Self {
            session,
            retained_values,
            retained: Mutex::new(initial),
        })
    }

    /// Neutral artifact identity shared with ephemeral sessions.
    pub fn artifact(&self) -> Result<Digest, ArtifactSessionError> {
        self.session.artifact()
    }

    /// Current immutable device generation identity.
    pub fn device(&self) -> Result<DeviceIdentity, ArtifactSessionError> {
        self.session.device()
    }

    /// Build empty transient bindings for the shared neutral artifact.
    pub fn bindings(&self) -> Result<BindingSet, ArtifactSessionError> {
        self.session.bindings()
    }

    /// Reacquire a device and rematerialize the authenticated artifact bytes.
    pub fn rematerialize(&self) -> Result<DeviceIdentity, ArtifactSessionError> {
        self.session.rematerialize()
    }

    /// Replace runtime-owned retained bytes before the next submission.
    ///
    /// # Errors
    ///
    /// Returns an error unless the update covers exactly every retained ABI value.
    pub fn replace_retained(
        &self,
        values: BTreeMap<ArtifactValueId, Vec<u8>>,
    ) -> Result<(), ArtifactSessionError> {
        if values.keys().copied().collect::<BTreeSet<_>>() != self.retained_values {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: replace exactly every retained artifact value.".to_string(),
            }
            .into());
        }
        *self
            .retained
            .lock()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))? = values;
        Ok(())
    }

    /// Submit transient bindings, merge retained state, and atomically retain completion state.
    pub fn submit_and_wait(
        &self,
        mut bindings: BindingSet,
    ) -> Result<Completion, ArtifactSessionError> {
        if bindings.artifact() != self.session.artifact()? {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: retained session bindings must name the session artifact digest."
                    .to_string(),
            }
            .into());
        }
        {
            let retained = self
                .retained
                .lock()
                .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
            for (value, bytes) in retained.iter() {
                bindings.insert(*value, BoundResource::Host(bytes.clone()));
            }
        }
        let completion = self.session.submit_and_wait(bindings)?;
        if completion.retained.keys().copied().collect::<BTreeSet<_>>() != self.retained_values {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: artifact completion must return exactly every retained ABI value."
                    .to_string(),
            }
            .into());
        }
        *self
            .retained
            .lock()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))? =
            completion.retained.clone();
        Ok(completion)
    }
}

fn validate_instance(
    admitted: &AdmittedArtifact,
    materializer: &dyn ArtifactMaterializer,
    instance: &dyn ArtifactInstance,
) -> Result<(), BackendError> {
    if instance.artifact() != admitted.neutral().digest()
        || instance.payload() != admitted.target_payload().digest()
        || instance.device() != materializer.device().identity()
    {
        return Err(BackendError::InvalidProgram {
            fix: "Fix: materialized instance identities must exactly match the admitted artifact, target payload, and acquired device generation.".to_string(),
        });
    }
    Ok(())
}

/// Artifact ABI resources the caller supplies host bytes for, in slot order,
/// each paired with its canonical resource record.
///
/// One fact decides the set: whether an artifact entry produces the value. A
/// value no entry produces has no other source, so its contents at launch are
/// the caller's. A value some entry produces is device state, however many
/// entries also read it, and a retained value's successor is produced even
/// though its predecessor is bound by the caller.
///
/// The earlier form asked for the values in `entry.outputs` that were absent
/// from `entry.inputs`. That is the same set on every representable artifact,
/// because a node's newly minted outputs can never appear among the values it
/// binds as inputs, but it reads as though the arity depended on how the
/// compiler grouped the graph. Stating the rule directly removes the question.
///
/// Access then removes what nothing reads: a write-only slot's contents at
/// launch are unobservable, so binding bytes to it would ask the caller for a
/// buffer no kernel reads.
///
/// Measurement and caller submission select the same set: a measured launch that
/// bound a different set would not be timing the launch the caller performs.
///
/// # Errors
///
/// Returns an error when an ABI slot names a value the resource set does not
/// carry. Both describe one graph, so a gap is a malformed artifact, and
/// assuming a byte count for the missing value binds a buffer at the wrong size.
fn host_input_resources(
    artifact: &Artifact,
) -> Result<Vec<(&ResourceAbiRecord, &ResourceRecord)>, BackendError> {
    let produced = entry_produced_values(artifact);
    let mut resources = Vec::new();
    for resource in &artifact.abi().resources {
        let record = artifact
            .resources()
            .iter()
            .find(|record| record.value == resource.value)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: artifact ABI slot {} names value {}, which the artifact resource set does not carry. Regenerate the artifact so its ABI and its resource set describe the same graph.",
                    resource.slot, resource.value.0
                ),
            })?;
        if !produced.contains(&resource.value) && kernel_reads_initial_bytes(resource.access) {
            resources.push((resource, record));
        }
    }
    resources.sort_unstable_by_key(|(resource, _)| resource.slot);
    Ok(resources)
}

/// Values some artifact entry produces.
fn entry_produced_values(artifact: &Artifact) -> BTreeSet<ArtifactValueId> {
    artifact
        .abi()
        .entries
        .iter()
        .flat_map(|entry| entry.outputs.iter().copied())
        .collect()
}

/// Whether the kernel reads what a slot holds at launch.
///
/// Exhaustive on purpose: a new access class must state whether its initial
/// contents are read before a caller can be asked for them, and a wildcard arm
/// would file it under whichever answer happened to be first.
fn kernel_reads_initial_bytes(access: AbiAccess) -> bool {
    match access {
        AbiAccess::ReadOnly | AbiAccess::Uniform | AbiAccess::ReadWrite => true,
        AbiAccess::WriteOnly => false,
    }
}

/// Compiler finalist evaluation on the acquired device.
///
/// The compiler decides which plans are finalists; this supplies the device half:
/// the registered target compiler, and one materialized launch per measurement
/// whose duration is the device timestamp the backend reports. Measured
/// compilation binds exact representative workload inputs for each host-input
/// resource, preventing traps from aborting compile-time device timing.
struct DeviceFinalists<'a> {
    compiler: &'a dyn TargetCompiler,
    materializer: &'a dyn ArtifactMaterializer,
    representative_inputs: &'a BTreeMap<GraphValueId, Vec<u8>>,
}

impl FinalistEvaluator for DeviceFinalists<'_> {
    fn target_compiler(&self) -> &dyn TargetCompiler {
        self.compiler
    }

    fn measure(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<u64, TargetCompileError> {
        let instance = self
            .materializer
            .materialize(artifact, payload)
            .map_err(measurement_failure)?;
        let mut bindings = BindingSet::new(artifact.digest());
        for (resource, record) in host_input_resources(artifact).map_err(measurement_failure)? {
            let byte_count = record.byte_count;
            let byte_count = usize::try_from(byte_count).map_err(|_| {
                TargetCompileError::Unsupported(format!(
                    "artifact value {} needs {byte_count} bytes, which exceeds host addressing",
                    resource.value.0
                ))
            })?;
            let bytes = self
                .representative_inputs
                .get(&GraphValueId(resource.value.0))
                .ok_or_else(|| {
                    TargetCompileError::Unsupported(format!(
                        "finalist measurement missing representative input for host-input resource `{}` (value {})",
                        record.name, resource.value.0
                    ))
                })?;
            if bytes.len() != byte_count {
                return Err(TargetCompileError::Unsupported(format!(
                    "finalist measurement representative input for resource `{}` (value {}) has {} bytes, but artifact requires {byte_count} bytes",
                    record.name, resource.value.0, bytes.len()
                )));
            }
            bindings.insert(resource.value, BoundResource::Host(bytes.clone()));
        }
        let completion = instance
            .submit(bindings)
            .and_then(|submission| submission.wait())
            .map_err(measurement_failure)?;
        completion.device_ns.ok_or_else(|| {
            TargetCompileError::Unsupported(
                "device reported no launch duration for a finalist measurement".to_string(),
            )
        })
    }
}

fn measurement_failure(error: BackendError) -> TargetCompileError {
    TargetCompileError::Unsupported(format!("finalist measurement failed on device: {error}"))
}
