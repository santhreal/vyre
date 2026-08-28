use std::collections::BTreeSet;
use std::sync::{Arc, RwLock};

use thiserror::Error;
use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BackendRegistration, BindingSet,
    BoundResource, Completion, DeviceIdentity, Resource, Submission,
};
use vyre_foundation::ir::{BufferAccess, BufferDecl, Program};
use vyre_megakernel::{
    AbiAccess, ArtifactEnvelope, ArtifactValueId, CompileError, Digest, ResourceLifetime,
    TargetCompileError, TargetEntryPoint, ValidatedCompileRequest,
};

use super::finalist::{host_input_resources, validate_instance, DeviceFinalists};
use super::workspace::ArtifactWorkspace;
use super::{admit_envelope, AdmittedArtifact, ArtifactAdmissionError};

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

pub(super) struct MaterializedArtifact {
    pub(super) admitted: AdmittedArtifact,
    pub(super) materializer: Arc<dyn ArtifactMaterializer>,
    pub(super) instance: Box<dyn ArtifactInstance>,
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
                name: "positional resident bindings for multi-entry artifacts".to_string(),
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
        require_every_entry_binding(target_entries, &typed)?;
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
                name: "positional resident bindings for multi-entry artifacts".to_string(),
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
        let canonical_by_name =
            state
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
        require_every_entry_binding(target_entries, &typed)?;
        Ok(typed)
    }

    /// Bind resident resources by exact resource name, over any entry count.
    ///
    /// Naming a value binds it for every entry point that declares it, so a
    /// multi-entry artifact needs no per-entry binding call.
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
        let canonical_by_name =
            state
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
                    fix: format!(
                        "Fix: artifact resources do not carry a canonical value named `{name}`."
                    ),
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
        require_every_entry_binding(target_entries, &typed)?;
        Ok(typed)
    }

    /// Allocate the workspace the artifact recorded for its own values.
    ///
    /// The plan is the compiler's. Every region is allocated at the recorded
    /// byte count, so the caller cannot resize, merge, or drop one.
    ///
    /// # Errors
    ///
    /// Returns the materializer rejection when a region cannot be allocated.
    pub fn allocate_workspace(&self) -> Result<ArtifactWorkspace, ArtifactSessionError> {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        Ok(ArtifactWorkspace::allocate(
            state.admitted.neutral().allocation(),
            state.materializer.as_ref(),
        )?)
    }

    /// Release every region of one workspace allocation.
    ///
    /// # Errors
    ///
    /// Returns the first materializer rejection after releasing the rest.
    pub fn free_workspace(&self, workspace: ArtifactWorkspace) -> Result<(), ArtifactSessionError> {
        let state = self
            .state
            .read()
            .map_err(|error| ArtifactSessionError::State(error.to_string()))?;
        Ok(workspace.free(state.materializer.as_ref())?)
    }

    /// Bind caller-owned resources by name over one allocated workspace.
    ///
    /// A workspace-owned value is bound from `workspace`, and naming one in
    /// `resources` is refused rather than overridden: the artifact allocated
    /// that region for the entry points that pass a value between themselves,
    /// and a caller buffer in its place is a wrong bind, not a substitution.
    ///
    /// # Errors
    ///
    /// Returns an invalid-program rejection when a named resource is not a
    /// canonical value, when the caller names a workspace-owned value, when two
    /// resources conflict for one value, and when an entry point declares a
    /// binding neither the caller nor the workspace supplied.
    pub fn resident_bindings_with_workspace<'a, I>(
        &self,
        workspace: &ArtifactWorkspace,
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
        let canonical_by_name =
            state
                .admitted
                .neutral()
                .canonical_value_by_name()
                .map_err(|collision| {
                    ArtifactSessionError::from(BackendError::InvalidProgram {
                        fix: collision.to_string(),
                    })
                })?;
        let mut typed = BindingSet::new(state.admitted.neutral().digest());
        for (value, resource) in workspace.bindings() {
            typed.insert(*value, BoundResource::Resident(resource.clone()));
        }
        for (name, resource) in resources {
            let value_id = canonical_by_name.get(name).copied().ok_or_else(|| {
                ArtifactSessionError::from(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: artifact resources do not carry a canonical value named `{name}`."
                    ),
                })
            })?;
            if workspace.owns(value_id) {
                return Err(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: resource `{name}` (canonical value {}) is workspace-owned; bind the artifact workspace and drop it from the caller bindings.",
                        value_id.0
                    ),
                }
                .into());
            }
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
        require_every_entry_binding(target_entries, &typed)?;
        Ok(typed)
    }

    /// Bind resident resources by exact canonical value identity, over any
    /// entry count.
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
        require_every_entry_binding(target_entries, &typed)?;
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

    pub(super) fn retained_values(
        &self,
    ) -> Result<BTreeSet<ArtifactValueId>, ArtifactSessionError> {
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

/// Refuse a binding set that leaves an entry point's declared resource unbound.
///
/// Every builder ends here, so one entry point missing a resource is the same
/// rejection whether the caller bound by position, by name, by value, or over a
/// workspace.
fn require_every_entry_binding(
    entries: &[TargetEntryPoint],
    typed: &BindingSet,
) -> Result<(), ArtifactSessionError> {
    for entry in entries {
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
    Ok(())
}
