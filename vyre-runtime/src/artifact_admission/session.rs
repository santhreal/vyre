use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::{Arc, RwLock};

use thiserror::Error;
use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BackendRegistration, BindingSet,
    BoundResource, Completion, DeviceIdentity, Resource, Submission,
};
use vyre_foundation::failure_domain::{
    govern_rwlock_read, govern_rwlock_write, RecoveryClass, TypedRecoveryError,
};
use vyre_foundation::ir::Program;
use vyre_megakernel::{
    AbiAccess, ArtifactEnvelope, ArtifactValueId, CompileError, Digest, ResourceLifetime,
    TargetCompileError, TargetEntryPoint, ValidatedCompileRequest,
};

use super::finalist::{host_input_resources, validate_instance, DeviceFinalists};
use super::ingestion::{
    ResourceDataSource, ResourceIngestionError, ResourceManifest, TypedResourceDataset,
};
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
    /// Typed resource ingestion or schema validation failed.
    #[error(transparent)]
    Ingestion(#[from] ResourceIngestionError),
    /// Runtime lifecycle state rejected the operation.
    #[error("artifact session state is inconsistent: {0}. Fix: discard and rebuild the session")]
    State(String),
    /// The session is not accepting operations because a panic left its
    /// device-bound state half written.
    ///
    /// The device context is fatal here: the state names the materialized
    /// instance the device holds, and a fresh device is the only way back.
    #[error(transparent)]
    Recovery(#[from] TypedRecoveryError),
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

/// The subsystem every poison report over one artifact session names.
const SESSION_OWNER: &str = "an artifact session";

/// The state every poison report over one artifact session names.
const SESSION_STATE: &str = "the materialized artifact and its device instance";

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

    /// Take the materialized artifact for reading, or report that the device
    /// context is fatal.
    ///
    /// WHY: the guarded value names the instance the device holds. A panic
    /// under this lock leaves that record half written, so the session rejects
    /// every later operation and the caller reacquires a device rather than
    /// reading what the panic left.
    fn read_state(
        &self,
    ) -> Result<std::sync::RwLockReadGuard<'_, MaterializedArtifact>, ArtifactSessionError> {
        Ok(govern_rwlock_read(
            &self.state,
            SESSION_OWNER,
            SESSION_STATE,
            RecoveryClass::DeviceContextFatal,
        )?)
    }

    /// Take the materialized artifact for replacement under the same contract
    /// as [`read_state`](Self::read_state).
    fn write_state(
        &self,
    ) -> Result<std::sync::RwLockWriteGuard<'_, MaterializedArtifact>, ArtifactSessionError> {
        Ok(govern_rwlock_write(
            &self.state,
            SESSION_OWNER,
            SESSION_STATE,
            RecoveryClass::DeviceContextFatal,
        )?)
    }

    /// Neutral artifact identity shared by every session and device generation.
    pub fn artifact(&self) -> Result<Digest, ArtifactSessionError> {
        let state = self.read_state()?;
        Ok(state.admitted.neutral().digest())
    }
    /// Exact authenticated target payload identity materialized by this session.
    pub fn payload(&self) -> Result<Digest, ArtifactSessionError> {
        let state = self.read_state()?;
        Ok(state.admitted.target_payload().digest())
    }

    /// Current immutable device generation identity.
    pub fn device(&self) -> Result<DeviceIdentity, ArtifactSessionError> {
        let state = self.read_state()?;
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
        let state = self.read_state()?;
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
        let mut state = self.write_state()?;
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
        let state = self.read_state()?;
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
        let state = self.read_state()?;
        Ok(state.materializer.allocate_resident(byte_len)?)
    }

    /// Upload bytes into one resource owned by this session's materializer.
    pub fn upload_resident(
        &self,
        resource: &Resource,
        bytes: &[u8],
    ) -> Result<(), ArtifactSessionError> {
        let state = self.read_state()?;
        Ok(state.materializer.upload_resident(resource, bytes)?)
    }

    /// Upload bytes at one offset into a resource owned by this session's materializer.
    pub fn upload_resident_at(
        &self,
        resource: &Resource,
        offset_bytes: usize,
        bytes: &[u8],
    ) -> Result<(), ArtifactSessionError> {
        let state = self.read_state()?;
        Ok(state
            .materializer
            .upload_resident_at(resource, offset_bytes, bytes)?)
    }

    /// Release one resource owned by this session's materializer.
    pub fn free_resident(&self, resource: Resource) -> Result<(), ArtifactSessionError> {
        let state = self.read_state()?;
        Ok(state.materializer.free_resident(resource)?)
    }

    /// Ingest a typed resource dataset, validating against the artifact ABI before
    /// performing any allocation, and publishing bindings only when the complete
    /// ABI is satisfied.
    ///
    /// # Errors
    ///
    /// Returns a validation rejection before allocation if any required resource
    /// is missing, unknown, or has a mismatched schema (dtype, element count,
    /// lifetime, access, generation, byte length, or digest). Returns a driver
    /// error if allocation or upload fails, rolling back all partial allocations.
    pub fn ingest(
        &self,
        dataset: &TypedResourceDataset,
    ) -> Result<BindingSet, ArtifactSessionError> {
        self.ingest_with_workspace_opt(None, dataset)
    }

    /// Ingest a typed resource dataset with an allocated artifact workspace.
    ///
    /// Values produced and consumed within the artifact are bound from `workspace`.
    /// Caller dataset providing a resource for a workspace-owned value is rejected
    /// rather than overridden.
    ///
    /// # Errors
    ///
    /// Returns a validation rejection if a workspace-owned value is overridden by
    /// the caller, if any non-workspace required resource is missing, or if any
    /// typed schema mismatches.
    pub fn ingest_with_workspace(
        &self,
        workspace: &ArtifactWorkspace,
        dataset: &TypedResourceDataset,
    ) -> Result<BindingSet, ArtifactSessionError> {
        self.ingest_with_workspace_opt(Some(workspace), dataset)
    }

    /// Ingest from an AOT resource manifest.
    pub fn ingest_manifest(
        &self,
        manifest: &ResourceManifest,
    ) -> Result<BindingSet, ArtifactSessionError> {
        let dataset = manifest.to_dataset()?;
        self.ingest(&dataset)
    }

    /// Ingest from an AOT resource manifest relative to a custom base directory.
    pub fn ingest_manifest_with_base_dir(
        &self,
        manifest: &ResourceManifest,
        base_dir: &Path,
    ) -> Result<BindingSet, ArtifactSessionError> {
        let dataset = manifest.to_dataset_with_base_dir(base_dir)?;
        self.ingest(&dataset)
    }

    /// Transactional implementation of typed resource ingestion.
    fn ingest_with_workspace_opt(
        &self,
        workspace: Option<&ArtifactWorkspace>,
        dataset: &TypedResourceDataset,
    ) -> Result<BindingSet, ArtifactSessionError> {
        let state = self.read_state()?;

        let neutral = state.admitted.neutral();
        let target_payload = state.admitted.target_payload();
        let entries = target_payload.entries();
        let device_identity = state.instance.device();
        let abi = neutral.abi();
        let resources = neutral.resources();

        // -------------------------------------------------------------------------
        // Phase 1: Pre-validation (Zero driver allocations, Zero uploads)
        // -------------------------------------------------------------------------

        // 1. Gather all required resources across all target entries.
        let mut required_by_entry: BTreeMap<ArtifactValueId, Vec<&TargetEntryPoint>> =
            BTreeMap::new();
        for entry in entries {
            for target_binding in &entry.resource_bindings {
                required_by_entry
                    .entry(target_binding.resource)
                    .or_default()
                    .push(entry);
            }
        }

        // 2. Check workspace overlap and missing resources.
        for (value_id, requesting_entries) in &required_by_entry {
            if let Some(ws) = workspace {
                if ws.owns(*value_id) {
                    if dataset.get(*value_id).is_some() {
                        return Err(ResourceIngestionError::WorkspaceOwnedCollision {
                            value: *value_id,
                        }
                        .into());
                    }
                    continue;
                }
            }
            if dataset.get(*value_id).is_none() {
                let entry_name = requesting_entries
                    .first()
                    .map(|e| e.name.clone())
                    .unwrap_or_default();
                return Err(ResourceIngestionError::MissingResource {
                    value: *value_id,
                    entry_name,
                }
                .into());
            }
        }

        // 3. Check all supplied resources in dataset against artifact ABI & schema.
        let mut prepared_payloads: BTreeMap<ArtifactValueId, Option<Vec<u8>>> = BTreeMap::new();

        for (val_id, typed_res) in dataset.iter() {
            let resource_record = resources
                .iter()
                .find(|r| r.value == *val_id)
                .ok_or(ResourceIngestionError::UnknownValue { value: *val_id })?;

            let abi_record = abi.resources.iter().find(|r| r.value == *val_id);

            if let Some(dtype) = &typed_res.dtype {
                if let Some(abi_rec) = abi_record {
                    if &abi_rec.dtype != dtype {
                        return Err(ResourceIngestionError::DtypeMismatch {
                            value: *val_id,
                            expected: abi_rec.dtype.clone(),
                            actual: dtype.clone(),
                        }
                        .into());
                    }
                }
            }

            if let Some(elem_count) = typed_res.element_count {
                if resource_record.element_count != elem_count {
                    return Err(ResourceIngestionError::ElementCountMismatch {
                        value: *val_id,
                        expected: resource_record.element_count,
                        actual: elem_count,
                    }
                    .into());
                }
            }

            if let Some(lifetime) = typed_res.lifetime {
                if resource_record.lifetime != lifetime {
                    return Err(ResourceIngestionError::LifetimeMismatch {
                        value: *val_id,
                        expected: resource_record.lifetime,
                        actual: lifetime,
                    }
                    .into());
                }
            }

            if let Some(access) = typed_res.access {
                if let Some(abi_rec) = abi_record {
                    if abi_rec.access != access {
                        return Err(ResourceIngestionError::AccessMismatch {
                            value: *val_id,
                            expected: abi_rec.access,
                            actual: access,
                        }
                        .into());
                    }
                }
            }

            if let Some(gen) = typed_res.generation {
                if device_identity.generation != gen {
                    return Err(ResourceIngestionError::GenerationMismatch {
                        value: *val_id,
                        expected: device_identity.generation,
                        actual: gen,
                    }
                    .into());
                }
            }

            let bytes_opt = typed_res.source.fetch_bytes()?;
            if let Some(bytes) = &bytes_opt {
                if resource_record.byte_count > 0
                    && bytes.len() as u64 != resource_record.byte_count
                {
                    return Err(ResourceIngestionError::ByteCountMismatch {
                        value: *val_id,
                        expected: resource_record.byte_count,
                        actual: bytes.len() as u64,
                    }
                    .into());
                }
                if let Some(expected_identity) = typed_res.identity {
                    let computed = Digest(*blake3::hash(bytes).as_bytes());
                    if computed != expected_identity {
                        return Err(ResourceIngestionError::IdentityMismatch {
                            value: *val_id,
                            expected: expected_identity,
                            actual: computed,
                        }
                        .into());
                    }
                }
            }

            prepared_payloads.insert(*val_id, bytes_opt);
        }

        // -------------------------------------------------------------------------
        // Phase 2: Transactional Allocation & Upload with Rollback Guard
        // -------------------------------------------------------------------------
        struct RollbackGuard<'a> {
            materializer: &'a dyn ArtifactMaterializer,
            allocated: Vec<Resource>,
            committed: bool,
        }

        impl<'a> Drop for RollbackGuard<'a> {
            fn drop(&mut self) {
                if !self.committed {
                    for resource in self.allocated.drain(..) {
                        let _ = self.materializer.free_resident(resource);
                    }
                }
            }
        }

        let mut guard = RollbackGuard {
            materializer: state.materializer.as_ref(),
            allocated: Vec::new(),
            committed: false,
        };

        let mut resolved_resources: BTreeMap<ArtifactValueId, Resource> = BTreeMap::new();

        for (val_id, typed_res) in dataset.iter() {
            match &typed_res.source {
                ResourceDataSource::Resident(existing_res) => {
                    resolved_resources.insert(*val_id, existing_res.clone());
                }
                _ => {
                    let bytes = prepared_payloads
                        .get(val_id)
                        .and_then(|b| b.as_ref())
                        .expect("Fix: prepare a payload in phase 1 for every non-resident dataset entry before resolving resources");

                    let resource =
                        state
                            .materializer
                            .allocate_resident(bytes.len())
                            .map_err(|err| ResourceIngestionError::AllocationFailed {
                                value: *val_id,
                                error: err.to_string(),
                            })?;

                    guard.allocated.push(resource.clone());

                    state
                        .materializer
                        .upload_resident(&resource, bytes)
                        .map_err(|err| ResourceIngestionError::UploadFailed {
                            value: *val_id,
                            error: err.to_string(),
                        })?;

                    resolved_resources.insert(*val_id, resource);
                }
            }
        }

        // -------------------------------------------------------------------------
        // Phase 3: BindingSet Construction and Publication
        // -------------------------------------------------------------------------
        let mut typed = BindingSet::new(neutral.digest());

        if let Some(ws) = workspace {
            for (value, resource) in ws.bindings() {
                typed.insert(*value, BoundResource::Resident(resource.clone()));
            }
        }

        for (val_id, resource) in resolved_resources {
            typed.insert(val_id, BoundResource::Resident(resource));
        }

        require_every_entry_binding(entries, &typed)?;

        guard.committed = true;
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
        let state = self.read_state()?;
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
        let state = self.read_state()?;
        Ok(workspace.free(state.materializer.as_ref())?)
    }

    /// Bind host inputs in canonical ABI slot order.
    pub fn host_bindings(&self, inputs: &[&[u8]]) -> Result<BindingSet, ArtifactSessionError> {
        let state = self.read_state()?;
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
        let state = self.read_state()?;
        let neutral = state.admitted.neutral();
        let lifetimes = neutral
            .resources()
            .iter()
            .map(|resource| (resource.value, resource.lifetime))
            .collect::<BTreeMap<_, _>>();
        let mut resources = neutral
            .abi()
            .resources
            .iter()
            .filter(|resource| {
                matches!(resource.access, AbiAccess::ReadWrite | AbiAccess::WriteOnly)
                    && matches!(
                        lifetimes.get(&resource.value),
                        Some(ResourceLifetime::Output | ResourceLifetime::Retained)
                    )
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
        let state = self.read_state()?;
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
        let state = self.read_state()?;
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

impl crate::StateOwnerRecovery for ArtifactSession {
    fn failure_domain(&self) -> crate::FailureDomain {
        crate::FailureDomain::DeviceContext
    }

    fn recovery_class(&self) -> crate::RecoveryClass {
        crate::RecoveryClass::DeviceContextFatal
    }
}
