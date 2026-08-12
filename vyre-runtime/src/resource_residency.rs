//! Backend-neutral immutable-resource, artifact, and mutable-state residency ownership.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, MutexGuard};

use thiserror::Error;
use vyre_driver::backend::{ArtifactInstance, ArtifactMaterializer, BackendError, Resource};

const ZERO_UPLOAD_CHUNK_BYTES: usize = 1024 * 1024;

/// Stable identity for one immutable resource set plus one compiler artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceSetKey {
    /// Content-verified immutable source identity.
    pub source_digest: [u8; 32],
    /// Neutral compiler artifact identity.
    pub artifact_digest: [u8; 32],
}

/// One immutable-resource upload admitted under a verified resource-set identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImmutableResourceUpload<'a> {
    /// Stable immutable resource name.
    pub name: &'a str,
    /// Exact resource bytes to upload.
    pub bytes: &'a [u8],
    /// Trusted BLAKE3 digest of `bytes`.
    pub blake3: [u8; 32],
}

/// One reusable authenticated artifact instance owned by a resident resource set.
#[derive(Clone)]
pub struct ArtifactInstanceBinding {
    name: String,
    instance: Arc<dyn ArtifactInstance>,
    byte_len: u64,
}

impl std::fmt::Debug for ArtifactInstanceBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ArtifactInstanceBinding")
            .field("name", &self.name)
            .field("artifact", &self.instance.artifact())
            .field("payload", &self.instance.payload())
            .field("device", self.instance.device())
            .field("byte_len", &self.byte_len)
            .finish()
    }
}

impl ArtifactInstanceBinding {
    /// Bind a named authenticated artifact instance and its resident byte estimate.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        instance: Arc<dyn ArtifactInstance>,
        byte_len: u64,
    ) -> Self {
        Self {
            name: name.into(),
            instance,
            byte_len,
        }
    }

    /// Stable artifact name inside the resource-set plan.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Reusable materialized artifact instance.
    #[must_use]
    pub fn instance(&self) -> &Arc<dyn ArtifactInstance> {
        &self.instance
    }

    /// Bytes charged to the residency budget for this artifact.
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

/// Complete cold-load request for one resource set and execution plan.
#[derive(Debug)]
pub struct ResourceSetAdmission<'a> {
    /// Immutable source and artifact identity.
    pub key: ResourceSetKey,
    /// Immutable resources to allocate and upload.
    pub immutable_resources: Vec<ImmutableResourceUpload<'a>>,
    /// Authenticated artifact instances retained for every state lease.
    pub artifacts: Vec<ArtifactInstanceBinding>,
}

/// Whether resource-set admission allocated resources or reused an exact resident key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceAdmissionStatus {
    /// Resources and artifacts were admitted for the first time.
    Cold,
    /// The exact source and artifact key was already resident.
    Warm,
}

/// Successful resource-set admission result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceSetLease {
    /// Resident resource-set identity.
    pub key: ResourceSetKey,
    /// Cold or warm admission result.
    pub status: ResourceAdmissionStatus,
}

/// Mutable state allocation owned by one resource set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MutableStateSpec<'a> {
    /// Stable cache or recurrent-state name.
    pub name: &'a str,
    /// Exact device allocation size.
    pub byte_len: usize,
}

/// Monotonic mutable-state identity. Zero is never issued.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StateId(pub u64);

/// Generation-checked access to mutable state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateLease {
    /// Monotonic mutable-state identity.
    pub id: StateId,
    /// Reset generation. Reset invalidates every earlier lease.
    pub generation: u64,
}

/// Resident-resource operations required by the ownership manager.
///
/// Production sessions use [`MaterializerResourceDevice`]. The trait also
/// permits deterministic fault injection in contract tests.
pub trait ResidentResourceDevice: Send + Sync {
    /// Allocate one resident resource.
    fn allocate(&self, byte_len: usize) -> Result<Resource, BackendError>;
    /// Upload immutable resources as one logical transfer.
    fn upload_many(&self, uploads: &[(&Resource, &[u8])]) -> Result<(), BackendError>;
    /// Upload one mutable-state byte range.
    fn upload_at(
        &self,
        resource: &Resource,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), BackendError>;
    /// Release one resident resource.
    fn free(&self, resource: Resource) -> Result<(), BackendError>;
}

/// Residency adapter bound to one artifact materializer device generation.
#[derive(Clone)]
pub struct MaterializerResourceDevice {
    materializer: Arc<dyn ArtifactMaterializer>,
}

impl MaterializerResourceDevice {
    /// Bind residency operations to one authenticated artifact materializer.
    #[must_use]
    pub fn new(materializer: Arc<dyn ArtifactMaterializer>) -> Self {
        Self { materializer }
    }
}

impl ResidentResourceDevice for MaterializerResourceDevice {
    fn allocate(&self, byte_len: usize) -> Result<Resource, BackendError> {
        self.materializer.allocate_resident(byte_len)
    }

    fn upload_many(&self, uploads: &[(&Resource, &[u8])]) -> Result<(), BackendError> {
        for (resource, bytes) in uploads {
            self.materializer.upload_resident(resource, bytes)?;
        }
        Ok(())
    }

    fn upload_at(
        &self,
        resource: &Resource,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), BackendError> {
        self.materializer
            .upload_resident_at(resource, offset, bytes)
    }

    fn free(&self, resource: Resource) -> Result<(), BackendError> {
        self.materializer.free_resident(resource)
    }
}

#[derive(Clone)]
struct ResidentImmutableResource {
    resource: Resource,
    byte_len: u64,
    digest: [u8; 32],
}

struct ResidentArtifact {
    instance: Arc<dyn ArtifactInstance>,
    artifact: [u8; 32],
    byte_len: u64,
}

struct ResidentResourceSet {
    immutable_resources: BTreeMap<String, ResidentImmutableResource>,
    artifacts: BTreeMap<String, ResidentArtifact>,
    accounted_bytes: u64,
    active_states: u64,
}

struct ResidentStateSet {
    resource_set: ResourceSetKey,
    generation: u64,
    states: BTreeMap<String, Resource>,
    state_sizes: BTreeMap<String, usize>,
    accounted_bytes: u64,
}

struct ResidencyState {
    resource_sets: BTreeMap<ResourceSetKey, ResidentResourceSet>,
    states: BTreeMap<StateId, ResidentStateSet>,
    next_state: u64,
    used_bytes: u64,
}

impl Default for ResidencyState {
    fn default() -> Self {
        Self {
            resource_sets: BTreeMap::new(),
            states: BTreeMap::new(),
            next_state: 1,
            used_bytes: 0,
        }
    }
}

/// One ownership boundary for immutable resources, authenticated artifact
/// instances, and mutable mutable state.
pub struct ResourceResidency {
    device: Arc<dyn ResidentResourceDevice>,
    budget_bytes: u64,
    state: Mutex<ResidencyState>,
}

impl std::fmt::Debug for ResourceResidency {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResourceResidency")
            .field("budget_bytes", &self.budget_bytes)
            .finish_non_exhaustive()
    }
}

impl ResourceResidency {
    /// Create a resource residency manager bound to one artifact materializer generation.
    #[must_use]
    pub fn new(materializer: Arc<dyn ArtifactMaterializer>, budget_bytes: u64) -> Self {
        Self::with_device(
            Arc::new(MaterializerResourceDevice::new(materializer)),
            budget_bytes,
        )
    }

    /// Create a residency manager over an explicit resident-resource boundary.
    #[must_use]
    pub fn with_device(device: Arc<dyn ResidentResourceDevice>, budget_bytes: u64) -> Self {
        Self {
            device,
            budget_bytes,
            state: Mutex::new(ResidencyState::default()),
        }
    }

    /// Admit immutable_resources and artifact instances or reuse an exact warm key.
    pub fn admit_resource_set(
        &self,
        request: ResourceSetAdmission<'_>,
    ) -> Result<ResourceSetLease, ResourceResidencyError> {
        validate_key(request.key)?;
        let prepared_immutable_resources =
            validate_immutable_resources(&request.immutable_resources)?;
        let prepared_artifacts = validate_artifacts(&request.artifacts)?;
        let requested_bytes =
            accounted_resource_set_bytes(&prepared_immutable_resources, &prepared_artifacts)?;
        let mut state = self.lock_state()?;

        if let Some(resident) = state.resource_sets.get(&request.key) {
            validate_warm_resource_set(
                resident,
                &prepared_immutable_resources,
                &prepared_artifacts,
            )?;
            return Ok(ResourceSetLease {
                key: request.key,
                status: ResourceAdmissionStatus::Warm,
            });
        }
        ensure_budget(
            state.used_bytes,
            requested_bytes,
            self.budget_bytes,
            "resource-set admission",
        )?;

        let mut allocations = Vec::with_capacity(prepared_immutable_resources.len());
        for immutable_resource in &prepared_immutable_resources {
            let byte_len = usize::try_from(immutable_resource.byte_len).map_err(|_| {
                ResourceResidencyError::ByteLengthOverflow {
                    context: format!("immutable_resource `{}`", immutable_resource.name),
                }
            })?;
            match self.device.allocate(byte_len) {
                Ok(resource) => allocations.push(resource),
                Err(error) => {
                    return Err(self.rollback_error(
                        allocations,
                        "immutable_resource allocation",
                        error,
                    ));
                }
            }
        }
        let uploads = allocations
            .iter()
            .zip(request.immutable_resources.iter())
            .map(|(resource, immutable_resource)| (resource, immutable_resource.bytes))
            .collect::<Vec<_>>();
        if let Err(error) = self.device.upload_many(&uploads) {
            return Err(self.rollback_error(allocations, "immutable_resource batch upload", error));
        }

        let immutable_resources = prepared_immutable_resources
            .into_iter()
            .zip(allocations)
            .map(|(immutable_resource, resource)| {
                (
                    immutable_resource.name,
                    ResidentImmutableResource {
                        resource,
                        byte_len: immutable_resource.byte_len,
                        digest: immutable_resource.digest,
                    },
                )
            })
            .collect();
        let artifacts = request
            .artifacts
            .into_iter()
            .map(|artifact| {
                (
                    artifact.name,
                    ResidentArtifact {
                        artifact: artifact.instance.artifact().0,
                        instance: artifact.instance,
                        byte_len: artifact.byte_len,
                    },
                )
            })
            .collect();
        state.used_bytes = state.used_bytes.checked_add(requested_bytes).ok_or(
            ResourceResidencyError::ByteLengthOverflow {
                context: "committed residency bytes".into(),
            },
        )?;
        state.resource_sets.insert(
            request.key,
            ResidentResourceSet {
                immutable_resources,
                artifacts,
                accounted_bytes: requested_bytes,
                active_states: 0,
            },
        );
        Ok(ResourceSetLease {
            key: request.key,
            status: ResourceAdmissionStatus::Cold,
        })
    }

    /// Begin one independently owned, zero-initialized mutable-state set.
    pub fn start_state(
        &self,
        resource_set: ResourceSetKey,
        specs: &[MutableStateSpec<'_>],
    ) -> Result<StateLease, ResourceResidencyError> {
        let prepared = validate_state_specs(specs)?;
        let requested_bytes = prepared.iter().try_fold(0_u64, |total, (_, bytes)| {
            total
                .checked_add(*bytes as u64)
                .ok_or(ResourceResidencyError::ByteLengthOverflow {
                    context: "state-state byte total".into(),
                })
        })?;
        let mut state = self.lock_state()?;
        if !state.resource_sets.contains_key(&resource_set) {
            return Err(ResourceResidencyError::ResourceSetNotResident { key: resource_set });
        }
        ensure_budget(
            state.used_bytes,
            requested_bytes,
            self.budget_bytes,
            "state admission",
        )?;
        let next_active_states = state
            .resource_sets
            .get(&resource_set)
            .ok_or(ResourceResidencyError::ResourceSetNotResident { key: resource_set })?
            .active_states
            .checked_add(1)
            .ok_or(ResourceResidencyError::StateIdentityOverflow)?;
        let id = StateId(state.next_state);
        state.next_state = state
            .next_state
            .checked_add(1)
            .ok_or(ResourceResidencyError::StateIdentityOverflow)?;

        let mut resources = Vec::with_capacity(prepared.len());
        for (_, byte_len) in &prepared {
            match self.device.allocate(*byte_len) {
                Ok(resource) => {
                    if let Err(error) = self.zero_resource(&resource, *byte_len) {
                        resources.push(resource);
                        return Err(self.rollback_error(resources, "state zeroing", error));
                    }
                    resources.push(resource);
                }
                Err(error) => {
                    return Err(self.rollback_error(resources, "state allocation", error));
                }
            }
        }
        let states = prepared
            .iter()
            .map(|(name, _)| name.clone())
            .zip(resources)
            .collect();
        let state_sizes = prepared.into_iter().collect();
        state.used_bytes = state.used_bytes.checked_add(requested_bytes).ok_or(
            ResourceResidencyError::ByteLengthOverflow {
                context: "committed state bytes".into(),
            },
        )?;
        state
            .resource_sets
            .get_mut(&resource_set)
            .ok_or(ResourceResidencyError::ResourceSetNotResident { key: resource_set })?
            .active_states = next_active_states;
        state.states.insert(
            id,
            ResidentStateSet {
                resource_set,
                generation: 0,
                states,
                state_sizes,
                accounted_bytes: requested_bytes,
            },
        );
        Ok(StateLease { id, generation: 0 })
    }

    /// Clone one current mutable-state resource for dispatch binding.
    pub fn mutable_state(
        &self,
        lease: StateLease,
        name: &str,
    ) -> Result<Resource, ResourceResidencyError> {
        let residency = self.lock_state()?;
        let resident_state = validate_state(&residency, lease)?;
        resident_state.states.get(name).cloned().ok_or_else(|| {
            ResourceResidencyError::StateNotFound {
                state: lease.id,
                name: name.to_string(),
            }
        })
    }

    /// Zero every mutable state and return a new generation lease.
    pub fn reset_state(&self, lease: StateLease) -> Result<StateLease, ResourceResidencyError> {
        let mut residency = self.lock_state()?;
        let resident_state = validate_state(&residency, lease)?;
        let reset_inputs = resident_state
            .states
            .iter()
            .map(|(name, resource)| {
                let byte_len = resident_state.state_sizes[name];
                (resource.clone(), byte_len)
            })
            .collect::<Vec<_>>();
        let next_generation = resident_state
            .generation
            .checked_add(1)
            .ok_or(ResourceResidencyError::StateGenerationOverflow { state: lease.id })?;
        for (resource, byte_len) in &reset_inputs {
            if let Err(error) = self.zero_resource(resource, *byte_len) {
                let removed = residency
                    .states
                    .remove(&lease.id)
                    .ok_or(ResourceResidencyError::StateLeaseNotFound { state: lease.id })?;
                release_state_accounting(&mut residency, &removed)?;
                let resources = removed.states.into_values().collect();
                return Err(self.rollback_error(resources, "state reset", error));
            }
        }
        residency
            .states
            .get_mut(&lease.id)
            .ok_or(ResourceResidencyError::StateLeaseNotFound { state: lease.id })?
            .generation = next_generation;
        Ok(StateLease {
            id: lease.id,
            generation: next_generation,
        })
    }

    /// Cancel a state lease and release all mutable state.
    pub fn cancel_state(&self, lease: StateLease) -> Result<(), ResourceResidencyError> {
        self.release_state(lease, "state cancellation")
    }

    /// Complete a state lease and release all mutable state.
    pub fn finish_state(&self, lease: StateLease) -> Result<(), ResourceResidencyError> {
        self.release_state(lease, "state completion")
    }

    /// Evict one resource_set only after every state lease has released its state.
    pub fn evict_resource_set(&self, key: ResourceSetKey) -> Result<(), ResourceResidencyError> {
        let mut state = self.lock_state()?;
        let resource_set = state
            .resource_sets
            .get(&key)
            .ok_or(ResourceResidencyError::ResourceSetNotResident { key })?;
        if resource_set.active_states != 0 {
            return Err(ResourceResidencyError::ResourceSetInUse {
                key,
                active_states: resource_set.active_states,
            });
        }
        let removed = state
            .resource_sets
            .remove(&key)
            .ok_or(ResourceResidencyError::ResourceSetNotResident { key })?;
        state.used_bytes = state
            .used_bytes
            .checked_sub(removed.accounted_bytes)
            .ok_or(ResourceResidencyError::AccountingUnderflow)?;
        let resources = removed
            .immutable_resources
            .into_values()
            .map(|immutable_resource| immutable_resource.resource)
            .collect::<Vec<_>>();
        self.release_resources(resources, "resource_set eviction")
    }

    /// Clone one resident immutable resource handle.
    pub fn immutable_resource(
        &self,
        key: ResourceSetKey,
        name: &str,
    ) -> Result<Resource, ResourceResidencyError> {
        let state = self.lock_state()?;
        state
            .resource_sets
            .get(&key)
            .ok_or(ResourceResidencyError::ResourceSetNotResident { key })?
            .immutable_resources
            .get(name)
            .map(|immutable_resource| immutable_resource.resource.clone())
            .ok_or_else(|| ResourceResidencyError::ImmutableResourceNotFound {
                key,
                name: name.to_string(),
            })
    }

    /// Clone one reusable authenticated artifact instance.
    pub fn artifact(
        &self,
        key: ResourceSetKey,
        name: &str,
    ) -> Result<Arc<dyn ArtifactInstance>, ResourceResidencyError> {
        let state = self.lock_state()?;
        state
            .resource_sets
            .get(&key)
            .ok_or(ResourceResidencyError::ResourceSetNotResident { key })?
            .artifacts
            .get(name)
            .map(|artifact| Arc::clone(&artifact.instance))
            .ok_or_else(|| ResourceResidencyError::ArtifactNotFound {
                key,
                name: name.to_string(),
            })
    }

    /// Replace a stale device-generation instance with the same neutral artifact.
    ///
    /// # Errors
    ///
    /// Returns an error when the resource_set or named artifact is absent, or when the
    /// replacement names a different neutral artifact.
    pub fn replace_artifact_instance(
        &self,
        key: ResourceSetKey,
        name: &str,
        instance: Arc<dyn ArtifactInstance>,
    ) -> Result<(), ResourceResidencyError> {
        let mut state = self.lock_state()?;
        let artifact = state
            .resource_sets
            .get_mut(&key)
            .ok_or(ResourceResidencyError::ResourceSetNotResident { key })?
            .artifacts
            .get_mut(name)
            .ok_or_else(|| ResourceResidencyError::ArtifactNotFound {
                key,
                name: name.to_string(),
            })?;
        if artifact.artifact != instance.artifact().0 {
            return Err(ResourceResidencyError::WarmResourceSetMismatch);
        }
        artifact.instance = instance;
        Ok(())
    }

    /// Total resource_set, artifact, and state bytes charged to the budget.
    pub fn used_bytes(&self) -> Result<u64, ResourceResidencyError> {
        Ok(self.lock_state()?.used_bytes)
    }

    /// Number of live state leases for one resident resource set.
    pub fn active_states(&self, key: ResourceSetKey) -> Result<u64, ResourceResidencyError> {
        Ok(self
            .lock_state()?
            .resource_sets
            .get(&key)
            .ok_or(ResourceResidencyError::ResourceSetNotResident { key })?
            .active_states)
    }

    fn release_state(
        &self,
        lease: StateLease,
        context: &'static str,
    ) -> Result<(), ResourceResidencyError> {
        let mut state = self.lock_state()?;
        validate_state(&state, lease)?;
        let removed = state
            .states
            .remove(&lease.id)
            .ok_or(ResourceResidencyError::StateLeaseNotFound { state: lease.id })?;
        release_state_accounting(&mut state, &removed)?;
        self.release_resources(removed.states.into_values().collect(), context)
    }

    fn zero_resource(&self, resource: &Resource, byte_len: usize) -> Result<(), BackendError> {
        let zeroes = vec![0_u8; ZERO_UPLOAD_CHUNK_BYTES.min(byte_len)];
        let mut offset = 0_usize;
        while offset < byte_len {
            let chunk = (byte_len - offset).min(zeroes.len());
            self.device.upload_at(resource, offset, &zeroes[..chunk])?;
            offset += chunk;
        }
        Ok(())
    }

    fn rollback_error(
        &self,
        resources: Vec<Resource>,
        operation: &'static str,
        source: BackendError,
    ) -> ResourceResidencyError {
        match self.release_resources(resources, "admission rollback") {
            Ok(()) => ResourceResidencyError::Backend {
                operation,
                detail: source.to_string(),
            },
            Err(cleanup) => ResourceResidencyError::Rollback {
                operation,
                detail: source.to_string(),
                cleanup: cleanup.to_string(),
            },
        }
    }

    fn release_resources(
        &self,
        resources: Vec<Resource>,
        context: &'static str,
    ) -> Result<(), ResourceResidencyError> {
        let mut failures = Vec::new();
        for resource in resources {
            if let Err(error) = self.device.free(resource) {
                failures.push(error.to_string());
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(ResourceResidencyError::Release {
                context,
                details: failures.join("; "),
            })
        }
    }

    fn lock_state(&self) -> Result<MutexGuard<'_, ResidencyState>, ResourceResidencyError> {
        self.state
            .lock()
            .map_err(|_| ResourceResidencyError::LockPoisoned)
    }
}

impl Drop for ResourceResidency {
    fn drop(&mut self) {
        let state = match self.state.get_mut() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut resources = std::mem::take(&mut state.states)
            .into_values()
            .flat_map(|state| state.states.into_values())
            .collect::<Vec<_>>();
        resources.extend(
            std::mem::take(&mut state.resource_sets)
                .into_values()
                .flat_map(|resource_set| resource_set.immutable_resources.into_values())
                .map(|immutable_resource| immutable_resource.resource),
        );
        for resource in resources {
            if let Err(error) = self.device.free(resource) {
                tracing::error!(
                    error = %error,
                    "resource_set residency drop could not release a backend resource"
                );
            }
        }
    }
}

struct PreparedImmutableResource {
    name: String,
    byte_len: u64,
    digest: [u8; 32],
}

struct ValidatedArtifact {
    name: String,
    artifact: [u8; 32],
    byte_len: u64,
}

fn validate_key(key: ResourceSetKey) -> Result<(), ResourceResidencyError> {
    if key.source_digest == [0; 32] || key.artifact_digest == [0; 32] {
        return Err(ResourceResidencyError::ZeroIdentity);
    }
    Ok(())
}

fn validate_immutable_resources(
    immutable_resources: &[ImmutableResourceUpload<'_>],
) -> Result<Vec<PreparedImmutableResource>, ResourceResidencyError> {
    let mut names = BTreeSet::new();
    let mut prepared = Vec::with_capacity(immutable_resources.len());
    for immutable_resource in immutable_resources {
        if immutable_resource.name.is_empty() || !names.insert(immutable_resource.name) {
            return Err(ResourceResidencyError::DuplicateOrEmptyName {
                kind: "immutable_resource",
                name: immutable_resource.name.to_string(),
            });
        }
        let actual = *blake3::hash(immutable_resource.bytes).as_bytes();
        if actual != immutable_resource.blake3 {
            return Err(ResourceResidencyError::ImmutableResourceDigestMismatch {
                name: immutable_resource.name.to_string(),
                actual,
                expected: immutable_resource.blake3,
            });
        }
        prepared.push(PreparedImmutableResource {
            name: immutable_resource.name.to_string(),
            byte_len: immutable_resource.bytes.len() as u64,
            digest: immutable_resource.blake3,
        });
    }
    Ok(prepared)
}

fn validate_artifacts(
    artifacts: &[ArtifactInstanceBinding],
) -> Result<Vec<ValidatedArtifact>, ResourceResidencyError> {
    let mut names = BTreeSet::new();
    let mut validated = Vec::with_capacity(artifacts.len());
    for artifact in artifacts {
        if artifact.name.is_empty() || !names.insert(artifact.name.as_str()) {
            return Err(ResourceResidencyError::DuplicateOrEmptyName {
                kind: "artifact",
                name: artifact.name.clone(),
            });
        }
        validated.push(ValidatedArtifact {
            name: artifact.name.clone(),
            artifact: artifact.instance.artifact().0,
            byte_len: artifact.byte_len,
        });
    }
    Ok(validated)
}

fn validate_state_specs(
    specs: &[MutableStateSpec<'_>],
) -> Result<Vec<(String, usize)>, ResourceResidencyError> {
    let mut names = BTreeSet::new();
    let mut prepared = Vec::with_capacity(specs.len());
    for spec in specs {
        if spec.name.is_empty() || !names.insert(spec.name) {
            return Err(ResourceResidencyError::DuplicateOrEmptyName {
                kind: "mutable state",
                name: spec.name.to_string(),
            });
        }
        if spec.byte_len == 0 {
            return Err(ResourceResidencyError::ZeroStateBytes {
                name: spec.name.to_string(),
            });
        }
        prepared.push((spec.name.to_string(), spec.byte_len));
    }
    Ok(prepared)
}

fn accounted_resource_set_bytes(
    immutable_resources: &[PreparedImmutableResource],
    artifacts: &[ValidatedArtifact],
) -> Result<u64, ResourceResidencyError> {
    immutable_resources
        .iter()
        .map(|immutable_resource| immutable_resource.byte_len)
        .chain(artifacts.iter().map(|artifact| artifact.byte_len))
        .try_fold(0_u64, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or(ResourceResidencyError::ByteLengthOverflow {
                    context: "resource-set and artifact byte total".into(),
                })
        })
}

fn validate_warm_resource_set(
    resident: &ResidentResourceSet,
    immutable_resources: &[PreparedImmutableResource],
    artifacts: &[ValidatedArtifact],
) -> Result<(), ResourceResidencyError> {
    if resident.immutable_resources.len() != immutable_resources.len()
        || resident.artifacts.len() != artifacts.len()
    {
        return Err(ResourceResidencyError::WarmResourceSetMismatch);
    }
    for immutable_resource in immutable_resources {
        let Some(existing) = resident.immutable_resources.get(&immutable_resource.name) else {
            return Err(ResourceResidencyError::WarmResourceSetMismatch);
        };
        if existing.byte_len != immutable_resource.byte_len
            || existing.digest != immutable_resource.digest
        {
            return Err(ResourceResidencyError::WarmResourceSetMismatch);
        }
    }
    for artifact in artifacts {
        let Some(existing) = resident.artifacts.get(&artifact.name) else {
            return Err(ResourceResidencyError::WarmResourceSetMismatch);
        };
        if existing.artifact != artifact.artifact || existing.byte_len != artifact.byte_len {
            return Err(ResourceResidencyError::WarmResourceSetMismatch);
        }
    }
    Ok(())
}

fn ensure_budget(
    used: u64,
    requested: u64,
    budget: u64,
    context: &'static str,
) -> Result<(), ResourceResidencyError> {
    let required =
        used.checked_add(requested)
            .ok_or(ResourceResidencyError::ByteLengthOverflow {
                context: context.into(),
            })?;
    if required > budget {
        return Err(ResourceResidencyError::OutOfMemory {
            context,
            used,
            requested,
            budget,
        });
    }
    Ok(())
}

fn validate_state(
    state: &ResidencyState,
    lease: StateLease,
) -> Result<&ResidentStateSet, ResourceResidencyError> {
    let state = state
        .states
        .get(&lease.id)
        .ok_or(ResourceResidencyError::StateLeaseNotFound { state: lease.id })?;
    if state.generation != lease.generation {
        return Err(ResourceResidencyError::StaleStateLease {
            state: lease.id,
            expected_generation: state.generation,
            actual_generation: lease.generation,
        });
    }
    Ok(state)
}

fn release_state_accounting(
    residency: &mut ResidencyState,
    resident_state: &ResidentStateSet,
) -> Result<(), ResourceResidencyError> {
    residency.used_bytes = residency
        .used_bytes
        .checked_sub(resident_state.accounted_bytes)
        .ok_or(ResourceResidencyError::AccountingUnderflow)?;
    let resource_set = residency
        .resource_sets
        .get_mut(&resident_state.resource_set)
        .ok_or(ResourceResidencyError::ResourceSetNotResident {
            key: resident_state.resource_set,
        })?;
    resource_set.active_states = resource_set
        .active_states
        .checked_sub(1)
        .ok_or(ResourceResidencyError::AccountingUnderflow)?;
    Ok(())
}

/// Resource residency admission or lifecycle failure.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ResourceResidencyError {
    /// A content or artifact identity is all zeroes.
    #[error("resident resource-set identity is zero. Fix: use verified source and compiler artifact digests")]
    ZeroIdentity,
    /// A request repeats or omits a stable name.
    #[error(
        "{kind} name `{name}` is empty or duplicated. Fix: provide one stable name per binding"
    )]
    DuplicateOrEmptyName {
        /// Binding class.
        kind: &'static str,
        /// Invalid name.
        name: String,
    },
    /// Immutable bytes disagree with their trusted digest.
    #[error("immutable_resource `{name}` does not match its trusted BLAKE3 digest")]
    ImmutableResourceDigestMismatch {
        /// Immutable resource name.
        name: String,
        /// Digest of supplied bytes.
        actual: [u8; 32],
        /// Trusted digest.
        expected: [u8; 32],
    },
    /// Byte arithmetic exceeded the supported domain.
    #[error("residency byte arithmetic overflowed for {context}. Fix: shard the admission")]
    ByteLengthOverflow {
        /// Failed arithmetic context.
        context: String,
    },
    /// Admission exceeds the explicit manager budget.
    #[error("{context} needs {requested} additional bytes with {used} already used, over budget {budget}. Fix: evict idle resource sets or reduce state capacity")]
    OutOfMemory {
        /// Admission class.
        context: &'static str,
        /// Currently accounted bytes.
        used: u64,
        /// Newly requested bytes.
        requested: u64,
        /// Hard budget.
        budget: u64,
    },
    /// Backend resource operation failed.
    #[error("{operation} failed: {detail}")]
    Backend {
        /// Failed operation.
        operation: &'static str,
        /// Backend diagnostic.
        detail: String,
    },
    /// Admission failed and one or more rollback frees also failed.
    #[error("{operation} failed: {detail}; rollback also failed: {cleanup}")]
    Rollback {
        /// Failed operation.
        operation: &'static str,
        /// Primary backend diagnostic.
        detail: String,
        /// Cleanup diagnostic.
        cleanup: String,
    },
    /// Releasing one or more unreachable resources failed.
    #[error("{context} could not release all resident resources: {details}")]
    Release {
        /// Lifecycle operation.
        context: &'static str,
        /// Joined backend diagnostics.
        details: String,
    },
    /// A warm request disagrees with its already resident key.
    #[error("warm resource-set request disagrees with resident immutable resource or artifact bindings. Fix: use a new artifact digest for a changed plan")]
    WarmResourceSetMismatch,
    /// Resource-set key is absent.
    #[error(
        "resource set {key:?} is not resident. Fix: admit the resource_set before starting or binding a state"
    )]
    ResourceSetNotResident {
        /// Missing resource-set key.
        key: ResourceSetKey,
    },
    /// Resource set still owns live states.
    #[error("resource set {key:?} has {active_states} active states. Fix: finish or cancel them before eviction")]
    ResourceSetInUse {
        /// Resource-set key.
        key: ResourceSetKey,
        /// Live state count.
        active_states: u64,
    },
    /// Mutable-state identity space is exhausted.
    #[error("state identity space is exhausted. Fix: restart the residency manager rather than reusing stale identities")]
    StateIdentityOverflow,
    /// Reset generation space is exhausted.
    #[error("state {state:?} generation space is exhausted. Fix: finish it and start a new state")]
    StateGenerationOverflow {
        /// Affected state.
        state: StateId,
    },
    /// State lease is absent, cancelled, or already finished.
    #[error("state {state:?} is not active. Fix: discard stale leases and start a new state")]
    StateLeaseNotFound {
        /// Missing state.
        state: StateId,
    },
    /// Lease predates the latest reset.
    #[error("state {state:?} lease generation {actual_generation} is stale; current generation is {expected_generation}")]
    StaleStateLease {
        /// State identity.
        state: StateId,
        /// Current generation.
        expected_generation: u64,
        /// Supplied generation.
        actual_generation: u64,
    },
    /// Mutable state name is absent.
    #[error("state {state:?} has no state `{name}`")]
    StateNotFound {
        /// State identity.
        state: StateId,
        /// Missing state name.
        name: String,
    },
    /// Mutable state cannot have a zero-byte allocation.
    #[error("mutable state `{name}` is zero bytes. Fix: omit unused state or provide its exact positive size")]
    ZeroStateBytes {
        /// Invalid state name.
        name: String,
    },
    /// Immutable resource name is absent.
    #[error("resident resource set {key:?} has no immutable resource `{name}`")]
    ImmutableResourceNotFound {
        /// Resource-set identity.
        key: ResourceSetKey,
        /// Missing immutable resource name.
        name: String,
    },
    /// Named artifact instance is absent.
    #[error("resident resource set {key:?} has no artifact `{name}`")]
    ArtifactNotFound {
        /// Resource-set identity.
        key: ResourceSetKey,
        /// Missing artifact name.
        name: String,
    },
    /// Internal counters would underflow.
    #[error(
        "residency accounting underflowed. Fix: stop using the manager and rebuild residency state"
    )]
    AccountingUnderflow,
    /// Another thread panicked while holding residency state.
    #[error(
        "residency state lock is poisoned. Fix: rebuild the manager before admitting more work"
    )]
    LockPoisoned,
}
