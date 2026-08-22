//! Backend-neutral immutable-resource, artifact, and mutable-state residency ownership.

mod admission;
mod device;
mod error;
mod validation;

use std::sync::{Arc, Mutex, MutexGuard};

pub use admission::{
    ArtifactInstanceBinding, ImmutableResourceUpload, MutableStateSpec, ResourceAdmissionStatus,
    ResourceSetAdmission, ResourceSetKey, ResourceSetLease, StateId, StateLease,
};
use admission::{
    ResidencyState, ResidentArtifact, ResidentImmutableResource, ResidentResourceSet,
    ResidentStateSet,
};
pub use device::{MaterializerResourceDevice, ResidentResourceDevice};
pub use error::ResourceResidencyError;
use validation::{
    accounted_resource_set_bytes, ensure_budget, release_state_accounting, validate_artifacts,
    validate_immutable_resources, validate_key, validate_state, validate_state_specs,
    validate_warm_resource_set,
};
use vyre_driver::{ArtifactInstance, ArtifactMaterializer, BackendError, Resource};

const ZERO_UPLOAD_CHUNK_BYTES: usize = 1024 * 1024;

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
                    artifact.name().to_string(),
                    ResidentArtifact {
                        artifact: artifact.instance().artifact().0,
                        instance: artifact.instance().clone(),
                        byte_len: artifact.byte_len(),
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
