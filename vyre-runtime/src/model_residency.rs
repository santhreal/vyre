//! Backend-neutral model, artifact, and per-sequence residency ownership.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, MutexGuard};

use thiserror::Error;
use vyre_driver::backend::{ArtifactInstance, ArtifactMaterializer, BackendError, Resource};

const ZERO_UPLOAD_CHUNK_BYTES: usize = 1024 * 1024;

/// Stable identity for one checkpoint plus one compiler artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelResidencyKey {
    /// Content-verified checkpoint identity.
    pub checkpoint_digest: [u8; 32],
    /// Neutral compiler artifact identity.
    pub artifact_digest: [u8; 32],
}

/// One immutable weight upload admitted under a verified model identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImmutableWeightUpload<'a> {
    /// Stable manifest tensor name.
    pub name: &'a str,
    /// Exact tensor bytes to upload.
    pub bytes: &'a [u8],
    /// Trusted BLAKE3 digest of `bytes`.
    pub blake3: [u8; 32],
}

/// One reusable authenticated artifact instance owned by a resident model.
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

    /// Stable artifact name inside the model plan.
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

/// Complete cold-load request for one model and execution plan.
#[derive(Debug)]
pub struct ModelAdmission<'a> {
    /// Checkpoint and artifact identity.
    pub key: ModelResidencyKey,
    /// Immutable tensors to allocate and upload.
    pub weights: Vec<ImmutableWeightUpload<'a>>,
    /// Authenticated artifact instances retained for every sequence.
    pub artifacts: Vec<ArtifactInstanceBinding>,
}

/// Whether model admission allocated resources or reused an exact resident key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelAdmissionStatus {
    /// Resources and artifacts were admitted for the first time.
    Cold,
    /// The exact checkpoint and artifact key was already resident.
    Warm,
}

/// Successful model admission result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelLease {
    /// Resident model identity.
    pub key: ModelResidencyKey,
    /// Cold or warm admission result.
    pub status: ModelAdmissionStatus,
}

/// Per-sequence mutable state allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceStateSpec<'a> {
    /// Stable cache or recurrent-state name.
    pub name: &'a str,
    /// Exact device allocation size.
    pub byte_len: usize,
}

/// Monotonic sequence identity. Zero is never issued.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SequenceId(pub u64);

/// Generation-checked access to mutable sequence state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequenceLease {
    /// Monotonic sequence identity.
    pub id: SequenceId,
    /// Reset generation. Reset invalidates every earlier lease.
    pub generation: u64,
}

/// Minimal resident-resource operations required by the ownership manager.
///
/// Production sessions use [`MaterializerResidencyDevice`]. The trait also
/// permits deterministic fault injection in contract tests.
pub trait ResidencyDevice: Send + Sync {
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
pub struct MaterializerResidencyDevice {
    materializer: Arc<dyn ArtifactMaterializer>,
}

impl MaterializerResidencyDevice {
    /// Bind residency operations to one authenticated artifact materializer.
    #[must_use]
    pub fn new(materializer: Arc<dyn ArtifactMaterializer>) -> Self {
        Self { materializer }
    }
}

impl ResidencyDevice for MaterializerResidencyDevice {
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
struct ResidentWeight {
    resource: Resource,
    byte_len: u64,
    digest: [u8; 32],
}

struct ResidentArtifact {
    instance: Arc<dyn ArtifactInstance>,
    artifact: [u8; 32],
    byte_len: u64,
}

struct ResidentModel {
    weights: BTreeMap<String, ResidentWeight>,
    artifacts: BTreeMap<String, ResidentArtifact>,
    accounted_bytes: u64,
    active_sequences: u64,
}

struct ResidentSequence {
    model: ModelResidencyKey,
    generation: u64,
    states: BTreeMap<String, Resource>,
    state_sizes: BTreeMap<String, usize>,
    accounted_bytes: u64,
}

struct ResidencyState {
    models: BTreeMap<ModelResidencyKey, ResidentModel>,
    sequences: BTreeMap<SequenceId, ResidentSequence>,
    next_sequence: u64,
    used_bytes: u64,
}

impl Default for ResidencyState {
    fn default() -> Self {
        Self {
            models: BTreeMap::new(),
            sequences: BTreeMap::new(),
            next_sequence: 1,
            used_bytes: 0,
        }
    }
}

/// One ownership boundary for immutable weights, authenticated artifact
/// instances, and mutable per-sequence state.
pub struct ModelResidency {
    device: Arc<dyn ResidencyDevice>,
    budget_bytes: u64,
    state: Mutex<ResidencyState>,
}

impl std::fmt::Debug for ModelResidency {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModelResidency")
            .field("budget_bytes", &self.budget_bytes)
            .finish_non_exhaustive()
    }
}

impl ModelResidency {
    /// Create a residency manager bound to one artifact materializer generation.
    #[must_use]
    pub fn new(materializer: Arc<dyn ArtifactMaterializer>, budget_bytes: u64) -> Self {
        Self::with_device(
            Arc::new(MaterializerResidencyDevice::new(materializer)),
            budget_bytes,
        )
    }

    /// Create a residency manager over an explicit residency device boundary.
    #[must_use]
    pub fn with_device(device: Arc<dyn ResidencyDevice>, budget_bytes: u64) -> Self {
        Self {
            device,
            budget_bytes,
            state: Mutex::new(ResidencyState::default()),
        }
    }

    /// Admit immutable weights and artifact instances or reuse an exact warm key.
    pub fn admit_model(
        &self,
        request: ModelAdmission<'_>,
    ) -> Result<ModelLease, ModelResidencyError> {
        validate_key(request.key)?;
        let prepared_weights = validate_weights(&request.weights)?;
        let prepared_artifacts = validate_artifacts(&request.artifacts)?;
        let requested_bytes = accounted_model_bytes(&prepared_weights, &prepared_artifacts)?;
        let mut state = self.lock_state()?;

        if let Some(resident) = state.models.get(&request.key) {
            validate_warm_model(resident, &prepared_weights, &prepared_artifacts)?;
            return Ok(ModelLease {
                key: request.key,
                status: ModelAdmissionStatus::Warm,
            });
        }
        ensure_budget(
            state.used_bytes,
            requested_bytes,
            self.budget_bytes,
            "model admission",
        )?;

        let mut allocations = Vec::with_capacity(prepared_weights.len());
        for weight in &prepared_weights {
            let byte_len = usize::try_from(weight.byte_len).map_err(|_| {
                ModelResidencyError::ByteLengthOverflow {
                    context: format!("weight `{}`", weight.name),
                }
            })?;
            match self.device.allocate(byte_len) {
                Ok(resource) => allocations.push(resource),
                Err(error) => {
                    return Err(self.rollback_error(allocations, "weight allocation", error));
                }
            }
        }
        let uploads = allocations
            .iter()
            .zip(request.weights.iter())
            .map(|(resource, weight)| (resource, weight.bytes))
            .collect::<Vec<_>>();
        if let Err(error) = self.device.upload_many(&uploads) {
            return Err(self.rollback_error(allocations, "weight batch upload", error));
        }

        let weights = prepared_weights
            .into_iter()
            .zip(allocations)
            .map(|(weight, resource)| {
                (
                    weight.name,
                    ResidentWeight {
                        resource,
                        byte_len: weight.byte_len,
                        digest: weight.digest,
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
            ModelResidencyError::ByteLengthOverflow {
                context: "committed residency bytes".into(),
            },
        )?;
        state.models.insert(
            request.key,
            ResidentModel {
                weights,
                artifacts,
                accounted_bytes: requested_bytes,
                active_sequences: 0,
            },
        );
        Ok(ModelLease {
            key: request.key,
            status: ModelAdmissionStatus::Cold,
        })
    }

    /// Begin one independently owned, zero-initialized sequence-state set.
    pub fn start_sequence(
        &self,
        model: ModelResidencyKey,
        specs: &[SequenceStateSpec<'_>],
    ) -> Result<SequenceLease, ModelResidencyError> {
        let prepared = validate_state_specs(specs)?;
        let requested_bytes = prepared.iter().try_fold(0_u64, |total, (_, bytes)| {
            total
                .checked_add(*bytes as u64)
                .ok_or(ModelResidencyError::ByteLengthOverflow {
                    context: "sequence-state byte total".into(),
                })
        })?;
        let mut state = self.lock_state()?;
        if !state.models.contains_key(&model) {
            return Err(ModelResidencyError::ModelNotResident { key: model });
        }
        ensure_budget(
            state.used_bytes,
            requested_bytes,
            self.budget_bytes,
            "sequence admission",
        )?;
        let next_active_sequences = state
            .models
            .get(&model)
            .ok_or(ModelResidencyError::ModelNotResident { key: model })?
            .active_sequences
            .checked_add(1)
            .ok_or(ModelResidencyError::SequenceIdentityOverflow)?;
        let id = SequenceId(state.next_sequence);
        state.next_sequence = state
            .next_sequence
            .checked_add(1)
            .ok_or(ModelResidencyError::SequenceIdentityOverflow)?;

        let mut resources = Vec::with_capacity(prepared.len());
        for (_, byte_len) in &prepared {
            match self.device.allocate(*byte_len) {
                Ok(resource) => {
                    if let Err(error) = self.zero_resource(&resource, *byte_len) {
                        resources.push(resource);
                        return Err(self.rollback_error(resources, "sequence zeroing", error));
                    }
                    resources.push(resource);
                }
                Err(error) => {
                    return Err(self.rollback_error(resources, "sequence allocation", error));
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
            ModelResidencyError::ByteLengthOverflow {
                context: "committed sequence bytes".into(),
            },
        )?;
        state
            .models
            .get_mut(&model)
            .ok_or(ModelResidencyError::ModelNotResident { key: model })?
            .active_sequences = next_active_sequences;
        state.sequences.insert(
            id,
            ResidentSequence {
                model,
                generation: 0,
                states,
                state_sizes,
                accounted_bytes: requested_bytes,
            },
        );
        Ok(SequenceLease { id, generation: 0 })
    }

    /// Clone one current mutable-state resource for dispatch binding.
    pub fn sequence_state(
        &self,
        lease: SequenceLease,
        name: &str,
    ) -> Result<Resource, ModelResidencyError> {
        let state = self.lock_state()?;
        let sequence = validate_sequence(&state, lease)?;
        sequence
            .states
            .get(name)
            .cloned()
            .ok_or_else(|| ModelResidencyError::StateNotFound {
                sequence: lease.id,
                name: name.to_string(),
            })
    }

    /// Zero every mutable state and return a new generation lease.
    pub fn reset_sequence(
        &self,
        lease: SequenceLease,
    ) -> Result<SequenceLease, ModelResidencyError> {
        let mut state = self.lock_state()?;
        let sequence = validate_sequence(&state, lease)?;
        let reset_inputs = sequence
            .states
            .iter()
            .map(|(name, resource)| {
                let byte_len = sequence.state_sizes[name];
                (resource.clone(), byte_len)
            })
            .collect::<Vec<_>>();
        let next_generation = sequence
            .generation
            .checked_add(1)
            .ok_or(ModelResidencyError::SequenceGenerationOverflow { sequence: lease.id })?;
        for (resource, byte_len) in &reset_inputs {
            if let Err(error) = self.zero_resource(resource, *byte_len) {
                let removed = state
                    .sequences
                    .remove(&lease.id)
                    .ok_or(ModelResidencyError::SequenceNotFound { sequence: lease.id })?;
                release_sequence_accounting(&mut state, &removed)?;
                let resources = removed.states.into_values().collect();
                return Err(self.rollback_error(resources, "sequence reset", error));
            }
        }
        state
            .sequences
            .get_mut(&lease.id)
            .ok_or(ModelResidencyError::SequenceNotFound { sequence: lease.id })?
            .generation = next_generation;
        Ok(SequenceLease {
            id: lease.id,
            generation: next_generation,
        })
    }

    /// Cancel a sequence and release all mutable state.
    pub fn cancel_sequence(&self, lease: SequenceLease) -> Result<(), ModelResidencyError> {
        self.release_sequence(lease, "sequence cancellation")
    }

    /// Complete a sequence and release all mutable state.
    pub fn finish_sequence(&self, lease: SequenceLease) -> Result<(), ModelResidencyError> {
        self.release_sequence(lease, "sequence completion")
    }

    /// Evict one model only after every sequence has released its state.
    pub fn evict_model(&self, key: ModelResidencyKey) -> Result<(), ModelResidencyError> {
        let mut state = self.lock_state()?;
        let model = state
            .models
            .get(&key)
            .ok_or(ModelResidencyError::ModelNotResident { key })?;
        if model.active_sequences != 0 {
            return Err(ModelResidencyError::ModelInUse {
                key,
                active_sequences: model.active_sequences,
            });
        }
        let removed = state
            .models
            .remove(&key)
            .ok_or(ModelResidencyError::ModelNotResident { key })?;
        state.used_bytes = state
            .used_bytes
            .checked_sub(removed.accounted_bytes)
            .ok_or(ModelResidencyError::AccountingUnderflow)?;
        let resources = removed
            .weights
            .into_values()
            .map(|weight| weight.resource)
            .collect::<Vec<_>>();
        self.release_resources(resources, "model eviction")
    }

    /// Clone one resident immutable weight handle.
    pub fn weight(
        &self,
        key: ModelResidencyKey,
        name: &str,
    ) -> Result<Resource, ModelResidencyError> {
        let state = self.lock_state()?;
        state
            .models
            .get(&key)
            .ok_or(ModelResidencyError::ModelNotResident { key })?
            .weights
            .get(name)
            .map(|weight| weight.resource.clone())
            .ok_or_else(|| ModelResidencyError::WeightNotFound {
                key,
                name: name.to_string(),
            })
    }

    /// Clone one reusable authenticated artifact instance.
    pub fn artifact(
        &self,
        key: ModelResidencyKey,
        name: &str,
    ) -> Result<Arc<dyn ArtifactInstance>, ModelResidencyError> {
        let state = self.lock_state()?;
        state
            .models
            .get(&key)
            .ok_or(ModelResidencyError::ModelNotResident { key })?
            .artifacts
            .get(name)
            .map(|artifact| Arc::clone(&artifact.instance))
            .ok_or_else(|| ModelResidencyError::ArtifactNotFound {
                key,
                name: name.to_string(),
            })
    }

    /// Replace a stale device-generation instance with the same neutral artifact.
    ///
    /// # Errors
    ///
    /// Returns an error when the model or named artifact is absent, or when the
    /// replacement names a different neutral artifact.
    pub fn replace_artifact_instance(
        &self,
        key: ModelResidencyKey,
        name: &str,
        instance: Arc<dyn ArtifactInstance>,
    ) -> Result<(), ModelResidencyError> {
        let mut state = self.lock_state()?;
        let artifact = state
            .models
            .get_mut(&key)
            .ok_or(ModelResidencyError::ModelNotResident { key })?
            .artifacts
            .get_mut(name)
            .ok_or_else(|| ModelResidencyError::ArtifactNotFound {
                key,
                name: name.to_string(),
            })?;
        if artifact.artifact != instance.artifact().0 {
            return Err(ModelResidencyError::WarmModelMismatch);
        }
        artifact.instance = instance;
        Ok(())
    }

    /// Total model, artifact, and sequence bytes charged to the budget.
    pub fn used_bytes(&self) -> Result<u64, ModelResidencyError> {
        Ok(self.lock_state()?.used_bytes)
    }

    /// Number of live sequences for one resident model.
    pub fn active_sequences(&self, key: ModelResidencyKey) -> Result<u64, ModelResidencyError> {
        Ok(self
            .lock_state()?
            .models
            .get(&key)
            .ok_or(ModelResidencyError::ModelNotResident { key })?
            .active_sequences)
    }

    fn release_sequence(
        &self,
        lease: SequenceLease,
        context: &'static str,
    ) -> Result<(), ModelResidencyError> {
        let mut state = self.lock_state()?;
        validate_sequence(&state, lease)?;
        let removed = state
            .sequences
            .remove(&lease.id)
            .ok_or(ModelResidencyError::SequenceNotFound { sequence: lease.id })?;
        release_sequence_accounting(&mut state, &removed)?;
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
    ) -> ModelResidencyError {
        match self.release_resources(resources, "admission rollback") {
            Ok(()) => ModelResidencyError::Backend {
                operation,
                detail: source.to_string(),
            },
            Err(cleanup) => ModelResidencyError::Rollback {
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
    ) -> Result<(), ModelResidencyError> {
        let mut failures = Vec::new();
        for resource in resources {
            if let Err(error) = self.device.free(resource) {
                failures.push(error.to_string());
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(ModelResidencyError::Release {
                context,
                details: failures.join("; "),
            })
        }
    }

    fn lock_state(&self) -> Result<MutexGuard<'_, ResidencyState>, ModelResidencyError> {
        self.state
            .lock()
            .map_err(|_| ModelResidencyError::LockPoisoned)
    }
}

impl Drop for ModelResidency {
    fn drop(&mut self) {
        let state = match self.state.get_mut() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut resources = std::mem::take(&mut state.sequences)
            .into_values()
            .flat_map(|sequence| sequence.states.into_values())
            .collect::<Vec<_>>();
        resources.extend(
            std::mem::take(&mut state.models)
                .into_values()
                .flat_map(|model| model.weights.into_values())
                .map(|weight| weight.resource),
        );
        for resource in resources {
            if let Err(error) = self.device.free(resource) {
                tracing::error!(
                    error = %error,
                    "model residency drop could not release a backend resource"
                );
            }
        }
    }
}

struct PreparedWeight {
    name: String,
    byte_len: u64,
    digest: [u8; 32],
}

struct ValidatedArtifact {
    name: String,
    artifact: [u8; 32],
    byte_len: u64,
}

fn validate_key(key: ModelResidencyKey) -> Result<(), ModelResidencyError> {
    if key.checkpoint_digest == [0; 32] || key.artifact_digest == [0; 32] {
        return Err(ModelResidencyError::ZeroIdentity);
    }
    Ok(())
}

fn validate_weights(
    weights: &[ImmutableWeightUpload<'_>],
) -> Result<Vec<PreparedWeight>, ModelResidencyError> {
    let mut names = BTreeSet::new();
    let mut prepared = Vec::with_capacity(weights.len());
    for weight in weights {
        if weight.name.is_empty() || !names.insert(weight.name) {
            return Err(ModelResidencyError::DuplicateOrEmptyName {
                kind: "weight",
                name: weight.name.to_string(),
            });
        }
        let actual = *blake3::hash(weight.bytes).as_bytes();
        if actual != weight.blake3 {
            return Err(ModelResidencyError::WeightDigestMismatch {
                name: weight.name.to_string(),
                actual,
                expected: weight.blake3,
            });
        }
        prepared.push(PreparedWeight {
            name: weight.name.to_string(),
            byte_len: weight.bytes.len() as u64,
            digest: weight.blake3,
        });
    }
    Ok(prepared)
}

fn validate_artifacts(
    artifacts: &[ArtifactInstanceBinding],
) -> Result<Vec<ValidatedArtifact>, ModelResidencyError> {
    let mut names = BTreeSet::new();
    let mut validated = Vec::with_capacity(artifacts.len());
    for artifact in artifacts {
        if artifact.name.is_empty() || !names.insert(artifact.name.as_str()) {
            return Err(ModelResidencyError::DuplicateOrEmptyName {
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
    specs: &[SequenceStateSpec<'_>],
) -> Result<Vec<(String, usize)>, ModelResidencyError> {
    let mut names = BTreeSet::new();
    let mut prepared = Vec::with_capacity(specs.len());
    for spec in specs {
        if spec.name.is_empty() || !names.insert(spec.name) {
            return Err(ModelResidencyError::DuplicateOrEmptyName {
                kind: "sequence state",
                name: spec.name.to_string(),
            });
        }
        if spec.byte_len == 0 {
            return Err(ModelResidencyError::ZeroStateBytes {
                name: spec.name.to_string(),
            });
        }
        prepared.push((spec.name.to_string(), spec.byte_len));
    }
    Ok(prepared)
}

fn accounted_model_bytes(
    weights: &[PreparedWeight],
    artifacts: &[ValidatedArtifact],
) -> Result<u64, ModelResidencyError> {
    weights
        .iter()
        .map(|weight| weight.byte_len)
        .chain(artifacts.iter().map(|artifact| artifact.byte_len))
        .try_fold(0_u64, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or(ModelResidencyError::ByteLengthOverflow {
                    context: "model and artifact byte total".into(),
                })
        })
}

fn validate_warm_model(
    resident: &ResidentModel,
    weights: &[PreparedWeight],
    artifacts: &[ValidatedArtifact],
) -> Result<(), ModelResidencyError> {
    if resident.weights.len() != weights.len() || resident.artifacts.len() != artifacts.len() {
        return Err(ModelResidencyError::WarmModelMismatch);
    }
    for weight in weights {
        let Some(existing) = resident.weights.get(&weight.name) else {
            return Err(ModelResidencyError::WarmModelMismatch);
        };
        if existing.byte_len != weight.byte_len || existing.digest != weight.digest {
            return Err(ModelResidencyError::WarmModelMismatch);
        }
    }
    for artifact in artifacts {
        let Some(existing) = resident.artifacts.get(&artifact.name) else {
            return Err(ModelResidencyError::WarmModelMismatch);
        };
        if existing.artifact != artifact.artifact || existing.byte_len != artifact.byte_len {
            return Err(ModelResidencyError::WarmModelMismatch);
        }
    }
    Ok(())
}

fn ensure_budget(
    used: u64,
    requested: u64,
    budget: u64,
    context: &'static str,
) -> Result<(), ModelResidencyError> {
    let required = used
        .checked_add(requested)
        .ok_or(ModelResidencyError::ByteLengthOverflow {
            context: context.into(),
        })?;
    if required > budget {
        return Err(ModelResidencyError::OutOfMemory {
            context,
            used,
            requested,
            budget,
        });
    }
    Ok(())
}

fn validate_sequence(
    state: &ResidencyState,
    lease: SequenceLease,
) -> Result<&ResidentSequence, ModelResidencyError> {
    let sequence = state
        .sequences
        .get(&lease.id)
        .ok_or(ModelResidencyError::SequenceNotFound { sequence: lease.id })?;
    if sequence.generation != lease.generation {
        return Err(ModelResidencyError::StaleSequenceLease {
            sequence: lease.id,
            expected_generation: sequence.generation,
            actual_generation: lease.generation,
        });
    }
    Ok(sequence)
}

fn release_sequence_accounting(
    state: &mut ResidencyState,
    sequence: &ResidentSequence,
) -> Result<(), ModelResidencyError> {
    state.used_bytes = state
        .used_bytes
        .checked_sub(sequence.accounted_bytes)
        .ok_or(ModelResidencyError::AccountingUnderflow)?;
    let model =
        state
            .models
            .get_mut(&sequence.model)
            .ok_or(ModelResidencyError::ModelNotResident {
                key: sequence.model,
            })?;
    model.active_sequences = model
        .active_sequences
        .checked_sub(1)
        .ok_or(ModelResidencyError::AccountingUnderflow)?;
    Ok(())
}

/// Model residency admission or lifecycle failure.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ModelResidencyError {
    /// A content or artifact identity is all zeroes.
    #[error("resident model identity is zero. Fix: use verified checkpoint and compiler artifact digests")]
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
    #[error("immutable weight `{name}` does not match its trusted BLAKE3 digest")]
    WeightDigestMismatch {
        /// Weight name.
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
    #[error("{context} needs {requested} additional bytes with {used} already used, over budget {budget}. Fix: evict idle models or reduce sequence capacity")]
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
    #[error("warm model request disagrees with resident weight or artifact bindings. Fix: use a new artifact digest for a changed plan")]
    WarmModelMismatch,
    /// Model key is absent.
    #[error(
        "model {key:?} is not resident. Fix: admit the model before starting or binding a sequence"
    )]
    ModelNotResident {
        /// Missing model key.
        key: ModelResidencyKey,
    },
    /// Model still owns live sequences.
    #[error("model {key:?} has {active_sequences} active sequences. Fix: finish or cancel them before eviction")]
    ModelInUse {
        /// Model key.
        key: ModelResidencyKey,
        /// Live sequence count.
        active_sequences: u64,
    },
    /// Sequence-state identity space is exhausted.
    #[error("sequence identity space is exhausted. Fix: restart the residency manager rather than reusing stale identities")]
    SequenceIdentityOverflow,
    /// Reset generation space is exhausted.
    #[error("sequence {sequence:?} generation space is exhausted. Fix: finish it and start a new sequence")]
    SequenceGenerationOverflow {
        /// Affected sequence.
        sequence: SequenceId,
    },
    /// Sequence is absent, cancelled, or already finished.
    #[error(
        "sequence {sequence:?} is not active. Fix: discard stale leases and start a new sequence"
    )]
    SequenceNotFound {
        /// Missing sequence.
        sequence: SequenceId,
    },
    /// Lease predates the latest reset.
    #[error("sequence {sequence:?} lease generation {actual_generation} is stale; current generation is {expected_generation}")]
    StaleSequenceLease {
        /// Sequence identity.
        sequence: SequenceId,
        /// Current generation.
        expected_generation: u64,
        /// Supplied generation.
        actual_generation: u64,
    },
    /// Sequence state name is absent.
    #[error("sequence {sequence:?} has no state `{name}`")]
    StateNotFound {
        /// Sequence identity.
        sequence: SequenceId,
        /// Missing state name.
        name: String,
    },
    /// Sequence state cannot have a zero-byte allocation.
    #[error("sequence state `{name}` is zero bytes. Fix: omit unused state or provide its exact positive size")]
    ZeroStateBytes {
        /// Invalid state name.
        name: String,
    },
    /// Immutable weight name is absent.
    #[error("resident model {key:?} has no weight `{name}`")]
    WeightNotFound {
        /// Model identity.
        key: ModelResidencyKey,
        /// Missing weight name.
        name: String,
    },
    /// Named artifact instance is absent.
    #[error("resident model {key:?} has no artifact `{name}`")]
    ArtifactNotFound {
        /// Model identity.
        key: ModelResidencyKey,
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
