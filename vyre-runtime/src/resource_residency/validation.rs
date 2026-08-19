use std::collections::BTreeSet;

use super::error::ResourceResidencyError;
use super::admission::{
    ArtifactInstanceBinding, ImmutableResourceUpload, MutableStateSpec, ResidencyState,
    ResidentResourceSet, ResidentStateSet, ResourceSetKey, StateLease,
};

pub(super) struct PreparedImmutableResource {
    pub(super) name: String,
    pub(super) byte_len: u64,
    pub(super) digest: [u8; 32],
}

pub(super) struct ValidatedArtifact {
    pub(super) name: String,
    pub(super) artifact: [u8; 32],
    pub(super) byte_len: u64,
}

pub(super) fn validate_key(key: ResourceSetKey) -> Result<(), ResourceResidencyError> {
    if key.source_digest == [0; 32] || key.artifact_digest == [0; 32] {
        return Err(ResourceResidencyError::ZeroIdentity);
    }
    Ok(())
}

pub(super) fn validate_immutable_resources(
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

pub(super) fn validate_artifacts(
    artifacts: &[ArtifactInstanceBinding],
) -> Result<Vec<ValidatedArtifact>, ResourceResidencyError> {
    let mut names = BTreeSet::new();
    let mut validated = Vec::with_capacity(artifacts.len());
    for artifact in artifacts {
        if artifact.name().is_empty() || !names.insert(artifact.name()) {
            return Err(ResourceResidencyError::DuplicateOrEmptyName {
                kind: "artifact",
                name: artifact.name().to_string(),
            });
        }
        validated.push(ValidatedArtifact {
            name: artifact.name().to_string(),
            artifact: artifact.instance().artifact().0,
            byte_len: artifact.byte_len(),
        });
    }
    Ok(validated)
}

pub(super) fn validate_state_specs(
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

pub(super) fn accounted_resource_set_bytes(
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

pub(super) fn validate_warm_resource_set(
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

pub(super) fn ensure_budget(
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

pub(super) fn validate_state(
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

pub(super) fn release_state_accounting(
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
