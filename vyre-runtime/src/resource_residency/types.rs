use std::collections::BTreeMap;
use std::sync::Arc;

use vyre_driver::{ArtifactInstance, Resource};

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
    pub(super) name: String,
    pub(super) instance: Arc<dyn ArtifactInstance>,
    pub(super) byte_len: u64,
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

#[derive(Clone)]
pub(super) struct ResidentImmutableResource {
    pub(super) resource: Resource,
    pub(super) byte_len: u64,
    pub(super) digest: [u8; 32],
}

pub(super) struct ResidentArtifact {
    pub(super) instance: Arc<dyn ArtifactInstance>,
    pub(super) artifact: [u8; 32],
    pub(super) byte_len: u64,
}

pub(super) struct ResidentResourceSet {
    pub(super) immutable_resources: BTreeMap<String, ResidentImmutableResource>,
    pub(super) artifacts: BTreeMap<String, ResidentArtifact>,
    pub(super) accounted_bytes: u64,
    pub(super) active_states: u64,
}

pub(super) struct ResidentStateSet {
    pub(super) resource_set: ResourceSetKey,
    pub(super) generation: u64,
    pub(super) states: BTreeMap<String, Resource>,
    pub(super) state_sizes: BTreeMap<String, usize>,
    pub(super) accounted_bytes: u64,
}

pub(super) struct ResidencyState {
    pub(super) resource_sets: BTreeMap<ResourceSetKey, ResidentResourceSet>,
    pub(super) states: BTreeMap<StateId, ResidentStateSet>,
    pub(super) next_state: u64,
    pub(super) used_bytes: u64,
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
