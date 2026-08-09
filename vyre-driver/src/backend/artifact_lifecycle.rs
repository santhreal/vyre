use std::collections::BTreeMap;

use vyre_megakernel::{Artifact, ArtifactValueId, Digest, TargetPayload, TargetPayloadFormat};

use super::BackendError;

/// Immutable identity of one acquired execution device generation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeviceIdentity {
    /// Stable registered backend identifier.
    pub backend: &'static str,
    /// Backend-local physical or logical device identifier.
    pub device: String,
    /// Monotonic generation that changes after device loss or reacquisition.
    pub generation: u64,
}

/// Acquired device identity, target compatibility, and health.
pub trait Device: Send + Sync {
    /// Immutable identity for this acquired generation.
    fn identity(&self) -> &DeviceIdentity;
    /// Exact target payload representation admitted by this device.
    fn target_format(&self) -> &TargetPayloadFormat;
    /// Whether new materialization and submission are currently allowed.
    fn is_healthy(&self) -> bool;
}

/// Host or resident bytes bound to one canonical artifact value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoundResource {
    /// Caller-owned bytes uploaded for this submission.
    Host(Vec<u8>),
    /// Backend-resident resource handle.
    Resident(super::Resource),
}

/// Complete typed bindings for one immutable artifact instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BindingSet {
    artifact: Digest,
    resources: BTreeMap<ArtifactValueId, BoundResource>,
}

impl BindingSet {
    /// Construct an empty binding set tied to one artifact identity.
    #[must_use]
    pub const fn new(artifact: Digest) -> Self {
        Self {
            artifact,
            resources: BTreeMap::new(),
        }
    }

    /// Artifact identity these bindings are valid for.
    #[must_use]
    pub const fn artifact(&self) -> Digest {
        self.artifact
    }

    /// Bind or replace one canonical value.
    pub fn insert(&mut self, value: ArtifactValueId, resource: BoundResource) {
        self.resources.insert(value, resource);
    }

    /// Exact canonical value bindings.
    #[must_use]
    pub const fn resources(&self) -> &BTreeMap<ArtifactValueId, BoundResource> {
        &self.resources
    }
}

/// Completed typed submission result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Completion {
    /// Artifact identity executed by this submission.
    pub artifact: Digest,
    /// Canonical output values keyed by artifact ABI identity.
    pub outputs: BTreeMap<ArtifactValueId, Vec<u8>>,
    /// Backend-measured device duration when available.
    pub device_ns: Option<u64>,
}

/// One in-flight submission against an immutable artifact instance.
pub trait Submission: Send + Sync {
    /// Non-blocking completion probe.
    fn is_ready(&self) -> bool;
    /// Wait for completion and typed readback.
    fn wait(self: Box<Self>) -> Result<Completion, BackendError>;
}

/// Device-native immutable executable and resource layout.
pub trait ArtifactInstance: Send + Sync {
    /// Neutral artifact identity implemented by this instance.
    fn artifact(&self) -> Digest;
    /// Exact payload identity materialized into this instance.
    fn payload(&self) -> Digest;
    /// Device generation that owns every native handle.
    fn device(&self) -> &DeviceIdentity;
    /// Validate bindings and submit one invocation.
    fn submit(&self, bindings: BindingSet) -> Result<Box<dyn Submission>, BackendError>;
}

/// Device-specific admission and native-handle construction.
pub trait ArtifactMaterializer: Send + Sync {
    /// Acquired target device.
    fn device(&self) -> &dyn Device;
    /// Materialize authenticated immutable target bytes.
    fn materialize(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<Box<dyn ArtifactInstance>, BackendError>;
}
