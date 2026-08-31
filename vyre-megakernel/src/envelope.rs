use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{
    allocation::DeviceSlot, failure, frame, frame::Frame, Artifact, ArtifactNodeId,
    ArtifactValueId, CompileError, CompilerFailureKind, Digest,
};

/// Current schema for the artifact envelope that carries neutral data and target payloads.
pub const ARTIFACT_ENVELOPE_SCHEMA_VERSION: u16 = 2;
/// Current schema for one target payload attachment.
pub const TARGET_PAYLOAD_SCHEMA_VERSION: u16 = 4;

/// Versioned identity of target bytes without assigning concrete target semantics.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetPayloadFormat {
    identity: String,
    version: u16,
}

impl TargetPayloadFormat {
    /// Construct a non-empty, non-zero format identity.
    pub fn new(identity: impl Into<String>, version: u16) -> Result<Self, CompileError> {
        let identity = identity.into();
        if identity.is_empty() {
            return Err(failure(
                CompilerFailureKind::MalformedTargetPayload,
                "target_payload.format.identity",
                "target payload format identity is empty",
                "supply the stable format identity owned by the target materializer",
            ));
        }
        if version == 0 {
            return Err(failure(
                CompilerFailureKind::TargetPayloadVersionSkew,
                "target_payload.format.version",
                "target payload format version zero is reserved",
                "supply a positive target format version",
            ));
        }
        Ok(Self { identity, version })
    }

    /// Stable target-payload format identity.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Exact target-payload format version.
    #[must_use]
    pub const fn version(&self) -> u16 {
        self.version
    }
}

/// Immutable capability profile used for pure target compilation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetProfile {
    identity: String,
    generation: u64,
    max_workgroup_size: [u32; 3],
    max_invocations_per_workgroup: u32,
    max_dynamic_shared_bytes: u32,
    subgroup_size: u32,
}

impl TargetProfile {
    /// Construct one target-owned compilation profile.
    pub fn new(
        identity: impl Into<String>,
        generation: u64,
        max_workgroup_size: [u32; 3],
        max_invocations_per_workgroup: u32,
        max_dynamic_shared_bytes: u32,
        subgroup_size: u32,
    ) -> Result<Self, CompileError> {
        let identity = identity.into();
        if identity.is_empty() {
            return Err(failure(
                CompilerFailureKind::MalformedTargetPayload,
                "target_payload.profile.identity",
                "target profile identity is empty",
                "supply the stable profile identity owned by the target compiler",
            ));
        }
        if generation == 0 {
            return Err(failure(
                CompilerFailureKind::TargetPayloadVersionSkew,
                "target_payload.profile.generation",
                "target profile generation zero is reserved",
                "supply a positive compiler/materializer generation",
            ));
        }
        if let Some(axis) = max_workgroup_size.iter().position(|extent| *extent == 0) {
            return Err(failure(
                CompilerFailureKind::MalformedTargetPayload,
                format!("target_payload.profile.max_workgroup_size[{axis}]"),
                "target profile workgroup limit is zero",
                "supply positive workgroup limits for every axis",
            ));
        }
        if max_invocations_per_workgroup == 0 {
            return Err(failure(
                CompilerFailureKind::MalformedTargetPayload,
                "target_payload.profile.max_invocations_per_workgroup",
                "target profile invocation limit is zero",
                "supply a positive invocation limit",
            ));
        }
        if subgroup_size != 0 && !subgroup_size.is_power_of_two() {
            return Err(failure(
                CompilerFailureKind::MalformedTargetPayload,
                "target_payload.profile.subgroup_size",
                "target profile subgroup width is not a power of two",
                "supply zero for no subgroup constraint or a power-of-two subgroup width",
            ));
        }
        Ok(Self {
            identity,
            generation,
            max_workgroup_size,
            max_invocations_per_workgroup,
            max_dynamic_shared_bytes,
            subgroup_size,
        })
    }

    /// Stable target-owned profile identity.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Compiler/materializer contract generation.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Maximum supported workgroup dimensions.
    #[must_use]
    pub const fn max_workgroup_size(&self) -> [u32; 3] {
        self.max_workgroup_size
    }

    /// Maximum supported invocations in one workgroup.
    #[must_use]
    pub const fn max_invocations_per_workgroup(&self) -> u32 {
        self.max_invocations_per_workgroup
    }

    /// Maximum dynamic shared bytes per entry.
    #[must_use]
    pub const fn max_dynamic_shared_bytes(&self) -> u32 {
        self.max_dynamic_shared_bytes
    }

    /// Required subgroup width, or zero when subgroup width is not constrained.
    #[must_use]
    pub const fn subgroup_size(&self) -> u32 {
        self.subgroup_size
    }
}

/// Neutral memory class required by one target resource binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetResourceMemory {
    /// Externally allocated storage.
    Global,
    /// Entry-local shared storage.
    Shared,
    /// Read-only constant storage.
    Constant,
}

/// Access required by one target resource binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetResourceAccess {
    /// Read-only access.
    ReadOnly,
    /// Write-only access.
    WriteOnly,
    /// Read and write access.
    ReadWrite,
}

/// Target binding metadata associated with one canonical neutral resource.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetResourceBinding {
    /// Canonical resource identity from [`Artifact::resources`].
    pub resource: ArtifactValueId,
    /// Target resource group or descriptor set.
    pub group: u32,
    /// Target entry binding slot within `group`.
    pub slot: u32,
    /// Required memory class.
    pub memory: TargetResourceMemory,
    /// Required access mode.
    pub access: TargetResourceAccess,
}

/// Metadata for one entry in an attached target payload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetEntryPoint {
    /// Stable entry name in the target payload.
    pub name: String,
    /// Canonical neutral node implemented by this entry.
    pub node: ArtifactNodeId,
    /// Exact target workgroup dimensions.
    pub workgroup_size: [u32; 3],
    /// Exact target grid dimensions.
    pub grid_size: [u32; 3],
    /// Entry-local dynamic shared byte requirement.
    pub dynamic_shared_bytes: u32,
    /// Bindings associated by identity with canonical neutral resources.
    pub resource_bindings: Vec<TargetResourceBinding>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetPayloadBody {
    schema_version: u16,
    device: DeviceSlot,
    neutral_artifact: Digest,
    format: TargetPayloadFormat,
    profile: TargetProfile,
    entries: Vec<TargetEntryPoint>,
    bytes: Vec<u8>,
}

/// Digest-bound target bytes attached to one exact neutral artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetPayload {
    body: TargetPayloadBody,
    digest: Digest,
}

impl TargetPayload {
    /// Validate and bind target bytes to an exact neutral artifact.
    pub fn new(
        neutral: &Artifact,
        format: TargetPayloadFormat,
        profile: TargetProfile,
        mut entries: Vec<TargetEntryPoint>,
        bytes: Vec<u8>,
    ) -> Result<Self, CompileError> {
        if bytes.is_empty() {
            return Err(failure(
                CompilerFailureKind::MalformedTargetPayload,
                "target_payload.bytes",
                "target payload bytes are empty",
                "attach non-empty bytes emitted for the declared target format",
            ));
        }
        entries.sort_by(|left, right| left.name.cmp(&right.name));
        for entry in &mut entries {
            entry
                .resource_bindings
                .sort_by_key(|binding| (binding.group, binding.slot));
        }
        validate_entries(neutral, &profile, &entries)?;
        let body = TargetPayloadBody {
            schema_version: TARGET_PAYLOAD_SCHEMA_VERSION,
            device: neutral.topology().anchor,
            neutral_artifact: neutral.digest(),
            format,
            profile,
            entries,
            bytes,
        };
        let digest = body_digest(&frame::TARGET_PAYLOAD, &body)?;
        Ok(Self { body, digest })
    }

    /// The same payload bound to another device of the same mesh.
    ///
    /// A mesh placement submits one payload per device. The bytes and entries are
    /// what the target compiler emitted for the format; the device states which
    /// member of the mesh runs them, and it participates in the payload identity
    /// so a payload cannot be submitted to a device it was not bound to.
    ///
    /// # Errors
    ///
    /// Returns when the rebound payload cannot be serialized.
    pub fn for_device(&self, device: DeviceSlot) -> Result<Self, CompileError> {
        let mut body = self.body.clone();
        body.device = device;
        let digest = body_digest(&frame::TARGET_PAYLOAD, &body)?;
        Ok(Self { body, digest })
    }

    /// Mesh device this payload is submitted to.
    #[must_use]
    pub const fn device(&self) -> DeviceSlot {
        self.body.device
    }

    /// Target payload attachment schema.
    #[must_use]
    pub const fn schema_version(&self) -> u16 {
        self.body.schema_version
    }

    /// Exact neutral artifact identity this payload implements.
    #[must_use]
    pub const fn neutral_artifact(&self) -> Digest {
        self.body.neutral_artifact
    }

    /// Versioned payload format identity.
    #[must_use]
    pub const fn format(&self) -> &TargetPayloadFormat {
        &self.body.format
    }

    /// Immutable target capability profile used during compilation.
    #[must_use]
    pub const fn profile(&self) -> &TargetProfile {
        &self.body.profile
    }

    /// Canonical entry metadata.
    #[must_use]
    pub fn entries(&self) -> &[TargetEntryPoint] {
        &self.body.entries
    }

    /// Target-owned opaque bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.body.bytes
    }

    /// Content identity covering association, format, entries, and bytes.
    #[must_use]
    pub const fn digest(&self) -> Digest {
        self.digest
    }

    /// Encode this authenticated target payload.
    pub fn to_bytes(&self) -> Result<Vec<u8>, CompileError> {
        let body = serde_json::to_vec(&self.body).map_err(serialization_failure)?;
        Ok(frame::TARGET_PAYLOAD
            .encode(self.body.schema_version, &body)?
            .bytes)
    }

    /// Decode and authenticate one target payload attachment.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CompileError> {
        let decoded = frame::TARGET_PAYLOAD.decode(bytes)?;
        let body: TargetPayloadBody = serde_json::from_slice(decoded.body).map_err(|error| {
            failure(
                CompilerFailureKind::MalformedTargetPayload,
                "target_payload.body",
                error.to_string(),
                "supply canonical target payload bytes emitted by this crate",
            )
        })?;
        if body.schema_version != decoded.version {
            return Err(failure(
                CompilerFailureKind::TargetPayloadVersionSkew,
                "target_payload.body.schema_version",
                "target payload body schema disagrees with its framing schema",
                "re-materialize the target payload instead of rewriting its framing",
            ));
        }
        let canonical = serde_json::to_vec(&body).map_err(serialization_failure)?;
        if canonical.as_slice() != decoded.body {
            return Err(failure(
                CompilerFailureKind::MalformedTargetPayload,
                "target_payload.body",
                "target payload body is not canonical JSON",
                "use bytes emitted by TargetPayload::to_bytes",
            ));
        }
        Ok(Self {
            body,
            digest: Digest(decoded.digest),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EnvelopeBody {
    schema_version: u16,
    neutral_artifact: Vec<u8>,
    target_payloads: Vec<Vec<u8>>,
}

/// Canonical versioned envelope containing one neutral artifact and target attachments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactEnvelope {
    neutral: Artifact,
    target_payloads: Vec<TargetPayload>,
}

impl ArtifactEnvelope {
    /// Start an envelope around one authenticated neutral artifact.
    #[must_use]
    pub fn new(neutral: Artifact) -> Self {
        Self {
            neutral,
            target_payloads: Vec::new(),
        }
    }

    /// Canonical neutral artifact.
    #[must_use]
    pub const fn neutral(&self) -> &Artifact {
        &self.neutral
    }

    /// Canonically ordered target payload attachments.
    #[must_use]
    pub fn target_payloads(&self) -> &[TargetPayload] {
        &self.target_payloads
    }

    /// Attach a validated target payload to its exact neutral artifact.
    pub fn attach_target_payload(&mut self, payload: TargetPayload) -> Result<(), CompileError> {
        validate_target_payload(&self.neutral, &payload)?;
        if self.target_payloads.iter().any(|existing| {
            existing.format() == payload.format() && existing.device() == payload.device()
        }) {
            return Err(failure(
                CompilerFailureKind::MalformedTargetPayload,
                "envelope.target_payloads",
                format!(
                    "duplicate target payload format {} version {} for device {}",
                    payload.format().identity(),
                    payload.format().version(),
                    payload.device().0
                ),
                "attach at most one payload for each format identity, version, and device",
            ));
        }
        self.target_payloads.push(payload);
        self.target_payloads.sort_by(|left, right| {
            left.format()
                .cmp(right.format())
                .then(left.device().cmp(&right.device()))
        });
        Ok(())
    }

    /// Device every single-device submission of this artifact uses.
    #[must_use]
    fn anchor_device(&self) -> DeviceSlot {
        self.neutral.topology().anchor
    }

    /// Return the canonical index of the anchor-device payload for one exact format.
    pub fn require_target_payload_index(
        &self,
        required: &TargetPayloadFormat,
    ) -> Result<usize, CompileError> {
        self.require_target_payload_index_for_device(required, self.anchor_device())
    }

    /// Return the canonical index of the payload one mesh device submits for an exact format.
    pub fn require_target_payload_index_for_device(
        &self,
        required: &TargetPayloadFormat,
        device: DeviceSlot,
    ) -> Result<usize, CompileError> {
        if let Some(index) = self
            .target_payloads
            .iter()
            .position(|payload| payload.format() == required && payload.device() == device)
        {
            return Ok(index);
        }
        if self
            .target_payloads
            .iter()
            .any(|payload| payload.format() == required)
        {
            return Err(failure(
                CompilerFailureKind::IncompatibleTargetPayload,
                "envelope.target_payloads.device",
                format!(
                    "format {} carries no payload for device {}",
                    required.identity(),
                    device.0
                ),
                "attach one payload for every device the artifact topology places work on",
            ));
        }
        if let Some(payload) = self
            .target_payloads
            .iter()
            .find(|payload| payload.format().identity() == required.identity())
        {
            return Err(failure(
                CompilerFailureKind::TargetPayloadVersionSkew,
                "envelope.target_payloads.format.version",
                format!(
                    "format {} version {} is incompatible; required version {}",
                    required.identity(),
                    payload.format().version(),
                    required.version()
                ),
                "materialize the neutral artifact with the exact required target format version",
            ));
        }
        Err(failure(
            CompilerFailureKind::IncompatibleTargetPayload,
            "envelope.target_payloads.format.identity",
            format!(
                "required target payload format {} is absent",
                required.identity()
            ),
            "attach a compatible payload or materialize one from the neutral artifact",
        ))
    }

    /// Return the anchor-device payload for one exact format identity and version.
    pub fn require_target_payload(
        &self,
        required: &TargetPayloadFormat,
    ) -> Result<&TargetPayload, CompileError> {
        self.require_target_payload_for_device(required, self.anchor_device())
    }

    /// Return the payload one mesh device submits for an exact format identity and version.
    pub fn require_target_payload_for_device(
        &self,
        required: &TargetPayloadFormat,
        device: DeviceSlot,
    ) -> Result<&TargetPayload, CompileError> {
        let index = self.require_target_payload_index_for_device(required, device)?;
        self.target_payloads.get(index).ok_or_else(|| {
            failure(
                CompilerFailureKind::MalformedArtifact,
                "envelope.target_payloads",
                "validated target payload index is outside the canonical attachment set",
                "discard the artifact and regenerate its canonical envelope",
            )
        })
    }

    /// Reject an attachment set that leaves one mesh device without its payload.
    ///
    /// A submission runs every device the topology places work on, so a format
    /// attached for one device and missing for another cannot be submitted at all.
    fn validate_device_coverage(&self) -> Result<(), CompileError> {
        let devices = self.neutral.topology().submission_devices();
        for payload in &self.target_payloads {
            for device in &devices {
                if !self
                    .target_payloads
                    .iter()
                    .any(|other| other.format() == payload.format() && other.device() == *device)
                {
                    return Err(failure(
                        CompilerFailureKind::IncompatibleTargetPayload,
                        "envelope.target_payloads.device",
                        format!(
                            "target payload format {} covers {} of {} topology device(s); device {} is absent",
                            payload.format().identity(),
                            self.target_payloads
                                .iter()
                                .filter(|other| other.format() == payload.format())
                                .count(),
                            devices.len(),
                            device.0
                        ),
                        "attach one payload per topology device for every attached format",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Encode the complete authenticated artifact envelope.
    pub fn to_bytes(&self) -> Result<Vec<u8>, CompileError> {
        let body = EnvelopeBody {
            schema_version: ARTIFACT_ENVELOPE_SCHEMA_VERSION,
            neutral_artifact: self.neutral.to_bytes()?,
            target_payloads: self
                .target_payloads
                .iter()
                .map(TargetPayload::to_bytes)
                .collect::<Result<_, _>>()?,
        };
        let body = serde_json::to_vec(&body).map_err(serialization_failure)?;
        Ok(frame::ENVELOPE
            .encode(ARTIFACT_ENVELOPE_SCHEMA_VERSION, &body)?
            .bytes)
    }

    /// Decode, authenticate, and validate a complete artifact envelope.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CompileError> {
        let decoded = frame::ENVELOPE.decode(bytes)?;
        let body: EnvelopeBody = serde_json::from_slice(decoded.body).map_err(|error| {
            failure(
                CompilerFailureKind::MalformedArtifact,
                "envelope.body",
                error.to_string(),
                "supply canonical envelope bytes emitted by this crate",
            )
        })?;
        if body.schema_version != decoded.version {
            return Err(failure(
                CompilerFailureKind::VersionSkew,
                "envelope.body.schema_version",
                "envelope body schema disagrees with its framing schema",
                "repackage the artifact instead of rewriting its framing",
            ));
        }
        let canonical = serde_json::to_vec(&body).map_err(serialization_failure)?;
        if canonical.as_slice() != decoded.body {
            return Err(failure(
                CompilerFailureKind::MalformedArtifact,
                "envelope.body",
                "envelope body is not canonical JSON",
                "use bytes emitted by ArtifactEnvelope::to_bytes",
            ));
        }
        let neutral = Artifact::from_bytes(&body.neutral_artifact)?;
        let mut envelope = Self::new(neutral);
        for payload_bytes in body.target_payloads {
            envelope.attach_target_payload(TargetPayload::from_bytes(&payload_bytes)?)?;
        }
        envelope.validate_device_coverage()?;
        Ok(envelope)
    }
}

fn validate_target_payload(
    neutral: &Artifact,
    payload: &TargetPayload,
) -> Result<(), CompileError> {
    if payload.schema_version() != TARGET_PAYLOAD_SCHEMA_VERSION {
        return Err(failure(
            CompilerFailureKind::TargetPayloadVersionSkew,
            "target_payload.schema_version",
            format!(
                "target payload schema {} is unsupported; expected {}",
                payload.schema_version(),
                TARGET_PAYLOAD_SCHEMA_VERSION
            ),
            "re-materialize the target payload with this envelope version",
        ));
    }
    if payload.neutral_artifact() != neutral.digest() {
        return Err(failure(
            CompilerFailureKind::TargetPayloadAssociationMismatch,
            "target_payload.neutral_artifact",
            "target payload names a different neutral artifact digest",
            "discard the payload and materialize bytes from this exact neutral artifact",
        ));
    }
    if !neutral
        .topology()
        .submission_devices()
        .contains(&payload.device())
    {
        return Err(failure(
            CompilerFailureKind::IncompatibleTargetPayload,
            "target_payload.device",
            format!(
                "target payload names device {}, which this artifact is not submitted to",
                payload.device().0
            ),
            "bind the payload to one of the devices this artifact is submitted to",
        ));
    }
    validate_entries(neutral, payload.profile(), payload.entries())?;
    let digest = body_digest(&frame::TARGET_PAYLOAD, &payload.body)?;
    if digest != payload.digest() {
        return Err(failure(
            CompilerFailureKind::TargetPayloadDigestMismatch,
            "target_payload.digest",
            "target payload identity does not match its association, metadata, and bytes",
            "discard the corrupted target payload and materialize it again",
        ));
    }
    Ok(())
}

fn validate_entries(
    neutral: &Artifact,
    profile: &TargetProfile,
    entries: &[TargetEntryPoint],
) -> Result<(), CompileError> {
    if entries.is_empty() {
        return Err(failure(
            CompilerFailureKind::MalformedTargetPayload,
            "target_payload.entries",
            "target payload has no entry metadata",
            "associate at least one payload entry with a canonical neutral node",
        ));
    }
    // Entry identity is the canonical node. The symbol name is scoped to the
    // module image that defines it, and every fusion group emits its own image
    // with its own entry symbol, so two groups both naming their entry `main` is
    // correct. Requiring the names to be unique across the payload rejected every
    // artifact with more than one fusion group.
    let mut nodes = BTreeSet::new();
    for (entry_index, entry) in entries.iter().enumerate() {
        let path = format!("target_payload.entries[{entry_index}]");
        if entry.name.is_empty() {
            return Err(failure(
                CompilerFailureKind::MalformedTargetPayload,
                format!("{path}.name"),
                "target entry name is empty",
                "supply the emitted entry symbol name",
            ));
        }
        if !nodes.insert(entry.node) {
            return Err(failure(
                CompilerFailureKind::MalformedTargetPayload,
                format!("{path}.node"),
                format!("duplicate target entry for node {}", entry.node.0),
                "associate each canonical neutral node with exactly one payload entry",
            ));
        }
        if !neutral.nodes().iter().any(|node| node.id == entry.node) {
            return Err(failure(
                CompilerFailureKind::TargetPayloadAssociationMismatch,
                format!("{path}.node"),
                format!("neutral artifact has no node {}", entry.node.0),
                "associate the target entry with a canonical neutral node identity",
            ));
        }
        let recorded = neutral
            .geometry()
            .iter()
            .find(|geometry| geometry.node == entry.node)
            .ok_or_else(|| {
                failure(
                    CompilerFailureKind::TargetPayloadAssociationMismatch,
                    format!("{path}.node"),
                    "target entry node has no canonical neutral geometry record",
                    "compile a complete neutral artifact before attaching target bytes",
                )
            })?;
        // The neutral record is the selected schedule projected onto one launch.
        // A payload that carried its own geometry could differ from it, and the
        // difference is a kernel launched at a shape nothing compiled.
        for (field, emitted, selected) in [
            (
                "workgroup_size",
                entry.workgroup_size,
                recorded.workgroup_size,
            ),
            ("grid_size", entry.grid_size, recorded.grid),
        ] {
            if emitted != selected {
                return Err(failure(
                    CompilerFailureKind::TargetPayloadAssociationMismatch,
                    format!("{path}.{field}"),
                    format!(
                        "target entry states {emitted:?} and the artifact selected {selected:?}"
                    ),
                    "attach the geometry the neutral artifact recorded",
                ));
            }
        }
        if entry.dynamic_shared_bytes != recorded.dynamic_shared_bytes {
            return Err(failure(
                CompilerFailureKind::TargetPayloadAssociationMismatch,
                format!("{path}.dynamic_shared_bytes"),
                format!(
                    "target entry states {} shared bytes and the artifact selected {}",
                    entry.dynamic_shared_bytes, recorded.dynamic_shared_bytes
                ),
                "attach the shared-memory requirement the neutral artifact recorded",
            ));
        }
        let limits = profile.max_workgroup_size();
        if let Some(axis) = entry
            .workgroup_size
            .iter()
            .zip(limits)
            .position(|(extent, limit)| *extent > limit)
        {
            return Err(failure(
                CompilerFailureKind::MalformedTargetPayload,
                format!("{path}.workgroup_size[{axis}]"),
                format!(
                    "target workgroup extent {} exceeds profile limit {}",
                    entry.workgroup_size[axis], limits[axis]
                ),
                "emit geometry admitted by the authenticated target profile",
            ));
        }
        let invocations = entry
            .workgroup_size
            .iter()
            .try_fold(1u32, |product, extent| product.checked_mul(*extent))
            .ok_or_else(|| {
                failure(
                    CompilerFailureKind::MalformedTargetPayload,
                    format!("{path}.workgroup_size"),
                    "target workgroup invocation count overflows u32",
                    "emit bounded workgroup geometry",
                )
            })?;
        if invocations > profile.max_invocations_per_workgroup() {
            return Err(failure(
                CompilerFailureKind::MalformedTargetPayload,
                format!("{path}.workgroup_size"),
                format!(
                    "target workgroup has {invocations} invocations, exceeding profile limit {}",
                    profile.max_invocations_per_workgroup()
                ),
                "emit geometry admitted by the authenticated target profile",
            ));
        }
        if entry.dynamic_shared_bytes > profile.max_dynamic_shared_bytes() {
            return Err(failure(
                CompilerFailureKind::MalformedTargetPayload,
                format!("{path}.dynamic_shared_bytes"),
                format!(
                    "target entry requires {} dynamic shared bytes, exceeding profile limit {}",
                    entry.dynamic_shared_bytes,
                    profile.max_dynamic_shared_bytes()
                ),
                "emit shared-memory requirements admitted by the authenticated target profile",
            ));
        }
        let mut slots = BTreeSet::new();
        for (binding_index, binding) in entry.resource_bindings.iter().enumerate() {
            let binding_path = format!("{path}.resource_bindings[{binding_index}]");
            if !slots.insert((binding.group, binding.slot)) {
                return Err(failure(
                    CompilerFailureKind::MalformedTargetPayload,
                    format!("{binding_path}.slot"),
                    format!(
                        "duplicate target binding group {} slot {}",
                        binding.group, binding.slot
                    ),
                    "associate each target binding group/slot exactly once",
                ));
            }
            if !neutral
                .resources()
                .iter()
                .any(|resource| resource.value == binding.resource)
            {
                return Err(failure(
                    CompilerFailureKind::TargetPayloadAssociationMismatch,
                    format!("{binding_path}.resource"),
                    format!("neutral artifact has no resource {}", binding.resource.0),
                    "bind only canonical resources from the associated neutral artifact",
                ));
            }
        }
    }
    Ok(())
}

fn body_digest<T: Serialize>(frame: &Frame, body: &T) -> Result<Digest, CompileError> {
    let body = serde_json::to_vec(body).map_err(serialization_failure)?;
    Ok(Digest(frame.digest(frame.version, &body)))
}

fn serialization_failure(error: serde_json::Error) -> CompileError {
    failure(
        CompilerFailureKind::MalformedArtifact,
        "envelope.serialization",
        error.to_string(),
        "report this deterministic canonical serialization failure",
    )
}
