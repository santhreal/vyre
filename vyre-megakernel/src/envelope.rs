use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::{
    failure, Artifact, ArtifactNodeId, ArtifactValueId, CompileError, CompilerFailureKind, Digest,
};

/// Current schema for the artifact envelope that carries neutral data and target payloads.
pub const ARTIFACT_ENVELOPE_SCHEMA_VERSION: u16 = 2;
/// Current schema for one target payload attachment.
pub const TARGET_PAYLOAD_SCHEMA_VERSION: u16 = 2;

const ENVELOPE_MAGIC: &[u8; 4] = b"VME0";
const TARGET_PAYLOAD_MAGIC: &[u8; 4] = b"VTP0";
const FRAME_HEADER_BYTES: usize = 10;
const DIGEST_BYTES: usize = 32;
const ENVELOPE_DIGEST_DOMAIN: &[u8] = b"vyre-megakernel-envelope-v2\0";
const TARGET_PAYLOAD_DIGEST_DOMAIN: &[u8] = b"vyre-megakernel-target-payload-v2\0";

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
    /// Target entry binding slot.
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
    /// Target grid dimensions. Workgroup dimensions remain in the neutral geometry record.
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
    neutral_artifact: Digest,
    format: TargetPayloadFormat,
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
            entry.resource_bindings.sort_by_key(|binding| binding.slot);
        }
        validate_entries(neutral, &entries)?;
        let body = TargetPayloadBody {
            schema_version: TARGET_PAYLOAD_SCHEMA_VERSION,
            neutral_artifact: neutral.digest(),
            format,
            entries,
            bytes,
        };
        let digest = body_digest(TARGET_PAYLOAD_DIGEST_DOMAIN, &body)?;
        Ok(Self { body, digest })
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
        encode_frame(
            TARGET_PAYLOAD_MAGIC,
            self.body.schema_version,
            TARGET_PAYLOAD_DIGEST_DOMAIN,
            &self.body,
        )
    }

    /// Decode and authenticate one target payload attachment.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CompileError> {
        let (version, body, encoded_digest) = decode_frame(
            bytes,
            TARGET_PAYLOAD_MAGIC,
            TARGET_PAYLOAD_SCHEMA_VERSION,
            TARGET_PAYLOAD_DIGEST_DOMAIN,
            "target_payload",
            CompilerFailureKind::TargetPayloadVersionSkew,
            CompilerFailureKind::TargetPayloadDigestMismatch,
        )?;
        let body: TargetPayloadBody = serde_json::from_slice(body).map_err(|error| {
            failure(
                CompilerFailureKind::MalformedTargetPayload,
                "target_payload.body",
                error.to_string(),
                "supply canonical target payload bytes emitted by this crate",
            )
        })?;
        if body.schema_version != version {
            return Err(failure(
                CompilerFailureKind::TargetPayloadVersionSkew,
                "target_payload.body.schema_version",
                "target payload body schema disagrees with its framing schema",
                "re-materialize the target payload instead of rewriting its framing",
            ));
        }
        let canonical = serde_json::to_vec(&body).map_err(serialization_failure)?;
        if canonical.as_slice() != &bytes[FRAME_HEADER_BYTES..FRAME_HEADER_BYTES + canonical.len()]
        {
            return Err(failure(
                CompilerFailureKind::MalformedTargetPayload,
                "target_payload.body",
                "target payload body is not canonical JSON",
                "use bytes emitted by TargetPayload::to_bytes",
            ));
        }
        Ok(Self {
            body,
            digest: Digest(encoded_digest),
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
        if self
            .target_payloads
            .iter()
            .any(|existing| existing.format() == payload.format())
        {
            return Err(failure(
                CompilerFailureKind::MalformedTargetPayload,
                "envelope.target_payloads",
                format!(
                    "duplicate target payload format {} version {}",
                    payload.format().identity(),
                    payload.format().version()
                ),
                "attach at most one payload for each exact format identity and version",
            ));
        }
        self.target_payloads.push(payload);
        self.target_payloads
            .sort_by(|left, right| left.format().cmp(right.format()));
        Ok(())
    }

    /// Return the canonical index of the payload compatible with one exact format.
    pub fn require_target_payload_index(
        &self,
        required: &TargetPayloadFormat,
    ) -> Result<usize, CompileError> {
        if let Some(index) = self
            .target_payloads
            .iter()
            .position(|payload| payload.format() == required)
        {
            return Ok(index);
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

    /// Return the payload compatible with one exact format identity and version.
    pub fn require_target_payload(
        &self,
        required: &TargetPayloadFormat,
    ) -> Result<&TargetPayload, CompileError> {
        let index = self.require_target_payload_index(required)?;
        self.target_payloads.get(index).ok_or_else(|| {
            failure(
                CompilerFailureKind::MalformedArtifact,
                "envelope.target_payloads",
                "validated target payload index is outside the canonical attachment set",
                "discard the artifact and regenerate its canonical envelope",
            )
        })
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
        encode_frame(
            ENVELOPE_MAGIC,
            body.schema_version,
            ENVELOPE_DIGEST_DOMAIN,
            &body,
        )
    }

    /// Decode, authenticate, and validate a complete artifact envelope.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CompileError> {
        let (version, body, _) = decode_frame(
            bytes,
            ENVELOPE_MAGIC,
            ARTIFACT_ENVELOPE_SCHEMA_VERSION,
            ENVELOPE_DIGEST_DOMAIN,
            "envelope",
            CompilerFailureKind::VersionSkew,
            CompilerFailureKind::DigestMismatch,
        )?;
        let body: EnvelopeBody = serde_json::from_slice(body).map_err(|error| {
            failure(
                CompilerFailureKind::MalformedArtifact,
                "envelope.body",
                error.to_string(),
                "supply canonical envelope bytes emitted by this crate",
            )
        })?;
        if body.schema_version != version {
            return Err(failure(
                CompilerFailureKind::VersionSkew,
                "envelope.body.schema_version",
                "envelope body schema disagrees with its framing schema",
                "repackage the artifact instead of rewriting its framing",
            ));
        }
        let canonical = serde_json::to_vec(&body).map_err(serialization_failure)?;
        if canonical.as_slice() != &bytes[FRAME_HEADER_BYTES..FRAME_HEADER_BYTES + canonical.len()]
        {
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
    validate_entries(neutral, payload.entries())?;
    let digest = body_digest(TARGET_PAYLOAD_DIGEST_DOMAIN, &payload.body)?;
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

fn validate_entries(neutral: &Artifact, entries: &[TargetEntryPoint]) -> Result<(), CompileError> {
    if entries.is_empty() {
        return Err(failure(
            CompilerFailureKind::MalformedTargetPayload,
            "target_payload.entries",
            "target payload has no entry metadata",
            "associate at least one payload entry with a canonical neutral node",
        ));
    }
    let mut names = BTreeSet::new();
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
        if !names.insert(entry.name.as_str()) {
            return Err(failure(
                CompilerFailureKind::MalformedTargetPayload,
                format!("{path}.name"),
                format!("duplicate target entry name {}", entry.name),
                "supply each target entry name exactly once",
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
        if !neutral
            .geometry()
            .iter()
            .any(|geometry| geometry.node == entry.node)
        {
            return Err(failure(
                CompilerFailureKind::TargetPayloadAssociationMismatch,
                format!("{path}.node"),
                "target entry node has no canonical neutral geometry record",
                "compile a complete neutral artifact before attaching target bytes",
            ));
        }
        if let Some(axis) = entry.grid_size.iter().position(|extent| *extent == 0) {
            return Err(failure(
                CompilerFailureKind::MalformedTargetPayload,
                format!("{path}.grid_size[{axis}]"),
                "target entry grid extent is zero",
                "materialize explicit positive target grid dimensions",
            ));
        }
        let mut slots = BTreeSet::new();
        let mut resources = BTreeSet::new();
        for (binding_index, binding) in entry.resource_bindings.iter().enumerate() {
            let binding_path = format!("{path}.resource_bindings[{binding_index}]");
            if !slots.insert(binding.slot) {
                return Err(failure(
                    CompilerFailureKind::MalformedTargetPayload,
                    format!("{binding_path}.slot"),
                    format!("duplicate target binding slot {}", binding.slot),
                    "associate each target binding slot exactly once",
                ));
            }
            if !resources.insert(binding.resource) {
                return Err(failure(
                    CompilerFailureKind::MalformedTargetPayload,
                    format!("{binding_path}.resource"),
                    format!(
                        "canonical resource {} is bound more than once",
                        binding.resource.0
                    ),
                    "associate each canonical resource with at most one entry binding",
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

fn encode_frame<T: Serialize>(
    magic: &[u8; 4],
    version: u16,
    domain: &[u8],
    body: &T,
) -> Result<Vec<u8>, CompileError> {
    let body = serde_json::to_vec(body).map_err(serialization_failure)?;
    let body_len = u32::try_from(body.len()).map_err(|_| {
        failure(
            CompilerFailureKind::MalformedArtifact,
            "envelope.body",
            "canonical body exceeds the u32 framing limit",
            "reduce or detach target payload bytes",
        )
    })?;
    let digest = digest_bytes(domain, version, &body);
    let mut bytes = Vec::with_capacity(FRAME_HEADER_BYTES + body.len() + DIGEST_BYTES);
    bytes.extend_from_slice(magic);
    bytes.extend_from_slice(&version.to_le_bytes());
    bytes.extend_from_slice(&body_len.to_le_bytes());
    bytes.extend_from_slice(&body);
    bytes.extend_from_slice(&digest);
    Ok(bytes)
}

fn decode_frame<'a>(
    bytes: &'a [u8],
    magic: &[u8; 4],
    expected_version: u16,
    domain: &[u8],
    path: &str,
    version_code: CompilerFailureKind,
    digest_code: CompilerFailureKind,
) -> Result<(u16, &'a [u8], [u8; 32]), CompileError> {
    if bytes.len() < FRAME_HEADER_BYTES + DIGEST_BYTES {
        return Err(failure(
            CompilerFailureKind::MalformedArtifact,
            format!("{path}.header"),
            "framed bytes are shorter than the fixed header and digest",
            "supply one complete canonical frame",
        ));
    }
    if &bytes[..4] != magic {
        return Err(failure(
            CompilerFailureKind::MalformedArtifact,
            format!("{path}.magic"),
            "framing magic is invalid",
            "supply bytes emitted for the expected artifact layer",
        ));
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != expected_version {
        return Err(failure(
            version_code,
            format!("{path}.schema_version"),
            format!("schema {version} is unsupported; expected {expected_version}"),
            "recompile or re-materialize with a compatible schema version",
        ));
    }
    let body_len = u32::from_le_bytes(bytes[6..10].try_into().expect("fixed frame slice")) as usize;
    let expected_len = FRAME_HEADER_BYTES
        .checked_add(body_len)
        .and_then(|len| len.checked_add(DIGEST_BYTES))
        .ok_or_else(|| {
            failure(
                CompilerFailureKind::MalformedArtifact,
                format!("{path}.body_length"),
                "framed body length overflowed addressable memory",
                "supply bounded canonical artifact bytes",
            )
        })?;
    if bytes.len() != expected_len {
        return Err(failure(
            CompilerFailureKind::MalformedArtifact,
            format!("{path}.body_length"),
            format!(
                "framing declares {expected_len} bytes but received {}",
                bytes.len()
            ),
            "supply exactly one complete canonical frame",
        ));
    }
    let body = &bytes[FRAME_HEADER_BYTES..FRAME_HEADER_BYTES + body_len];
    let expected_digest = digest_bytes(domain, version, body);
    let encoded_digest: [u8; 32] = bytes[FRAME_HEADER_BYTES + body_len..]
        .try_into()
        .expect("validated digest length");
    if expected_digest != encoded_digest {
        return Err(failure(
            digest_code,
            format!("{path}.digest"),
            "framed body does not match its content identity",
            "discard the corrupted bytes and regenerate them",
        ));
    }
    Ok((version, body, encoded_digest))
}

fn body_digest<T: Serialize>(domain: &[u8], body: &T) -> Result<Digest, CompileError> {
    let body = serde_json::to_vec(body).map_err(serialization_failure)?;
    Ok(Digest(digest_bytes(
        domain,
        TARGET_PAYLOAD_SCHEMA_VERSION,
        &body,
    )))
}

fn digest_bytes(domain: &[u8], version: u16, body: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&version.to_le_bytes());
    hasher.update(&(body.len() as u64).to_le_bytes());
    hasher.update(body);
    *hasher.finalize().as_bytes()
}

fn serialization_failure(error: serde_json::Error) -> CompileError {
    failure(
        CompilerFailureKind::MalformedArtifact,
        "envelope.serialization",
        error.to_string(),
        "report this deterministic canonical serialization failure",
    )
}
