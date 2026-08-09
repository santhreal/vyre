//! AOT package handle for the canonical megakernel artifact envelope.

pub use vyre_driver::Target;
use vyre_megakernel::{
    ArtifactEnvelope, CompileError as EnvelopeError, TargetPayload, TargetPayloadFormat,
};

/// Target payload format version emitted by this AOT package revision.
pub const AOT_TARGET_PAYLOAD_FORMAT_VERSION: u16 = 1;

/// A packaged canonical artifact plus AOT-only producer metadata.
#[derive(Debug, Clone)]
pub struct CompiledArtifact {
    /// Concrete target selected through the driver registry.
    pub target: Target,
    envelope: ArtifactEnvelope,
    aot_version: String,
    vsa_fingerprint: Vec<u32>,
}

impl CompiledArtifact {
    /// Construct an AOT package only when the canonical envelope carries the selected target.
    pub fn new(
        target: Target,
        envelope: ArtifactEnvelope,
        aot_version: impl Into<String>,
        vsa_fingerprint: Vec<u32>,
    ) -> Result<Self, EnvelopeError> {
        envelope.require_target_payload(&target_payload_format(target)?)?;
        Ok(Self {
            target,
            envelope,
            aot_version: aot_version.into(),
            vsa_fingerprint,
        })
    }

    /// Canonical neutral artifact and attached target payloads.
    #[must_use]
    pub const fn envelope(&self) -> &ArtifactEnvelope {
        &self.envelope
    }

    /// Exact compatible target payload selected by this package.
    pub fn target_payload(&self) -> Result<&TargetPayload, EnvelopeError> {
        self.envelope
            .require_target_payload(&target_payload_format(self.target)?)
    }

    /// AOT producer version retained as package provenance.
    #[must_use]
    pub fn aot_version(&self) -> &str {
        &self.aot_version
    }

    /// Approximate optimized-program fingerprint retained for cache discovery.
    #[must_use]
    pub fn vsa_fingerprint(&self) -> &[u32] {
        &self.vsa_fingerprint
    }

    /// Exact total bytes from the canonical neutral resource envelope.
    #[must_use]
    pub fn total_buffer_bytes(&self) -> u64 {
        self.envelope.neutral().resource_envelope().total_bytes
    }
}

/// Canonical target payload format identity for one driver target.
pub fn target_payload_format(target: Target) -> Result<TargetPayloadFormat, EnvelopeError> {
    TargetPayloadFormat::new(target.aot_target_id(), AOT_TARGET_PAYLOAD_FORMAT_VERSION)
}
