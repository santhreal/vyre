use thiserror::Error;
use vyre_megakernel::{
    CompileError, Diagnostic, MegakernelArtifact, MegakernelArtifactEnvelope, TargetPayload,
    TargetPayloadFormat,
};

/// Failure to authenticate an artifact envelope or select its exact required payload.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("artifact admission rejected: {source}")]
pub struct ArtifactAdmissionError {
    #[source]
    source: CompileError,
}

impl ArtifactAdmissionError {
    /// Canonical structured diagnostic produced while decoding or selecting the payload.
    #[must_use]
    pub const fn diagnostic(&self) -> &Diagnostic {
        &self.source.diagnostic
    }

    /// Recover the canonical error without flattening its diagnostic context.
    #[must_use]
    pub fn into_compile_error(self) -> CompileError {
        self.source
    }
}

impl From<CompileError> for ArtifactAdmissionError {
    fn from(source: CompileError) -> Self {
        Self { source }
    }
}

/// Authenticated canonical envelope with one caller-selected exact payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmittedArtifact {
    envelope: MegakernelArtifactEnvelope,
    target_payload_index: usize,
}

impl AdmittedArtifact {
    /// Borrow the authenticated canonical envelope.
    #[must_use]
    pub const fn envelope(&self) -> &MegakernelArtifactEnvelope {
        &self.envelope
    }

    /// Borrow the canonical backend-neutral artifact.
    #[must_use]
    pub const fn neutral(&self) -> &MegakernelArtifact {
        self.envelope.neutral()
    }

    /// Borrow the exact target payload selected during admission.
    #[must_use]
    pub fn target_payload(&self) -> &TargetPayload {
        &self.envelope.target_payloads()[self.target_payload_index]
    }

    /// Consume the admission result and recover its owned canonical envelope.
    #[must_use]
    pub fn into_envelope(self) -> MegakernelArtifactEnvelope {
        self.envelope
    }
}

/// Decode and authenticate canonical envelope bytes, then require one exact payload format.
pub fn admit_artifact(
    envelope_bytes: &[u8],
    required_format: &TargetPayloadFormat,
) -> Result<AdmittedArtifact, ArtifactAdmissionError> {
    let envelope = MegakernelArtifactEnvelope::from_bytes(envelope_bytes)?;
    let selected = envelope.require_target_payload(required_format)?;
    let target_payload_index = envelope
        .target_payloads()
        .iter()
        .position(|payload| std::ptr::eq(payload, selected))
        .expect("canonical payload selection must borrow an attached payload");

    Ok(AdmittedArtifact {
        envelope,
        target_payload_index,
    })
}
