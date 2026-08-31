mod finalist;
mod mesh;
mod portfolio;
mod retained;
mod session;
mod workspace;

pub use mesh::{MeshSession, MeshSessionError, MeshSubmission};
pub use portfolio::{admit_portfolio, AdmittedPortfolio};
pub use retained::RetainedArtifactSession;
pub use session::{ArtifactSession, ArtifactSessionError};
pub use workspace::ArtifactWorkspace;

use thiserror::Error;
use vyre_megakernel::allocation::DeviceSlot;
use vyre_megakernel::{
    Artifact, ArtifactEnvelope, CompileError, Diagnostic, TargetPayload, TargetPayloadFormat,
};

use crate::pipeline_cache::{PipelineCacheStore, PipelineFingerprint};

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
    envelope: ArtifactEnvelope,
    target_payload_index: usize,
    device_payload_indices: Vec<(DeviceSlot, usize)>,
}

impl AdmittedArtifact {
    /// Borrow the authenticated canonical envelope.
    #[must_use]
    pub const fn envelope(&self) -> &ArtifactEnvelope {
        &self.envelope
    }

    /// Borrow the canonical backend-neutral artifact.
    #[must_use]
    pub const fn neutral(&self) -> &Artifact {
        self.envelope.neutral()
    }

    /// Borrow the exact target payload selected during admission.
    #[must_use]
    pub fn target_payload(&self) -> &TargetPayload {
        &self.envelope.target_payloads()[self.target_payload_index]
    }

    /// Devices this artifact is submitted to, in slot order.
    #[must_use]
    pub fn submission_devices(&self) -> Vec<DeviceSlot> {
        self.device_payload_indices
            .iter()
            .map(|(device, _)| *device)
            .collect()
    }

    /// Borrow the payload one mesh device submits.
    ///
    /// # Errors
    ///
    /// Returns when the artifact is not submitted to `device`.
    pub fn target_payload_for_device(
        &self,
        device: DeviceSlot,
    ) -> Result<&TargetPayload, ArtifactAdmissionError> {
        let index = match self
            .device_payload_indices
            .iter()
            .find(|(slot, _)| *slot == device)
        {
            Some((_, index)) => *index,
            None => self
                .envelope
                .require_target_payload_index_for_device(self.target_payload().format(), device)?,
        };
        Ok(&self.envelope.target_payloads()[index])
    }

    /// Consume the admission result and recover its owned canonical envelope.
    #[must_use]
    pub fn into_envelope(self) -> ArtifactEnvelope {
        self.envelope
    }
}

/// Decode and authenticate canonical envelope bytes, then require one exact payload format.
pub fn admit_artifact(
    envelope_bytes: &[u8],
    required_format: &TargetPayloadFormat,
) -> Result<AdmittedArtifact, ArtifactAdmissionError> {
    let envelope = ArtifactEnvelope::from_bytes(envelope_bytes)?;
    admit_envelope(envelope, required_format)
}

/// Authenticate an already-decoded canonical envelope and require one exact payload format.
///
/// Prefer this when a producer such as AOT packaging has already decoded the envelope
/// and only the exact target-format selection remains.
pub fn admit_envelope(
    envelope: ArtifactEnvelope,
    required_format: &TargetPayloadFormat,
) -> Result<AdmittedArtifact, ArtifactAdmissionError> {
    let target_payload_index = envelope.require_target_payload_index(required_format)?;
    let mut device_payload_indices = Vec::new();
    for device in envelope.neutral().topology().submission_devices() {
        let index = envelope.require_target_payload_index_for_device(required_format, device)?;
        device_payload_indices.push((device, index));
    }
    Ok(AdmittedArtifact {
        envelope,
        target_payload_index,
        device_payload_indices,
    })
}

/// Load verified cache payload bytes and admit them as a canonical envelope.
///
/// `DiskCache` / `PipelineCacheStore` are format-agnostic blob stores. AOT
/// writes `ArtifactEnvelope` bytes as the payload (plus the store's
/// own BLAKE3 footer, stripped by [`PipelineCacheStore::get`]). Callers that
/// treat a cache hit as executable MUST run this helper (or
/// [`admit_artifact`] on the payload) before dispatch. A miss is `Ok(None)`.
///
/// # Errors
///
/// Returns [`ArtifactAdmissionError`] when payload bytes are present but are
/// not an authentic envelope with the required target format.
pub fn admit_cached_artifact(
    store: &dyn PipelineCacheStore,
    fingerprint: &PipelineFingerprint,
    required_format: &TargetPayloadFormat,
) -> Result<Option<AdmittedArtifact>, ArtifactAdmissionError> {
    let Some(payload) = store.get(fingerprint) else {
        return Ok(None);
    };
    admit_artifact(&payload, required_format).map(Some)
}
