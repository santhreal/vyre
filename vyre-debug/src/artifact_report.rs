use serde::{Deserialize, Serialize};
use vyre::compiler::{ArtifactAbi, ArtifactEnvelope, CompileError, SelectedPlan};

/// Stable debug projection for one authenticated target payload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetPayloadReport {
    /// Target payload format identity.
    pub format: String,
    /// Target payload format version.
    pub format_version: u16,
    /// Authenticated payload digest.
    pub digest: String,
    /// Number of compiler-selected target entries.
    pub entries: usize,
    /// Encoded target-module bytes.
    pub bytes: usize,
}

/// Compiler-owned identities, selected plan, ABI, and attached target payloads.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactReport {
    /// Neutral artifact digest.
    pub artifact: String,
    /// Resolved logical-program digest.
    pub source_graph: String,
    /// Graph digest before any binding resolved it.
    pub semantic_graph: String,
    /// Validated compile-request digest.
    pub request: String,
    /// Compiler version recorded by the artifact.
    pub compiler_version: String,
    /// Exact bounded schedule selected by the compiler.
    pub selected_plan: SelectedPlan,
    /// Canonical ABI projected to every target.
    pub abi: ArtifactAbi,
    /// Canonically ordered authenticated target payload reports.
    pub targets: Vec<TargetPayloadReport>,
}

impl ArtifactReport {
    /// Authenticate envelope bytes and project compiler-owned diagnostic state.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CompileError> {
        let envelope = ArtifactEnvelope::from_bytes(bytes)?;
        Ok(Self::from_envelope(&envelope))
    }

    /// Project compiler-owned diagnostic state from an authenticated envelope.
    #[must_use]
    pub fn from_envelope(envelope: &ArtifactEnvelope) -> Self {
        let artifact = envelope.neutral();
        let provenance = artifact.provenance();
        let targets = envelope
            .target_payloads()
            .iter()
            .map(|payload| TargetPayloadReport {
                format: payload.format().identity().to_string(),
                format_version: payload.format().version(),
                digest: digest_hex(payload.digest().as_bytes()),
                entries: payload.entries().len(),
                bytes: payload.bytes().len(),
            })
            .collect();
        Self {
            artifact: digest_hex(artifact.digest().as_bytes()),
            source_graph: digest_hex(provenance.source_graph.as_bytes()),
            semantic_graph: digest_hex(provenance.semantic_graph.as_bytes()),
            request: digest_hex(provenance.request.as_bytes()),
            compiler_version: provenance.compiler_version.clone(),
            selected_plan: artifact.selected_plan().clone(),
            abi: artifact.abi().clone(),
            targets,
        }
    }
}

fn digest_hex(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}
