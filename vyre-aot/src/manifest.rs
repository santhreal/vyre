//! AOT package manifest for one canonical megakernel artifact envelope.

use serde::{Deserialize, Serialize};

use crate::artifact::TargetId;

/// Top-level package manifest written beside the canonical envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// AOT package schema version.
    pub schema: String,
    /// AOT producer version.
    pub aot_version: String,
    /// Caller-supplied package name.
    pub artifact_name: String,
    /// Opaque concrete-driver target used to select the attached payload.
    pub target: TargetId,
    /// Concrete-driver-owned target payload format selected for this package.
    pub target_payload_format: String,
    /// Compressed canonical envelope filename within the package.
    pub envelope_file: String,
    /// Compression applied to the envelope file.
    pub envelope_compression: String,
    /// SHA-256 of the uncompressed canonical envelope bytes.
    pub envelope_sha256_hex: String,
    /// Exact canonical neutral artifact identity.
    pub neutral_artifact_digest_hex: String,
    /// Exact attached target payload identity.
    pub target_payload_digest_hex: String,
    /// Compressed weights filename within the package.
    pub weights_file: String,
    /// Compression applied to the weights file.
    pub weights_compression: String,
    /// SHA-256 of the uncompressed weights bytes.
    pub weights_sha256_hex: String,
    /// Free-form package notes.
    #[serde(default)]
    pub notes: String,
}

impl Manifest {
    /// Package schema written by this build.
    pub const SCHEMA_VERSION: &'static str = "vyre-aot-manifest-v4";
}
