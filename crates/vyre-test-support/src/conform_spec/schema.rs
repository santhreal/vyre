//! Serializable conformance case, result, and replay-capsule schemas.

use std::fmt;

use serde::{Deserialize, Serialize};
use vyre_spec::schema_registry::SchemaId;

/// Supported replay-capsule schema version.
pub const REPLAY_CAPSULE_SCHEMA_VERSION: u32 = SchemaId::ReplayCapsule.version_u32();

/// A named conformance input case.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConformanceCase {
    /// Stable case name used as the canonical corpus sort key.
    pub name: String,
    /// Logical input buffers in declaration order.
    pub inputs: Vec<Vec<u8>>,
}

/// The result of one operation/executor conformance pair.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConformanceResult {
    /// Stable operation identifier.
    pub op_id: String,
    /// Executor identifier: a registered backend id, or the reference oracle.
    pub executor_id: String,
    /// Whether every executed case matched the reference.
    pub passed: bool,
    /// Human-readable result or failure diagnostic.
    pub message: String,
    /// Deterministic reproduction data for a mismatch.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replay_capsule: Option<ReplayCapsule>,
}

/// Deterministic data needed to reproduce one conformance mismatch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayCapsule {
    /// Replay schema version. Only [`REPLAY_CAPSULE_SCHEMA_VERSION`] is supported.
    #[serde(deserialize_with = "deserialize_replay_schema_version")]
    pub schema_version: u32,
    /// Stable operation identifier.
    pub op_id: String,
    /// Backend that diverged from the reference.
    pub backend_id: String,
    /// Zero-based witness case index.
    pub case_index: usize,
    /// Command that reproduces this mismatch.
    pub replay_command: String,
    /// Blake3 digest of canonical program bytes.
    pub program_blake3: String,
    /// Blake3 digest of witness input bytes.
    pub witness_input_blake3: String,
    /// Blake3 digest of reference output bytes.
    pub reference_output_blake3: String,
    /// Blake3 digest of backend output bytes.
    pub backend_output_blake3: String,
    /// Witness input buffers encoded as lowercase hexadecimal.
    pub witness_input_buffers_hex: Vec<String>,
    /// Reference output buffers encoded as lowercase hexadecimal.
    pub reference_output_buffers_hex: Vec<String>,
    /// Backend output buffers encoded as lowercase hexadecimal.
    pub backend_output_buffers_hex: Vec<String>,
    /// Number of witness input buffers.
    pub witness_input_count: usize,
    /// Number of reference output buffers.
    pub reference_output_count: usize,
    /// Number of backend output buffers.
    pub backend_output_count: usize,
    /// First observed mismatch.
    pub first_mismatch: ReplayMismatch,
    /// Deterministic minimization summary.
    pub minimization: ReplayMinimization,
}

impl ReplayCapsule {
    /// Reject a replay capsule from an unsupported schema version.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaVersionError`] when `schema_version` is not the current
    /// replay schema version.
    pub fn validate_schema_version(&self) -> Result<(), SchemaVersionError> {
        validate_u32_version(
            "replay capsule",
            self.schema_version,
            REPLAY_CAPSULE_SCHEMA_VERSION,
        )
    }
}

/// Location and shape of the first replay mismatch.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayMismatch {
    /// Stable mismatch kind (`output_count`, `output_length`, or `byte`).
    pub kind: String,
    /// Output buffer index, when applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_index: Option<usize>,
    /// Byte index within the output buffer, when applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_index: Option<usize>,
    /// Reference buffer length, when applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_len: Option<usize>,
    /// Backend buffer length, when applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend_len: Option<usize>,
    /// Reference byte, when applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_byte: Option<u8>,
    /// Backend byte, when applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend_byte: Option<u8>,
}

/// Replay-case minimization metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplayMinimization {
    /// Stable minimization strategy name.
    pub strategy: String,
    /// Number of cases before minimization.
    pub original_case_count: usize,
    /// Number of retained cases.
    pub retained_case_count: usize,
}

/// Explicit incompatibility between an artifact's schema and this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaVersionError {
    artifact: &'static str,
    found: String,
    supported: String,
}

impl SchemaVersionError {
    fn new(artifact: &'static str, found: impl Into<String>, supported: impl Into<String>) -> Self {
        Self {
            artifact,
            found: found.into(),
            supported: supported.into(),
        }
    }

    /// Artifact kind whose version was rejected.
    #[must_use]
    pub const fn artifact(&self) -> &'static str {
        self.artifact
    }

    /// Version present in the rejected artifact.
    #[must_use]
    pub fn found(&self) -> &str {
        &self.found
    }

    /// Version supported by this crate.
    #[must_use]
    pub fn supported(&self) -> &str {
        &self.supported
    }
}

impl fmt::Display for SchemaVersionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "unsupported {} schema version `{}`; supported version is `{}`",
            self.artifact, self.found, self.supported
        )
    }
}

impl std::error::Error for SchemaVersionError {}

fn validate_u32_version(
    artifact: &'static str,
    found: u32,
    supported: u32,
) -> Result<(), SchemaVersionError> {
    if found == supported {
        Ok(())
    } else {
        Err(SchemaVersionError::new(
            artifact,
            found.to_string(),
            supported.to_string(),
        ))
    }
}

fn deserialize_replay_schema_version<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let version = u32::deserialize(deserializer)?;
    validate_u32_version("replay capsule", version, REPLAY_CAPSULE_SCHEMA_VERSION)
        .map_err(serde::de::Error::custom)?;
    Ok(version)
}
