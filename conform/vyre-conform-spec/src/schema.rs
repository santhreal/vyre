//! Stable, serializable conformance artifact schemas.
//!
//! This module owns data only. Execution, signing, dispatch, and enforcement
//! remain the responsibility of conformance consumers.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Supported per-operation and bundle certificate schema version.
pub const CERTIFICATE_SCHEMA_VERSION: &str = "0.4.1";

/// Supported replay-capsule schema version.
pub const REPLAY_CAPSULE_SCHEMA_VERSION: u32 = 1;

/// A named conformance input case.
///
/// `inputs` contains one raw byte buffer per logical input. The field order is
/// part of the JSON contract and must not be changed without a schema migration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConformanceCase {
    /// Stable case name used as the canonical corpus sort key.
    pub name: String,
    /// Logical input buffers in declaration order.
    pub inputs: Vec<Vec<u8>>,
}

/// The result of one operation/backend conformance pair.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConformanceResult {
    /// Stable operation identifier.
    pub op_id: String,
    /// Backend that executed the case set.
    pub backend_id: String,
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

/// Conformance certificate for one operation/backend pair.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Certificate {
    /// Certificate schema version.
    #[serde(deserialize_with = "deserialize_certificate_schema_version")]
    pub version: String,
    /// Stable operation identifier.
    pub op_id: String,
    /// Program wire-format version at issue time.
    pub wire_format_version: u32,
    /// Blake3 digest of canonical program wire bytes.
    pub program_blake3: String,
    /// Blake3 digest of canonical witness bytes.
    pub witness_set_blake3: String,
    /// Backend that produced the certificate.
    pub backend_id: String,
    /// Backend crate version string.
    pub backend_version: String,
    /// Laws verified to hold.
    pub laws_verified: Vec<String>,
    /// ISO 8601 UTC timestamp string.
    pub timestamp: String,
    /// Ed25519 signature over the canonical JSON body, encoded as hex.
    pub signature_ed25519: String,
    /// Ed25519 public key, encoded as hex.
    pub pubkey: String,
}

impl Certificate {
    /// Construct an unsigned certificate with deterministic placeholder fields.
    #[must_use]
    pub fn new(
        op_id: impl Into<String>,
        backend_id: impl Into<String>,
        backend_version: impl Into<String>,
        laws_verified: Vec<String>,
    ) -> Self {
        Self {
            version: CERTIFICATE_SCHEMA_VERSION.to_string(),
            op_id: op_id.into(),
            wire_format_version: 1,
            program_blake3: "TBD".to_string(),
            witness_set_blake3: "TBD".to_string(),
            backend_id: backend_id.into(),
            backend_version: backend_version.into(),
            laws_verified,
            timestamp: "1970-01-01T00:00:00Z".to_string(),
            signature_ed25519: "TBD".to_string(),
            pubkey: "TBD".to_string(),
        }
    }

    /// Serialize to the established pretty-printed JSON representation.
    ///
    /// # Errors
    ///
    /// Returns the underlying serde error if serialization fails.
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// Reject a certificate from an unsupported schema version.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaVersionError`] when `version` is not the current
    /// certificate schema version.
    pub fn validate_schema_version(&self) -> Result<(), SchemaVersionError> {
        validate_string_version("certificate", &self.version, CERTIFICATE_SCHEMA_VERSION)
    }
}

/// Conformance certificate for a complete compiled bundle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BundleCertificate {
    /// Certificate schema version.
    #[serde(deserialize_with = "deserialize_bundle_certificate_schema_version")]
    pub version: String,
    /// Blake3 digest of canonical bundle wire bytes.
    pub bundle_blake3: String,
    /// Blake3 digest of the canonical input corpus stream.
    pub corpus_blake3: String,
    /// Blake3 digest of the canonical reference-output stream.
    pub reference_output_blake3: String,
    /// Number of witness inputs.
    pub witness_count: u64,
    /// ISO 8601 UTC timestamp string.
    pub timestamp: String,
    /// Ed25519 signature over the canonical JSON body, encoded as hex.
    pub signature_ed25519: String,
    /// Ed25519 public key, encoded as hex.
    pub pubkey: String,
}

impl BundleCertificate {
    /// Reject a bundle certificate from an unsupported schema version.
    ///
    /// # Errors
    ///
    /// Returns [`SchemaVersionError`] when `version` is not the current
    /// certificate schema version.
    pub fn validate_schema_version(&self) -> Result<(), SchemaVersionError> {
        validate_string_version(
            "bundle certificate",
            &self.version,
            CERTIFICATE_SCHEMA_VERSION,
        )
    }
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

fn validate_string_version(
    artifact: &'static str,
    found: &str,
    supported: &str,
) -> Result<(), SchemaVersionError> {
    if found == supported {
        Ok(())
    } else {
        Err(SchemaVersionError::new(artifact, found, supported))
    }
}

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

fn deserialize_certificate_schema_version<'de, D>(
    deserializer: D,
) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let version = String::deserialize(deserializer)?;
    validate_string_version("certificate", &version, CERTIFICATE_SCHEMA_VERSION)
        .map_err(serde::de::Error::custom)?;
    Ok(version)
}

fn deserialize_bundle_certificate_schema_version<'de, D>(
    deserializer: D,
) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let version = String::deserialize(deserializer)?;
    validate_string_version(
        "bundle certificate",
        &version,
        CERTIFICATE_SCHEMA_VERSION,
    )
    .map_err(serde::de::Error::custom)?;
    Ok(version)
}

fn deserialize_replay_schema_version<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let version = u32::deserialize(deserializer)?;
    validate_u32_version(
        "replay capsule",
        version,
        REPLAY_CAPSULE_SCHEMA_VERSION,
    )
    .map_err(serde::de::Error::custom)?;
    Ok(version)
}
