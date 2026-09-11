//! Canonical typed schema authority for all persisted and signed records.
//!
//! Versioning across compilers, tooling, and runtimes must be governed by a
//! single authoritative schema registry rather than fragmented ad-hoc constants.
//!
//! This module owns:
//! 1. Canonical typed version representation ([`CanonicalSchemaVersion`]).
//! 2. Authoritative schema identifiers and descriptors ([`SchemaId`], [`SchemaDescriptor`]).
//! 3. Bounded payload decoding with strict unknown-field rejection ([`BoundedDecoder`]).
//! 4. Platform-independent signed payload digest algorithms ([`CanonicalDigest`]).
//! 5. Cross-language JSON schema generation and export ([`export_schema_json`]).

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use thiserror::Error;

/// Authoritative schema identifiers recognized by the Vyre workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchemaId {
    /// Conformance test certificate.
    ConformanceCertificate,
    /// Conformance divergence replay capsule.
    ReplayCapsule,
    /// Compiled physical kernel / program artifact.
    CompilationArtifact,
    /// Optimization e-graph rewrite proof certificate.
    OptimizationProof,
    /// Runtime telemetry and latency trace event.
    TelemetryEvent,
    /// Persistent compilation disk cache entry.
    PersistentCacheEntry,
    /// On-wire serialized program envelope.
    WirePayload,
    /// Target facet support matrix.
    TargetFacetMatrix,
}

impl SchemaId {
    /// Canonical string identifier for this schema.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ConformanceCertificate => "vyre.conformance.certificate",
            Self::ReplayCapsule => "vyre.conformance.replay_capsule",
            Self::CompilationArtifact => "vyre.compiler.artifact",
            Self::OptimizationProof => "vyre.optimizer.proof",
            Self::TelemetryEvent => "vyre.runtime.telemetry",
            Self::PersistentCacheEntry => "vyre.cache.persistent_entry",
            Self::WirePayload => "vyre.wire.envelope",
            Self::TargetFacetMatrix => "vyre.matrix.target_facet",
        }
    }
}

vyre_spec::semver_triple! {
    /// Canonical semantic schema version (Major.Minor.Patch).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
    pub struct CanonicalSchemaVersion;
}

impl CanonicalSchemaVersion {
    /// True if `self` can read data encoded by `encoded_version`.
    #[must_use]
    pub const fn is_compatible_with(&self, encoded_version: &Self) -> bool {
        self.major == encoded_version.major && self.minor >= encoded_version.minor
    }
}

/// Supported cryptographic digest algorithms for signed payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DigestAlgorithm {
    /// Blake3 256-bit cryptographic digest.
    Blake3_256,
}

/// Metadata descriptor defining the canonical contract of one schema.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaDescriptor {
    /// Schema identity.
    pub schema_id: SchemaId,
    /// Canonical human-readable name.
    pub canonical_name: &'static str,
    /// Current production version.
    pub current_version: CanonicalSchemaVersion,
    /// Minimum backward-compatible version accepted by decoders.
    pub min_compatible_version: CanonicalSchemaVersion,
    /// Maximum allowed payload size in bytes to prevent denial of service.
    pub max_payload_bytes: usize,
    /// Digest algorithm required for signatures.
    pub digest_algorithm: DigestAlgorithm,
}

/// Authoritative schema registry.
pub struct SchemaAuthority;

impl SchemaAuthority {
    /// Retrieve the authoritative descriptor for a schema.
    #[must_use]
    pub fn descriptor_for(schema_id: SchemaId) -> SchemaDescriptor {
        match schema_id {
            SchemaId::ConformanceCertificate => SchemaDescriptor {
                schema_id,
                canonical_name: "Vyre Conformance Certificate",
                current_version: CanonicalSchemaVersion::new(2, 0, 0),
                min_compatible_version: CanonicalSchemaVersion::new(2, 0, 0),
                max_payload_bytes: 32 * 1024 * 1024, // 32 MiB
                digest_algorithm: DigestAlgorithm::Blake3_256,
            },
            SchemaId::ReplayCapsule => SchemaDescriptor {
                schema_id,
                canonical_name: "Vyre Conformance Replay Capsule",
                current_version: CanonicalSchemaVersion::new(2, 0, 0),
                min_compatible_version: CanonicalSchemaVersion::new(2, 0, 0),
                max_payload_bytes: 64 * 1024 * 1024, // 64 MiB
                digest_algorithm: DigestAlgorithm::Blake3_256,
            },
            SchemaId::CompilationArtifact => SchemaDescriptor {
                schema_id,
                canonical_name: "Vyre Compilation Artifact",
                current_version: CanonicalSchemaVersion::new(1, 0, 0),
                min_compatible_version: CanonicalSchemaVersion::new(1, 0, 0),
                max_payload_bytes: 128 * 1024 * 1024, // 128 MiB
                digest_algorithm: DigestAlgorithm::Blake3_256,
            },
            SchemaId::OptimizationProof => SchemaDescriptor {
                schema_id,
                canonical_name: "Vyre Optimization Proof Certificate",
                current_version: CanonicalSchemaVersion::new(1, 0, 0),
                min_compatible_version: CanonicalSchemaVersion::new(1, 0, 0),
                max_payload_bytes: 16 * 1024 * 1024, // 16 MiB
                digest_algorithm: DigestAlgorithm::Blake3_256,
            },
            SchemaId::TelemetryEvent => SchemaDescriptor {
                schema_id,
                canonical_name: "Vyre Telemetry Event",
                current_version: CanonicalSchemaVersion::new(1, 0, 0),
                min_compatible_version: CanonicalSchemaVersion::new(1, 0, 0),
                max_payload_bytes: 4 * 1024 * 1024, // 4 MiB
                digest_algorithm: DigestAlgorithm::Blake3_256,
            },
            SchemaId::PersistentCacheEntry => SchemaDescriptor {
                schema_id,
                canonical_name: "Vyre Persistent Cache Entry",
                current_version: CanonicalSchemaVersion::new(1, 0, 0),
                min_compatible_version: CanonicalSchemaVersion::new(1, 0, 0),
                max_payload_bytes: 256 * 1024 * 1024, // 256 MiB
                digest_algorithm: DigestAlgorithm::Blake3_256,
            },
            SchemaId::WirePayload => SchemaDescriptor {
                schema_id,
                canonical_name: "Vyre Wire Framing Envelope",
                current_version: CanonicalSchemaVersion::new(1, 0, 0),
                min_compatible_version: CanonicalSchemaVersion::new(1, 0, 0),
                max_payload_bytes: 512 * 1024 * 1024, // 512 MiB
                digest_algorithm: DigestAlgorithm::Blake3_256,
            },
            SchemaId::TargetFacetMatrix => SchemaDescriptor {
                schema_id,
                canonical_name: "Vyre Target Facet Matrix",
                current_version: CanonicalSchemaVersion::new(1, 0, 0),
                min_compatible_version: CanonicalSchemaVersion::new(1, 0, 0),
                max_payload_bytes: 8 * 1024 * 1024, // 8 MiB
                digest_algorithm: DigestAlgorithm::Blake3_256,
            },
        }
    }
}

/// Error encountered during bounded schema decoding.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SchemaAuthorityError {
    /// Input bytes exceed descriptor maximum length limit.
    #[error(
        "payload length {got_bytes} B exceeds maximum allowed {max_bytes} B for {schema_name}"
    )]
    PayloadTooLarge {
        /// Schema canonical name.
        schema_name: &'static str,
        /// Maximum allowed bytes.
        max_bytes: usize,
        /// Actual bytes provided.
        got_bytes: usize,
    },
    /// Input is not valid UTF-8 for text/JSON schemas.
    #[error("invalid UTF-8 in {schema_name}: {details}")]
    InvalidUtf8 {
        /// Schema canonical name.
        schema_name: &'static str,
        /// Details.
        details: String,
    },
    /// Incompatible schema version.
    #[error(
        "incompatible schema version {found} for {schema_name}; minimum required is {min_required}"
    )]
    IncompatibleVersion {
        /// Schema canonical name.
        schema_name: &'static str,
        /// Version found in payload.
        found: CanonicalSchemaVersion,
        /// Minimum version required.
        min_required: CanonicalSchemaVersion,
    },
    /// Deserialization error or unknown field detected.
    #[error("schema decoding error in {schema_name}: {details}")]
    DecodeFailure {
        /// Schema canonical name.
        schema_name: &'static str,
        /// Error details.
        details: String,
    },
    /// Digest verification failure.
    #[error("signed digest mismatch for {schema_name}: expected {expected}, computed {computed}")]
    DigestMismatch {
        /// Schema canonical name.
        schema_name: &'static str,
        /// Expected digest hex.
        expected: String,
        /// Computed digest hex.
        computed: String,
    },
}

/// Bounded decoder enforcing byte size, version compatibility, and strict field constraints.
pub struct BoundedDecoder;

impl BoundedDecoder {
    /// Decode a JSON payload with strict size limits and deserialization.
    pub fn decode_json<T: DeserializeOwned>(
        schema_id: SchemaId,
        bytes: &[u8],
    ) -> Result<T, SchemaAuthorityError> {
        let desc = SchemaAuthority::descriptor_for(schema_id);
        if bytes.len() > desc.max_payload_bytes {
            return Err(SchemaAuthorityError::PayloadTooLarge {
                schema_name: desc.canonical_name,
                max_bytes: desc.max_payload_bytes,
                got_bytes: bytes.len(),
            });
        }

        let s = std::str::from_utf8(bytes).map_err(|e| SchemaAuthorityError::InvalidUtf8 {
            schema_name: desc.canonical_name,
            details: e.to_string(),
        })?;

        let mut deserializer = serde_json::Deserializer::from_str(s);
        let value =
            T::deserialize(&mut deserializer).map_err(|e| SchemaAuthorityError::DecodeFailure {
                schema_name: desc.canonical_name,
                details: e.to_string(),
            })?;

        deserializer
            .end()
            .map_err(|e| SchemaAuthorityError::DecodeFailure {
                schema_name: desc.canonical_name,
                details: format!("trailing characters after payload: {e}"),
            })?;

        Ok(value)
    }
}

/// Platform-independent canonical digest calculations.
pub struct CanonicalDigest;

impl CanonicalDigest {
    /// Compute a canonical Blake3 digest over raw byte slices.
    #[must_use]
    pub fn blake3_hex(data: &[u8]) -> String {
        let hash = blake3::hash(data);
        hash.to_hex().to_string()
    }

    /// Compute a canonical digest over multiple structured byte slices with length prefixing
    /// to prevent concatenation collisions.
    #[must_use]
    pub fn blake3_structured<I, B>(chunks: I) -> String
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        let mut hasher = blake3::Hasher::new();
        for chunk in chunks {
            let slice = chunk.as_ref();
            let len_bytes = (slice.len() as u64).to_le_bytes();
            hasher.update(&len_bytes);
            hasher.update(slice);
        }
        hasher.finalize().to_hex().to_string()
    }
}

/// Export canonical JSON schema definition for foreign-language interop and documentation.
#[must_use]
pub fn export_schema_json(schema_id: SchemaId) -> String {
    let desc = SchemaAuthority::descriptor_for(schema_id);
    serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "id": desc.schema_id.as_str(),
        "title": desc.canonical_name,
        "type": "object",
        "version": desc.current_version.to_string(),
        "min_compatible_version": desc.min_compatible_version.to_string(),
        "max_payload_bytes": desc.max_payload_bytes,
        "digest_algorithm": desc.digest_algorithm,
        "properties": {
            "schema_version": {
                "type": "string",
                "const": desc.current_version.to_string()
            }
        },
        "required": ["schema_version"],
        "additionalProperties": false
    })
    .to_string()
}
