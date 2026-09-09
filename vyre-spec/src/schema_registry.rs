//! Declarative schema registry for all persisted, transmitted, cached, signed, and evidence records.
//!
//! WHY: closes the class "schema authority is fragmented and version constants drift".
//! Every persisted record, wire frame, proof certificate, and cache entry declares
//! fixed-width types, canonical field numbers, bounds, identity fields, and signature domain separators
//! in one central declarative registry.

use core::fmt;

use crate::compatibility::{CompatibilityDisposition, ProtocolDomain, ProtocolVersion};

/// Globally unique schema identifier for every persisted, signed, cached, and transmitted record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
#[repr(u32)]
pub enum SchemaId {
    /// Conformance test execution certificate.
    ConformanceCertificate = 1,
    /// Megakernel compiled artifact payload.
    ArtifactPayload = 2,
    /// Selected schedule and transform record.
    ScheduleRecord = 3,
    /// Proof and verification receipt.
    ProofReceipt = 4,
    /// Hardware performance measurement record.
    MeasurementRecord = 5,
    /// Causal execution trace event.
    TraceEvent = 6,
    /// Intermediate compilation and lowering cache entry.
    CacheEntry = 7,
    /// Canonical runtime and compiler configuration receipt.
    ConfigReceipt = 8,
    /// Serialized operation wire metadata.
    WireOpMetadata = 9,
    /// Engine invariant digest and verification descriptor.
    InvariantDigest = 10,
    /// Cross-engine structural analysis fact record.
    AnalysisFact = 11,
    /// Dialect extension schema declaration.
    ExtensionSchema = 12,
    /// Conformance mismatch reproduction replay capsule.
    ReplayCapsule = 13,
    /// Conformance certificate for a compiled bundle.
    BundleCertificate = 14,
    /// Signed prove command artifact.
    ProveArtifact = 15,
    /// Proof execution plan summary artifact.
    ProofPlanArtifact = 16,
    /// Safetensors checkpoint index metadata.
    SafetensorIndex = 17,
    /// AOT package manifest.
    AotManifest = 18,
    /// Diagnostic compiler artifact report.
    ArtifactReport = 19,
    /// Serialized program binary wire framing envelope.
    WireFraming = 20,
    /// Target facet compatibility matrix.
    TargetFacetMatrix = 21,
    /// Execution causal receipt.
    CausalReceipt = 22,
}

impl SchemaId {
    /// All schema identifiers in the registry.
    pub const ALL: &'static [Self] = &[
        Self::ConformanceCertificate,
        Self::ArtifactPayload,
        Self::ScheduleRecord,
        Self::ProofReceipt,
        Self::MeasurementRecord,
        Self::TraceEvent,
        Self::CacheEntry,
        Self::ConfigReceipt,
        Self::WireOpMetadata,
        Self::InvariantDigest,
        Self::AnalysisFact,
        Self::ExtensionSchema,
        Self::ReplayCapsule,
        Self::BundleCertificate,
        Self::ProveArtifact,
        Self::ProofPlanArtifact,
        Self::SafetensorIndex,
        Self::AotManifest,
        Self::ArtifactReport,
        Self::WireFraming,
        Self::TargetFacetMatrix,
        Self::CausalReceipt,
    ];

    /// Canonical string identifier for this schema.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ConformanceCertificate => "conformance_certificate",
            Self::ArtifactPayload => "artifact_payload",
            Self::ScheduleRecord => "schedule_record",
            Self::ProofReceipt => "proof_receipt",
            Self::MeasurementRecord => "measurement_record",
            Self::TraceEvent => "trace_event",
            Self::CacheEntry => "cache_entry",
            Self::ConfigReceipt => "config_receipt",
            Self::WireOpMetadata => "wire_op_metadata",
            Self::InvariantDigest => "invariant_digest",
            Self::AnalysisFact => "analysis_fact",
            Self::ExtensionSchema => "extension_schema",
            Self::ReplayCapsule => "replay_capsule",
            Self::BundleCertificate => "bundle_certificate",
            Self::ProveArtifact => "prove_artifact",
            Self::ProofPlanArtifact => "proof_plan_artifact",
            Self::SafetensorIndex => "safetensor_index",
            Self::AotManifest => "aot_manifest",
            Self::ArtifactReport => "artifact_report",
            Self::WireFraming => "wire_framing",
            Self::TargetFacetMatrix => "target_facet_matrix",
            Self::CausalReceipt => "causal_receipt",
        }
    }

    /// Protocol domain associated with this schema identifier.
    #[must_use]
    pub const fn domain(self) -> ProtocolDomain {
        match self {
            Self::WireFraming | Self::TargetFacetMatrix => ProtocolDomain::PublicWire,
            Self::WireOpMetadata
            | Self::AnalysisFact
            | Self::ExtensionSchema
            | Self::ConfigReceipt => ProtocolDomain::Catalog,
            Self::ConformanceCertificate
            | Self::ProofReceipt
            | Self::InvariantDigest
            | Self::ReplayCapsule
            | Self::BundleCertificate
            | Self::ProveArtifact
            | Self::ProofPlanArtifact => ProtocolDomain::Proof,
            Self::ScheduleRecord => ProtocolDomain::Schedule,
            Self::ArtifactPayload
            | Self::CacheEntry
            | Self::SafetensorIndex
            | Self::AotManifest
            | Self::ArtifactReport => ProtocolDomain::Artifact,
            Self::MeasurementRecord => ProtocolDomain::Measurement,
            Self::TraceEvent | Self::CausalReceipt => ProtocolDomain::RuntimeProtocol,
        }
    }
}

impl fmt::Display for SchemaId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Primitive and composite field types in canonical schemas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FieldType {
    /// Unsigned 8-bit integer.
    U8,
    /// Unsigned 16-bit integer (little-endian).
    U16,
    /// Unsigned 32-bit integer (little-endian).
    U32,
    /// Unsigned 64-bit integer (little-endian).
    U64,
    /// Signed 32-bit integer (little-endian).
    I32,
    /// Signed 64-bit integer (little-endian).
    I64,
    /// IEEE-754 32-bit float.
    F32,
    /// IEEE-754 64-bit float.
    F64,
    /// Boolean (encoded as 0 or 1 single byte).
    Bool,
    /// Fixed-length byte array of N bytes.
    FixedBytes(usize),
    /// Length-prefixed variable byte slice.
    VarBytes,
    /// UTF-8 encoded string.
    Utf8String,
    /// Repeated element list with max bound.
    List(&'static FieldType),
}

/// A canonical field definition in a schema entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CanonicalField {
    /// 1-based canonical field number (must be strictly monotonically increasing).
    pub number: u32,
    /// Canonical field name.
    pub name: &'static str,
    /// Fixed-width field type.
    pub field_type: FieldType,
    /// Whether this field is included in the record's cryptographic identity digest.
    pub is_identity: bool,
    /// Whether this field is strictly required.
    pub required: bool,
}

/// Policy for handling absent fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefaultsPolicy {
    /// No implicit or absent fields allowed; all declared fields must be explicitly encoded.
    NoDefaults,
    /// Strict explicit default only; missing optional fields must encode an explicit None tag.
    ExplicitDefaultOnly,
    /// Unknown fields are preserved only where the entry says that is safe.
    PreserveUnknownSafe,
}

/// Resource and depth bounds enforced during decoding and validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchemaBounds {
    /// Maximum payload size in bytes.
    pub max_bytes: usize,
    /// Maximum structural nesting depth.
    pub max_depth: usize,
    /// Maximum number of list elements.
    pub max_elements: usize,
}

/// Declarative schema definition for one registered schema id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchemaDefinition {
    /// Schema unique identifier.
    pub id: SchemaId,
    /// Semantic version of this schema format.
    pub semver: ProtocolVersion,
    /// Canonical field definitions in exact ascending order.
    pub fields: &'static [CanonicalField],
    /// Defaults policy.
    pub defaults_policy: DefaultsPolicy,
    /// Hard bounds on size, depth, and element counts.
    pub bounds: SchemaBounds,
    /// Signature domain separator preventing cross-schema cryptographic confusion.
    pub domain_separator: &'static str,
    /// Compatibility disposition for this schema version.
    pub compatibility: CompatibilityDisposition,
    /// Known stale fixture encodings from earlier versions for rejection testing.
    pub stale_fixtures: &'static [&'static str],
    /// Crate owning this schema definition.
    pub owning_package: &'static str,
}

impl SchemaDefinition {
    /// Protocol domain for this schema definition.
    #[must_use]
    pub const fn protocol_domain(&self) -> ProtocolDomain {
        self.id.domain()
    }

    /// Validate structural invariants of this schema definition.
    #[must_use]
    pub fn validate_invariants(&self) -> bool {
        let mut last_num = 0;
        let mut has_identity = false;
        for field in self.fields {
            if field.number <= last_num {
                return false; // Field numbers must be strictly increasing
            }
            last_num = field.number;
            if field.name.is_empty() {
                return false;
            }
            if field.is_identity {
                has_identity = true;
            }
        }
        has_identity
            && !self.domain_separator.is_empty()
            && self.bounds.max_bytes > 0
            && self.bounds.max_depth > 0
            && self.bounds.max_elements > 0
            && !self.owning_package.is_empty()
    }

    /// Generate structured markdown documentation for this schema.
    #[must_use]
    pub fn generate_documentation(&self) -> String {
        use core::fmt::Write as _;
        let mut doc = String::new();
        let _ = writeln!(
            doc,
            "# Schema: `{}` (ID: {})",
            self.id.as_str(),
            self.id as u32
        );
        let _ = writeln!(doc, "- **Version**: `{}`", self.semver);
        let _ = writeln!(doc, "- **Owning Package**: `{}`", self.owning_package);
        let _ = writeln!(doc, "- **Domain Separator**: `{}`", self.domain_separator);
        let _ = writeln!(
            doc,
            "- **Bounds**: max_bytes={}, max_depth={}, max_elements={}",
            self.bounds.max_bytes, self.bounds.max_depth, self.bounds.max_elements
        );
        let _ = writeln!(doc, "\n| Field # | Name | Type | Identity | Required |");
        let _ = writeln!(doc, "|---|---|---|---|---|");
        for field in self.fields {
            let _ = writeln!(
                doc,
                "| {} | `{}` | `{:?}` | {} | {} |",
                field.number, field.name, field.field_type, field.is_identity, field.required
            );
        }
        doc
    }

    /// Generate deterministic fuzz grammar rule in BNF format.
    #[must_use]
    pub fn fuzz_grammar(&self) -> String {
        use core::fmt::Write as _;
        let mut grammar = String::new();
        let name = self.id.as_str();
        let _ = writeln!(
            grammar,
            "<{name}_record> ::= \"VYRE\" <u32_id_{}> <semver_{}> <fields_{name}>",
            self.id as u32, self.semver
        );
        let mut fields_str = String::new();
        for field in self.fields {
            let _ = write!(fields_str, " <field_{}_{}>", name, field.number);
        }
        let _ = writeln!(grammar, "<fields_{name}> ::={fields_str}");
        for field in self.fields {
            let _ = writeln!(
                grammar,
                "<field_{}_{}> ::= <u32_field_num_{}> <type_{:?}>",
                name, field.number, field.number, field.field_type
            );
        }
        grammar
    }
}

static CONFORMANCE_CERT_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "schema_version",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 2,
        name: "certificate_id",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 3,
        name: "backend_id",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 4,
        name: "pass_count",
        field_type: FieldType::U64,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 5,
        name: "fail_count",
        field_type: FieldType::U64,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 6,
        name: "timestamp_utc",
        field_type: FieldType::U64,
        is_identity: false,
        required: true,
    },
];

static ARTIFACT_PAYLOAD_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "schema_version",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 2,
        name: "artifact_hash",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 3,
        name: "target_backend",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 4,
        name: "bytecode",
        field_type: FieldType::VarBytes,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 5,
        name: "entrypoint",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 6,
        name: "required_workgroup_size",
        field_type: FieldType::FixedBytes(12),
        is_identity: true,
        required: true,
    },
];

static SCHEDULE_RECORD_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "schema_version",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 2,
        name: "schedule_id",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 3,
        name: "fusion_plan",
        field_type: FieldType::VarBytes,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 4,
        name: "tiling_x",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 5,
        name: "tiling_y",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
];

static PROOF_RECEIPT_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "schema_version",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 2,
        name: "proof_digest",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 3,
        name: "checker_identity",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 4,
        name: "verified_claims",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
];

static MEASUREMENT_RECORD_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "schema_version",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 2,
        name: "workload_hash",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 3,
        name: "duration_nanos",
        field_type: FieldType::U64,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 4,
        name: "warm_iterations",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
];

static TRACE_EVENT_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "schema_version",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 2,
        name: "event_id",
        field_type: FieldType::U64,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 3,
        name: "phase_tag",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 4,
        name: "timestamp_ns",
        field_type: FieldType::U64,
        is_identity: false,
        required: true,
    },
];

static CACHE_ENTRY_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "schema_version",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 2,
        name: "key_hash",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 3,
        name: "value_payload",
        field_type: FieldType::VarBytes,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 4,
        name: "generation_id",
        field_type: FieldType::U64,
        is_identity: true,
        required: true,
    },
];

static CONFIG_RECEIPT_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "schema_version",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 2,
        name: "config_hash",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 3,
        name: "behavior_hash",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 4,
        name: "resolved_keys",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
];

static WIRE_OP_METADATA_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "schema_version",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 2,
        name: "op_id",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 3,
        name: "category",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 4,
        name: "input_count",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
];

static INVARIANT_DIGEST_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "schema_version",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 2,
        name: "invariant_id",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 3,
        name: "digest",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
];

static ANALYSIS_FACT_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "schema_version",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 2,
        name: "fact_kind",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 3,
        name: "provenance_hash",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
];

static EXTENSION_SCHEMA_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "schema_version",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 2,
        name: "extension_id",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 3,
        name: "extension_name",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 4,
        name: "op_count",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 5,
        name: "proof_digest",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
];

static REPLAY_CAPSULE_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "schema_version",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 2,
        name: "op_id",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 3,
        name: "backend_id",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 4,
        name: "case_index",
        field_type: FieldType::U64,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 5,
        name: "replay_command",
        field_type: FieldType::Utf8String,
        is_identity: false,
        required: true,
    },
    CanonicalField {
        number: 6,
        name: "program_blake3",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 7,
        name: "witness_input_blake3",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 8,
        name: "reference_output_blake3",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 9,
        name: "backend_output_blake3",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
];

static BUNDLE_CERT_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "schema_version",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 2,
        name: "bundle_blake3",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 3,
        name: "corpus_blake3",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 4,
        name: "reference_output_blake3",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 5,
        name: "witness_count",
        field_type: FieldType::U64,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 6,
        name: "timestamp",
        field_type: FieldType::Utf8String,
        is_identity: false,
        required: true,
    },
    CanonicalField {
        number: 7,
        name: "pubkey",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
];

static PROVE_ARTIFACT_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "wire_format_version",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 2,
        name: "program_hash",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 3,
        name: "backend_id",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 4,
        name: "plan_digest",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 5,
        name: "pair_count",
        field_type: FieldType::U64,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 6,
        name: "law_count",
        field_type: FieldType::U64,
        is_identity: true,
        required: true,
    },
];

static PROOF_PLAN_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "wire_format_version",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 2,
        name: "catalog_hash",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 3,
        name: "execution_hash",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 4,
        name: "backend_count",
        field_type: FieldType::U64,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 5,
        name: "op_count",
        field_type: FieldType::U64,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 6,
        name: "pair_count",
        field_type: FieldType::U64,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 7,
        name: "witness_case_count",
        field_type: FieldType::U64,
        is_identity: true,
        required: true,
    },
];

static SAFETENSOR_INDEX_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "schema_version",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 2,
        name: "framing_version",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 3,
        name: "total_tensor_count",
        field_type: FieldType::U64,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 4,
        name: "shard_count",
        field_type: FieldType::U64,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 5,
        name: "total_bytes",
        field_type: FieldType::U64,
        is_identity: true,
        required: true,
    },
];

static AOT_MANIFEST_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "schema_version",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 2,
        name: "schema_name",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 3,
        name: "aot_version",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 4,
        name: "artifact_name",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 5,
        name: "envelope_sha256_hex",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 6,
        name: "neutral_artifact_digest_hex",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 7,
        name: "target_payload_digest_hex",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 8,
        name: "weights_sha256_hex",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
];

static ARTIFACT_REPORT_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "format_version",
        field_type: FieldType::U16,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 2,
        name: "artifact_digest",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 3,
        name: "source_graph",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 4,
        name: "semantic_graph",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 5,
        name: "compiler_version",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 6,
        name: "target_count",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
];

static WIRE_FRAMING_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "wire_format_version",
        field_type: FieldType::U16,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 2,
        name: "magic",
        field_type: FieldType::FixedBytes(4),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 3,
        name: "dialect_manifest_len",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 4,
        name: "body_len",
        field_type: FieldType::U64,
        is_identity: true,
        required: true,
    },
];

static TARGET_FACET_MATRIX_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "schema_version",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 2,
        name: "platform_name",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 3,
        name: "facet_count",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 4,
        name: "matrix_digest",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
];

static CAUSAL_RECEIPT_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "schema_version",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 2,
        name: "session_id",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 3,
        name: "receipt_id",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 4,
        name: "causality_digest",
        field_type: FieldType::FixedBytes(32),
        is_identity: true,
        required: true,
    },
];

static STALE_CERT_FIXTURES: &[&str] = &[
    "vyre-conformance-certificate-v0",
    "vyre-conformance-certificate-v1",
];
static STALE_ARTIFACT_FIXTURES: &[&str] = &["vyre-artifact-v0"];
static STALE_SCHEDULE_FIXTURES: &[&str] = &["vyre-schedule-v0"];
static STALE_PROOF_FIXTURES: &[&str] = &["vyre-proof-receipt-v0"];
static STALE_MEASUREMENT_FIXTURES: &[&str] = &["vyre-measurement-v0"];
static STALE_TRACE_FIXTURES: &[&str] = &["vyre-trace-v0"];
static STALE_CACHE_FIXTURES: &[&str] = &["vyre-cache-v0"];
static STALE_CONFIG_FIXTURES: &[&str] = &["vyre-config-v0"];
static STALE_WIRE_OP_FIXTURES: &[&str] = &["vyre-wire-op-v0"];
static STALE_INVARIANT_FIXTURES: &[&str] = &["vyre-invariant-v0"];
static STALE_ANALYSIS_FIXTURES: &[&str] = &["vyre-fact-v0"];
static STALE_EXTENSION_FIXTURES: &[&str] = &["vyre-ext-schema-v0"];
static STALE_REPLAY_FIXTURES: &[&str] = &["vyre-replay-capsule-v1"];
static STALE_BUNDLE_FIXTURES: &[&str] = &["vyre-conformance-certificate-v1"];
static STALE_PROVE_FIXTURES: &[&str] = &["vyre-prove-artifact-v1"];
static STALE_PROOF_PLAN_FIXTURES: &[&str] = &["vyre-proof-plan-v0"];
static STALE_SAFETENSOR_FIXTURES: &[&str] = &["vyre-safetensors-v0"];
static STALE_AOT_FIXTURES: &[&str] = &[
    "vyre-aot-manifest-v3",
    "vyre-aot-manifest-v2",
    "vyre-aot-manifest-v1",
];
static STALE_REPORT_FIXTURES: &[&str] = &["vyre-artifact-report-v0"];
static STALE_WIRE_FRAMING_FIXTURES: &[&str] = &["vyre-wire-v3", "vyre-wire-v2", "vyre-wire-v1"];
static STALE_TARGET_FACET_FIXTURES: &[&str] = &["vyre-target-facet-v0"];
static STALE_CAUSAL_FIXTURES: &[&str] = &["vyre-causal-receipt-v0"];

/// The complete declarative schema registry.
pub const CANONICAL_SCHEMA_REGISTRY: &[SchemaDefinition] = &[
    SchemaDefinition {
        id: SchemaId::ConformanceCertificate,
        semver: ProtocolVersion::V1_0_0,
        fields: CONFORMANCE_CERT_FIELDS,
        defaults_policy: DefaultsPolicy::NoDefaults,
        bounds: SchemaBounds {
            max_bytes: 65536,
            max_depth: 4,
            max_elements: 1024,
        },
        domain_separator: "VYRE_CONFORMANCE_CERT_V1",
        compatibility: CompatibilityDisposition::Supported,
        stale_fixtures: STALE_CERT_FIXTURES,
        owning_package: "conform/vyre-conform",
    },
    SchemaDefinition {
        id: SchemaId::ArtifactPayload,
        semver: ProtocolVersion::V1_0_0,
        fields: ARTIFACT_PAYLOAD_FIELDS,
        defaults_policy: DefaultsPolicy::NoDefaults,
        bounds: SchemaBounds {
            max_bytes: 67108864,
            max_depth: 4,
            max_elements: 65536,
        },
        domain_separator: "VYRE_ARTIFACT_PAYLOAD_V1",
        compatibility: CompatibilityDisposition::Supported,
        stale_fixtures: STALE_ARTIFACT_FIXTURES,
        owning_package: "vyre-megakernel",
    },
    SchemaDefinition {
        id: SchemaId::ScheduleRecord,
        semver: ProtocolVersion::V1_0_0,
        fields: SCHEDULE_RECORD_FIELDS,
        defaults_policy: DefaultsPolicy::NoDefaults,
        bounds: SchemaBounds {
            max_bytes: 1048576,
            max_depth: 4,
            max_elements: 4096,
        },
        domain_separator: "VYRE_SCHEDULE_RECORD_V1",
        compatibility: CompatibilityDisposition::Supported,
        stale_fixtures: STALE_SCHEDULE_FIXTURES,
        owning_package: "vyre-foundation",
    },
    SchemaDefinition {
        id: SchemaId::ProofReceipt,
        semver: ProtocolVersion::V1_0_0,
        fields: PROOF_RECEIPT_FIELDS,
        defaults_policy: DefaultsPolicy::NoDefaults,
        bounds: SchemaBounds {
            max_bytes: 65536,
            max_depth: 4,
            max_elements: 1024,
        },
        domain_separator: "VYRE_PROOF_RECEIPT_V1",
        compatibility: CompatibilityDisposition::Supported,
        stale_fixtures: STALE_PROOF_FIXTURES,
        owning_package: "vyre-spec",
    },
    SchemaDefinition {
        id: SchemaId::MeasurementRecord,
        semver: ProtocolVersion::V1_0_0,
        fields: MEASUREMENT_RECORD_FIELDS,
        defaults_policy: DefaultsPolicy::NoDefaults,
        bounds: SchemaBounds {
            max_bytes: 65536,
            max_depth: 4,
            max_elements: 1024,
        },
        domain_separator: "VYRE_MEASUREMENT_RECORD_V1",
        compatibility: CompatibilityDisposition::Supported,
        stale_fixtures: STALE_MEASUREMENT_FIXTURES,
        owning_package: "vyre-bench",
    },
    SchemaDefinition {
        id: SchemaId::TraceEvent,
        semver: ProtocolVersion::V1_0_0,
        fields: TRACE_EVENT_FIELDS,
        defaults_policy: DefaultsPolicy::NoDefaults,
        bounds: SchemaBounds {
            max_bytes: 65536,
            max_depth: 4,
            max_elements: 1024,
        },
        domain_separator: "VYRE_TRACE_EVENT_V1",
        compatibility: CompatibilityDisposition::Supported,
        stale_fixtures: STALE_TRACE_FIXTURES,
        owning_package: "vyre-runtime",
    },
    SchemaDefinition {
        id: SchemaId::CacheEntry,
        semver: ProtocolVersion::V1_0_0,
        fields: CACHE_ENTRY_FIELDS,
        defaults_policy: DefaultsPolicy::NoDefaults,
        bounds: SchemaBounds {
            max_bytes: 67108864,
            max_depth: 4,
            max_elements: 65536,
        },
        domain_separator: "VYRE_CACHE_ENTRY_V1",
        compatibility: CompatibilityDisposition::Supported,
        stale_fixtures: STALE_CACHE_FIXTURES,
        owning_package: "vyre-runtime",
    },
    SchemaDefinition {
        id: SchemaId::ConfigReceipt,
        semver: ProtocolVersion::V1_0_0,
        fields: CONFIG_RECEIPT_FIELDS,
        defaults_policy: DefaultsPolicy::NoDefaults,
        bounds: SchemaBounds {
            max_bytes: 65536,
            max_depth: 4,
            max_elements: 1024,
        },
        domain_separator: "VYRE_CONFIG_RECEIPT_V1",
        compatibility: CompatibilityDisposition::Supported,
        stale_fixtures: STALE_CONFIG_FIXTURES,
        owning_package: "vyre-foundation",
    },
    SchemaDefinition {
        id: SchemaId::WireOpMetadata,
        semver: ProtocolVersion::V1_0_0,
        fields: WIRE_OP_METADATA_FIELDS,
        defaults_policy: DefaultsPolicy::NoDefaults,
        bounds: SchemaBounds {
            max_bytes: 65536,
            max_depth: 4,
            max_elements: 1024,
        },
        domain_separator: "VYRE_WIRE_OP_METADATA_V1",
        compatibility: CompatibilityDisposition::Supported,
        stale_fixtures: STALE_WIRE_OP_FIXTURES,
        owning_package: "vyre-spec",
    },
    SchemaDefinition {
        id: SchemaId::InvariantDigest,
        semver: ProtocolVersion::V1_0_0,
        fields: INVARIANT_DIGEST_FIELDS,
        defaults_policy: DefaultsPolicy::NoDefaults,
        bounds: SchemaBounds {
            max_bytes: 65536,
            max_depth: 4,
            max_elements: 1024,
        },
        domain_separator: "VYRE_INVARIANT_DIGEST_V1",
        compatibility: CompatibilityDisposition::Supported,
        stale_fixtures: STALE_INVARIANT_FIXTURES,
        owning_package: "vyre-spec",
    },
    SchemaDefinition {
        id: SchemaId::AnalysisFact,
        semver: ProtocolVersion::V1_0_0,
        fields: ANALYSIS_FACT_FIELDS,
        defaults_policy: DefaultsPolicy::NoDefaults,
        bounds: SchemaBounds {
            max_bytes: 65536,
            max_depth: 4,
            max_elements: 1024,
        },
        domain_separator: "VYRE_ANALYSIS_FACT_V1",
        compatibility: CompatibilityDisposition::Supported,
        stale_fixtures: STALE_ANALYSIS_FIXTURES,
        owning_package: "vyre-spec",
    },
    SchemaDefinition {
        id: SchemaId::ExtensionSchema,
        semver: ProtocolVersion::V1_0_0,
        fields: EXTENSION_SCHEMA_FIELDS,
        defaults_policy: DefaultsPolicy::NoDefaults,
        bounds: SchemaBounds {
            max_bytes: 1048576,
            max_depth: 4,
            max_elements: 4096,
        },
        domain_separator: "VYRE_EXTENSION_SCHEMA_V1",
        compatibility: CompatibilityDisposition::Supported,
        stale_fixtures: STALE_EXTENSION_FIXTURES,
        owning_package: "vyre-spec",
    },
    SchemaDefinition {
        id: SchemaId::ReplayCapsule,
        semver: ProtocolVersion::new(2, 0, 0),
        fields: REPLAY_CAPSULE_FIELDS,
        defaults_policy: DefaultsPolicy::NoDefaults,
        bounds: SchemaBounds {
            max_bytes: 67108864,
            max_depth: 4,
            max_elements: 65536,
        },
        domain_separator: "VYRE_REPLAY_CAPSULE_V2",
        compatibility: CompatibilityDisposition::Supported,
        stale_fixtures: STALE_REPLAY_FIXTURES,
        owning_package: "conform/vyre-conform-spec",
    },
    SchemaDefinition {
        id: SchemaId::BundleCertificate,
        semver: ProtocolVersion::new(2, 0, 0),
        fields: BUNDLE_CERT_FIELDS,
        defaults_policy: DefaultsPolicy::NoDefaults,
        bounds: SchemaBounds {
            max_bytes: 1048576,
            max_depth: 4,
            max_elements: 4096,
        },
        domain_separator: "VYRE_BUNDLE_CERT_V2",
        compatibility: CompatibilityDisposition::Supported,
        stale_fixtures: STALE_BUNDLE_FIXTURES,
        owning_package: "conform/vyre-conform-spec",
    },
    SchemaDefinition {
        id: SchemaId::ProveArtifact,
        semver: ProtocolVersion::new(2, 0, 0),
        fields: PROVE_ARTIFACT_FIELDS,
        defaults_policy: DefaultsPolicy::NoDefaults,
        bounds: SchemaBounds {
            max_bytes: 33554432,
            max_depth: 4,
            max_elements: 32768,
        },
        domain_separator: "VYRE_PROVE_ARTIFACT_V2",
        compatibility: CompatibilityDisposition::Supported,
        stale_fixtures: STALE_PROVE_FIXTURES,
        owning_package: "conform/vyre-conform",
    },
    SchemaDefinition {
        id: SchemaId::ProofPlanArtifact,
        semver: ProtocolVersion::new(1, 0, 0),
        fields: PROOF_PLAN_FIELDS,
        defaults_policy: DefaultsPolicy::NoDefaults,
        bounds: SchemaBounds {
            max_bytes: 1048576,
            max_depth: 4,
            max_elements: 4096,
        },
        domain_separator: "VYRE_PROOF_PLAN_V1",
        compatibility: CompatibilityDisposition::Supported,
        stale_fixtures: STALE_PROOF_PLAN_FIXTURES,
        owning_package: "conform/vyre-conform",
    },
    SchemaDefinition {
        id: SchemaId::SafetensorIndex,
        semver: ProtocolVersion::new(1, 0, 0),
        fields: SAFETENSOR_INDEX_FIELDS,
        defaults_policy: DefaultsPolicy::NoDefaults,
        bounds: SchemaBounds {
            max_bytes: 67108864,
            max_depth: 4,
            max_elements: 1000000,
        },
        domain_separator: "VYRE_SAFETENSOR_INDEX_V1",
        compatibility: CompatibilityDisposition::Supported,
        stale_fixtures: STALE_SAFETENSOR_FIXTURES,
        owning_package: "vyre-safetensors",
    },
    SchemaDefinition {
        id: SchemaId::AotManifest,
        semver: ProtocolVersion::new(4, 0, 0),
        fields: AOT_MANIFEST_FIELDS,
        defaults_policy: DefaultsPolicy::NoDefaults,
        bounds: SchemaBounds {
            max_bytes: 1048576,
            max_depth: 4,
            max_elements: 4096,
        },
        domain_separator: "VYRE_AOT_MANIFEST_V4",
        compatibility: CompatibilityDisposition::Supported,
        stale_fixtures: STALE_AOT_FIXTURES,
        owning_package: "vyre-aot",
    },
    SchemaDefinition {
        id: SchemaId::ArtifactReport,
        semver: ProtocolVersion::new(1, 0, 0),
        fields: ARTIFACT_REPORT_FIELDS,
        defaults_policy: DefaultsPolicy::NoDefaults,
        bounds: SchemaBounds {
            max_bytes: 16777216,
            max_depth: 4,
            max_elements: 8192,
        },
        domain_separator: "VYRE_ARTIFACT_REPORT_V1",
        compatibility: CompatibilityDisposition::Supported,
        stale_fixtures: STALE_REPORT_FIXTURES,
        owning_package: "vyre-debug",
    },
    SchemaDefinition {
        id: SchemaId::WireFraming,
        semver: ProtocolVersion::new(8, 0, 0),
        fields: WIRE_FRAMING_FIELDS,
        defaults_policy: DefaultsPolicy::NoDefaults,
        bounds: SchemaBounds {
            max_bytes: 536870912,
            max_depth: 32,
            max_elements: 1048576,
        },
        domain_separator: "VYRE_WIRE_FRAMING_V8",
        compatibility: CompatibilityDisposition::Supported,
        stale_fixtures: STALE_WIRE_FRAMING_FIXTURES,
        owning_package: "vyre-foundation",
    },
    SchemaDefinition {
        id: SchemaId::TargetFacetMatrix,
        semver: ProtocolVersion::new(1, 0, 0),
        fields: TARGET_FACET_MATRIX_FIELDS,
        defaults_policy: DefaultsPolicy::NoDefaults,
        bounds: SchemaBounds {
            max_bytes: 8388608,
            max_depth: 4,
            max_elements: 4096,
        },
        domain_separator: "VYRE_TARGET_FACET_MATRIX_V1",
        compatibility: CompatibilityDisposition::Supported,
        stale_fixtures: STALE_TARGET_FACET_FIXTURES,
        owning_package: "vyre-foundation",
    },
    SchemaDefinition {
        id: SchemaId::CausalReceipt,
        semver: ProtocolVersion::new(1, 0, 0),
        fields: CAUSAL_RECEIPT_FIELDS,
        defaults_policy: DefaultsPolicy::NoDefaults,
        bounds: SchemaBounds {
            max_bytes: 4194304,
            max_depth: 4,
            max_elements: 2048,
        },
        domain_separator: "VYRE_CAUSAL_RECEIPT_V1",
        compatibility: CompatibilityDisposition::Supported,
        stale_fixtures: STALE_CAUSAL_FIXTURES,
        owning_package: "vyre-foundation",
    },
];

/// Declarative schema registry manager.
pub struct SchemaRegistry;

impl SchemaRegistry {
    /// Retrieve schema definition by unique schema id.
    #[must_use]
    pub fn lookup(id: SchemaId) -> Option<&'static SchemaDefinition> {
        CANONICAL_SCHEMA_REGISTRY.iter().find(|def| def.id == id)
    }

    /// Return all registered schema definitions.
    #[must_use]
    pub const fn all() -> &'static [SchemaDefinition] {
        CANONICAL_SCHEMA_REGISTRY
    }
}
