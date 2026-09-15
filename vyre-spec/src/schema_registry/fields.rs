//! The canonical field table for every registered schema.
//!
//! One `static` per schema id, each an ordered list of `CanonicalField`. Field
//! number, type, identity flag and requirement are the wire contract, so the
//! tables are data with no logic around them and are kept apart from the
//! registry that reads them.

use super::{CanonicalField, FieldType};

/// Field 1 of every schema whose first field states the version it was written
/// at. The number, type and identity flag are part of the wire contract, so the
/// entry is stated once and shared by every table that opens with it.
pub(super) const SCHEMA_VERSION_FIELD: CanonicalField = CanonicalField {
    number: 1,
    name: "schema_version",
    field_type: FieldType::U32,
    is_identity: true,
    required: true,
};

pub(super) static CONFORMANCE_CERT_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "version",
        field_type: FieldType::Utf8String,
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
        name: "wire_format_version",
        field_type: FieldType::U32,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 4,
        name: "program_blake3",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 5,
        name: "witness_set_blake3",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 6,
        name: "backend_id",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 7,
        name: "backend_version",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 8,
        name: "laws_verified",
        field_type: FieldType::List(&FieldType::Utf8String),
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 9,
        name: "timestamp",
        field_type: FieldType::Utf8String,
        is_identity: false,
        required: true,
    },
    CanonicalField {
        number: 10,
        name: "signature_ed25519",
        field_type: FieldType::Utf8String,
        is_identity: false,
        required: true,
    },
    CanonicalField {
        number: 11,
        name: "pubkey",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
];

pub(super) static ARTIFACT_PAYLOAD_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
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

pub(super) static SCHEDULE_RECORD_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
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

pub(super) static PROOF_RECEIPT_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
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

pub(super) static MEASUREMENT_RECORD_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
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

pub(super) static TRACE_EVENT_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
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

pub(super) static CACHE_ENTRY_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
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

pub(super) static CONFIG_RECEIPT_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
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

pub(super) static WIRE_OP_METADATA_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
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

pub(super) static INVARIANT_DIGEST_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
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

pub(super) static ANALYSIS_FACT_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
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

pub(super) static EXTENSION_SCHEMA_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
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

pub(super) static REPLAY_CAPSULE_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
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

pub(super) static BUNDLE_CERT_FIELDS: &[CanonicalField] = &[
    CanonicalField {
        number: 1,
        name: "version",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 2,
        name: "bundle_blake3",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 3,
        name: "corpus_blake3",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
    CanonicalField {
        number: 4,
        name: "reference_output_blake3",
        field_type: FieldType::Utf8String,
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
        name: "signature_ed25519",
        field_type: FieldType::Utf8String,
        is_identity: false,
        required: true,
    },
    CanonicalField {
        number: 8,
        name: "pubkey",
        field_type: FieldType::Utf8String,
        is_identity: true,
        required: true,
    },
];

pub(super) static PROVE_ARTIFACT_FIELDS: &[CanonicalField] = &[
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

pub(super) static PROOF_PLAN_FIELDS: &[CanonicalField] = &[
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

pub(super) static SAFETENSOR_INDEX_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
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

pub(super) static AOT_MANIFEST_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
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

pub(super) static ARTIFACT_REPORT_FIELDS: &[CanonicalField] = &[
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

pub(super) static WIRE_FRAMING_FIELDS: &[CanonicalField] = &[
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

pub(super) static TARGET_FACET_MATRIX_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
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

pub(super) static CAUSAL_RECEIPT_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
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
