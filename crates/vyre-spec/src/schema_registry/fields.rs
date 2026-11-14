//! The canonical field table for every registered schema.
//!
//! One `static` per schema id, each an ordered list of `CanonicalField`. Field
//! number, type, identity flag and requirement are the wire contract, so the
//! tables are data with no logic around them and are kept apart from the
//! registry that reads them.

use super::{CanonicalField, FieldType};

const fn id_field(number: u32, name: &'static str, field_type: FieldType) -> CanonicalField {
    CanonicalField {
        number,
        name,
        field_type,
        is_identity: true,
        required: true,
    }
}

const fn req_field(number: u32, name: &'static str, field_type: FieldType) -> CanonicalField {
    CanonicalField {
        number,
        name,
        field_type,
        is_identity: false,
        required: true,
    }
}

const fn opt_field(number: u32, name: &'static str, field_type: FieldType) -> CanonicalField {
    CanonicalField {
        number,
        name,
        field_type,
        is_identity: false,
        required: false,
    }
}

/// Field 1 of every schema whose first field states the version it was written
/// at. The number, type and identity flag are part of the wire contract, so the
/// entry is stated once and shared by every table that opens with it.
pub(super) const SCHEMA_VERSION_FIELD: CanonicalField =
    id_field(1, "schema_version", FieldType::U32);

/// Canonical field order of the shared structured diagnostic record.
///
/// `schema_version`, `severity`, `code` and `stage` form the identity: two
/// renderings of one failure are the same record when those four agree, and
/// every other field is context the surfaces render around them.
pub(super) static DIAGNOSTIC_RECORD_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
    id_field(2, "severity", FieldType::Utf8String),
    id_field(3, "code", FieldType::Utf8String),
    id_field(4, "stage", FieldType::Utf8String),
    opt_field(5, "compiler_level", FieldType::Utf8String),
    req_field(6, "message", FieldType::Utf8String),
    opt_field(7, "location", FieldType::Utf8String),
    opt_field(8, "artifact_id", FieldType::Utf8String),
    opt_field(9, "target", FieldType::Utf8String),
    opt_field(10, "device", FieldType::Utf8String),
    opt_field(11, "suggested_fix", FieldType::Utf8String),
    opt_field(12, "cause_chain", FieldType::List(&FieldType::Utf8String)),
    req_field(13, "retry", FieldType::Utf8String),
    opt_field(
        14,
        "context_values",
        FieldType::List(&FieldType::Utf8String),
    ),
    opt_field(15, "doc_url", FieldType::Utf8String),
    opt_field(16, "notes", FieldType::List(&FieldType::Utf8String)),
];

pub(super) static CONFORMANCE_CERT_FIELDS: &[CanonicalField] = &[
    id_field(1, "version", FieldType::Utf8String),
    id_field(2, "op_id", FieldType::Utf8String),
    id_field(3, "wire_format_version", FieldType::U32),
    id_field(4, "program_blake3", FieldType::Utf8String),
    id_field(5, "witness_set_blake3", FieldType::Utf8String),
    id_field(6, "backend_id", FieldType::Utf8String),
    id_field(7, "backend_version", FieldType::Utf8String),
    id_field(8, "laws_verified", FieldType::List(&FieldType::Utf8String)),
    req_field(9, "timestamp", FieldType::Utf8String),
    req_field(10, "signature_ed25519", FieldType::Utf8String),
    id_field(11, "pubkey", FieldType::Utf8String),
];

pub(super) static ARTIFACT_PAYLOAD_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
    id_field(2, "artifact_hash", FieldType::FixedBytes(32)),
    id_field(3, "target_backend", FieldType::Utf8String),
    id_field(4, "bytecode", FieldType::VarBytes),
    id_field(5, "entrypoint", FieldType::Utf8String),
    id_field(6, "required_workgroup_size", FieldType::FixedBytes(12)),
];

pub(super) static SCHEDULE_RECORD_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
    id_field(2, "schedule_id", FieldType::FixedBytes(32)),
    id_field(3, "fusion_plan", FieldType::VarBytes),
    id_field(4, "tiling_x", FieldType::U32),
    id_field(5, "tiling_y", FieldType::U32),
];

pub(super) static PROOF_RECEIPT_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
    id_field(2, "proof_digest", FieldType::FixedBytes(32)),
    id_field(3, "checker_identity", FieldType::Utf8String),
    id_field(4, "verified_claims", FieldType::U32),
];

pub(super) static MEASUREMENT_RECORD_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
    id_field(2, "workload_hash", FieldType::FixedBytes(32)),
    id_field(3, "duration_nanos", FieldType::U64),
    id_field(4, "warm_iterations", FieldType::U32),
];

pub(super) static TRACE_EVENT_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
    id_field(2, "event_id", FieldType::U64),
    id_field(3, "phase_tag", FieldType::U32),
    req_field(4, "timestamp_ns", FieldType::U64),
];

pub(super) static CACHE_ENTRY_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
    id_field(2, "key_hash", FieldType::FixedBytes(32)),
    id_field(3, "value_payload", FieldType::VarBytes),
    id_field(4, "generation_id", FieldType::U64),
];

pub(super) static CONFIG_RECEIPT_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
    id_field(2, "config_hash", FieldType::FixedBytes(32)),
    id_field(3, "behavior_hash", FieldType::FixedBytes(32)),
    id_field(4, "resolved_keys", FieldType::U32),
];

pub(super) static WIRE_OP_METADATA_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
    id_field(2, "op_id", FieldType::Utf8String),
    id_field(3, "category", FieldType::U32),
    id_field(4, "input_count", FieldType::U32),
];

pub(super) static INVARIANT_DIGEST_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
    id_field(2, "invariant_id", FieldType::Utf8String),
    id_field(3, "digest", FieldType::FixedBytes(32)),
];

pub(super) static ANALYSIS_FACT_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
    id_field(2, "fact_kind", FieldType::U32),
    id_field(3, "provenance_hash", FieldType::FixedBytes(32)),
];

pub(super) static EXTENSION_SCHEMA_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
    id_field(2, "extension_id", FieldType::FixedBytes(32)),
    id_field(3, "extension_name", FieldType::Utf8String),
    id_field(4, "op_count", FieldType::U32),
    id_field(5, "proof_digest", FieldType::FixedBytes(32)),
];

pub(super) static REPLAY_CAPSULE_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
    id_field(2, "op_id", FieldType::Utf8String),
    id_field(3, "backend_id", FieldType::Utf8String),
    id_field(4, "case_index", FieldType::U64),
    req_field(5, "replay_command", FieldType::Utf8String),
    id_field(6, "program_blake3", FieldType::FixedBytes(32)),
    id_field(7, "witness_input_blake3", FieldType::FixedBytes(32)),
    id_field(8, "reference_output_blake3", FieldType::FixedBytes(32)),
    id_field(9, "backend_output_blake3", FieldType::FixedBytes(32)),
];

pub(super) static BUNDLE_CERT_FIELDS: &[CanonicalField] = &[
    id_field(1, "version", FieldType::Utf8String),
    id_field(2, "bundle_blake3", FieldType::Utf8String),
    id_field(3, "corpus_blake3", FieldType::Utf8String),
    id_field(4, "reference_output_blake3", FieldType::Utf8String),
    id_field(5, "witness_count", FieldType::U64),
    req_field(6, "timestamp", FieldType::Utf8String),
    req_field(7, "signature_ed25519", FieldType::Utf8String),
    id_field(8, "pubkey", FieldType::Utf8String),
];

pub(super) static PROVE_ARTIFACT_FIELDS: &[CanonicalField] = &[
    id_field(1, "wire_format_version", FieldType::U32),
    id_field(2, "program_hash", FieldType::FixedBytes(32)),
    id_field(3, "backend_id", FieldType::Utf8String),
    id_field(4, "plan_digest", FieldType::FixedBytes(32)),
    id_field(5, "pair_count", FieldType::U64),
    id_field(6, "law_count", FieldType::U64),
];

pub(super) static PROOF_PLAN_FIELDS: &[CanonicalField] = &[
    id_field(1, "wire_format_version", FieldType::U32),
    id_field(2, "catalog_hash", FieldType::FixedBytes(32)),
    id_field(3, "execution_hash", FieldType::FixedBytes(32)),
    id_field(4, "backend_count", FieldType::U64),
    id_field(5, "op_count", FieldType::U64),
    id_field(6, "pair_count", FieldType::U64),
    id_field(7, "witness_case_count", FieldType::U64),
];

pub(super) static SAFETENSOR_INDEX_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
    id_field(2, "framing_version", FieldType::Utf8String),
    id_field(3, "total_tensor_count", FieldType::U64),
    id_field(4, "shard_count", FieldType::U64),
    id_field(5, "total_bytes", FieldType::U64),
];

pub(super) static AOT_MANIFEST_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
    id_field(2, "schema_name", FieldType::Utf8String),
    id_field(3, "aot_version", FieldType::Utf8String),
    id_field(4, "artifact_name", FieldType::Utf8String),
    id_field(5, "envelope_sha256_hex", FieldType::FixedBytes(32)),
    id_field(6, "neutral_artifact_digest_hex", FieldType::FixedBytes(32)),
    id_field(7, "target_payload_digest_hex", FieldType::FixedBytes(32)),
    id_field(8, "weights_sha256_hex", FieldType::FixedBytes(32)),
];

pub(super) static ARTIFACT_REPORT_FIELDS: &[CanonicalField] = &[
    id_field(1, "format_version", FieldType::U16),
    id_field(2, "artifact_digest", FieldType::FixedBytes(32)),
    id_field(3, "source_graph", FieldType::FixedBytes(32)),
    id_field(4, "semantic_graph", FieldType::FixedBytes(32)),
    id_field(5, "compiler_version", FieldType::Utf8String),
    id_field(6, "target_count", FieldType::U32),
];

pub(super) static WIRE_FRAMING_FIELDS: &[CanonicalField] = &[
    id_field(1, "wire_format_version", FieldType::U16),
    id_field(2, "magic", FieldType::FixedBytes(4)),
    id_field(3, "dialect_manifest_len", FieldType::U32),
    id_field(4, "body_len", FieldType::U64),
];

pub(super) static TARGET_FACET_MATRIX_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
    id_field(2, "platform_name", FieldType::Utf8String),
    id_field(3, "facet_count", FieldType::U32),
    id_field(4, "matrix_digest", FieldType::FixedBytes(32)),
];

pub(super) static CAUSAL_RECEIPT_FIELDS: &[CanonicalField] = &[
    SCHEMA_VERSION_FIELD,
    id_field(2, "session_id", FieldType::FixedBytes(32)),
    id_field(3, "receipt_id", FieldType::FixedBytes(32)),
    id_field(4, "causality_digest", FieldType::FixedBytes(32)),
];
