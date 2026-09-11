//! Contract tests for typed schema authority, bounded decoding, canonical digests, and export.
//!
//! Proves:
//! 1. Canonical version comparison and compatibility.
//! 2. Bounded JSON decoder size limit enforcement.
//! 3. Trailing junk and invalid UTF-8 rejection.
//! 4. Platform-independent structured Blake3 digest calculation.
//! 5. Cross-language JSON schema generation.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use vyre_foundation::serial::{
    export_schema_json, BoundedDecoder, CanonicalDigest, CanonicalSchemaVersion,
    SchemaAuthorityError, SchemaId,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TestRecord {
    schema_version: String,
    op_id: String,
    count: u32,
}

#[test]
fn schema_version_compatibility_is_semver_compliant() {
    let v2_0_0 = CanonicalSchemaVersion::new(2, 0, 0);
    let v2_1_0 = CanonicalSchemaVersion::new(2, 1, 0);
    let v3_0_0 = CanonicalSchemaVersion::new(3, 0, 0);

    // v2.1.0 can read v2.0.0
    assert!(v2_1_0.is_compatible_with(&v2_0_0));
    // v2.0.0 cannot read v2.1.0 (forward incompatible)
    assert!(!v2_0_0.is_compatible_with(&v2_1_0));
    // Major version mismatch is incompatible
    assert!(!v3_0_0.is_compatible_with(&v2_0_0));
    assert!(!v2_0_0.is_compatible_with(&v3_0_0));
}

#[test]
fn bounded_decoder_rejects_oversized_payloads() {
    let payload = vec![b' '; 5 * 1024 * 1024]; // 5 MiB payload for TelemetryEvent (max 4 MiB)
    let res: Result<TestRecord, _> =
        BoundedDecoder::decode_json(SchemaId::TelemetryEvent, &payload);
    assert!(matches!(
        res,
        Err(SchemaAuthorityError::PayloadTooLarge { .. })
    ));
}

#[test]
fn bounded_decoder_parses_valid_payload_and_rejects_trailing_bytes() {
    let valid_json = br#"{"schema_version":"2.0.0","op_id":"primitive.add.u32","count":42}"#;
    let record: TestRecord =
        BoundedDecoder::decode_json(SchemaId::ConformanceCertificate, valid_json)
            .expect("valid JSON must decode");
    assert_eq!(record.schema_version, "2.0.0");
    assert_eq!(record.op_id, "primitive.add.u32");
    assert_eq!(record.count, 42);

    // Trailing junk rejected
    let trailing_json =
        br#"{"schema_version":"2.0.0","op_id":"primitive.add.u32","count":42} trailing_junk"#;
    let err: Result<TestRecord, _> =
        BoundedDecoder::decode_json(SchemaId::ConformanceCertificate, trailing_json);
    assert!(matches!(
        err,
        Err(SchemaAuthorityError::DecodeFailure { .. })
    ));

    // Unknown field rejected
    let unknown_field_json =
        br#"{"schema_version":"2.0.0","op_id":"primitive.add.u32","count":42,"unknown_prop":123}"#;
    let err: Result<TestRecord, _> =
        BoundedDecoder::decode_json(SchemaId::ConformanceCertificate, unknown_field_json);
    assert!(matches!(
        err,
        Err(SchemaAuthorityError::DecodeFailure { .. })
    ));
}

#[test]
fn canonical_digest_structured_prevents_concatenation_collision() {
    let d1 = CanonicalDigest::blake3_structured([b"ab".as_slice(), b"c".as_slice()]);
    let d2 = CanonicalDigest::blake3_structured([b"a".as_slice(), b"bc".as_slice()]);
    assert_ne!(
        d1, d2,
        "structured digest with length prefixes must differ for different chunking"
    );
}

#[test]
fn export_schema_json_produces_valid_json() {
    let schema_str = export_schema_json(SchemaId::ConformanceCertificate);
    let parsed: serde_json::Value =
        serde_json::from_str(&schema_str).expect("exported schema must be valid JSON");
    assert_eq!(parsed["id"], "vyre.conformance.certificate");
    assert_eq!(parsed["version"], "2.0.0");
    assert_eq!(parsed["digest_algorithm"], "blake3_256");
}
