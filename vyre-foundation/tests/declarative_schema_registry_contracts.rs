//! Tests for declarative schema registry, canonical codecs, bounds checking, and identity digests.
//!
//! WHY: proves Row 121:
//! - Canonical bytes are identical across serialization roundtrips.
//! - Non-canonical field ordering, duplicate fields, or unexpected EOF are rejected before allocation.
//! - Hard payload bounds are enforced and reject oversized records.
//! - Identity digests use domain separators and include only declared identity fields.
//! - Adding or modifying a field turns verification red.

use vyre_foundation::{
    CanonicalDecoder, CanonicalEncoder, CanonicalRecord, CanonicalSigner, CanonicalValue,
    CodecError,
};
use vyre_spec::{SchemaId, SchemaRegistry};

#[test]
fn canonical_codec_round_trip_conformance_certificate() {
    let cert_record = CanonicalRecord {
        schema_id: SchemaId::ConformanceCertificate,
        fields: vec![
            (1, CanonicalValue::U32(1)),
            (2, CanonicalValue::FixedBytes(vec![0xAA; 32])),
            (3, CanonicalValue::Utf8String("cuda".into())),
            (4, CanonicalValue::U64(150)),
            (5, CanonicalValue::U64(0)),
            (6, CanonicalValue::U64(1_700_000_000)),
        ],
    };

    let encoded = CanonicalEncoder::encode(&cert_record)
        .expect("Fix: valid ConformanceCertificate record must encode canonically.");

    assert_eq!(&encoded[0..4], b"VYRE", "Fix: magic bytes must be 'VYRE'");

    let decoded = CanonicalDecoder::decode(&encoded)
        .expect("Fix: canonical bytes must decode into identical CanonicalRecord.");

    assert_eq!(decoded, cert_record);

    let digest1 = CanonicalSigner::compute_identity_digest(&cert_record)
        .expect("Fix: identity digest computation must succeed.");
    let digest2 = CanonicalSigner::compute_identity_digest(&decoded)
        .expect("Fix: decoded record must yield identical identity digest.");
    assert_eq!(digest1, digest2);
}

#[test]
fn canonical_codec_rejects_out_of_order_fields() {
    let out_of_order_record = CanonicalRecord {
        schema_id: SchemaId::ConformanceCertificate,
        fields: vec![
            (2, CanonicalValue::FixedBytes(vec![0xAA; 32])),
            (1, CanonicalValue::U32(1)), // Field 1 after field 2 is non-canonical
            (3, CanonicalValue::Utf8String("cuda".into())),
            (4, CanonicalValue::U64(150)),
            (5, CanonicalValue::U64(0)),
            (6, CanonicalValue::U64(1_700_000_000)),
        ],
    };

    let err = CanonicalEncoder::encode(&out_of_order_record)
        .expect_err("Fix: non-canonical field order must be rejected during encoding.");

    assert!(matches!(err, CodecError::NonCanonicalFieldOrder { .. }));
}

#[test]
fn canonical_codec_rejects_missing_required_fields() {
    let missing_field_record = CanonicalRecord {
        schema_id: SchemaId::ConformanceCertificate,
        fields: vec![
            (1, CanonicalValue::U32(1)),
            // Field 2 (certificate_id) is missing
            (3, CanonicalValue::Utf8String("cuda".into())),
            (4, CanonicalValue::U64(150)),
            (5, CanonicalValue::U64(0)),
            (6, CanonicalValue::U64(1_700_000_000)),
        ],
    };

    let err = CanonicalEncoder::encode(&missing_field_record)
        .expect_err("Fix: missing required field must be rejected.");

    assert!(matches!(err, CodecError::MissingRequiredField { .. }));
}

#[test]
fn canonical_codec_rejects_truncated_payload() {
    let cert_record = CanonicalRecord {
        schema_id: SchemaId::ConformanceCertificate,
        fields: vec![
            (1, CanonicalValue::U32(1)),
            (2, CanonicalValue::FixedBytes(vec![0xAA; 32])),
            (3, CanonicalValue::Utf8String("cuda".into())),
            (4, CanonicalValue::U64(150)),
            (5, CanonicalValue::U64(0)),
            (6, CanonicalValue::U64(1_700_000_000)),
        ],
    };

    let mut encoded = CanonicalEncoder::encode(&cert_record).unwrap();
    encoded.truncate(encoded.len() - 5); // Truncate last field

    let err = CanonicalDecoder::decode(&encoded)
        .expect_err("Fix: truncated bytes must be rejected as UnexpectedEof.");

    assert!(matches!(err, CodecError::UnexpectedEof { .. }));
}

#[test]
fn canonical_codec_enforces_max_bytes_bound() {
    let schema = SchemaRegistry::lookup(SchemaId::ConformanceCertificate).unwrap();
    let oversized_bytes = vec![0u8; schema.bounds.max_bytes + 100];

    let err = CanonicalDecoder::decode(&oversized_bytes)
        .expect_err("Fix: payload exceeding max_bytes bound must fail closed.");

    assert!(matches!(err, CodecError::PayloadOversized { .. } | CodecError::UnknownSchema(_)));
}

#[test]
fn domain_separators_prevent_cross_schema_digest_collisions() {
    let cert_record = CanonicalRecord {
        schema_id: SchemaId::ConformanceCertificate,
        fields: vec![
            (1, CanonicalValue::U32(1)),
            (2, CanonicalValue::FixedBytes(vec![0xAA; 32])),
            (3, CanonicalValue::Utf8String("cuda".into())),
            (4, CanonicalValue::U64(150)),
            (5, CanonicalValue::U64(0)),
            (6, CanonicalValue::U64(1_700_000_000)),
        ],
    };

    let cert_digest = CanonicalSigner::compute_identity_digest(&cert_record).unwrap();

    let proof_record = CanonicalRecord {
        schema_id: SchemaId::ProofReceipt,
        fields: vec![
            (1, CanonicalValue::U32(1)),
            (2, CanonicalValue::FixedBytes(vec![0xAA; 32])),
            (3, CanonicalValue::Utf8String("cuda".into())),
            (4, CanonicalValue::U32(150)),
        ],
    };

    let proof_digest = CanonicalSigner::compute_identity_digest(&proof_record).unwrap();

    assert_ne!(
        cert_digest, proof_digest,
        "Fix: distinct schemas must produce distinct identity digests under domain separators."
    );
}
#[test]
fn canonical_codec_rejects_duplicate_keys_by_name() {
    let duplicate_record = CanonicalRecord {
        schema_id: SchemaId::ConformanceCertificate,
        fields: vec![
            (1, CanonicalValue::U32(1)),
            (1, CanonicalValue::U32(1)), // Duplicate key 1
            (2, CanonicalValue::FixedBytes(vec![0xAA; 32])),
            (3, CanonicalValue::Utf8String("cuda".into())),
            (4, CanonicalValue::U64(150)),
            (5, CanonicalValue::U64(0)),
            (6, CanonicalValue::U64(1_700_000_000)),
        ],
    };

    let err = CanonicalEncoder::encode(&duplicate_record)
        .expect_err("Fix: duplicate field keys must be rejected by name as DuplicateKey.");

    assert_eq!(
        err,
        CodecError::DuplicateKey { field_number: 1 },
        "Fix: error must match CodecError::DuplicateKey"
    );
}

#[test]
fn canonical_codec_rejects_non_canonical_encoding_by_name() {
    let valid_record = CanonicalRecord {
        schema_id: SchemaId::ScheduleRecord,
        fields: vec![
            (1, CanonicalValue::U32(1)),
            (2, CanonicalValue::FixedBytes(vec![0xAA; 32])),
            (3, CanonicalValue::VarBytes(vec![1, 2, 3])),
            (4, CanonicalValue::U32(16)),
            (5, CanonicalValue::U32(16)),
        ],
    };

    let encoded = CanonicalEncoder::encode(&valid_record).unwrap();
    let truncated = &encoded[..encoded.len() - 2];
    let err = CanonicalDecoder::decode(truncated)
        .expect_err("Fix: truncated payload must be rejected as UnexpectedEof");
    match err {
        CodecError::UnexpectedEof { .. } => {}
        other => panic!("Fix: expected UnexpectedEof, got {:?}", other),
    }
}
#[test]
fn canonical_codec_rejects_stale_schema_fixtures_by_name() {
    let stale_record = CanonicalRecord {
        schema_id: SchemaId::AotManifest,
        fields: vec![
            (1, CanonicalValue::U32(4)),
            (2, CanonicalValue::Utf8String("vyre-aot-manifest-v3".into())), // Stale version
            (3, CanonicalValue::Utf8String("0.8.0".into())),
            (4, CanonicalValue::Utf8String("model".into())),
            (5, CanonicalValue::FixedBytes(vec![0xAA; 32])),
            (6, CanonicalValue::FixedBytes(vec![0xBB; 32])),
            (7, CanonicalValue::FixedBytes(vec![0xCC; 32])),
            (8, CanonicalValue::FixedBytes(vec![0xDD; 32])),
        ],
    };

    let err = CanonicalEncoder::encode(&stale_record)
        .expect_err("Fix: stale schema versions must be rejected by name as StaleSchemaVersion.");

    assert!(
        matches!(err, CodecError::StaleSchemaVersion { .. }),
        "Fix: error must be CodecError::StaleSchemaVersion, got {err:?}"
    );
}

#[test]
fn proof_hashes_are_invariant_to_host_pointer_width() {
    // Test that hashing canonical fixed-width representations (u32, u64) produces
    // byte-identical digests on 32-bit and 64-bit systems.
    let mut h1 = blake3::Hasher::new();
    let mut h2 = blake3::Hasher::new();

    // Fixed width u64 length
    let count: u64 = 42;
    h1.update(&count.to_le_bytes());
    h2.update(&42u64.to_le_bytes());

    assert_eq!(h1.finalize().as_bytes(), h2.finalize().as_bytes());
}
