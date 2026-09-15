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
    export_schema_json, BoundedDecoder, CanonicalDigest, CanonicalSchemaVersion, DigestAlgorithm,
    SchemaAuthority, SchemaAuthorityError, SchemaDescriptor, SchemaId,
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

/// One payload declaring `version`, otherwise shaped like a real record.
fn payload_at(version: &str) -> Vec<u8> {
    format!(r#"{{"schema_version":"{version}","op_id":"primitive.add.u32","count":42}}"#)
        .into_bytes()
}

/// WHY: closes the class "a record written by another build is decoded and
/// served". `BoundedDecoder` documents version enforcement, the exported JSON
/// Schema declares `schema_version` required and pinned, and
/// `SchemaAuthorityError::IncompatibleVersion` exists for the refusal, but the
/// decoder checked only the payload's size. Every persisted, cached, signed
/// and transmitted schema in this authority was therefore readable across a
/// version boundary, which is how a fixed bug returns from a cache after the
/// fix ships.
///
/// Every case runs for every `SchemaId`, taken from `SchemaId::ALL` at run
/// time, so a new schema is covered without editing this test.
///
/// What this does NOT catch: no registered schema currently sets
/// `min_compatible_version` above `x.0.0`, so the floor is exercised through
/// the same-major comparison rather than by a payload below it.
#[test]
fn a_record_this_build_cannot_read_is_refused_for_every_schema() {
    for &id in SchemaId::ALL {
        let desc = SchemaAuthority::descriptor_for(id);
        let current = desc.current_version;

        let accepted: TestRecord = BoundedDecoder::decode_json(id, &payload_at(&current.to_string()))
            .unwrap_or_else(|error| {
                panic!("{id:?} must read a record at its own current version {current}: {error}")
            });
        assert_eq!(accepted.schema_version, current.to_string());

        for newer in [
            CanonicalSchemaVersion::new(current.major, current.minor + 1, 0),
            CanonicalSchemaVersion::new(current.major + 1, 0, 0),
        ] {
            let refused: Result<TestRecord, _> =
                BoundedDecoder::decode_json(id, &payload_at(&newer.to_string()));
            assert!(
                matches!(
                    refused,
                    Err(SchemaAuthorityError::IncompatibleVersion { found, .. }) if found == newer
                ),
                "{id:?} accepted a record at {newer}, which is newer than the {current} it reads: \
                 {refused:?}"
            );
        }

        if current.major > 1 {
            let older = CanonicalSchemaVersion::new(current.major - 1, 0, 0);
            let refused: Result<TestRecord, _> =
                BoundedDecoder::decode_json(id, &payload_at(&older.to_string()));
            assert!(
                matches!(
                    refused,
                    Err(SchemaAuthorityError::IncompatibleVersion { found, .. }) if found == older
                ),
                "{id:?} accepted a record at the retired major version {older}: {refused:?}"
            );
        }
    }
}

/// WHY: a payload with no `schema_version`, or one that is not a triple, states
/// nothing about which contract it was written against. Serving it would mean
/// deciding on its behalf, so it is refused rather than assumed current.
#[test]
fn a_record_that_states_no_usable_version_is_refused() {
    let unversioned = br#"{"op_id":"primitive.add.u32","count":42}"#;
    let refused: Result<TestRecord, _> =
        BoundedDecoder::decode_json(SchemaId::ConformanceCertificate, unversioned);
    assert!(
        matches!(refused, Err(SchemaAuthorityError::DecodeFailure { .. })),
        "a payload with no schema_version must be refused: {refused:?}"
    );

    for malformed in ["2", "2.0", "2.0.0.1", "two.0.0", "", "v2.0.0"] {
        let refused: Result<TestRecord, _> =
            BoundedDecoder::decode_json(SchemaId::ConformanceCertificate, &payload_at(malformed));
        assert!(
            matches!(refused, Err(SchemaAuthorityError::DecodeFailure { .. })),
            "`{malformed}` is not a Major.Minor.Patch triple and must be refused: {refused:?}"
        );
    }
}

/// A descriptor reading `floor..=current`, for exercising both bounds.
fn descriptor_reading(
    floor: CanonicalSchemaVersion,
    current: CanonicalSchemaVersion,
) -> SchemaDescriptor {
    SchemaDescriptor {
        schema_id: SchemaId::ConformanceCertificate,
        canonical_name: "test descriptor",
        current_version: current,
        min_compatible_version: floor,
        max_payload_bytes: 1024,
        digest_algorithm: DigestAlgorithm::Blake3_256,
    }
}

/// WHY: the admission decision has two independent bounds and a record outside
/// either one is read as fields it does not have. No registered schema sets a
/// floor above `x.0.0`, so a case routed through `descriptor_for` can only ever
/// exercise the ceiling; dropping the floor comparison entirely would leave
/// every such test green. These build the descriptor directly so both bounds
/// fail independently.
#[test]
fn admission_honours_both_the_floor_and_the_ceiling() {
    let desc = descriptor_reading(
        CanonicalSchemaVersion::new(2, 2, 0),
        CanonicalSchemaVersion::new(2, 5, 0),
    );

    for readable in [
        CanonicalSchemaVersion::new(2, 2, 0),
        CanonicalSchemaVersion::new(2, 3, 7),
        CanonicalSchemaVersion::new(2, 5, 0),
    ] {
        assert!(
            desc.admits(&readable),
            "{readable} is within 2.2.0..=2.5.0 and must be admitted"
        );
    }

    for retired in [
        CanonicalSchemaVersion::new(2, 1, 9),
        CanonicalSchemaVersion::new(2, 0, 0),
    ] {
        assert!(
            !desc.admits(&retired),
            "{retired} is below the 2.2.0 floor and must be refused"
        );
    }

    for unreadable in [
        CanonicalSchemaVersion::new(2, 6, 0),
        CanonicalSchemaVersion::new(3, 0, 0),
        CanonicalSchemaVersion::new(1, 9, 9),
    ] {
        assert!(
            !desc.admits(&unreadable),
            "{unreadable} is outside 2.2.0..=2.5.0 and must be refused"
        );
    }

    // The patch component never decides admission: it records a fix that
    // changed no field.
    let pinned = descriptor_reading(
        CanonicalSchemaVersion::new(1, 0, 0),
        CanonicalSchemaVersion::new(1, 0, 3),
    );
    assert!(pinned.admits(&CanonicalSchemaVersion::new(1, 0, 9)));
}

/// WHY: `SchemaDescriptor::admits` compares the floor's minor alone, which is
/// correct only while every descriptor states one major for both its floor and
/// its current version. A descriptor whose floor named an older major would
/// make the floor silently stop deciding and widen what the decoder accepts, so
/// the precondition is checked here rather than re-asserted defensively on
/// every decode.
///
/// The descriptors come from `SchemaId::ALL` at run time, so a new schema is
/// held to this without editing the test.
#[test]
fn every_descriptor_states_a_readable_range() {
    for &id in SchemaId::ALL {
        let desc = SchemaAuthority::descriptor_for(id);
        let floor = desc.min_compatible_version;
        let current = desc.current_version;
        assert_eq!(
            floor.major, current.major,
            "{id:?} reads major {} but retires from major {}, so the floor cannot decide any \
             record the ceiling admits",
            current.major, floor.major
        );
        assert!(
            floor <= current,
            "{id:?} has a floor of {floor} above its current {current}, which reads nothing"
        );
        assert!(
            desc.admits(&current),
            "{id:?} must admit a record at its own current version {current}"
        );
        assert!(
            !desc.admits(&CanonicalSchemaVersion::new(current.major + 1, 0, 0)),
            "{id:?} must refuse the next major"
        );
    }
}
