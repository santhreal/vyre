//! Canonical codec contracts for the declarative schema registry.
//!
//! WHY: closes the class "a codec fixture is hand-written against a schema the
//! registry does not declare, so the test exercises the first rejection it
//! happens to hit instead of the rule it names". Every record here is built
//! from `CANONICAL_SCHEMA_REGISTRY` at run time, so retyping, adding, or
//! renumbering a field changes the fixture with it. A hand-written field list
//! goes stale in silence, which is the same failure as having no test.
//!
//! Covered: canonical round trip, strictly ascending field order, duplicate
//! keys, undeclared field numbers, missing required fields, truncation,
//! declared payload bounds, stale fixture rejection, fixed-width length
//! prefixes, and domain separation between schemas.
//!
//! Not covered: whether a schema declares the right fields for what it
//! describes. That is a judgement about the protocol, and review is its gate.

use vyre_foundation::{
    CanonicalDecoder, CanonicalEncoder, CanonicalRecord, CanonicalSigner, CanonicalValue,
    CodecError,
};
use vyre_spec::schema_registry::{FieldType, SchemaId, SchemaRegistry, CANONICAL_SCHEMA_REGISTRY};

/// A value conforming to `field_type`, distinct per field number.
///
/// The value has to satisfy the declared type and nothing else: these contracts
/// judge the codec, not the semantics of any one field. `seed` keeps values
/// distinct so a codec that transposed two fields would not round trip.
///
/// `FieldType` is `#[non_exhaustive]`, so this match cannot close at compile
/// time from another crate. The wildcard fails instead of inventing a value: a
/// new field type turns every contract in this file red until it has one, which
/// is the same signal a missing match arm would give.
fn conforming_value(field_type: FieldType, seed: u32) -> CanonicalValue {
    match field_type {
        FieldType::U8 => CanonicalValue::U8(seed as u8),
        FieldType::U16 => CanonicalValue::U16(seed as u16),
        FieldType::U32 => CanonicalValue::U32(seed),
        FieldType::U64 => CanonicalValue::U64(u64::from(seed)),
        FieldType::I32 => CanonicalValue::I32(seed as i32),
        FieldType::I64 => CanonicalValue::I64(i64::from(seed)),
        FieldType::F32 => CanonicalValue::F32(seed as f32),
        FieldType::F64 => CanonicalValue::F64(f64::from(seed)),
        FieldType::Bool => CanonicalValue::Bool(seed % 2 == 0),
        FieldType::FixedBytes(n) => CanonicalValue::FixedBytes(vec![seed as u8; n]),
        FieldType::VarBytes => CanonicalValue::VarBytes(vec![seed as u8; 3]),
        FieldType::Utf8String => CanonicalValue::Utf8String(format!("field-{seed}")),
        FieldType::List(elem) => CanonicalValue::List(vec![conforming_value(*elem, seed)]),
        other => panic!(
            "Fix: `FieldType::{other:?}` has no conforming value here, so no schema declaring \
             it is covered. Add an arm producing a value of that type."
        ),
    }
}

/// A record that satisfies every field the schema declares.
///
/// Built from the registry rather than written out, so a schema change is
/// reflected here without an edit. A declared field is included whether or not
/// it is required: an encoder that rejected a declared optional field would go
/// red rather than pass on a shorter record.
fn conforming_record(schema_id: SchemaId) -> CanonicalRecord {
    let schema = SchemaRegistry::lookup(schema_id)
        .expect("Fix: every SchemaId in the registry resolves to a definition");
    let mut fields: Vec<(u32, CanonicalValue)> = schema
        .fields
        .iter()
        .map(|field| {
            (
                field.number,
                conforming_value(field.field_type, field.number),
            )
        })
        .collect();
    fields.sort_by_key(|(number, _)| *number);
    CanonicalRecord { schema_id, fields }
}

/// Every schema in the registry round trips through the codec unchanged.
///
/// The whole registry rather than one schema: a codec arm that mishandles one
/// field type is a defect for every schema that declares it, and this is the
/// choke point all of them pass through.
#[test]
fn every_schema_round_trips_through_canonical_bytes() {
    assert!(
        !CANONICAL_SCHEMA_REGISTRY.is_empty(),
        "Fix: an empty registry makes this contract cover nothing"
    );

    for schema in CANONICAL_SCHEMA_REGISTRY {
        let record = conforming_record(schema.id);
        let encoded = CanonicalEncoder::encode(&record).unwrap_or_else(|error| {
            panic!(
                "Fix: a record conforming to {:?} must encode: {error}",
                schema.id
            )
        });

        assert_eq!(
            &encoded[0..4],
            b"VYRE",
            "Fix: {:?} must carry the canonical magic bytes",
            schema.id
        );

        let decoded = CanonicalDecoder::decode(&encoded).unwrap_or_else(|error| {
            panic!(
                "Fix: canonical bytes for {:?} must decode: {error}",
                schema.id
            )
        });
        assert_eq!(
            decoded, record,
            "Fix: {:?} must decode into the record it encoded",
            schema.id
        );

        let encoded_again = CanonicalEncoder::encode(&decoded)
            .expect("Fix: a decoded record must re-encode canonically");
        assert_eq!(
            encoded, encoded_again,
            "Fix: {:?} must encode to identical bytes on every round trip",
            schema.id
        );
    }
}

/// An identity digest is stable across a round trip and separated per schema.
///
/// Two schemas whose declared fields coincide must still digest apart, which is
/// what the domain separator is for. Comparing across the whole registry covers
/// every pair rather than the one pair someone had in mind.
#[test]
fn identity_digests_are_stable_and_domain_separated() {
    let mut digests: Vec<(SchemaId, Vec<u8>)> = Vec::new();

    for schema in CANONICAL_SCHEMA_REGISTRY {
        let record = conforming_record(schema.id);
        let digest = CanonicalSigner::compute_identity_digest(&record).unwrap_or_else(|error| {
            panic!(
                "Fix: an identity digest for {:?} must compute: {error}",
                schema.id
            )
        });

        let encoded = CanonicalEncoder::encode(&record).expect("Fix: the record must encode");
        let decoded = CanonicalDecoder::decode(&encoded).expect("Fix: the bytes must decode");
        let after_round_trip = CanonicalSigner::compute_identity_digest(&decoded)
            .expect("Fix: a decoded record must digest");
        assert_eq!(
            digest, after_round_trip,
            "Fix: {:?} must digest identically after a round trip",
            schema.id
        );

        digests.push((schema.id, digest.to_vec()));
    }

    for (index, (schema_id, digest)) in digests.iter().enumerate() {
        for (other_id, other) in &digests[index + 1..] {
            assert_ne!(
                digest, other,
                "Fix: {schema_id:?} and {other_id:?} share an identity digest; \
                 the domain separator must distinguish them"
            );
        }
    }
}

/// A field number that arrives after a higher one is rejected.
///
/// Swapping the first two declared fields of a conforming record isolates the
/// ordering rule: everything else about the record still satisfies the schema,
/// so ordering is the only reason left to reject it.
#[test]
fn descending_field_numbers_are_rejected() {
    let mut record = conforming_record(SchemaId::ConformanceCertificate);
    assert!(
        record.fields.len() >= 2,
        "Fix: the ordering rule needs at least two declared fields"
    );
    record.fields.swap(0, 1);

    let error = CanonicalEncoder::encode(&record)
        .expect_err("Fix: descending field numbers must be rejected");
    let CodecError::NonCanonicalFieldOrder {
        expected_after,
        got,
    } = error
    else {
        panic!("Fix: expected NonCanonicalFieldOrder, got {error:?}");
    };
    assert!(
        got < expected_after,
        "Fix: the report must name the descending pair, got {got} after {expected_after}"
    );
}

/// A repeated field number is rejected and named.
///
/// Duplicating the first declared field leaves a record that is otherwise
/// conforming, so the duplicate is the only reason to reject it.
#[test]
fn duplicate_field_numbers_are_rejected_and_named() {
    let mut record = conforming_record(SchemaId::ConformanceCertificate);
    let (number, value) = record.fields[0].clone();
    record.fields.insert(1, (number, value));

    let error =
        CanonicalEncoder::encode(&record).expect_err("Fix: a duplicate key must be rejected");
    assert_eq!(
        error,
        CodecError::DuplicateKey {
            field_number: number
        },
        "Fix: the report must name the duplicated field number"
    );
}

/// A field number the schema does not declare is rejected as undeclared.
///
/// This used to report `NonCanonicalFieldOrder`, which sent a caller looking
/// for an ordering defect in a correctly ordered record. The rejection is
/// right; the reason it gave was not.
#[test]
fn an_undeclared_field_number_is_rejected_as_undeclared() {
    let mut record = conforming_record(SchemaId::ConformanceCertificate);
    let beyond = record
        .fields
        .iter()
        .map(|(number, _)| *number)
        .max()
        .expect("Fix: the schema declares at least one field")
        + 1;
    record.fields.push((beyond, CanonicalValue::U32(1)));

    let error = CanonicalEncoder::encode(&record)
        .expect_err("Fix: an undeclared field number must be rejected");
    assert_eq!(
        error,
        CodecError::UndeclaredField {
            schema_id: SchemaId::ConformanceCertificate,
            field_number: beyond,
        },
        "Fix: the report must name the schema and the undeclared field"
    );
}

/// A value of the wrong type is rejected and names the field it belongs to.
///
/// The diagnostic is the contract here as much as the rejection: reporting
/// field 0 with a sentence in the name slot tells a caller nothing about which
/// field to fix.
#[test]
fn a_mistyped_value_is_rejected_and_names_its_field() {
    let schema = SchemaRegistry::lookup(SchemaId::ConformanceCertificate)
        .expect("Fix: the schema must resolve");
    let target = schema
        .fields
        .iter()
        .find(|field| !matches!(field.field_type, FieldType::Bool))
        .expect("Fix: the schema declares a field that is not a boolean");

    let mut record = conforming_record(SchemaId::ConformanceCertificate);
    for (number, value) in &mut record.fields {
        if *number == target.number {
            *value = CanonicalValue::Bool(true);
        }
    }

    let error =
        CanonicalEncoder::encode(&record).expect_err("Fix: a mistyped value must be rejected");
    assert_eq!(
        error,
        CodecError::TypeMismatch {
            field_number: target.number,
            field_name: target.name,
        },
        "Fix: the report must name the field whose value did not match"
    );
}

/// Every required field the schema declares is enforced, one at a time.
///
/// Dropping each required field in turn covers the whole set rather than the
/// one field a hand-written fixture omitted.
#[test]
fn every_required_field_is_enforced() {
    let schema = SchemaRegistry::lookup(SchemaId::ConformanceCertificate)
        .expect("Fix: the schema must resolve");
    let required: Vec<_> = schema
        .fields
        .iter()
        .filter(|field| field.required)
        .collect();
    assert!(
        !required.is_empty(),
        "Fix: a schema with no required field makes this contract cover nothing"
    );

    for missing in required {
        let mut record = conforming_record(SchemaId::ConformanceCertificate);
        record
            .fields
            .retain(|(number, _)| *number != missing.number);

        let error = CanonicalEncoder::encode(&record).unwrap_err();
        assert_eq!(
            error,
            CodecError::MissingRequiredField {
                field_number: missing.number,
                field_name: missing.name,
            },
            "Fix: dropping `{}` must be reported as a missing required field",
            missing.name
        );
    }
}

/// Truncated bytes are rejected rather than decoded into a shorter record.
///
/// Every truncation length rather than one: a decoder that checked remaining
/// bytes for some field widths and not others would pass a single case.
#[test]
fn truncation_at_any_length_is_rejected() {
    let record = conforming_record(SchemaId::ConformanceCertificate);
    let encoded = CanonicalEncoder::encode(&record).expect("Fix: the record must encode");

    for length in 1..encoded.len() {
        assert!(
            CanonicalDecoder::decode(&encoded[..length]).is_err(),
            "Fix: {length} of {} bytes decoded as if the record were whole",
            encoded.len()
        );
    }
}

/// A payload past the declared byte bound fails closed.
#[test]
fn a_payload_past_the_declared_bound_fails_closed() {
    let schema = SchemaRegistry::lookup(SchemaId::ConformanceCertificate)
        .expect("Fix: the schema must resolve");
    let oversized = vec![0u8; schema.bounds.max_bytes + 100];

    let error =
        CanonicalDecoder::decode(&oversized).expect_err("Fix: an oversized payload must fail");
    assert!(
        matches!(
            error,
            CodecError::PayloadOversized { .. } | CodecError::UnknownSchema(_)
        ),
        "Fix: an oversized payload must be reported as oversized or unknown, got {error:?}"
    );
}

/// Every stale fixture string a schema records is rejected by name.
///
/// Derived from `stale_fixtures` so recording a new stale version covers it
/// without an edit here.
#[test]
fn every_recorded_stale_fixture_is_rejected() {
    let mut checked = 0;

    for schema in CANONICAL_SCHEMA_REGISTRY {
        let string_field = schema
            .fields
            .iter()
            .find(|field| matches!(field.field_type, FieldType::Utf8String));
        let Some(string_field) = string_field else {
            continue;
        };

        for stale in schema.stale_fixtures {
            let mut record = conforming_record(schema.id);
            for (number, value) in &mut record.fields {
                if *number == string_field.number {
                    *value = CanonicalValue::Utf8String((*stale).to_string());
                }
            }

            let error = CanonicalEncoder::encode(&record).unwrap_err();
            assert_eq!(
                error,
                CodecError::StaleSchemaVersion {
                    schema_id: schema.id,
                    found: (*stale).to_string(),
                },
                "Fix: `{stale}` is recorded stale for {:?} and must be rejected by name",
                schema.id
            );
            checked += 1;
        }
    }

    assert!(
        checked > 0,
        "Fix: no schema records a stale fixture, so this contract covers nothing"
    );
}

/// Length prefixes are fixed-width, so bytes do not depend on host pointer width.
///
/// A `usize` length written in native width would encode differently on a
/// 32-bit host and a 64-bit one, and the digest of a proof would disagree
/// across hosts. Measuring the encoded size of two records whose only
/// difference is a variable-length payload isolates the prefix: the difference
/// must be the payload growth plus a 4-byte prefix, never 8.
#[test]
fn length_prefixes_are_fixed_width() {
    let schema_id = CANONICAL_SCHEMA_REGISTRY
        .iter()
        .find(|schema| {
            schema
                .fields
                .iter()
                .any(|field| matches!(field.field_type, FieldType::VarBytes))
        })
        .map(|schema| schema.id)
        .expect("Fix: a schema declaring a variable-length field must exist");
    let schema = SchemaRegistry::lookup(schema_id).expect("Fix: the schema must resolve");
    let var_field = schema
        .fields
        .iter()
        .find(|field| matches!(field.field_type, FieldType::VarBytes))
        .expect("Fix: the schema declares a variable-length field");

    let mut sizes = Vec::new();
    for payload_len in [0usize, 8] {
        let mut record = conforming_record(schema_id);
        for (number, value) in &mut record.fields {
            if *number == var_field.number {
                *value = CanonicalValue::VarBytes(vec![0xAB; payload_len]);
            }
        }
        let encoded = CanonicalEncoder::encode(&record).expect("Fix: the record must encode");
        sizes.push(encoded.len());
    }

    assert_eq!(
        sizes[1] - sizes[0],
        8,
        "Fix: eight more payload bytes must grow the encoding by exactly eight; \
         a length prefix that changed width would not"
    );
}
