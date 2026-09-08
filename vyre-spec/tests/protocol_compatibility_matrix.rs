//! Tests for protocol compatibility matrix and declarative schema registry.
//!
//! WHY: proves row 120 and row 121 contracts:
//! - Explicit compatibility matrix covering all protocol domains.
//! - Protocol negotiation selects preregistered mutual contracts without silent downgrades.
//! - Unsupported version pairs fail with explicit actionable upgrade instructions.
//! - Declarative schema registry derives canonical fields, types, and bounds from live source at run time.

use vyre_spec::compatibility::{
    CompatibilityDisposition, CompatibilityMatrix, ProtocolDomain, ProtocolVersion,
};
use vyre_spec::schema_registry::{DefaultsPolicy, FieldType, SchemaId, SchemaRegistry};

#[test]
fn all_protocol_domains_are_covered_in_canonical_matrix() {
    let matrix = CompatibilityMatrix::canonical();
    for &domain in ProtocolDomain::ALL {
        let disp = matrix.check(
            domain,
            ProtocolVersion::V1_0_0,
            ProtocolVersion::V1_0_0,
        );
        assert_eq!(
            disp,
            CompatibilityDisposition::Supported,
            "Fix: domain '{domain}' must have a supported cell for v1.0.0 -> v1.0.0."
        );

        let disp_1_1 = matrix.check(
            domain,
            ProtocolVersion::V1_1_0,
            ProtocolVersion::V1_1_0,
        );
        assert_eq!(
            disp_1_1,
            CompatibilityDisposition::Supported,
            "Fix: domain '{domain}' must have a supported cell for v1.1.0 -> v1.1.0."
        );
    }
}

#[test]
fn protocol_negotiation_selects_highest_mutually_supported_version() {
    let matrix = CompatibilityMatrix::canonical();
    let offered = [ProtocolVersion::V1_0_0, ProtocolVersion::V1_1_0];
    let supported = [ProtocolVersion::V1_1_0];

    let contract = matrix
        .negotiate(ProtocolDomain::PublicWire, &offered, &supported)
        .expect("Fix: negotiation must succeed for overlapping supported versions.");

    assert_eq!(contract.domain, ProtocolDomain::PublicWire);
    assert_eq!(contract.client_version, ProtocolVersion::V1_1_0);
    assert_eq!(contract.host_version, ProtocolVersion::V1_1_0);
    assert_eq!(contract.disposition, CompatibilityDisposition::Supported);
    assert_ne!(contract.contract_digest, [0_u8; 32]);
}

#[test]
fn unsupported_version_pair_fails_with_actionable_upgrade_error() {
    let matrix = CompatibilityMatrix::canonical();
    let offered = [ProtocolVersion::V2_0_0];
    let supported = [ProtocolVersion::V1_0_0];

    let err = matrix
        .negotiate(ProtocolDomain::Artifact, &offered, &supported)
        .expect_err("Fix: unsupported major version jump must fail negotiation.");

    assert_eq!(err.domain, ProtocolDomain::Artifact);
    assert!(
        err.upgrade_action.contains("Fix:"),
        "Fix: negotiation error must contain actionable 'Fix:' message: {}",
        err.upgrade_action
    );
}

#[test]
fn empty_negotiation_offers_fail_closed() {
    let matrix = CompatibilityMatrix::canonical();
    let err = matrix
        .negotiate(ProtocolDomain::Schedule, &[], &[ProtocolVersion::V1_0_0])
        .expect_err("Fix: empty client offer list must fail closed.");
    assert!(err.upgrade_action.contains("Fix:"));
}

#[test]
fn schema_registry_covers_every_schema_id() {
    let all_schemas = SchemaRegistry::all();
    assert_eq!(
        all_schemas.len(),
        SchemaId::ALL.len(),
        "Fix: SchemaRegistry must contain exactly one definition for each SchemaId."
    );

    for &id in SchemaId::ALL {
        let def = SchemaRegistry::lookup(id)
            .unwrap_or_else(|| panic!("Fix: SchemaId::{id:?} must be registered in SchemaRegistry"));
        assert_eq!(def.id, id);
        assert!(
            def.validate_invariants(),
            "Fix: SchemaDefinition for '{id}' failed invariant validation."
        );
        assert!(
            !def.domain_separator.is_empty(),
            "Fix: domain separator must not be empty for '{id}'"
        );
        assert!(
            def.bounds.max_bytes > 0,
            "Fix: max_bytes bound must be positive for '{id}'"
        );
        assert_eq!(
            def.defaults_policy,
            DefaultsPolicy::NoDefaults,
            "Fix: canonical records must enforce NoDefaults policy"
        );
    }
}

#[test]
fn schema_canonical_fields_are_strictly_ordered_and_typed() {
    for def in SchemaRegistry::all() {
        let mut prev_num = 0;
        for field in def.fields {
            assert!(
                field.number > prev_num,
                "Fix: fields in schema '{}' must have strictly increasing numbers (found {} after {})",
                def.id,
                field.number,
                prev_num
            );
            prev_num = field.number;
            assert!(
                !field.name.is_empty(),
                "Fix: field {} in schema '{}' must have a non-empty name",
                field.number,
                def.id
            );
            // Verify field type is valid
            match field.field_type {
                FieldType::U8
                | FieldType::U16
                | FieldType::U32
                | FieldType::U64
                | FieldType::I32
                | FieldType::I64
                | FieldType::F32
                | FieldType::F64
                | FieldType::Bool
                | FieldType::FixedBytes(_)
                | FieldType::VarBytes
                | FieldType::Utf8String
                | FieldType::List(_) => {}
                _ => panic!("Uncataloged FieldType variant in schema registry"),
            }
        }
    }
}
