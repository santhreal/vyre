//! Contract tests for restored scan database wire header and decode budget types.

use vyre_foundation::serial::wire::decode::{
    validate_scan_construct_decode_budget, validate_scan_database_decode_budget,
    ScanConstructDecodeBudget, ScanConstructDecodeShape, ScanDatabaseDecodeBudget,
    ScanDatabaseDecodeShape,
};
use vyre_foundation::serial::wire::encode::{
    decode_compatible_scan_database_header, decode_scan_database_header,
    decode_scan_database_header_with_compatibility, encode_scan_database_header,
    ScanDatabaseCompatibilityRecord, ScanDatabaseHeader, ScanDatabaseMode,
    ScanDatabaseReaderCompatibility, ScanDatabaseSectionHeader, ScanDatabaseSectionKind,
    UnsupportedScanFeature, SCAN_DATABASE_HEADER_MAGIC, SCAN_DATABASE_HEADER_VERSION,
};

#[test]
fn scan_database_header_round_trip() {
    assert_eq!(SCAN_DATABASE_HEADER_VERSION, 1);
    let header = ScanDatabaseHeader {
        pattern_set_digest: [0x42; 32],
        compiler_version: "vyre-scan-0.7.2".to_string(),
        mode: ScanDatabaseMode::Streaming,
        table_sections: vec![
            ScanDatabaseSectionHeader {
                kind: ScanDatabaseSectionKind::LiteralTable,
                offset: 0,
                byte_len: 128,
                section_digest: 0x11223344,
            },
            ScanDatabaseSectionHeader {
                kind: ScanDatabaseSectionKind::AutomataTable,
                offset: 128,
                byte_len: 512,
                section_digest: 0x55667788,
            },
        ],
        unsupported_features: vec![UnsupportedScanFeature {
            pattern_index: 3,
            feature: "lookaround assertion".to_string(),
        }],
        compatibility: ScanDatabaseCompatibilityRecord {
            construct_tier_digest: 0xaabb,
            dialect_digest: 0xccdd,
            reader_compatibility: ScanDatabaseReaderCompatibility::Compatible,
        },
    };

    let encoded = encode_scan_database_header(&header).expect("encode header");
    assert!(encoded.starts_with(SCAN_DATABASE_HEADER_MAGIC));

    let decoded = decode_scan_database_header(&encoded).expect("decode header");
    assert_eq!(decoded, header);
}

#[test]
fn scan_database_compatibility_decode_check() {
    let header = ScanDatabaseHeader {
        pattern_set_digest: [0x11; 32],
        compiler_version: "0.7.2".to_string(),
        mode: ScanDatabaseMode::Block,
        table_sections: vec![],
        unsupported_features: vec![],
        compatibility: ScanDatabaseCompatibilityRecord {
            construct_tier_digest: 0x1234,
            dialect_digest: 0x5678,
            reader_compatibility: ScanDatabaseReaderCompatibility::Compatible,
        },
    };

    let encoded = encode_scan_database_header(&header).expect("encode header");

    // Compiler version and mode match succeeds
    let decoded =
        decode_compatible_scan_database_header(&encoded, "0.7.2", ScanDatabaseMode::Block)
            .expect("compatible version and mode");
    assert_eq!(decoded, header);

    // Mismatched compiler version fails
    let err = decode_compatible_scan_database_header(&encoded, "0.7.1", ScanDatabaseMode::Block)
        .unwrap_err();
    assert!(err.contains("Fix:"));

    // Full compatibility check with tier, dialect, and reader class succeeds
    let decoded_full = decode_scan_database_header_with_compatibility(
        &encoded,
        "0.7.2",
        ScanDatabaseMode::Block,
        0x1234,
        0x5678,
        &[ScanDatabaseReaderCompatibility::Compatible],
    )
    .expect("compatible tier and dialect");
    assert_eq!(decoded_full, header);

    // Mismatched dialect fails when exact compatibility is required
    let err = decode_scan_database_header_with_compatibility(
        &encoded,
        "0.7.2",
        ScanDatabaseMode::Block,
        0x1234,
        0x9999,
        &[ScanDatabaseReaderCompatibility::Compatible],
    )
    .unwrap_err();
    assert!(err.contains("Fix:"));
}

#[test]
fn scan_database_decode_budget_enforcement() {
    let header = ScanDatabaseHeader {
        pattern_set_digest: [0u8; 32],
        compiler_version: "test".to_string(),
        mode: ScanDatabaseMode::Block,
        table_sections: vec![ScanDatabaseSectionHeader {
            kind: ScanDatabaseSectionKind::AutomataTable,
            offset: 0,
            byte_len: 2048,
            section_digest: 1,
        }],
        unsupported_features: vec![],
        compatibility: ScanDatabaseCompatibilityRecord {
            construct_tier_digest: 1,
            dialect_digest: 2,
            reader_compatibility: ScanDatabaseReaderCompatibility::Compatible,
        },
    };

    let shape = ScanDatabaseDecodeShape {
        state_count: 50,
        transition_count: 100,
        verifier_fragment_bytes: 0,
    };

    let strict_budget = ScanDatabaseDecodeBudget {
        max_total_table_bytes: 1024, // header has 2048
        ..ScanDatabaseDecodeBudget::default()
    };

    let err = validate_scan_database_decode_budget(&header, shape, strict_budget).unwrap_err();
    assert!(err.to_string().contains("exceed budget"));

    let construct_shape = ScanConstructDecodeShape {
        construct_id: "unicode_classes",
        states: 10,
        transitions: 20,
        literal_bytes: 0,
        capture_slots: 0,
        unicode_table_bytes: 4096,
        verifier_fragment_bytes: 0,
    };

    let construct_budget = ScanConstructDecodeBudget {
        construct_id: "unicode_classes",
        max_states: 10,
        max_transitions: 20,
        max_literal_bytes: 0,
        max_capture_slots: 0,
        max_unicode_table_bytes: 1024,
        max_verifier_fragment_bytes: 0,
    };

    let construct_err =
        validate_scan_construct_decode_budget(construct_shape, construct_budget).unwrap_err();
    assert!(construct_err.to_string().contains("exceeds budget"));
}
