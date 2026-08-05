//! Positive, negative, boundary, and adversarial contracts for conformance schemas.

use vyre_conform_spec::{
    BundleCertificate, Certificate, ConformanceCase, ConformanceResult, ReplayCapsule,
    ReplayMinimization, ReplayMismatch, CERTIFICATE_SCHEMA_VERSION,
    REPLAY_CAPSULE_SCHEMA_VERSION,
};

fn replay_capsule() -> ReplayCapsule {
    ReplayCapsule {
        schema_version: REPLAY_CAPSULE_SCHEMA_VERSION,
        op_id: "primitive.add.u32".to_string(),
        backend_id: "cpu-ref".to_string(),
        case_index: 0,
        replay_command: "vyre-conform dispatch --backend cpu-ref --ops primitive.add.u32"
            .to_string(),
        program_blake3: "01".repeat(32),
        witness_input_blake3: "02".repeat(32),
        reference_output_blake3: "03".repeat(32),
        backend_output_blake3: "04".repeat(32),
        witness_input_buffers_hex: vec!["00000000".to_string()],
        reference_output_buffers_hex: vec!["01000000".to_string()],
        backend_output_buffers_hex: vec!["02000000".to_string()],
        witness_input_count: 1,
        reference_output_count: 1,
        backend_output_count: 1,
        first_mismatch: ReplayMismatch {
            kind: "byte".to_string(),
            output_index: Some(0),
            byte_index: Some(0),
            reference_len: Some(4),
            backend_len: Some(4),
            reference_byte: Some(1),
            backend_byte: Some(2),
        },
        minimization: ReplayMinimization {
            strategy: "single_witness_case".to_string(),
            original_case_count: 8,
            retained_case_count: 1,
        },
    }
}

#[test]
fn case_round_trip_retains_exact_bytes() {
    let case = ConformanceCase {
        name: "edge".to_string(),
        inputs: vec![vec![0, 255], Vec::new()],
    };

    let bytes = serde_json::to_vec(&case).expect("case must serialize");
    assert_eq!(bytes, br#"{"name":"edge","inputs":[[0,255],[]]}"#);
    let decoded: ConformanceCase = serde_json::from_slice(&bytes).expect("case must deserialize");
    assert_eq!(decoded, case);
    assert_eq!(serde_json::to_vec(&decoded).unwrap(), bytes);
}

#[test]
fn result_round_trip_retains_exact_bytes() {
    let result = ConformanceResult {
        op_id: "primitive.add.u32".to_string(),
        backend_id: "cpu-ref".to_string(),
        passed: true,
        message: "1 case passed".to_string(),
        replay_capsule: None,
    };

    let bytes = serde_json::to_vec(&result).expect("result must serialize");
    assert_eq!(
        bytes,
        br#"{"op_id":"primitive.add.u32","backend_id":"cpu-ref","passed":true,"message":"1 case passed"}"#
    );
    let decoded: ConformanceResult =
        serde_json::from_slice(&bytes).expect("result must deserialize");
    assert_eq!(decoded, result);
    assert_eq!(serde_json::to_vec(&decoded).unwrap(), bytes);
}

#[test]
fn result_with_replay_capsule_round_trip_retains_exact_bytes() {
    let result = ConformanceResult {
        op_id: "primitive.add.u32".to_string(),
        backend_id: "wgpu".to_string(),
        passed: false,
        message: "mismatch".to_string(),
        replay_capsule: Some(replay_capsule()),
    };

    let bytes = serde_json::to_vec(&result).expect("failure result must serialize");
    let decoded: ConformanceResult =
        serde_json::from_slice(&bytes).expect("failure result must deserialize");
    assert_eq!(decoded, result);
    assert_eq!(serde_json::to_vec(&decoded).unwrap(), bytes);
}

#[test]
fn certificate_round_trip_retains_established_bytes() {
    let certificate = Certificate::new(
        "primitive.add.u32",
        "cpu-ref",
        "0.7.2",
        vec!["Associative".to_string()],
    );

    let bytes = serde_json::to_vec(&certificate).expect("certificate must serialize");
    assert_eq!(
        bytes,
        br#"{"version":"0.4.1","op_id":"primitive.add.u32","wire_format_version":1,"program_blake3":"TBD","witness_set_blake3":"TBD","backend_id":"cpu-ref","backend_version":"0.7.2","laws_verified":["Associative"],"timestamp":"1970-01-01T00:00:00Z","signature_ed25519":"TBD","pubkey":"TBD"}"#
    );
    let decoded: Certificate =
        serde_json::from_slice(&bytes).expect("certificate must deserialize");
    assert_eq!(decoded, certificate);
    assert_eq!(serde_json::to_vec(&decoded).unwrap(), bytes);
}

#[test]
fn bundle_certificate_round_trip_retains_established_bytes() {
    let certificate = BundleCertificate {
        version: CERTIFICATE_SCHEMA_VERSION.to_string(),
        bundle_blake3: "01".repeat(32),
        corpus_blake3: "02".repeat(32),
        reference_output_blake3: "03".repeat(32),
        witness_count: 0,
        timestamp: "1970-01-01T00:00:00Z".to_string(),
        signature_ed25519: "TBD".to_string(),
        pubkey: "TBD".to_string(),
    };

    let bytes = serde_json::to_vec(&certificate).expect("bundle certificate must serialize");
    let decoded: BundleCertificate =
        serde_json::from_slice(&bytes).expect("bundle certificate must deserialize");
    assert_eq!(decoded, certificate);
    assert_eq!(serde_json::to_vec(&decoded).unwrap(), bytes);
}

#[test]
fn certificate_version_skew_fails_explicitly() {
    let json = br#"{"version":"0.4.2","op_id":"op","wire_format_version":1,"program_blake3":"TBD","witness_set_blake3":"TBD","backend_id":"ref","backend_version":"0","laws_verified":[],"timestamp":"1970-01-01T00:00:00Z","signature_ed25519":"TBD","pubkey":"TBD"}"#;

    let error = serde_json::from_slice::<Certificate>(json).expect_err("version skew must fail");
    let message = error.to_string();
    assert!(message.contains("unsupported certificate schema version `0.4.2`"));
    assert!(message.contains(CERTIFICATE_SCHEMA_VERSION));
}

#[test]
fn bundle_certificate_version_skew_fails_explicitly() {
    let json = br#"{"version":"0.3.9","bundle_blake3":"","corpus_blake3":"","reference_output_blake3":"","witness_count":0,"timestamp":"","signature_ed25519":"","pubkey":""}"#;

    let error = serde_json::from_slice::<BundleCertificate>(json)
        .expect_err("bundle certificate version skew must fail");
    assert!(error
        .to_string()
        .contains("unsupported bundle certificate schema version `0.3.9`"));
}

#[test]
fn replay_version_boundaries_and_adversarial_values_fail_explicitly() {
    let valid = serde_json::to_value(replay_capsule()).expect("replay must serialize");
    for unsupported in [0_u64, 2, u64::from(u32::MAX)] {
        let mut skewed = valid.clone();
        skewed["schema_version"] = unsupported.into();
        let error = serde_json::from_value::<ReplayCapsule>(skewed)
            .expect_err("unsupported replay version must fail");
        assert!(error
            .to_string()
            .contains("unsupported replay capsule schema version"));
    }
}

#[test]
fn adversarial_certificate_version_string_is_not_accepted_as_compatible() {
    let mut value = serde_json::to_value(Certificate::new("op", "ref", "0", vec![]))
        .expect("certificate must serialize");
    value["version"] = "0.4.1\u{0}suffix".into();

    let error = serde_json::from_value::<Certificate>(value)
        .expect_err("an embedded-NUL version must not compare equal");
    assert!(error.to_string().contains("unsupported certificate schema version"));
}
