//! Positive, negative, boundary, and adversarial contracts for conformance schemas.

use vyre_test_support::conform_spec::{
    ConformanceCase, ConformanceResult, ReplayCapsule, ReplayMinimization, ReplayMismatch,
    REPLAY_CAPSULE_SCHEMA_VERSION,
};

fn replay_capsule() -> ReplayCapsule {
    ReplayCapsule {
        schema_version: REPLAY_CAPSULE_SCHEMA_VERSION,
        op_id: "primitive.add.u32".to_string(),
        backend_id: "wgpu".to_string(),
        case_index: 0,
        replay_command: "cargo test -p vyre-test-support --features conform".to_string(),
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
        executor_id: "reference-oracle".to_string(),
        passed: true,
        message: "1 case passed".to_string(),
        replay_capsule: None,
    };

    let bytes = serde_json::to_vec(&result).expect("result must serialize");
    assert_eq!(
        bytes,
        br#"{"op_id":"primitive.add.u32","executor_id":"reference-oracle","passed":true,"message":"1 case passed"}"#
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
        executor_id: "wgpu".to_string(),
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
fn replay_version_boundaries_and_adversarial_values_fail_explicitly() {
    let valid = serde_json::to_value(replay_capsule()).expect("replay must serialize");
    for unsupported in [0_u64, 1, 3, u64::from(u32::MAX)] {
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
fn runtime_registry_entry_derivation_covers_all_conformance_schemas() {
    use vyre_spec::schema_registry::{SchemaId, SchemaRegistry};

    let registry_ids: Vec<SchemaId> = SchemaRegistry::all().iter().map(|d| d.id).collect();
    let required_conformance_schemas = [SchemaId::ReplayCapsule, SchemaId::ProofReceipt];

    for schema_id in required_conformance_schemas {
        assert!(
            registry_ids.contains(&schema_id),
            "Registry is missing required conformance schema definition for {:?}",
            schema_id
        );
        let def = SchemaRegistry::lookup(schema_id).expect("schema must be present in registry");
        assert!(
            def.validate_invariants(),
            "schema invariants violated for {:?}",
            schema_id
        );
    }
}
