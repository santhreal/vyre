//! Contracts for `vyre_conform_spec::protocol`.

use vyre_conform_spec::{
    hash_outputs, ulp_distance_f32, verify_receipts_for_certificate, CasePayload,
    CertificateRejection, DeviceLease, NumericalMismatch, NumericalPolicy, WorkerBudget,
    WorkerMode, WorkerReceipt, WorkerRequest, WorkerStatus,
};

#[test]
fn worker_budget_defaults_and_customization() {
    let default_budget = WorkerBudget::default_budget();
    assert_eq!(default_budget.wall_timeout_ms, 120_000);
    assert_eq!(default_budget.cpu_timeout_ms, 120_000);
    assert_eq!(default_budget.max_memory_bytes, 1024 * 1024 * 1024);
    assert_eq!(default_budget.max_artifact_bytes, 64 * 1024 * 1024);

    let customized = default_budget
        .with_wall_timeout_ms(5_000)
        .with_max_memory_bytes(512 * 1024 * 1024)
        .with_max_artifact_bytes(8 * 1024 * 1024)
        .with_max_output_bytes(4 * 1024 * 1024);
    assert_eq!(customized.wall_timeout_ms, 5_000);
    assert_eq!(customized.max_memory_bytes, 512 * 1024 * 1024);
    assert_eq!(customized.max_artifact_bytes, 8 * 1024 * 1024);
    assert_eq!(customized.max_output_bytes, 4 * 1024 * 1024);
}

#[test]
fn device_lease_serialization_round_trip() {
    let lease = DeviceLease::new(
        "lease-001",
        "cuda",
        0,
        "NVIDIA GeForce RTX 4090",
        "tok-abcdef123456",
    );
    let serialized = serde_json::to_string(&lease).expect("serialize lease");
    let deserialized: DeviceLease =
        serde_json::from_str(&serialized).expect("deserialize lease");
    assert_eq!(lease, deserialized);
}

#[test]
fn worker_request_construction_and_serialization() {
    let case = CasePayload::new(
        "op.test",
        vec![1, 2, 3],
        vec![vec![4, 5, 6]],
        NumericalPolicy::Exact,
        None,
    );
    let budget = WorkerBudget::default_budget();
    let req_ref = WorkerRequest::new_reference(
        "req-01",
        case.clone(),
        budget,
        "bin-comp-1",
        "bin-run-1",
        "env-1",
    );
    assert_eq!(req_ref.mode, WorkerMode::Reference);
    assert!(req_ref.device_lease.is_none());

    let lease = DeviceLease::new("lease-1", "cuda", 0, "GPU-0", "tok-1");
    let req_prod = WorkerRequest::new_production(
        "req-02",
        case,
        lease,
        budget,
        "bin-comp-1",
        "bin-run-1",
        Some("facts-1".to_string()),
        "env-1",
    );
    assert_eq!(req_prod.mode, WorkerMode::Production);
    assert!(req_prod.device_lease.is_some());

    let serialized = serde_json::to_string(&req_prod).expect("serialize");
    let deserialized: WorkerRequest = serde_json::from_str(&serialized).expect("deserialize");
    assert_eq!(req_prod, deserialized);
}

#[test]
fn content_addressed_case_payload_is_deterministic_and_distinct() {
    let wire1 = vec![1, 2, 3, 4];
    let inputs1 = vec![vec![10, 20], vec![30, 40]];
    let policy1 = NumericalPolicy::Exact;
    let case1_a = CasePayload::new("op.add", wire1.clone(), inputs1.clone(), policy1, None);
    let case1_b = CasePayload::new("op.add", wire1.clone(), inputs1.clone(), policy1, None);
    assert_eq!(case1_a.case_id, case1_b.case_id);

    let case2_diff_op = CasePayload::new("op.sub", wire1.clone(), inputs1.clone(), policy1, None);
    assert_ne!(case1_a.case_id, case2_diff_op.case_id);

    let case3_diff_inputs =
        CasePayload::new("op.add", wire1.clone(), vec![vec![10, 21], vec![30, 40]], policy1, None);
    assert_ne!(case1_a.case_id, case3_diff_inputs.case_id);

    let case4_diff_policy = CasePayload::new(
        "op.add",
        wire1.clone(),
        inputs1.clone(),
        NumericalPolicy::Float32Ulps { max_ulps: 4 },
        None,
    );
    assert_ne!(case1_a.case_id, case4_diff_policy.case_id);

    let case5_diff_schedule = CasePayload::new(
        "op.add",
        wire1,
        inputs1,
        policy1,
        Some("tiled".to_string()),
    );
    assert_ne!(case1_a.case_id, case5_diff_schedule.case_id);
}

#[test]
fn numerical_policy_exact_and_ulp_contracts() {
    let exact = NumericalPolicy::Exact;
    let out_a = vec![vec![1, 2, 3, 4]];
    let out_b = vec![vec![1, 2, 3, 4]];
    assert!(exact.check_agreement(&out_a, &out_b).is_ok());

    let out_diff_byte = vec![vec![1, 2, 3, 5]];
    assert_eq!(
        exact.check_agreement(&out_a, &out_diff_byte),
        Err(NumericalMismatch::ExactByteMismatch {
            buffer_idx: 0,
            byte_idx: 3,
            expected: 4,
            found: 5,
        })
    );

    let out_diff_count = vec![vec![1, 2, 3, 4], vec![5, 6]];
    assert_eq!(
        exact.check_agreement(&out_a, &out_diff_count),
        Err(NumericalMismatch::BufferCount {
            expected: 1,
            found: 2,
        })
    );

    let ulp_policy = NumericalPolicy::Float32Ulps { max_ulps: 2 };
    let f1: f32 = 1.0;
    let f1_bits = f1.to_bits();
    let f1_plus_1ulp_bits = f1_bits + 1;
    let f1_plus_3ulps_bits = f1_bits + 3;

    let buf_base = vec![f1_bits.to_le_bytes().to_vec()];
    let buf_1ulp = vec![f1_plus_1ulp_bits.to_le_bytes().to_vec()];
    let buf_3ulps = vec![f1_plus_3ulps_bits.to_le_bytes().to_vec()];

    assert!(ulp_policy.check_agreement(&buf_base, &buf_1ulp).is_ok());
    assert_eq!(
        ulp_policy.check_agreement(&buf_base, &buf_3ulps),
        Err(NumericalMismatch::LaneMismatch {
            buffer_idx: 0,
            lane_idx: 0,
            expected: f1_bits,
            found: f1_plus_3ulps_bits,
            distance: 3,
        })
    );

    // Test NaN / Inf distances return u32::MAX
    let nan_bits = f32::NAN.to_bits();
    assert_eq!(ulp_distance_f32(f1_bits, nan_bits), u32::MAX);
    let neg_f1_bits = (-1.0f32).to_bits();
    assert_eq!(ulp_distance_f32(f1_bits, neg_f1_bits), u32::MAX);
}

#[test]
fn worker_receipt_authentication_and_tamper_detection() {
    let secret = b"super-secret-worker-auth-key-123456";
    let outputs = vec![vec![10, 20, 30, 40]];
    let outputs_hash = hash_outputs(&outputs);

    let mut receipt = WorkerReceipt {
        receipt_version: WorkerReceipt::SCHEMA_VERSION,
        request_id: "req-001".to_string(),
        case_id: "case-abcdef".to_string(),
        mode: WorkerMode::Production,
        status: WorkerStatus::Success,
        outputs: outputs.clone(),
        outputs_blake3: outputs_hash,
        artifact_blake3: Some("art-111".to_string()),
        payload_blake3: Some("pay-222".to_string()),
        target_facts_blake3: Some("facts-333".to_string()),
        compiler_binary_blake3: "bin-compiler-444".to_string(),
        runner_binary_blake3: "bin-runner-555".to_string(),
        environment_blake3: "env-666".to_string(),
        device_lease_id: Some("lease-001".to_string()),
        elapsed_ms: 42,
        peak_memory_bytes: 1024 * 1024,
        auth_tag: String::new(),
    };
    receipt.auth_tag = receipt.compute_auth_tag(secret);

    assert!(receipt.verify_auth_tag(secret));
    assert!(!receipt.verify_auth_tag(b"wrong-secret"));

    // Tampering with any field invalidates auth_tag
    let mut tampered = receipt.clone();
    tampered.compiler_binary_blake3 = "tampered-binary".to_string();
    assert!(!tampered.verify_auth_tag(secret));

    let mut tampered_outputs = receipt.clone();
    tampered_outputs.outputs_blake3 = "tampered-outputs-hash".to_string();
    assert!(!tampered_outputs.verify_auth_tag(secret));
}

#[test]
fn certificate_verification_rejection_matrix() {
    let secret = b"shared-secret-for-testing-cert-issuance";
    let outputs = vec![vec![1, 2, 3, 4]];
    let outputs_hash = hash_outputs(&outputs);
    let policy = NumericalPolicy::Exact;

    let mut ref_receipt = WorkerReceipt {
        receipt_version: WorkerReceipt::SCHEMA_VERSION,
        request_id: "req-ref".to_string(),
        case_id: "case-alpha".to_string(),
        mode: WorkerMode::Reference,
        status: WorkerStatus::Success,
        outputs: outputs.clone(),
        outputs_blake3: outputs_hash.clone(),
        artifact_blake3: None,
        payload_blake3: None,
        target_facts_blake3: None,
        compiler_binary_blake3: "bin-compiler-1".to_string(),
        runner_binary_blake3: "bin-runner-1".to_string(),
        environment_blake3: "env-std".to_string(),
        device_lease_id: None,
        elapsed_ms: 10,
        peak_memory_bytes: 1024,
        auth_tag: String::new(),
    };
    ref_receipt.auth_tag = ref_receipt.compute_auth_tag(secret);

    let mut prod_receipt = WorkerReceipt {
        receipt_version: WorkerReceipt::SCHEMA_VERSION,
        request_id: "req-prod".to_string(),
        case_id: "case-alpha".to_string(),
        mode: WorkerMode::Production,
        status: WorkerStatus::Success,
        outputs: outputs.clone(),
        outputs_blake3: outputs_hash.clone(),
        artifact_blake3: Some("art-1".to_string()),
        payload_blake3: Some("pay-1".to_string()),
        target_facts_blake3: Some("facts-cuda-4090".to_string()),
        compiler_binary_blake3: "bin-compiler-1".to_string(),
        runner_binary_blake3: "bin-runner-1".to_string(),
        environment_blake3: "env-std".to_string(),
        device_lease_id: Some("lease-1".to_string()),
        elapsed_ms: 25,
        peak_memory_bytes: 4096,
        auth_tag: String::new(),
    };
    prod_receipt.auth_tag = prod_receipt.compute_auth_tag(secret);

    // Valid receipts pass
    assert!(verify_receipts_for_certificate(
        &ref_receipt,
        &prod_receipt,
        "bin-compiler-1",
        Some("facts-cuda-4090"),
        "env-std",
        &policy,
        secret
    )
    .is_ok());

    // Mismatched case ID
    let mut mismatched_case = prod_receipt.clone();
    mismatched_case.case_id = "case-beta".to_string();
    mismatched_case.auth_tag = mismatched_case.compute_auth_tag(secret);
    assert_eq!(
        verify_receipts_for_certificate(
            &ref_receipt,
            &mismatched_case,
            "bin-compiler-1",
            Some("facts-cuda-4090"),
            "env-std",
            &policy,
            secret
        ),
        Err(CertificateRejection::MismatchedCaseIdentity {
            ref_case: "case-alpha".to_string(),
            prod_case: "case-beta".to_string(),
        })
    );

    // Mismatched binary
    assert_eq!(
        verify_receipts_for_certificate(
            &ref_receipt,
            &prod_receipt,
            "expected-different-binary",
            Some("facts-cuda-4090"),
            "env-std",
            &policy,
            secret
        ),
        Err(CertificateRejection::MismatchedBinary {
            expected: "expected-different-binary".to_string(),
            found: "bin-compiler-1".to_string(),
        })
    );

    // Mismatched target facts
    assert_eq!(
        verify_receipts_for_certificate(
            &ref_receipt,
            &prod_receipt,
            "bin-compiler-1",
            Some("facts-expected-wgpu"),
            "env-std",
            &policy,
            secret
        ),
        Err(CertificateRejection::MismatchedTargetFacts {
            expected: "facts-expected-wgpu".to_string(),
            found: "facts-cuda-4090".to_string(),
        })
    );

    // Mismatched environment
    assert_eq!(
        verify_receipts_for_certificate(
            &ref_receipt,
            &prod_receipt,
            "bin-compiler-1",
            Some("facts-cuda-4090"),
            "env-expected-different",
            &policy,
            secret
        ),
        Err(CertificateRejection::MismatchedEnvironment {
            expected: "env-expected-different".to_string(),
            found: "env-std".to_string(),
        })
    );

    // Invalid receipt auth
    let mut bad_auth_prod = prod_receipt.clone();
    bad_auth_prod.auth_tag = "bad-auth-hex".to_string();
    assert_eq!(
        verify_receipts_for_certificate(
            &ref_receipt,
            &bad_auth_prod,
            "bin-compiler-1",
            Some("facts-cuda-4090"),
            "env-std",
            &policy,
            secret
        ),
        Err(CertificateRejection::InvalidReceiptAuth {
            mode: "production".to_string(),
        })
    );

    // Worker failure
    let mut failed_prod = prod_receipt.clone();
    failed_prod.status = WorkerStatus::Timeout {
        elapsed_ms: 120_000,
        budget_ms: 120_000,
    };
    failed_prod.auth_tag = failed_prod.compute_auth_tag(secret);
    assert_eq!(
        verify_receipts_for_certificate(
            &ref_receipt,
            &failed_prod,
            "bin-compiler-1",
            Some("facts-cuda-4090"),
            "env-std",
            &policy,
            secret
        ),
        Err(CertificateRejection::WorkerFailed {
            mode: "production".to_string(),
            status: WorkerStatus::Timeout {
                elapsed_ms: 120_000,
                budget_ms: 120_000,
            },
        })
    );
}
