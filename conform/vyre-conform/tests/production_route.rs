//! Production conformance step bounds and worker process isolation contracts.
//!
//! A coordinator schedules content-addressed cases into disposable worker processes
//! with explicit CPU, GPU, memory, artifact, output, and wall budgets. Workers hold
//! one device lease, emit one authenticated report, and are terminated and replaced
//! on timeout, panic, driver loss, leak, or protocol violation. No timed-out thread
//! survives a case.

use std::time::{Duration, Instant};

use vyre::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_conform::{DeviceLeaseManager, WorkerCoordinator};
use vyre_conform_spec::{
    hash_outputs, verify_receipts_for_certificate, CasePayload, CertificateRejection,
    NumericalMismatch, NumericalPolicy, WorkerBudget, WorkerMode, WorkerReceipt, WorkerStatus,
};

/// Operation identity used in bounded step tests.
const LIFECYCLE_OP_ID: &str = "vyre-conform::production_route::session_lifecycle";

fn make_infinite_loop_program() -> Program {
    // 4-billion-iteration loop that never completes in milliseconds
    Program::wrapped(
        vec![
            BufferDecl::read("in", 0, DataType::U32).with_count(1),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![
            Node::Loop {
                var: "i".into(),
                from: Expr::u32(0),
                to: Expr::u32(u32::MAX),
                body: vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
            },
            Node::Return,
        ],
    )
    .with_entry_op_id(LIFECYCLE_OP_ID)
}

fn make_identity_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("in", 0, DataType::U32).with_count(1),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![
            Node::store("out", Expr::u32(0), Expr::load("in", Expr::u32(0))),
            Node::Return,
        ],
    )
    .with_entry_op_id("vyre-conform::production_route::identity")
}

#[test]
fn timed_out_worker_is_reaped_and_next_case_observes_clean_worker() {
    let coordinator = WorkerCoordinator::new();
    let infinite_program = make_infinite_loop_program();
    let wire_bytes = infinite_program
        .to_wire()
        .expect("infinite program wire encode");

    let hanging_case = CasePayload::new(
        LIFECYCLE_OP_ID,
        wire_bytes,
        vec![42u32.to_le_bytes().to_vec()],
        NumericalPolicy::Exact,
        None,
    );

    // Set tight 100ms wall timeout budget
    let budget = WorkerBudget::fast_test_budget(100);
    let started = Instant::now();

    let timeout_receipt = coordinator.execute_reference(&hanging_case, Some(budget));
    let elapsed = started.elapsed();

    // Assert the bound: must return close to the ceiling, never hang
    assert!(
        elapsed >= Duration::from_millis(80),
        "Fix: bounded execution must wait for the budget; elapsed was {elapsed:?}"
    );
    assert!(
        elapsed < Duration::from_secs(10),
        "Fix: bounded execution must return once deadline elapses; elapsed was {elapsed:?}"
    );

    match timeout_receipt.status {
        WorkerStatus::Timeout {
            elapsed_ms,
            budget_ms,
        } => {
            assert_eq!(budget_ms, 100);
            assert!(elapsed_ms >= 80);
        }
        other => panic!("expected WorkerStatus::Timeout, got {other:?}"),
    }

    // Now execute a clean identity case through the same coordinator.
    // Proves that the timed-out worker process was reaped and that the next
    // case observes a completely clean, responsive worker!
    let identity_program = make_identity_program();
    let id_wire = identity_program
        .to_wire()
        .expect("identity program wire encode");

    let clean_case = CasePayload::new(
        "vyre-conform::production_route::identity",
        id_wire,
        vec![99u32.to_le_bytes().to_vec()],
        NumericalPolicy::Exact,
        None,
    );

    let clean_receipt = coordinator.execute_reference(&clean_case, Some(WorkerBudget::default_budget()));
    assert_eq!(clean_receipt.status, WorkerStatus::Success);
    assert_eq!(
        clean_receipt.outputs,
        vec![99u32.to_le_bytes().to_vec()],
        "Fix: clean worker must execute subsequent case correctly"
    );
}

#[test]
fn contaminated_or_panicking_worker_cannot_influence_later_cases() {
    let coordinator = WorkerCoordinator::new();

    // Malformed program wire bytes that trigger decode failure in worker
    let bad_case = CasePayload::new(
        "vyre-conform::production_route::bad",
        vec![0xff, 0xff, 0xff, 0xff],
        vec![vec![0]],
        NumericalPolicy::Exact,
        None,
    );

    let bad_receipt = coordinator.execute_reference(&bad_case, None);
    assert!(
        matches!(bad_receipt.status, WorkerStatus::ExecutionError { .. }),
        "Fix: bad case must report ExecutionError, got {:?}",
        bad_receipt.status
    );

    // Next case on the coordinator must run cleanly and succeed
    let identity_program = make_identity_program();
    let id_wire = identity_program
        .to_wire()
        .expect("identity wire encode");

    let clean_case = CasePayload::new(
        "vyre-conform::production_route::identity",
        id_wire,
        vec![77u32.to_le_bytes().to_vec()],
        NumericalPolicy::Exact,
        None,
    );

    let clean_receipt = coordinator.execute_reference(&clean_case, None);
    assert_eq!(clean_receipt.status, WorkerStatus::Success);
    assert_eq!(clean_receipt.outputs, vec![77u32.to_le_bytes().to_vec()]);
}

#[test]
fn certificate_naming_mismatched_receipts_binaries_or_target_facts_is_rejected_by_name() {
    let secret = b"test-verification-secret-key-000";
    let outputs = vec![vec![1, 2, 3, 4]];
    let outputs_hash = hash_outputs(&outputs);
    let policy = NumericalPolicy::Exact;

    let mut ref_receipt = WorkerReceipt {
        receipt_version: WorkerReceipt::SCHEMA_VERSION,
        request_id: "req-ref-1".to_string(),
        case_id: "case-001".to_string(),
        mode: WorkerMode::Reference,
        status: WorkerStatus::Success,
        outputs: outputs.clone(),
        outputs_blake3: outputs_hash.clone(),
        artifact_blake3: None,
        payload_blake3: None,
        target_facts_blake3: None,
        compiler_binary_blake3: "bin-compiler-current".to_string(),
        runner_binary_blake3: "bin-runner-current".to_string(),
        environment_blake3: "env-linux-x86_64".to_string(),
        device_lease_id: None,
        elapsed_ms: 15,
        peak_memory_bytes: 1024,
        auth_tag: String::new(),
    };
    ref_receipt.auth_tag = ref_receipt.compute_auth_tag(secret);

    let mut prod_receipt = WorkerReceipt {
        receipt_version: WorkerReceipt::SCHEMA_VERSION,
        request_id: "req-prod-1".to_string(),
        case_id: "case-001".to_string(),
        mode: WorkerMode::Production,
        status: WorkerStatus::Success,
        outputs: outputs.clone(),
        outputs_blake3: outputs_hash.clone(),
        artifact_blake3: Some("art-digest-01".to_string()),
        payload_blake3: Some("pay-digest-01".to_string()),
        target_facts_blake3: Some("facts-cuda-ad102".to_string()),
        compiler_binary_blake3: "bin-compiler-current".to_string(),
        runner_binary_blake3: "bin-runner-current".to_string(),
        environment_blake3: "env-linux-x86_64".to_string(),
        device_lease_id: Some("lease-cuda-000001".to_string()),
        elapsed_ms: 30,
        peak_memory_bytes: 4096,
        auth_tag: String::new(),
    };
    prod_receipt.auth_tag = prod_receipt.compute_auth_tag(secret);

    // 1. Valid receipts verify cleanly
    assert!(verify_receipts_for_certificate(
        &ref_receipt,
        &prod_receipt,
        "bin-compiler-current",
        Some("facts-cuda-ad102"),
        "env-linux-x86_64",
        &policy,
        secret
    )
    .is_ok());

    // 2. Mismatched case identity rejected by name
    let mut bad_case_prod = prod_receipt.clone();
    bad_case_prod.case_id = "case-002-stale".to_string();
    bad_case_prod.auth_tag = bad_case_prod.compute_auth_tag(secret);
    assert_eq!(
        verify_receipts_for_certificate(
            &ref_receipt,
            &bad_case_prod,
            "bin-compiler-current",
            Some("facts-cuda-ad102"),
            "env-linux-x86_64",
            &policy,
            secret
        ),
        Err(CertificateRejection::MismatchedCaseIdentity {
            ref_case: "case-001".to_string(),
            prod_case: "case-002-stale".to_string(),
        })
    );

    // 3. Mismatched binary rejected by name
    assert_eq!(
        verify_receipts_for_certificate(
            &ref_receipt,
            &prod_receipt,
            "bin-compiler-different",
            Some("facts-cuda-ad102"),
            "env-linux-x86_64",
            &policy,
            secret
        ),
        Err(CertificateRejection::MismatchedBinary {
            expected: "bin-compiler-different".to_string(),
            found: "bin-compiler-current".to_string(),
        })
    );

    // 4. Mismatched target facts rejected by name
    assert_eq!(
        verify_receipts_for_certificate(
            &ref_receipt,
            &prod_receipt,
            "bin-compiler-current",
            Some("facts-wgpu-vulkan"),
            "env-linux-x86_64",
            &policy,
            secret
        ),
        Err(CertificateRejection::MismatchedTargetFacts {
            expected: "facts-wgpu-vulkan".to_string(),
            found: "facts-cuda-ad102".to_string(),
        })
    );

    // 5. Mismatched environment rejected by name
    assert_eq!(
        verify_receipts_for_certificate(
            &ref_receipt,
            &prod_receipt,
            "bin-compiler-current",
            Some("facts-cuda-ad102"),
            "env-macos-aarch64",
            &policy,
            secret
        ),
        Err(CertificateRejection::MismatchedEnvironment {
            expected: "env-macos-aarch64".to_string(),
            found: "env-linux-x86_64".to_string(),
        })
    );

    // 6. Divergent outputs rejected under exact policy
    let mut divergent_prod = prod_receipt.clone();
    divergent_prod.outputs = vec![vec![1, 2, 3, 5]];
    divergent_prod.outputs_blake3 = hash_outputs(&divergent_prod.outputs);
    divergent_prod.auth_tag = divergent_prod.compute_auth_tag(secret);
    assert_eq!(
        verify_receipts_for_certificate(
            &ref_receipt,
            &divergent_prod,
            "bin-compiler-current",
            Some("facts-cuda-ad102"),
            "env-linux-x86_64",
            &policy,
            secret
        ),
        Err(CertificateRejection::NumericalPolicyViolation(
            NumericalMismatch::ExactByteMismatch {
                buffer_idx: 0,
                byte_idx: 3,
                expected: 4,
                found: 5,
            }
        ))
    );
}

#[test]
fn device_lease_manager_acquire_release_and_quarantine() {
    let manager = DeviceLeaseManager::new();
    let lease1 = manager.acquire_lease("cuda").expect("acquire lease 1");
    assert_eq!(lease1.backend_id, "cuda");
    assert_eq!(lease1.device_ordinal, 0);

    let lease2 = manager.acquire_lease("cuda").expect("acquire lease 2");
    assert_ne!(lease1.lease_id, lease2.lease_id);

    manager.release_lease(&lease1);
    manager.quarantine_lease(&lease2, "worker leaked memory");
    assert!(manager.is_quarantined(&lease2.lease_id));
}
