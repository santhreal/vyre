//! Proves that backend support is a production-path certificate join.
//!
//! WHY: An operation cannot be claimed supported by registration or tier alone.
//! Support requires a join of seven concrete production-path facts:
//! 1. Validation
//! 2. Emission
//! 3. Native compilation
//! 4. Materialization
//! 5. Hostile bindings
//! 6. Device execution
//! 7. Independent oracle agreement
//!
//! Missing or failing ANY single stage means the operation is unsupported.

use vyre_driver::support_certificate::{
    FactStatus, ProductionPathStage, SupportCertificate, SupportCertificateRegistry, SupportStatus,
    SUPPORT_CERTIFICATE_SCHEMA,
};
use vyre_foundation::ir::OpId;

fn sample_fully_proven_certificate(backend: &str, target: &str, op: &str) -> SupportCertificate {
    let mut cert = SupportCertificate::new(backend, target, OpId::from(op));
    for stage in ProductionPathStage::ALL {
        cert = cert.with_proven_stage(
            stage,
            format!("proof-digest-{}-{}", backend, stage.name()),
            Some(format!("verified on {backend} production path")),
        );
    }
    cert
}

#[test]
fn support_certificate_schema_is_versioned() {
    assert_eq!(SUPPORT_CERTIFICATE_SCHEMA, 1);
}

#[test]
fn empty_certificate_is_unsupported_naming_validation() {
    let cert = SupportCertificate::new("cuda", "cuda", OpId::from("vyre-libs::math::add"));
    let status = cert.evaluate();
    assert!(!status.is_supported());
    match status {
        SupportStatus::Unsupported {
            missing_stage,
            reason,
        } => {
            assert_eq!(missing_stage, ProductionPathStage::Validation);
            assert!(
                reason.contains("validation"),
                "reason must name unproven validation: {reason}"
            );
        }
        SupportStatus::Supported { .. } => panic!("empty certificate must not be supported"),
    }
}

#[test]
fn fully_proven_certificate_is_supported() {
    let cert = sample_fully_proven_certificate("cuda", "cuda", "vyre-libs::math::add");
    let status = cert.evaluate();
    assert!(status.is_supported());
    match status {
        SupportStatus::Supported { certificate_digest } => {
            assert!(!certificate_digest.is_empty());
        }
        SupportStatus::Unsupported { .. } => panic!("fully proven certificate must be supported"),
    }
}

#[test]
fn missing_any_single_production_path_stage_fails_support_join() {
    // Mutation test: for each of the 7 stages, build a certificate where all OTHER
    // 6 stages are proven, but this stage is unproven. The join MUST reject support
    // and name the exact missing stage.
    for missing_stage in ProductionPathStage::ALL {
        let mut cert =
            SupportCertificate::new("cuda", "cuda", OpId::from("vyre-libs::math::test_op"));
        for stage in ProductionPathStage::ALL {
            if stage != missing_stage {
                cert = cert.with_proven_stage(stage, format!("proof-{}", stage.name()), None);
            }
        }
        let status = cert.evaluate();
        assert!(
            !status.is_supported(),
            "certificate missing stage `{}` must not be supported",
            missing_stage.name()
        );
        match status {
            SupportStatus::Unsupported {
                missing_stage: reported_stage,
                reason,
            } => {
                assert_eq!(
                    reported_stage, missing_stage,
                    "reported missing stage must match omitted stage"
                );
                assert!(
                    reason.contains(missing_stage.name()),
                    "reason must cite omitted stage `{}`: {reason}",
                    missing_stage.name()
                );
            }
            SupportStatus::Supported { .. } => {
                panic!(
                    "certificate missing `{}` was incorrectly supported",
                    missing_stage.name()
                );
            }
        }
    }
}

#[test]
fn failed_stage_reports_failure_reason() {
    for failed_stage in ProductionPathStage::ALL {
        let mut cert = SupportCertificate::new(
            "wgpu",
            "wgpu",
            OpId::from("vyre-libs::security::taint_pollution"),
        );
        for stage in ProductionPathStage::ALL {
            if stage == failed_stage {
                cert = cert.with_failed_stage(
                    stage,
                    format!("device error: `{}` failed verification", stage.name()),
                );
            } else {
                cert = cert.with_proven_stage(stage, format!("proof-{}", stage.name()), None);
            }
        }
        let status = cert.evaluate();
        assert!(!status.is_supported());
        match status {
            SupportStatus::Unsupported {
                missing_stage,
                reason,
            } => {
                assert_eq!(missing_stage, failed_stage);
                assert!(
                    reason.contains(&format!("Stage `{}` failed", failed_stage.name())),
                    "reason must state failure: {reason}"
                );
            }
            SupportStatus::Supported { .. } => {
                panic!("certificate with failed stage must not be supported")
            }
        }
    }
}

#[test]
fn support_certificate_registry_filters_unsupported_operations() {
    let registry = SupportCertificateRegistry::global();
    let op_supported = OpId::from("vyre-libs::bitset::and");
    let op_unsupported = OpId::from("vyre-libs::security::grid_sync_test");

    let cert_good =
        sample_fully_proven_certificate("mock_backend", "mock_target", op_supported.as_ref());
    let mut cert_bad =
        sample_fully_proven_certificate("mock_backend", "mock_target", op_unsupported.as_ref());
    cert_bad.facts.insert(
        ProductionPathStage::DeviceExecution,
        vyre_driver::support_certificate::ProductionPathFact {
            stage: ProductionPathStage::DeviceExecution,
            status: FactStatus::Failed {
                reason: "GridSync not supported in mock runtime".to_string(),
            },
        },
    );

    registry.register_certificate(cert_good);
    registry.register_certificate(cert_bad);

    assert!(registry
        .evaluate_support("mock_backend", &op_supported)
        .is_supported());
    assert!(!registry
        .evaluate_support("mock_backend", &op_unsupported)
        .is_supported());

    let supported_set = registry.supported_ops_for_backend("mock_backend");
    assert!(supported_set.contains(&op_supported));
    assert!(!supported_set.contains(&op_unsupported));
}

#[test]
fn runtime_derived_operation_and_target_sets_require_certificate_join() {
    use vyre_foundation::operation::OperationRegistry;

    let op_registry = OperationRegistry::global();
    let backends = vyre_driver::registered_backends().expect("Fix: backend registry must load");

    assert!(
        op_registry.iter().len() > 0,
        "Fix: OperationRegistry must contain registered operations"
    );
    assert!(
        !backends.is_empty(),
        "Fix: registered_backends must contain registered backends"
    );

    let cert_registry = SupportCertificateRegistry::global();

    // For every backend and every operation, evaluate support without a certificate
    // and verify it fails closed (Unsupported).
    for backend in backends {
        for op in op_registry.iter() {
            let op_id = OpId::from(op.id);
            // Query an unregistered combination
            let status =
                cert_registry.evaluate_support(&format!("unregistered_{}", backend.id), &op_id);
            assert!(
                !status.is_supported(),
                "Fix: unregistered backend `{}` and op `{}` must evaluate to unsupported",
                backend.id,
                op_id
            );
        }
    }
}

#[test]
fn device_execution_stage_cannot_be_fabricated_and_requires_hardware_proof() {
    let mut cert = SupportCertificate::new("cuda", "cuda", OpId::from("vyre-libs::math::fma"));

    // Record host-side stages 1-5 as proven
    cert = cert.with_proven_stage(ProductionPathStage::Validation, "val-digest", None);
    cert = cert.with_proven_stage(ProductionPathStage::Emission, "emit-digest", None);
    cert = cert.with_proven_stage(ProductionPathStage::NativeCompilation, "nvrtc-digest", None);
    cert = cert.with_proven_stage(ProductionPathStage::Materialization, "module-digest", None);
    cert = cert.with_proven_stage(ProductionPathStage::HostileBindings, "hostile-digest", None);

    // Leave DeviceExecution and OracleAgreement as Unproven (as when no physical GPU is present)
    let status = cert.evaluate();
    assert!(
        !status.is_supported(),
        "Fix: certificate without device execution proof must not be supported"
    );

    match status {
        SupportStatus::Unsupported {
            missing_stage,
            reason,
        } => {
            assert_eq!(
                missing_stage,
                ProductionPathStage::DeviceExecution,
                "Fix: evaluation must report DeviceExecution as the first missing stage"
            );
            assert!(
                reason.contains("device_execution"),
                "Fix: reason must explicitly state device_execution is unproven: {reason}"
            );
        }
        SupportStatus::Supported { .. } => {
            panic!("unproven device execution must not be supported")
        }
    }
}
