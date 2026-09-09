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
