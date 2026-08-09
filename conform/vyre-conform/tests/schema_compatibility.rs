//! Runner consumption contracts for schema-owned conformance artifacts.

use vyre_conform::{
    verify_cert_signature_hex, verify_structural, BundleCertError, CertificateError,
};
use vyre_conform_spec::{BundleCertificate, Certificate, CERTIFICATE_SCHEMA_VERSION};

#[test]
fn runner_accepts_the_supported_certificate_schema_before_field_validation() {
    let certificate = Certificate::new("op", "cpu-ref", "0.7.2", vec![]);

    let error = verify_structural(&certificate).expect_err("unsigned fields must still fail");
    assert!(matches!(error, CertificateError::UnsetField(_)));
}

#[test]
fn runner_rejects_per_operation_certificate_version_skew_explicitly() {
    let mut certificate = Certificate::new("op", "cpu-ref", "0.7.2", vec![]);
    certificate.version = "0.4.2".to_string();

    let error = verify_structural(&certificate).expect_err("version skew must fail first");
    assert!(matches!(
        error,
        CertificateError::UnsupportedSchemaVersion(version) if version == "0.4.2"
    ));
}

#[test]
fn runner_rejects_bundle_certificate_version_skew_before_signature_fields() {
    let certificate = BundleCertificate {
        version: "999999999999999999999999999999999999".to_string(),
        bundle_blake3: String::new(),
        corpus_blake3: String::new(),
        reference_output_blake3: String::new(),
        witness_count: 0,
        timestamp: String::new(),
        signature_ed25519: "TBD".to_string(),
        pubkey: "TBD".to_string(),
    };

    let error = verify_cert_signature_hex(&certificate, "")
        .expect_err("unsupported schema must fail before malformed signature fields");
    assert!(matches!(
        error,
        BundleCertError::UnsupportedSchemaVersion(version)
            if version == "999999999999999999999999999999999999"
    ));
}

#[test]
fn runner_supported_version_constant_matches_serialized_certificate() {
    let certificate = Certificate::new("op", "cpu-ref", "0.7.2", vec![]);
    let value = serde_json::to_value(certificate).expect("certificate must serialize");
    assert_eq!(value["version"].as_str(), Some(CERTIFICATE_SCHEMA_VERSION));
}
