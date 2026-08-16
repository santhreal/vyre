//! Regression coverage for AOT packaging of canonical compiler envelopes.

mod fixture_target;

use vyre_aot::{package_artifact, TargetId};

#[test]
fn package_and_read_round_trip_the_canonical_envelope() {
    let envelope = fixture_target::compiled_artifact();
    let neutral_digest = envelope.neutral().digest();
    let payload_digest = envelope.target_payloads()[0].digest();
    let target_bytes = envelope.target_payloads()[0].bytes().to_vec();
    let directory = tempfile::tempdir().expect("temporary package directory must exist");

    package_artifact(
        directory.path(),
        &envelope,
        fixture_target::fixture_target(),
        &[9; 32],
        "canonical-package",
        "regression fixture",
    )
    .expect("canonical package must write");
    let (manifest, decoded) =
        vyre_aot::read_bundle_artifact(directory.path()).expect("canonical package must read");

    assert_eq!(manifest.schema, "vyre-aot-manifest-v4");
    assert_eq!(manifest.artifact_name, "canonical-package");
    assert_eq!(manifest.target, fixture_target::fixture_target());
    assert_eq!(decoded.neutral().digest(), neutral_digest);
    assert_eq!(decoded.target_payloads()[0].digest(), payload_digest);
    assert_eq!(decoded.target_payloads()[0].bytes(), target_bytes);
    assert_eq!(decoded.neutral().resources()[0].value.0, 0);
    assert_eq!(decoded.neutral().resources()[1].value.0, 1);
    assert_eq!(decoded.neutral().geometry()[0].workgroup_size, [64, 1, 1]);
}

/// Persisted target identities are owned opaque data and survive JSON round trips.
#[test]
fn manifest_target_identity_is_opaque_and_owned() {
    let target = TargetId::from_owned("external-package-target".to_string())
        .expect("target identity must validate");
    let bytes = serde_json::to_vec(&target).expect("target identity must serialize");
    let decoded: TargetId =
        serde_json::from_slice(&bytes).expect("target identity must deserialize");
    assert_eq!(decoded, target);
}
