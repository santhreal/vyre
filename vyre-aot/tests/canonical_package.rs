//! Regression coverage for AOT packaging of canonical compiler envelopes.

mod common;

use vyre_aot::{package_artifact, read_bundle_artifact, Target};

fn ptx_payload(envelope: &vyre_aot::ArtifactEnvelope) -> &vyre_aot::TargetPayload {
    envelope
        .target_payloads()
        .iter()
        .find(|payload| payload.format().identity() == Target::Ptx.aot_target_id())
        .expect("fixture must carry one PTX payload")
}

/// AOT package/read preserves canonical IDs, target bytes, and manifest identities.
#[test]
fn package_and_read_round_trip_the_canonical_envelope() {
    let envelope = common::compiled_artifact();
    let neutral_digest = envelope.neutral().digest();
    let payload_digest = ptx_payload(&envelope).digest();
    let target_bytes = ptx_payload(&envelope).bytes().to_vec();
    let directory = tempfile::tempdir().expect("temporary package directory must exist");

    package_artifact(
        directory.path(),
        &envelope,
        Target::Ptx,
        &[9; 32],
        "canonical-package",
        "regression fixture",
    )
    .expect("canonical package must write");
    let (manifest, decoded) =
        read_bundle_artifact(directory.path()).expect("canonical package must read");

    assert_eq!(manifest.schema, "vyre-aot-manifest-v3");
    assert_eq!(manifest.artifact_name, "canonical-package");
    assert_eq!(manifest.target, Target::Ptx);
    assert_eq!(decoded.neutral().digest(), neutral_digest);
    assert_eq!(ptx_payload(&decoded).digest(), payload_digest);
    assert_eq!(ptx_payload(&decoded).bytes(), target_bytes);
    assert_eq!(decoded.neutral().resources()[0].value.0, 0);
    assert_eq!(decoded.neutral().resources()[1].value.0, 1);
    assert_eq!(decoded.neutral().geometry()[0].workgroup_size, [64, 1, 1]);
}
