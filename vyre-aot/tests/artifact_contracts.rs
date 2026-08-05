//! AOT handle contracts for the canonical megakernel artifact envelope.

mod common;

use vyre_aot::{target_payload_format, Target};

#[test]
fn target_extensions_remain_stable() {
    assert_eq!(Target::Ptx.extension(), "secondary_text");
    assert_eq!(Target::SpirV.extension(), "spv");
}

#[test]
fn targets_round_trip_through_serde() {
    for target in [Target::Ptx, Target::SpirV] {
        let bytes = serde_json::to_vec(&target).expect("target must serialize");
        let decoded: Target = serde_json::from_slice(&bytes).expect("target must deserialize");
        assert_eq!(decoded, target);
    }
}

/// Regression: AOT size accounting must read the canonical neutral resource envelope.
#[test]
fn total_buffer_bytes_comes_from_canonical_resources() {
    let artifact = common::compiled_artifact();
    assert_eq!(artifact.total_buffer_bytes(), 256 * 4 + 64 * 4);
    assert_eq!(
        artifact.total_buffer_bytes(),
        artifact.envelope().neutral().resource_envelope().total_bytes
    );
}

/// Regression: selected target bytes must be read from the exact canonical attachment.
#[test]
fn compiled_artifact_selects_the_canonical_target_payload() {
    let artifact = common::compiled_artifact();
    let payload = artifact
        .target_payload()
        .expect("selected target attachment must exist");

    assert_eq!(payload.format(), &target_payload_format(Target::Ptx).unwrap());
    assert_eq!(payload.bytes(), b"target-payload-fixture");
    assert_eq!(payload.neutral_artifact(), artifact.envelope().neutral().digest());
    assert_eq!(payload.entries()[0].name, "main");
    assert_eq!(payload.entries()[0].resource_bindings[0].slot, 0);
    assert_eq!(payload.entries()[0].resource_bindings[1].slot, 1);
}
