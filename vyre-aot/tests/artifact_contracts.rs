//! AOT contracts for canonical compiler artifact envelopes.

mod common;

use vyre_aot::Target;

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
    let envelope = common::compiled_artifact();
    assert_eq!(
        envelope.neutral().resource_envelope().total_bytes,
        256 * 4 + 64 * 4
    );
}

/// Regression: selected target bytes must be read from the exact canonical attachment.
#[test]
fn envelope_selects_the_canonical_target_payload() {
    let envelope = common::compiled_artifact();
    let payload = envelope
        .target_payloads()
        .iter()
        .find(|payload| payload.format().identity() == Target::Ptx.aot_target_id())
        .expect("selected target attachment must exist");

    assert_eq!(payload.format().identity(), "secondary_text");
    assert_eq!(payload.format().version(), 1);
    let modules = vyre_megakernel::target::TargetModuleBundle::from_bytes(payload.bytes())
        .expect("target module bundle must decode");
    assert_eq!(modules.modules[0].bytes, b"target-payload-fixture");
    assert_eq!(payload.neutral_artifact(), envelope.neutral().digest());
    assert_eq!(payload.entries()[0].name, "main");
    assert_eq!(payload.entries()[0].resource_bindings[0].slot, 0);
    assert_eq!(payload.entries()[0].resource_bindings[1].slot, 1);
}
