//! AOT contracts for canonical compiler artifact envelopes.

mod fixture_target;

use vyre_aot::TargetId;

#[test]
fn target_ids_round_trip_through_serde_without_static_leaks() {
    let target = TargetId::from_owned("external-aot-target".to_string())
        .expect("opaque target identity must validate");
    let bytes = serde_json::to_vec(&target).expect("target must serialize");
    let decoded: TargetId = serde_json::from_slice(&bytes).expect("target must deserialize");
    assert_eq!(decoded, target);
    assert_eq!(decoded.as_str(), "external-aot-target");
}

/// WHY: persisted target identities must reject malformed routing keys before
/// registry lookup or cache association.
#[test]
fn target_ids_reject_invalid_deserialized_spellings() {
    for encoded in [r#""""#, r#"" target""#, r#""target ""#] {
        let error = serde_json::from_str::<TargetId>(encoded)
            .expect_err("invalid persisted target identity must fail");
        assert!(
            error
                .to_string()
                .contains("target identity must be non-empty"),
            "unexpected target validation error: {error}"
        );
    }
}

/// Regression: AOT size accounting must read the canonical neutral resource envelope.
#[test]
fn total_buffer_bytes_comes_from_canonical_resources() {
    let envelope = fixture_target::compiled_artifact();
    assert_eq!(
        envelope.neutral().resource_envelope().total_bytes,
        256 * 4 + 64 * 4
    );
}

/// Regression: selected target bytes must be read from the exact canonical attachment.
#[test]
fn envelope_selects_the_canonical_target_payload() {
    let envelope = fixture_target::compiled_artifact();
    let payload = envelope
        .target_payloads()
        .iter()
        .find(|payload| payload.format().identity() == "fixture-target-format")
        .expect("selected target attachment must exist");

    assert_eq!(payload.format().identity(), "fixture-target-format");
    assert_eq!(payload.format().version(), 1);
    let modules = vyre_megakernel::TargetModuleBundle::from_bytes(payload.bytes())
        .expect("target module bundle must decode");
    assert_eq!(modules.modules[0].bytes, b"target-payload-fixture");
    assert_eq!(payload.neutral_artifact(), envelope.neutral().digest());
    assert_eq!(payload.entries()[0].name, "main");
    assert_eq!(payload.entries()[0].resource_bindings[0].slot, 0);
    assert_eq!(payload.entries()[0].resource_bindings[1].slot, 1);
}
