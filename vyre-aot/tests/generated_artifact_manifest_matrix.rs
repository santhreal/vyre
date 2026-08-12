//! Generated canonical envelope and AOT packaging-manifest matrix.

mod common;

use vyre_aot::Manifest;

fn target_payload(envelope: &vyre_aot::ArtifactEnvelope) -> &vyre_aot::TargetPayload {
    envelope
        .target_payloads()
        .first()
        .expect("fixture target payload must exist")
}

fn generated_manifest(seed: u32, envelope: &vyre_aot::ArtifactEnvelope) -> Manifest {
    let neutral = envelope.neutral().digest();
    let payload = target_payload(envelope).digest();
    let digest_hex = |bytes: &[u8; 32]| {
        bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    };
    Manifest {
        schema: Manifest::SCHEMA_VERSION.to_string(),
        aot_version: vyre_aot::VERSION.to_string(),
        artifact_name: format!("artifact_{seed:08x}"),
        target: common::fixture_target(),
        target_payload_format: target_payload(envelope).format().identity().to_string(),
        envelope_file: format!("artifact_{seed:08x}.vmk.lzma"),
        envelope_compression: "lzma".to_string(),
        envelope_sha256_hex: format!("{:064x}", seed as u64),
        neutral_artifact_digest_hex: digest_hex(neutral.as_bytes()),
        target_payload_digest_hex: digest_hex(payload.as_bytes()),
        weights_file: format!("weights_{seed:08x}.brotli"),
        weights_compression: "brotli-11".to_string(),
        weights_sha256_hex: format!("{:064x}", seed.wrapping_mul(17) as u64),
        notes: format!("generated seed {seed}"),
    }
}

/// Generated regression: canonical envelope bytes and identities remain stable across repeated reads.
#[test]
fn generated_canonical_envelopes_round_trip() {
    let envelope = common::compiled_artifact();
    let expected_neutral = envelope.neutral().digest();
    let expected_payload = target_payload(&envelope).digest();
    let expected_bytes = target_payload(&envelope).bytes().to_vec();

    for seed in 0..512_u32 {
        let envelope_bytes = envelope.to_bytes().unwrap();
        let decoded = vyre_megakernel::ArtifactEnvelope::from_bytes(&envelope_bytes)
            .expect("generated canonical envelope must parse");
        let payload = target_payload(&decoded);
        assert_eq!(decoded.neutral().digest(), expected_neutral, "seed {seed}");
        assert_eq!(payload.digest(), expected_payload, "seed {seed}");
        assert_eq!(payload.bytes(), expected_bytes, "seed {seed}");
    }
}

/// Generated regression: AOT manifests retain exact canonical identities without resource copies.
#[test]
fn generated_manifests_round_trip_canonical_identity_fields() {
    let envelope = common::compiled_artifact();
    for seed in 0..512_u32 {
        let manifest = generated_manifest(seed.wrapping_mul(0x9e37_79b9), &envelope);
        let bytes = serde_json::to_vec(&manifest).expect("generated manifest must serialize");
        let decoded: Manifest =
            serde_json::from_slice(&bytes).expect("generated manifest must parse");
        assert_eq!(decoded, manifest);
        assert_eq!(
            decoded.neutral_artifact_digest_hex,
            generated_manifest(seed.wrapping_mul(0x9e37_79b9), &envelope)
                .neutral_artifact_digest_hex
        );
        assert_eq!(
            decoded.target_payload_digest_hex,
            generated_manifest(seed.wrapping_mul(0x9e37_79b9), &envelope).target_payload_digest_hex
        );
    }
}
