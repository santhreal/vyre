//! Generated canonical envelope and AOT packaging-manifest matrix.

mod common;

use vyre_aot::{Manifest, Target};

fn generated_manifest(seed: u32, artifact: &vyre_aot::CompiledArtifact) -> Manifest {
    let neutral = artifact.envelope().neutral().digest();
    let payload = artifact.target_payload().unwrap().digest();
    let digest_hex = |bytes: &[u8; 32]| {
        bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    };
    Manifest {
        schema: Manifest::SCHEMA_VERSION.to_string(),
        aot_version: artifact.aot_version().to_string(),
        artifact_name: format!("artifact_{seed:08x}"),
        target: Target::Ptx,
        envelope_file: format!("artifact_{seed:08x}.vmk.lzma"),
        envelope_compression: "lzma".to_string(),
        envelope_sha256_hex: format!("{:064x}", seed as u64),
        neutral_artifact_digest_hex: digest_hex(neutral.as_bytes()),
        target_payload_digest_hex: digest_hex(payload.as_bytes()),
        weights_file: format!("weights_{seed:08x}.brotli"),
        weights_compression: "brotli-11".to_string(),
        weights_sha256_hex: format!("{:064x}", seed.wrapping_mul(17) as u64),
        notes: format!("generated seed {seed}"),
        vsa_fingerprint: artifact.vsa_fingerprint().to_vec(),
    }
}

/// Generated regression: canonical envelope bytes and identities remain stable across repeated reads.
#[test]
fn generated_canonical_envelopes_round_trip() {
    let artifact = common::compiled_artifact();
    let expected_neutral = artifact.envelope().neutral().digest();
    let expected_payload = artifact.target_payload().unwrap().digest();
    let expected_bytes = artifact.target_payload().unwrap().bytes().to_vec();

    for seed in 0..512_u32 {
        let envelope_bytes = artifact.envelope().to_bytes().unwrap();
        let decoded = vyre_megakernel::ArtifactEnvelope::from_bytes(&envelope_bytes)
            .expect("generated canonical envelope must parse");
        let payload = decoded
            .require_target_payload(&vyre_aot::target_payload_format(Target::Ptx).unwrap())
            .expect("generated canonical payload must remain compatible");
        assert_eq!(decoded.neutral().digest(), expected_neutral, "seed {seed}");
        assert_eq!(payload.digest(), expected_payload, "seed {seed}");
        assert_eq!(payload.bytes(), expected_bytes, "seed {seed}");
    }
}

/// Generated regression: AOT manifests retain exact canonical identities without resource copies.
#[test]
fn generated_manifests_round_trip_canonical_identity_fields() {
    let artifact = common::compiled_artifact();
    for seed in 0..512_u32 {
        let manifest = generated_manifest(seed.wrapping_mul(0x9e37_79b9), &artifact);
        let bytes = serde_json::to_vec(&manifest).expect("generated manifest must serialize");
        let decoded: Manifest =
            serde_json::from_slice(&bytes).expect("generated manifest must parse");
        assert_eq!(decoded, manifest);
        assert_eq!(
            decoded.neutral_artifact_digest_hex,
            generated_manifest(seed.wrapping_mul(0x9e37_79b9), &artifact)
                .neutral_artifact_digest_hex
        );
        assert_eq!(
            decoded.target_payload_digest_hex,
            generated_manifest(seed.wrapping_mul(0x9e37_79b9), &artifact).target_payload_digest_hex
        );
    }
}
