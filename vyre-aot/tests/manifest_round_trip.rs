//! Manifest serialization contract for canonical artifact packages.

use vyre_aot::artifact::TargetId;
use vyre_aot::manifest::Manifest;

/// Regression: the packaging manifest must preserve canonical artifact identities exactly.
#[test]
fn manifest_round_trips_through_serde_json() {
    let original = Manifest {
        schema: Manifest::SCHEMA_VERSION.to_string(),
        aot_version: "0.7.2".to_string(),
        artifact_name: "test-artifact".to_string(),
        target: TargetId::from_owned("external-manifest-target".to_string()).unwrap(),
        target_payload_format: "external-payload-format".to_string(),
        envelope_file: "artifact.vmk.lzma".to_string(),
        envelope_compression: "lzma".to_string(),
        envelope_sha256_hex: "11".repeat(32),
        neutral_artifact_digest_hex: "22".repeat(32),
        target_payload_digest_hex: "33".repeat(32),
        weights_file: "weights.brotli".to_string(),
        weights_compression: "brotli-11".to_string(),
        weights_sha256_hex: "44".repeat(32),
        notes: "round-trip".to_string(),
    };

    let bytes = serde_json::to_vec_pretty(&original).expect("manifest must serialize");
    let decoded: Manifest = serde_json::from_slice(&bytes).expect("manifest must deserialize");
    assert_eq!(decoded, original);
    assert_eq!(decoded.neutral_artifact_digest_hex, "22".repeat(32));
    assert_eq!(decoded.target_payload_digest_hex, "33".repeat(32));
}
