//! Canonical conformance certificate artifact contract.
//!
//! The `prove --out` command must refuse to emit a certificate when any
//! selected production target diverges from the independent reference engine.
//! Successful acquisition alone is not conformance; every selected witness is
//! compiled, materialized, submitted, read back, and compared before signing.

use std::process::Command;

use ed25519_dalek::{Signer, SigningKey};
use serde_json::Value;

/// The backend a device lane pins the proof to.
///
/// Only the live-GPU tests read it, so it is admitted with them.
#[cfg(feature = "device-tests")]
fn selected_backend_override() -> Option<String> {
    std::env::var("VYRE_BACKEND")
        .ok()
        .filter(|value| !value.trim().is_empty())
}
fn conform_binary() -> &'static str {
    env!("CARGO_BIN_EXE_vyre-conform")
}

/// A synthetic shard is built from the same types the binary signs, so a field
/// added to the certificate cannot be dropped here and leave a signature that
/// no longer covers what a reader rebuilds.
use vyre_conform::certificate_wire::{
    LawRecord, ProofPlanSummary, ProofSelectionSummary, ProveArtifact, ProveSignableBody,
};
use vyre_conform_spec::ConformanceResult;

fn write_signed_shard(
    path: &std::path::Path,
    catalog_hash: &str,
    execution_hash: &str,
    program_hash: &str,
    pairs: Value,
    laws: Value,
) {
    let pairs_vec: Vec<ConformanceResult> =
        serde_json::from_value(pairs.clone()).expect("pairs deserialize");
    let laws_vec: Vec<LawRecord> = serde_json::from_value(laws.clone()).expect("laws deserialize");
    let plan = ProofPlanSummary {
        backend_count: 1,
        op_count: pairs_vec.len(),
        pair_count: pairs_vec.len(),
        witness_case_count: pairs_vec.len(),
        catalog_hash: catalog_hash.to_string(),
        execution_hash: execution_hash.to_string(),
        selection: ProofSelectionSummary {
            backend_filter: "cuda".to_string(),
            ops_filter: "all".to_string(),
            shard_index: Some(0),
            shard_count: Some(2),
            universe_backend_count: 3,
            universe_op_count: 2,
            selected_backend_count: 1,
            selected_op_count: pairs_vec.len(),
            unavailable_backends: Vec::new(),
        },
    };
    let key = SigningKey::from_bytes(&[7u8; 32]);
    let signable = ProveSignableBody {
        wire_format_version: 2u32,
        program_hash,
        backend_id: "all",
        plan: &plan,
        pairs: &pairs_vec,
        laws: &laws_vec,
    };
    let signable_bytes =
        serde_json::to_vec(&signable).expect("Fix: synthetic shard should serialize");
    let signature = key.sign(&signable_bytes);
    let artifact = serde_json::json!({
        "wire_format_version": 2u32,
        "program_hash": program_hash,
        "backend_id": "all",
        "plan": serde_json::to_value(&plan).expect("plan to_value"),
        "signature": hex::encode(signature.to_bytes()),
        "public_key": hex::encode(key.verifying_key().to_bytes()),
        "pairs": pairs,
        "laws": laws,
    });
    std::fs::write(
        path,
        serde_json::to_string_pretty(&artifact).expect("Fix: synthetic shard should serialize"),
    )
    .expect("Fix: synthetic shard should be writable");
}

/// Verify one parsed certificate through the type its writer signs.
///
/// Deserializing into [`ProveArtifact`] is what makes the check whole: a field
/// the certificate carries and this reader does not know about fails the parse
/// rather than silently leaving the signature covering different bytes.
fn verify_certificate_signature(parsed: &Value) {
    let artifact: ProveArtifact = serde_json::from_value(parsed.clone())
        .expect("Fix: certificate must parse as the artifact its writer signed");
    artifact
        .verify_signature()
        .expect("Fix: certificate Ed25519 signature must verify over the canonical body");
}

fn merge_shards(
    merged: &std::path::Path,
    shard_a: &std::path::Path,
    shard_b: &std::path::Path,
) -> Value {
    let status = Command::new(conform_binary())
        .args(["merge", "--out"])
        .arg(merged)
        .arg(shard_a)
        .arg(shard_b)
        .status()
        .expect("Fix: the built vyre-conform binary must launch");
    assert!(
        status.success(),
        "Fix: merge must accept signed certificate shards"
    );

    let merged_json =
        std::fs::read_to_string(merged).expect("Fix: merge must write a readable artifact");
    serde_json::from_str(&merged_json).expect("Fix: merged artifact must be valid JSON")
}

#[cfg(feature = "device-tests")]
mod gpu_certificate_contracts;
mod merge_contracts;
mod prove_failure_contracts;
mod release_script_contracts;
mod shard_plan_contracts;
