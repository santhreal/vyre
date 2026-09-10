//! Canonical conformance certificate artifact contract.
//!
//! The `prove --out` command must refuse to emit a certificate when any
//! selected production target diverges from the independent reference engine.
//! Successful acquisition alone is not conformance; every selected witness is
//! compiled, materialized, submitted, read back, and compared before signing.

use std::process::Command;

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
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

#[derive(serde::Serialize, serde::Deserialize)]
struct SyntheticSelectionSummary {
    backend_filter: String,
    ops_filter: String,
    shard_index: Option<usize>,
    shard_count: Option<usize>,
    universe_backend_count: usize,
    universe_op_count: usize,
    selected_backend_count: usize,
    selected_op_count: usize,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SyntheticPlanSummary {
    backend_count: usize,
    op_count: usize,
    pair_count: usize,
    witness_case_count: usize,
    catalog_hash: String,
    execution_hash: String,
    selection: SyntheticSelectionSummary,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SyntheticConformanceResult {
    op_id: String,
    executor_id: String,
    passed: bool,
    message: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SyntheticLawRecord {
    op_id: String,
    law: String,
    witness: String,
    cases: usize,
}

#[derive(serde::Serialize)]
struct SyntheticSignableBody<'a> {
    wire_format_version: u32,
    program_hash: &'a str,
    backend_id: &'a str,
    plan: &'a SyntheticPlanSummary,
    pairs: &'a [SyntheticConformanceResult],
    laws: &'a [SyntheticLawRecord],
}

fn write_signed_shard(
    path: &std::path::Path,
    catalog_hash: &str,
    execution_hash: &str,
    program_hash: &str,
    pairs: Value,
    laws: Value,
) {
    let pairs_vec: Vec<SyntheticConformanceResult> =
        serde_json::from_value(pairs.clone()).expect("pairs deserialize");
    let laws_vec: Vec<SyntheticLawRecord> =
        serde_json::from_value(laws.clone()).expect("laws deserialize");
    let plan = SyntheticPlanSummary {
        backend_count: 1,
        op_count: pairs_vec.len(),
        pair_count: pairs_vec.len(),
        witness_case_count: pairs_vec.len(),
        catalog_hash: catalog_hash.to_string(),
        execution_hash: execution_hash.to_string(),
        selection: SyntheticSelectionSummary {
            backend_filter: "cuda".to_string(),
            ops_filter: "all".to_string(),
            shard_index: Some(0),
            shard_count: Some(2),
            universe_backend_count: 3,
            universe_op_count: 2,
            selected_backend_count: 1,
            selected_op_count: pairs_vec.len(),
        },
    };
    let key = SigningKey::from_bytes(&[7u8; 32]);
    let signable = SyntheticSignableBody {
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
        "plan": serde_json::to_value(&signable.plan).expect("plan to_value"),
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

fn verify_certificate_signature(parsed: &Value) {
    let signature_hex = parsed["signature"]
        .as_str()
        .expect("Fix: certificate must carry signature");
    let public_key_hex = parsed["public_key"]
        .as_str()
        .expect("Fix: certificate must carry public_key");
    let signature_bytes =
        hex::decode(signature_hex).expect("Fix: certificate signature must be hex");
    let public_key_bytes =
        hex::decode(public_key_hex).expect("Fix: certificate public key must be hex");
    let signature = Signature::from_slice(&signature_bytes)
        .expect("Fix: certificate signature must be a 64-byte Ed25519 signature");
    let public_key_array: [u8; 32] = public_key_bytes
        .as_slice()
        .try_into()
        .expect("Fix: certificate public key must be 32 bytes");
    let verifying_key = VerifyingKey::from_bytes(&public_key_array)
        .expect("Fix: certificate public key must be a valid Ed25519 verifying key");
    let plan: SyntheticPlanSummary =
        serde_json::from_value(parsed["plan"].clone()).expect("plan deserialize");
    let pairs_vec: Vec<SyntheticConformanceResult> =
        serde_json::from_value(parsed["pairs"].clone()).expect("pairs deserialize");
    let laws_vec: Vec<SyntheticLawRecord> =
        serde_json::from_value(parsed["laws"].clone()).expect("laws deserialize");
    let signable = SyntheticSignableBody {
        wire_format_version: parsed["wire_format_version"].as_u64().expect("version") as u32,
        program_hash: parsed["program_hash"].as_str().expect("program_hash"),
        backend_id: parsed["backend_id"].as_str().expect("backend_id"),
        plan: &plan,
        pairs: &pairs_vec,
        laws: &laws_vec,
    };
    let signable_bytes =
        serde_json::to_vec(&signable).expect("Fix: certificate signable body must serialize");
    verifying_key
        .verify(&signable_bytes, &signature)
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
