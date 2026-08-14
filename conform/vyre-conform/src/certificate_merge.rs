//! The `merge` subcommand: signature verification of prove shards and emission of one
//! re-signed merged artifact.

use crate::artifact_json::{
    read_prove_artifact_bounded, string_field, u32_field, usize_field, value_field,
    write_json_artifact,
};
use crate::proof_options::next_option_value;
use crate::proof_plan::{hash_proof_plan, ProofPlanSummary, ProofSelectionSummary};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Serialize)]
struct MergedProveArtifact {
    pub(crate) wire_format_version: u32,
    pub(crate) program_hash: String,
    pub(crate) backend_id: String,
    pub(crate) plan: ProofPlanSummary,
    pub(crate) signature: String,
    pub(crate) public_key: String,
    pub(crate) pairs: Vec<serde_json::Value>,
}

struct VerifiedShard {
    pub(crate) path: String,
    pub(crate) value: serde_json::Value,
    pub(crate) catalog_hash: String,
    pub(crate) execution_hash: String,
    pub(crate) program_hash: String,
    pub(crate) witness_case_count: usize,
    pub(crate) pair_count: usize,
    pub(crate) universe_backend_count: usize,
    pub(crate) universe_op_count: usize,
}

pub(crate) fn merge_certificates(args: impl IntoIterator<Item = String>) -> Result<(), String> {
    let mut out = None::<String>;
    let mut paths = Vec::new();
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--out" => out = Some(next_option_value(&mut it, "--out")?),
            other => paths.push(other.to_string()),
        }
    }
    let out = out.ok_or_else(|| {
        "missing --out for merge. Fix: run `vyre-conform merge --out <merged.json> <prove-shard.json>...`."
            .to_string()
    })?;
    if paths.is_empty() {
        return Err(
            "merge refused to emit: no certificates were provided. Fix: pass one or more signed prove artifacts."
                .to_string(),
        );
    }

    let mut catalog_hash = None::<String>;
    let mut source_hashes = Vec::with_capacity(paths.len());
    let mut pair_map = BTreeMap::<(String, String), serde_json::Value>::new();
    let mut unique_backends = BTreeSet::<String>::new();
    let mut unique_ops = BTreeSet::<String>::new();
    let mut witness_case_count = 0usize;
    let mut universe_backend_count = 0usize;
    let mut universe_op_count = 0usize;
    let mut merge_hasher = blake3::Hasher::new();
    merge_hasher.update(b"vyre-conform/proof-merge/v1");

    for path in paths {
        let shard = read_and_verify_shard(&path)?;
        match &catalog_hash {
            Some(expected) if expected != &shard.catalog_hash => {
                return Err(format!(
                    "merge refused `{path}`: catalog_hash `{}` differs from `{expected}`. Fix: only merge shards produced from the same executable registry.",
                    shard.catalog_hash
                ));
            }
            None => catalog_hash = Some(shard.catalog_hash.clone()),
            _ => {}
        }
        merge_hasher.update(shard.program_hash.as_bytes());
        merge_hasher.update(shard.execution_hash.as_bytes());
        source_hashes.push(shard.program_hash.clone());
        witness_case_count = witness_case_count.saturating_add(shard.witness_case_count);
        universe_backend_count = universe_backend_count.max(shard.universe_backend_count);
        universe_op_count = universe_op_count.max(shard.universe_op_count);

        let pairs = value_field(&shard.value, "pairs", &shard.path)?
            .as_array()
            .ok_or_else(|| {
                format!(
                    "certificate `{}` has non-array `pairs`. Fix: only merge prove artifacts.",
                    shard.path
                )
            })?;
        if pairs.len() != shard.pair_count {
            return Err(format!(
                "certificate `{}` plan pair_count={} but pairs.len()={}. Fix: regenerate the shard; the signed plan must match the body.",
                shard.path,
                shard.pair_count,
                pairs.len()
            ));
        }
        for pair in pairs {
            let backend = string_field(pair, "backend_id", &shard.path)?.to_string();
            let op = string_field(pair, "op_id", &shard.path)?.to_string();
            let passed = value_field(pair, "passed", &shard.path)?
                .as_bool()
                .ok_or_else(|| {
                    format!(
                        "certificate `{}` pair ({backend}, {op}) has non-boolean `passed`. Fix: regenerate the shard.",
                        shard.path
                    )
                })?;
            if !passed {
                return Err(format!(
                    "merge refused failing pair ({backend}, {op}) from `{}`. Fix: repair the backend/op divergence before merging.",
                    shard.path
                ));
            }
            let key = (backend.clone(), op.clone());
            unique_backends.insert(backend);
            unique_ops.insert(op);
            if pair_map.insert(key.clone(), pair.clone()).is_some() {
                return Err(format!(
                    "merge refused duplicate pair ({}, {}) from `{}`. Fix: merge disjoint shards or remove duplicate certificates.",
                    key.0, key.1, shard.path
                ));
            }
        }
    }

    let catalog_hash = catalog_hash.ok_or_else(|| {
        "merge refused to emit: no catalog hash was observed. Fix: pass valid prove artifacts."
            .to_string()
    })?;
    let pairs = pair_map.into_values().collect::<Vec<_>>();
    for pair in &pairs {
        merge_hasher.update(string_field(pair, "backend_id", "merged artifact")?.as_bytes());
        merge_hasher.update(string_field(pair, "op_id", "merged artifact")?.as_bytes());
        merge_hasher.update(string_field(pair, "message", "merged artifact")?.as_bytes());
    }
    for source_hash in &source_hashes {
        merge_hasher.update(source_hash.as_bytes());
    }
    let execution_hash = merge_hasher.finalize().to_hex().to_string();
    let plan = ProofPlanSummary {
        backend_count: unique_backends.len(),
        op_count: unique_ops.len(),
        pair_count: pairs.len(),
        witness_case_count,
        catalog_hash,
        execution_hash,
        selection: ProofSelectionSummary {
            backend_filter: "merged".to_string(),
            ops_filter: "merged".to_string(),
            shard_index: None,
            shard_count: Some(source_hashes.len()),
            universe_backend_count,
            universe_op_count,
            selected_backend_count: unique_backends.len(),
            selected_op_count: unique_ops.len(),
        },
    };

    let mut program_hasher = blake3::Hasher::new();
    program_hasher.update(b"vyre-conform/merge/v1");
    hash_proof_plan(&mut program_hasher, &plan);
    for pair in &pairs {
        program_hasher.update(string_field(pair, "backend_id", "merged artifact")?.as_bytes());
        program_hasher.update(string_field(pair, "op_id", "merged artifact")?.as_bytes());
        program_hasher.update(string_field(pair, "message", "merged artifact")?.as_bytes());
    }
    let program_hash = program_hasher.finalize().to_hex().to_string();

    use rand_core::RngCore;
    let mut seed = [0u8; 32];
    rand_core::OsRng.fill_bytes(&mut seed);
    let key = SigningKey::from_bytes(&seed);
    let signable = serde_json::json!({
        "wire_format_version": 1u32,
        "program_hash": program_hash,
        "backend_id": "merged",
        "plan": &plan,
        "pairs": &pairs,
    });
    let signable_bytes = serde_json::to_vec(&signable).map_err(|error| {
        format!("failed to serialize merged prove artifact body: {error}. Fix: keep certificate fields JSON-serializable.")
    })?;
    let signature = key.sign(&signable_bytes);
    let artifact = MergedProveArtifact {
        wire_format_version: 1,
        program_hash,
        backend_id: "merged".to_string(),
        plan,
        signature: hex::encode(signature.to_bytes()),
        public_key: hex::encode(key.verifying_key().to_bytes()),
        pairs,
    };
    let json = serde_json::to_string_pretty(&artifact).map_err(|error| {
        format!("failed to serialize merged prove artifact: {error}. Fix: keep certificate fields JSON-serializable.")
    })?;
    write_json_artifact(&out, json, "merged prove artifact")
}

fn read_and_verify_shard(path: &str) -> Result<VerifiedShard, String> {
    let json = read_prove_artifact_bounded(path)?;
    let value: serde_json::Value = serde_json::from_str(&json).map_err(|error| {
        format!(
            "failed to parse certificate `{path}`: {error}. Fix: pass a valid JSON prove artifact."
        )
    })?;
    let wire_format_version = u32_field(&value, "wire_format_version", path)?;
    if wire_format_version != 1 {
        return Err(format!(
            "certificate `{path}` has wire_format_version {wire_format_version}. Fix: merge only v1 prove artifacts."
        ));
    }
    let program_hash = string_field(&value, "program_hash", path)?.to_string();
    let signature_hex = string_field(&value, "signature", path)?;
    let public_key_hex = string_field(&value, "public_key", path)?;
    let signature_bytes = hex::decode(signature_hex).map_err(|error| {
        format!("certificate `{path}` signature is not hex: {error}. Fix: regenerate the shard.")
    })?;
    let public_key_bytes = hex::decode(public_key_hex).map_err(|error| {
        format!("certificate `{path}` public_key is not hex: {error}. Fix: regenerate the shard.")
    })?;
    let signature = Signature::from_slice(&signature_bytes).map_err(|error| {
        format!("certificate `{path}` signature is invalid: {error}. Fix: regenerate the shard.")
    })?;
    let public_key_array: [u8; 32] = public_key_bytes.as_slice().try_into().map_err(|_| {
        format!(
            "certificate `{path}` public_key must decode to 32 bytes. Fix: regenerate the shard."
        )
    })?;
    let verifying_key = VerifyingKey::from_bytes(&public_key_array).map_err(|error| {
        format!("certificate `{path}` public_key is invalid: {error}. Fix: regenerate the shard.")
    })?;
    let signable = serde_json::json!({
        "wire_format_version": value["wire_format_version"].clone(),
        "program_hash": value["program_hash"].clone(),
        "backend_id": value["backend_id"].clone(),
        "plan": value["plan"].clone(),
        "pairs": value["pairs"].clone(),
    });
    let signable_bytes = serde_json::to_vec(&signable).map_err(|error| {
        format!("failed to serialize certificate `{path}` signable body: {error}. Fix: regenerate the shard.")
    })?;
    verifying_key
        .verify(&signable_bytes, &signature)
        .map_err(|error| {
            format!("certificate `{path}` signature verification failed: {error}. Fix: discard the tampered shard and rerun prove.")
        })?;

    let plan = value_field(&value, "plan", path)?;
    let selection = value_field(plan, "selection", path)?;
    let catalog_hash = string_field(plan, "catalog_hash", path)?.to_string();
    let execution_hash = string_field(plan, "execution_hash", path)?.to_string();
    let witness_case_count = usize_field(plan, "witness_case_count", path)?;
    let pair_count = usize_field(plan, "pair_count", path)?;
    let universe_backend_count = usize_field(selection, "universe_backend_count", path)?;
    let universe_op_count = usize_field(selection, "universe_op_count", path)?;

    let pairs = value_field(&value, "pairs", path)?
        .as_array()
        .ok_or_else(|| format!("certificate `{path}` has non-array pairs. Fix: regenerate it."))?;
    if pairs.is_empty() {
        return Err(format!(
            "certificate `{path}` has no pairs. Fix: prove artifacts must contain executable parity pairs."
        ));
    }
    for pair in pairs {
        let backend = string_field(pair, "backend_id", path)?;
        let op = string_field(pair, "op_id", path)?;
        let passed = value_field(pair, "passed", path)?
            .as_bool()
            .ok_or_else(|| {
                format!(
                    "certificate `{path}` pair ({backend}, {op}) has non-boolean `passed`. Fix: regenerate the shard."
                )
            })?;
        if !passed {
            return Err(format!(
                "certificate `{path}` contains failing pair ({backend}, {op}). Fix: repair the divergence before merging."
            ));
        }
    }

    Ok(VerifiedShard {
        path: path.to_string(),
        value,
        catalog_hash,
        execution_hash,
        program_hash,
        witness_case_count,
        pair_count,
        universe_backend_count,
        universe_op_count,
    })
}
