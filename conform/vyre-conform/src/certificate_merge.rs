//! The `merge` subcommand: signature verification of prove shards and emission of one
//! re-signed merged artifact.

use crate::artifact_json::{read_prove_artifact_bounded, write_json_artifact};
use crate::proof_options::next_option_value;
use crate::proof_plan::{hash_proof_plan, ProofPlanSummary, ProofSelectionSummary};
use crate::prove_command::{LawRecord, ProveArtifact, ProveSignableBody};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use vyre_conform_spec::ConformanceResult;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct MergedProveArtifact {
    pub(crate) wire_format_version: u32,
    pub(crate) program_hash: String,
    pub(crate) backend_id: String,
    pub(crate) plan: ProofPlanSummary,
    pub(crate) signature: String,
    pub(crate) public_key: String,
    pub(crate) pairs: Vec<ConformanceResult>,
    pub(crate) laws: Vec<LawRecord>,
}

struct VerifiedShard {
    pub(crate) path: String,
    pub(crate) artifact: ProveArtifact,
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
    let mut pair_map = BTreeMap::<(String, String), ConformanceResult>::new();
    let mut unique_backends = BTreeSet::<String>::new();
    let mut unique_ops = BTreeSet::<String>::new();
    let mut witness_case_count = 0usize;
    let mut law_map = BTreeMap::<(String, String), LawRecord>::new();
    let mut universe_backend_count = 0usize;
    let mut universe_op_count = 0usize;
    let mut merge_hasher = blake3::Hasher::new();
    merge_hasher.update(b"vyre-conform/proof-merge/v1");

    for path in paths {
        let shard = read_and_verify_shard(&path)?;
        match &catalog_hash {
            Some(expected) if expected != &shard.artifact.plan.catalog_hash => {
                return Err(format!(
                    "merge refused `{path}`: catalog_hash `{}` differs from `{expected}`. Fix: only merge shards produced from the same executable registry.",
                    shard.artifact.plan.catalog_hash
                ));
            }
            None => catalog_hash = Some(shard.artifact.plan.catalog_hash.clone()),
            _ => {}
        }
        merge_hasher.update(shard.artifact.program_hash.as_bytes());
        merge_hasher.update(shard.artifact.plan.execution_hash.as_bytes());
        source_hashes.push(shard.artifact.program_hash.clone());
        witness_case_count =
            witness_case_count.saturating_add(shard.artifact.plan.witness_case_count);
        universe_backend_count =
            universe_backend_count.max(shard.artifact.plan.selection.universe_backend_count);
        universe_op_count = universe_op_count.max(shard.artifact.plan.selection.universe_op_count);

        if shard.artifact.pairs.len() != shard.artifact.plan.pair_count {
            return Err(format!(
                "certificate `{}` plan pair_count={} but pairs.len()={}. Fix: regenerate the shard; the signed plan must match the body.",
                shard.path,
                shard.artifact.plan.pair_count,
                shard.artifact.pairs.len()
            ));
        }
        for pair in shard.artifact.pairs {
            if !pair.passed {
                return Err(format!(
                    "merge refused failing pair ({}, {}) from `{}`. Fix: repair the backend/op divergence before merging.",
                    pair.executor_id, pair.op_id, shard.path
                ));
            }
            let key = (pair.executor_id.clone(), pair.op_id.clone());
            unique_backends.insert(pair.executor_id.clone());
            unique_ops.insert(pair.op_id.clone());
            if pair_map.insert(key.clone(), pair).is_some() {
                return Err(format!(
                    "merge refused duplicate pair ({}, {}) from `{}`. Fix: merge disjoint shards or remove duplicate certificates.",
                    key.0, key.1, shard.path
                ));
            }
        }

        for law in shard.artifact.laws {
            if let Some(existing) =
                law_map.insert((law.op_id.clone(), law.law.clone()), law.clone())
            {
                if existing != law {
                    return Err(format!(
                        "merge refused law ({}, {}) from `{}`: shards disagree about its proof. Fix: re-run prove on one registry revision.",
                        law.op_id, law.law, shard.path
                    ));
                }
            }
        }
    }

    let catalog_hash = catalog_hash.ok_or_else(|| {
        "merge refused to emit: no catalog hash was observed. Fix: pass valid prove artifacts."
            .to_string()
    })?;
    let pairs = pair_map.into_values().collect::<Vec<_>>();
    let laws = law_map.into_values().collect::<Vec<_>>();
    for pair in &pairs {
        merge_hasher.update(pair.executor_id.as_bytes());
        merge_hasher.update(pair.op_id.as_bytes());
        merge_hasher.update(pair.message.as_bytes());
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
    program_hasher.update(b"vyre-conform/merge/v2");
    hash_proof_plan(&mut program_hasher, &plan);
    for pair in &pairs {
        program_hasher.update(pair.executor_id.as_bytes());
        program_hasher.update(pair.op_id.as_bytes());
        program_hasher.update(pair.message.as_bytes());
    }
    for law in &laws {
        program_hasher.update(law.op_id.as_bytes());
        program_hasher.update(law.law.as_bytes());
        program_hasher.update(law.witness.as_bytes());
        program_hasher.update(&(law.cases as u64).to_le_bytes());
    }
    let program_hash = program_hasher.finalize().to_hex().to_string();

    use rand_core::RngCore;
    let mut seed = [0u8; 32];
    rand_core::OsRng.fill_bytes(&mut seed);
    let key = SigningKey::from_bytes(&seed);
    let signable = ProveSignableBody {
        wire_format_version: vyre_spec::schema_registry::SchemaId::ProveArtifact.version_u32(),
        program_hash: &program_hash,
        backend_id: "merged",
        plan: &plan,
        pairs: &pairs,
        laws: &laws,
    };
    let signable_bytes = serde_json::to_vec(&signable).map_err(|error| {
        format!("failed to serialize merged prove artifact body: {error}. Fix: keep certificate fields JSON-serializable.")
    })?;
    let signature = key.sign(&signable_bytes);
    let artifact = MergedProveArtifact {
        wire_format_version: vyre_spec::schema_registry::SchemaId::ProveArtifact.version_u32(),
        program_hash,
        backend_id: "merged".to_string(),
        plan,
        signature: hex::encode(signature.to_bytes()),
        public_key: hex::encode(key.verifying_key().to_bytes()),
        pairs,
        laws,
    };
    let json = serde_json::to_string_pretty(&artifact).map_err(|error| {
        format!("failed to serialize merged prove artifact: {error}. Fix: keep certificate fields JSON-serializable.")
    })?;
    write_json_artifact(&out, json, "merged prove artifact")
}

fn read_and_verify_shard(path: &str) -> Result<VerifiedShard, String> {
    let json = read_prove_artifact_bounded(path)?;
    let artifact: ProveArtifact = serde_json::from_str(&json).map_err(|error| {
        format!(
            "failed to parse certificate `{path}`: {error}. Fix: pass a valid JSON prove artifact."
        )
    })?;
    if artifact.wire_format_version
        != vyre_spec::schema_registry::SchemaId::ProveArtifact.version_u32()
    {
        return Err(format!(
            "certificate `{path}` has wire_format_version {}. Fix: merge only v2 prove artifacts, which carry the proven-law roster.",
            artifact.wire_format_version
        ));
    }
    let signature_bytes = hex::decode(&artifact.signature).map_err(|error| {
        format!("certificate `{path}` signature is not hex: {error}. Fix: regenerate the shard.")
    })?;
    let public_key_bytes = hex::decode(&artifact.public_key).map_err(|error| {
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
    let signable = ProveSignableBody {
        wire_format_version: artifact.wire_format_version,
        program_hash: &artifact.program_hash,
        backend_id: &artifact.backend_id,
        plan: &artifact.plan,
        pairs: &artifact.pairs,
        laws: &artifact.laws,
    };
    let signable_bytes = serde_json::to_vec(&signable).map_err(|error| {
        format!("failed to serialize certificate `{path}` signable body: {error}. Fix: regenerate the shard.")
    })?;
    verifying_key
        .verify(&signable_bytes, &signature)
        .map_err(|error| {
            format!("certificate `{path}` signature verification failed: {error}. Fix: discard the tampered shard and rerun prove.")
        })?;

    if artifact.pairs.is_empty() {
        return Err(format!(
            "certificate `{path}` has no pairs. Fix: prove artifacts must contain executable parity pairs."
        ));
    }
    for pair in &artifact.pairs {
        if !pair.passed {
            return Err(format!(
                "certificate `{path}` contains failing pair ({}, {}). Fix: repair the divergence before merging.",
                pair.executor_id, pair.op_id
            ));
        }
    }

    Ok(VerifiedShard {
        path: path.to_string(),
        artifact,
    })
}
