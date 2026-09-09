//! The `prove` subcommand: certificate defaults, proof execution, and Ed25519 signing of
//! the emitted artifact.

use std::collections::BTreeSet;

use crate::artifact_json::write_json_artifact;
use vyre_conform::backend_selection::{select_backends, semantic_execution_backends};
use crate::operation_selection::{select_entries, unified_entries};
use crate::proof_options::parse_proof_options;
use crate::proof_plan::{hash_proof_plan, proof_plan_summary, ProofPlanSummary};
use crate::proof_scheduler::{
    prepare_entries_in_parallel, proof_worker_count, prove_backends_in_parallel,
};
use crate::proof_timing::{emit_proof_timing, ProofTimingReport};
use ed25519_dalek::{Signer, SigningKey};
use serde::{Deserialize, Serialize};
use vyre_conform::law_proof::{prove_declared_laws, LawVerdict};
use vyre_conform_spec::ConformanceResult;

pub(crate) const DEFAULT_CERTIFICATE_DIR: &str = ".internals/certs/";

pub(crate) const DEFAULT_CERTIFICATE_FILE: &str = "prove.json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ProveArtifact {
    pub(crate) wire_format_version: u32,
    pub(crate) program_hash: String,
    pub(crate) backend_id: String,
    pub(crate) plan: ProofPlanSummary,
    pub(crate) signature: String,
    pub(crate) public_key: String,
    pub(crate) pairs: Vec<ConformanceResult>,
    pub(crate) laws: Vec<LawRecord>,
}

/// One declared law and the witness that proved it on the reference oracle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct LawRecord {
    pub(crate) op_id: String,
    pub(crate) law: String,
    pub(crate) witness: String,
    pub(crate) cases: usize,
}

/// Typed signable body for prove artifacts ensuring identical field order without ad-hoc JSON indexing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ProveSignableBody<'a> {
    pub(crate) wire_format_version: u32,
    pub(crate) program_hash: &'a str,
    pub(crate) backend_id: &'a str,
    pub(crate) plan: &'a ProofPlanSummary,
    pub(crate) pairs: &'a [ConformanceResult],
    pub(crate) laws: &'a [LawRecord],
}

/// Prove every declared law of every selected operation, refusing the
/// certificate when the oracle refutes one.
///
/// A law the declared buffer shape cannot exercise is not recorded here: the
/// roster of those pairs, and the payload each one is missing, is a source
/// contract the conformance suite judges. What belongs in a certificate is what
/// this run executed.
fn prove_selected_laws(selected: &[&'static str]) -> Result<Vec<LawRecord>, String> {
    let selected: BTreeSet<&str> = selected.iter().copied().collect();
    let mut records = Vec::new();
    let mut rejected = Vec::new();
    for entry in vyre_registry_link::operation::live_operation_registry().iter() {
        if !selected.contains(entry.id) {
            continue;
        }
        for proof in prove_declared_laws(&entry) {
            match proof.verdict {
                LawVerdict::Holds { cases } => records.push(LawRecord {
                    op_id: proof.op_id.to_string(),
                    law: proof.law.to_string(),
                    witness: proof
                        .witness
                        .map_or("none", vyre_conform::LawWitness::name)
                        .to_string(),
                    cases,
                }),
                LawVerdict::Refuted { case, detail } => rejected.push(format!(
                    "  - ({}, {}): refuted on fixture case {case}: {detail}",
                    proof.op_id, proof.law
                )),
                LawVerdict::Unrunnable { reason } => rejected.push(format!(
                    "  - ({}, {}): proof could not run: {reason}",
                    proof.op_id, proof.law
                )),
                LawVerdict::Unproven { .. } => {}
            }
        }
    }
    if rejected.is_empty() {
        Ok(records)
    } else {
        Err(format!(
            "{} declared law(s) did not survive the reference oracle:\n{}\nFix: correct the operation, correct its fixtures, or remove a declaration the oracle refutes.",
            rejected.len(),
            rejected.join("\n")
        ))
    }
}

pub(crate) fn prove(args: impl IntoIterator<Item = String>) -> Result<(), String> {
    let total_started = std::time::Instant::now();
    let options = parse_proof_options("prove", args)?;
    let out = options
        .out
        .as_deref()
        .map(str::to_owned)
        .unwrap_or_else(|| {
            std::path::Path::new(
                options
                    .certificates_dir
                    .as_deref()
                    .unwrap_or(DEFAULT_CERTIFICATE_DIR),
            )
            .join(DEFAULT_CERTIFICATE_FILE)
            .to_string_lossy()
            .into_owned()
        });

    let all_backends = semantic_execution_backends()?;
    if all_backends.is_empty() {
        return Err(
            "prove refused to emit the certificate: no dispatch-capable backend is linked into this binary. \
             Fix: build with `--features gpu` (or another backend feature) so a backend that implements \
             real dispatch registers itself via `inventory::submit!(BackendCapability { dispatches: true, .. })`. \
             Emission-only backends are filtered out because they cannot execute Programs \
             against vyre-reference."
                .to_string(),
        );
    }
    // Every reference oracle is filtered out of `all_backends`, so a reference
    // backend cannot reach the proof: selection is where that is decided and
    // where the refusal is worded.
    let backends = select_backends(&all_backends, &options.backend_filter)
        .map_err(|reason| format!("prove refused to emit the certificate: {reason}"))?;
    let all_entries = unified_entries();
    let entries = select_entries(&all_entries, &options.ops_filter, options.shard)?;
    let selected_op_count = entries.len();
    let worker_count = proof_worker_count(selected_op_count);
    let prepare_started = std::time::Instant::now();
    let prepared = prepare_entries_in_parallel(entries, &backends);
    let prepare_elapsed = prepare_started.elapsed();
    let prepared_entries = prepared.entries;
    let mut pairs = prepared.pairs;
    let mut any_failed = prepared.any_failed;
    let backend_started = std::time::Instant::now();
    for backend_pairs in prove_backends_in_parallel(&backends, &prepared_entries) {
        for pair in backend_pairs {
            if !pair.passed {
                any_failed = true;
            }
            pairs.push(pair);
        }
    }
    let backend_elapsed = backend_started.elapsed();
    if any_failed {
        use std::fmt::Write;
        let mut failing_count = 0usize;
        let mut failing_detail = String::new();
        for pair in pairs.iter().filter(|pair| !pair.passed) {
            if !failing_detail.is_empty() {
                failing_detail.push('\n');
            }
            let _ = write!(
                &mut failing_detail,
                "  - ({}, {}): {}",
                pair.backend_id, pair.op_id, pair.message
            );
            failing_count += 1;
        }
        return Err(format!(
            "prove refused to emit `{out}` because {} (backend, op) pair(s) diverged from vyre-reference:\n{}\nFix: resolve every failing pair before re-running prove.",
            failing_count,
            failing_detail
        ));
    }

    let selected_ops: Vec<&'static str> = prepared_entries
        .iter()
        .map(|prepared| prepared.id)
        .collect();
    let laws = prove_selected_laws(&selected_ops)
        .map_err(|reason| format!("prove refused to emit `{out}`: {reason}"))?;

    let plan = proof_plan_summary(
        &all_backends,
        &all_entries,
        &backends,
        &prepared_entries,
        pairs.len(),
        &options,
    );

    let signing_started = std::time::Instant::now();
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vyre-conform/prove/v2");
    hash_proof_plan(&mut hasher, &plan);
    for pair in &pairs {
        hasher.update(pair.op_id.as_bytes());
        hasher.update(pair.backend_id.as_bytes());
        hasher.update(&[u8::from(pair.passed)]);
        hasher.update(pair.message.as_bytes());
    }
    for law in &laws {
        hasher.update(law.op_id.as_bytes());
        hasher.update(law.law.as_bytes());
        hasher.update(law.witness.as_bytes());
        hasher.update(&(law.cases as u64).to_le_bytes());
    }
    let program_hash = hasher.finalize().to_hex().to_string();

    // The prior derivation
    // hashed `program_hash:pid:SystemTime::now()` into the Ed25519
    // seed. All three inputs are attacker-guessable (program_hash is
    // public, pid is ~2^22, SystemTime has microsecond resolution)
    // so an attacker who knew approximate CI runtime could brute-force
    // the seed and forge signed artifacts. The signature was
    // security theater.
    //
    // Use OS randomness instead. This makes every cert non-reproducible
    // (a feature  -  two runs of `prove` MUST produce different keys)
    // and removes the brute-force attack surface entirely. If a user
    // later needs reproducibility, they can thread a high-entropy
    // secret through an env var + HKDF; the insecure derivation above
    // is never the right answer.
    use rand_core::RngCore;
    let mut seed = [0u8; 32];
    rand_core::OsRng.fill_bytes(&mut seed);
    let key = SigningKey::from_bytes(&seed);
    let signable = ProveSignableBody {
        wire_format_version: 2,
        program_hash: &program_hash,
        backend_id: "all",
        plan: &plan,
        pairs: &pairs,
        laws: &laws,
    };
    let signable_bytes = serde_json::to_vec(&signable).map_err(|error| {
        format!("failed to serialize prove artifact body: {error}. Fix: keep certificate fields JSON-serializable.")
    })?;
    let signature = key.sign(&signable_bytes);
    let emitted_pair_count = pairs.len();
    let artifact = ProveArtifact {
        wire_format_version: 2,
        program_hash,
        backend_id: "all".to_string(),
        plan,
        signature: hex::encode(signature.to_bytes()),
        public_key: hex::encode(key.verifying_key().to_bytes()),
        pairs,
        laws,
    };
    let json = serde_json::to_string_pretty(&artifact).map_err(|error| {
        format!("failed to serialize prove artifact: {error}. Fix: keep certificate fields JSON-serializable.")
    })?;
    let signing_elapsed = signing_started.elapsed();
    let result = write_json_artifact(&out, json, "prove artifact");
    if result.is_ok() {
        emit_proof_timing(ProofTimingReport {
            out: &out,
            backend_count: backends.len(),
            selected_op_count,
            prepared_op_count: prepared_entries.len(),
            pair_count: emitted_pair_count,
            worker_count,
            prepare_elapsed,
            backend_elapsed,
            signing_elapsed,
            total_elapsed: total_started.elapsed(),
        });
    }
    result
}
