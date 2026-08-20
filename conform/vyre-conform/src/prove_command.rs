//! The `prove` subcommand: certificate defaults, proof execution, and Ed25519 signing of
//! the emitted artifact.

use crate::artifact_json::write_json_artifact;
use crate::backend_selection::{dispatch_capable_backends, select_backends};
use crate::operation_selection::{select_entries, unified_entries};
use crate::proof_options::parse_proof_options;
use crate::proof_plan::{hash_proof_plan, proof_plan_summary, ProofPlanSummary};
use crate::proof_scheduler::{
    prepare_entries_in_parallel, proof_worker_count, prove_backends_in_parallel,
};
use crate::proof_timing::{emit_proof_timing, ProofTimingReport};
use ed25519_dalek::{Signer, SigningKey};
use serde::Serialize;
use vyre_conform_spec::ConformanceResult;

pub(crate) const DEFAULT_CERTIFICATE_DIR: &str = ".internals/certs/";

pub(crate) const DEFAULT_CERTIFICATE_FILE: &str = "prove.json";

#[derive(Debug, Serialize)]
struct ProveArtifact {
    pub(crate) wire_format_version: u32,
    pub(crate) program_hash: String,
    pub(crate) backend_id: String,
    pub(crate) plan: ProofPlanSummary,
    pub(crate) signature: String,
    pub(crate) public_key: String,
    pub(crate) pairs: Vec<ConformanceResult>,
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

    let all_backends = dispatch_capable_backends()?;
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
    let backends = select_backends(&all_backends, &options.backend_filter)?;
    if !backends.iter().any(|backend| !backend.reference_oracle) {
        return Err(
            "prove refused to emit the certificate: the selected backend set only contains reference dispatch backends. \
             Fix: build with `--features gpu` so certificate generation proves at least one real GPU backend \
             against vyre-reference instead of certifying the reference executor against itself."
                .to_string(),
        );
    }
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
    hasher.update(b"vyre-conform/prove/v1");
    hash_proof_plan(&mut hasher, &plan);
    for pair in &pairs {
        hasher.update(pair.op_id.as_bytes());
        hasher.update(pair.backend_id.as_bytes());
        hasher.update(&[u8::from(pair.passed)]);
        hasher.update(pair.message.as_bytes());
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
    let signable = serde_json::json!({
        "wire_format_version": 1u32,
        "program_hash": program_hash,
        "backend_id": "all",
        "plan": &plan,
        "pairs": &pairs,
    });
    let signable_bytes = serde_json::to_vec(&signable).map_err(|error| {
        format!("failed to serialize prove artifact body: {error}. Fix: keep certificate fields JSON-serializable.")
    })?;
    let signature = key.sign(&signable_bytes);
    let emitted_pair_count = pairs.len();
    let artifact = ProveArtifact {
        wire_format_version: 1,
        program_hash,
        backend_id: "all".to_string(),
        plan,
        signature: hex::encode(signature.to_bytes()),
        public_key: hex::encode(key.verifying_key().to_bytes()),
        pairs,
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
