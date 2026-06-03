//! Parity-cert artifact (TEST-034).
//!
//! See `contracts/release.md`. A reviewer must be able to run ONE
//! command and produce a signed JSON certificate that proves every op
//! passes every registered backend's byte-identity dispatch against the
//! CPU reference. `prove --out` is the load-bearing gate: it MUST
//! refuse to emit a certificate when any (backend, op) pair diverges
//! from `vyre-reference` byte-for-byte. Acquisition success is not
//! parity  -  TEST-034 was filed because the earlier implementation
//! stopped at `backend.factory()` and never dispatched anything.

use std::process::Command;

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde_json::Value;

#[test]
fn prove_refuses_certificate_when_backend_cannot_dispatch() {
    // In the default build only SPIR-V (emission-only, no device) and
    // photonic (non-dispatching hardware substrate) are linked. Neither can execute a program, so
    // `prove` MUST refuse to emit the certificate.
    let out_path = std::env::temp_dir().join(format!(
        "vyre-conform-prove-refuses-{}.json",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&out_path);
    let output = Command::new("cargo")
        .args([
            "run",
            "-p",
            "vyre-conform-runner",
            "--no-default-features",
            "--quiet",
            "--",
            "prove",
            "--out",
        ])
        .arg(&out_path)
        .output()
        .expect("Fix: cargo must be available in PATH");
    assert!(
        !output.status.success(),
        "TEST-034: prove without a dispatch-capable backend must exit non-zero; stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("refused to emit"),
        "TEST-034: prove must explain why it refused to emit the certificate; stderr={stderr}"
    );
    assert!(
        !out_path.exists(),
        "TEST-034: prove must not leave a certificate file on disk when parity fails"
    );
}

// Drives `cargo run -p vyre-conform-runner --features gpu -- prove`; GPU
// backend acquisition failures are release-host failures, not skipped tests.
#[test]
fn prove_emits_signed_certificate_on_gpu_build() {
    let out = tempfile::NamedTempFile::new().expect("tempfile");
    let status = Command::new("cargo")
        .env("VYRE_CONFORM_PROOF_WORKERS", "16")
        .args([
            "run",
            "-p",
            "vyre-conform-runner",
            "--features",
            "gpu",
            "--quiet",
            "--",
            "prove",
            "--out",
        ])
        .arg(out.path())
        .status()
        .expect("Fix: cargo must be available in PATH");
    assert!(
        status.success(),
        "TEST-034: `cargo run -p vyre-conform-runner --features gpu -- prove --out <path>` must succeed on a live GPU"
    );

    let cert =
        std::fs::read_to_string(out.path()).expect("Fix: prove must produce a readable artifact");
    let parsed: Value = serde_json::from_str(&cert).expect("TEST-034: artifact must be valid JSON");
    for required in &[
        "wire_format_version",
        "program_hash",
        "backend_id",
        "plan",
        "signature",
        "public_key",
        "pairs",
    ] {
        assert!(
            parsed.get(required).is_some(),
            "TEST-034: certificate missing required field `{required}`"
        );
    }
    let pairs = parsed
        .get("pairs")
        .and_then(|v| v.as_array())
        .expect("TEST-034: certificate must embed a pairs array");
    let plan = parsed
        .get("plan")
        .and_then(|v| v.as_object())
        .expect("Fix: signed certificate must embed an executable proof plan summary");
    for required in &[
        "backend_count",
        "op_count",
        "pair_count",
        "witness_case_count",
        "catalog_hash",
        "execution_hash",
        "selection",
    ] {
        assert!(
            plan.get(*required).is_some(),
            "Fix: proof plan summary missing `{required}`"
        );
    }
    assert!(
        !pairs.is_empty(),
        "TEST-034: pairs array must include every registered (backend, op) witness"
    );
    for pair in pairs {
        let passed = pair
            .get("passed")
            .and_then(|v| v.as_bool())
            .expect("TEST-034: every pair must carry a boolean `passed` field");
        assert!(
            passed,
            "TEST-034: prove emitted a certificate containing a failing pair: {pair}"
        );
    }

    let mut by_backend =
        std::collections::BTreeMap::<String, std::collections::BTreeSet<String>>::new();
    for pair in pairs {
        let backend = pair["backend_id"]
            .as_str()
            .expect("Fix: certificate pair must carry backend_id")
            .to_string();
        let op = pair["op_id"]
            .as_str()
            .expect("Fix: certificate pair must carry op_id")
            .to_string();
        by_backend.entry(backend).or_default().insert(op);
    }
    for required_backend in ["cuda", "wgpu", "cpu-ref"] {
        let ops = by_backend.get(required_backend).unwrap_or_else(|| {
            panic!("Fix: signed certificate must include backend `{required_backend}`.")
        });
        assert!(
            ops.len() >= 300,
            "Fix: signed certificate backend `{required_backend}` must cover the catalog-scale executable registry, got {} ops.",
            ops.len()
        );
    }
    let cuda_ops = by_backend
        .get("cuda")
        .expect("Fix: signed certificate must include cuda ops");
    for backend in ["wgpu", "cpu-ref"] {
        let ops = by_backend
            .get(backend)
            .unwrap_or_else(|| panic!("Fix: signed certificate must include `{backend}` ops."));
        assert_eq!(
            ops, cuda_ops,
            "Fix: signed certificate backend `{backend}` must cover the same executable op set as cuda."
        );
    }

    let signature_hex = parsed["signature"]
        .as_str()
        .expect("Fix: signed certificate must carry signature");
    let public_key_hex = parsed["public_key"]
        .as_str()
        .expect("Fix: signed certificate must carry public_key");
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
    let signable = serde_json::json!({
        "wire_format_version": parsed["wire_format_version"].clone(),
        "program_hash": parsed["program_hash"].clone(),
        "backend_id": parsed["backend_id"].clone(),
        "plan": parsed["plan"].clone(),
        "pairs": parsed["pairs"].clone(),
    });
    let signable_bytes =
        serde_json::to_vec(&signable).expect("Fix: certificate signable body must serialize");
    verifying_key
        .verify(&signable_bytes, &signature)
        .expect("Fix: certificate Ed25519 signature must verify over the canonical prove body");
}

#[test]
fn prove_emits_signed_cuda_release_certificate_on_gpu_build() {
    let out = tempfile::NamedTempFile::new().expect("tempfile");
    let status = Command::new("cargo")
        .env("VYRE_CONFORM_PROOF_WORKERS", "16")
        .args([
            "run",
            "-p",
            "vyre-conform-runner",
            "--features",
            "gpu",
            "--quiet",
            "--",
            "prove",
            "--backend",
            "cuda",
            "--out",
        ])
        .arg(out.path())
        .status()
        .expect("Fix: cargo must be available in PATH");
    assert!(
        status.success(),
        "Fix: CUDA is the release path; `prove --backend cuda` must produce a signed certificate on a live GPU."
    );

    let cert =
        std::fs::read_to_string(out.path()).expect("Fix: prove must produce a readable artifact");
    let parsed: Value = serde_json::from_str(&cert).expect("Fix: artifact must be valid JSON");
    let pairs = parsed["pairs"]
        .as_array()
        .expect("Fix: CUDA certificate must carry executable parity pairs.");
    assert!(
        pairs.len() >= 300,
        "Fix: CUDA release certificate must cover the catalog-scale executable registry, got {} pairs.",
        pairs.len()
    );
    for pair in pairs {
        assert_eq!(
            pair["backend_id"].as_str(),
            Some("cuda"),
            "Fix: CUDA release certificate must be filtered to the CUDA backend."
        );
        assert_eq!(
            pair["passed"].as_bool(),
            Some(true),
            "Fix: CUDA release certificate must not contain failing pairs: {pair}"
        );
    }
    assert_eq!(
        parsed["plan"]["backend_count"].as_u64(),
        Some(1),
        "Fix: CUDA release certificate must prove exactly one selected backend."
    );
    assert_eq!(
        parsed["plan"]["selection"]["selected_backend_count"].as_u64(),
        Some(1),
        "Fix: CUDA release certificate selection metadata must stay aligned with --backend cuda."
    );
    verify_certificate_signature(&parsed);
}

#[test]
fn prove_merges_live_gpu_certificate_shards() {
    let dir = tempfile::tempdir().expect("tempdir");
    let shard_a = dir.path().join("live-shard-a.json");
    let shard_b = dir.path().join("live-shard-b.json");
    let merged = dir.path().join("live-merged.json");

    for (shard, path) in [("0/64", &shard_a), ("1/64", &shard_b)] {
        let status = Command::new("cargo")
            .env("VYRE_CONFORM_PROOF_WORKERS", "16")
            .args([
                "run",
                "-p",
                "vyre-conform-runner",
                "--features",
                "gpu",
                "--quiet",
                "--",
                "prove",
                "--shard",
                shard,
                "--out",
            ])
            .arg(path)
            .status()
            .expect("Fix: cargo must be available in PATH");
        assert!(
            status.success(),
            "Fix: live GPU proof shard {shard} must emit a signed certificate."
        );
    }

    let status = Command::new("cargo")
        .args([
            "run",
            "-p",
            "vyre-conform-runner",
            "--no-default-features",
            "--quiet",
            "--",
            "merge",
            "--out",
        ])
        .arg(&merged)
        .arg(&shard_a)
        .arg(&shard_b)
        .status()
        .expect("Fix: cargo must be available in PATH");
    assert!(
        status.success(),
        "Fix: merge must accept live signed GPU proof shards."
    );

    let merged_json =
        std::fs::read_to_string(&merged).expect("Fix: merge must write a readable artifact");
    let parsed: Value =
        serde_json::from_str(&merged_json).expect("Fix: merged artifact must be valid JSON");
    assert_eq!(
        parsed["backend_id"].as_str(),
        Some("merged"),
        "Fix: merged live certificate must use the aggregate backend id."
    );
    let pairs = parsed["pairs"]
        .as_array()
        .expect("Fix: merged live certificate must carry pair results.");
    assert!(
        pairs.len() >= 30,
        "Fix: merged live certificate shards must cover multiple real GPU/backend pairs, got {}.",
        pairs.len()
    );
    let mut backends = std::collections::BTreeSet::new();
    for pair in pairs {
        let backend = pair["backend_id"]
            .as_str()
            .expect("Fix: merged live pair must carry backend_id");
        backends.insert(backend.to_string());
        assert_eq!(
            pair["passed"].as_bool(),
            Some(true),
            "Fix: merged live certificate must not contain failing pairs: {pair}"
        );
    }
    for required in ["cuda", "wgpu", "cpu-ref"] {
        assert!(
            backends.contains(required),
            "Fix: merged live GPU shards must preserve backend `{required}`."
        );
    }
    assert_eq!(
        parsed["plan"]["pair_count"].as_u64(),
        Some(pairs.len() as u64),
        "Fix: merged live plan pair_count must match carried pairs."
    );
    verify_certificate_signature(&parsed);
}

#[test]
fn release_shard_script_keeps_prove_merge_backend_and_worker_controls() {
    let script = include_str!("../../../scripts/prove-release-shards.sh");
    for required in [
        "VYRE_RELEASE_SHARDS",
        "VYRE_RELEASE_BACKEND",
        "VYRE_CONFORM_PROOF_WORKERS",
        "prove",
        "--shard",
        "--backend",
        "merge",
        "--no-default-features",
        "merged.json",
    ] {
        assert!(
            script.contains(required),
            "Fix: release shard automation must keep `{required}` wired so GPU proof evidence remains reproducible."
        );
    }
}

#[test]
fn plan_emits_deterministic_shard_manifest_without_dispatch() {
    let out = tempfile::NamedTempFile::new().expect("tempfile");
    let status = Command::new("cargo")
        .args([
            "run",
            "-p",
            "vyre-conform-runner",
            "--features",
            "gpu",
            "--quiet",
            "--",
            "plan",
            "--shard",
            "0/64",
            "--out",
        ])
        .arg(out.path())
        .status()
        .expect("Fix: cargo must be available in PATH");
    assert!(
        status.success(),
        "Fix: proof planning must not acquire or dispatch a backend; it should only emit the selected executable shard manifest."
    );

    let plan_json =
        std::fs::read_to_string(out.path()).expect("Fix: plan must produce a readable artifact");
    let parsed: Value =
        serde_json::from_str(&plan_json).expect("Fix: plan artifact must be valid JSON");
    let plan = parsed["plan"]
        .as_object()
        .expect("Fix: plan artifact must include a plan object");
    let selection = plan["selection"]
        .as_object()
        .expect("Fix: plan summary must include selection metadata");
    assert_eq!(
        selection["shard_index"].as_u64(),
        Some(0),
        "Fix: plan must preserve the selected shard index."
    );
    assert_eq!(
        selection["shard_count"].as_u64(),
        Some(64),
        "Fix: plan must preserve the selected shard count."
    );
    assert!(
        plan["catalog_hash"].as_str().is_some(),
        "Fix: plan must carry a full-catalog hash shared by every shard."
    );
    assert!(
        plan["execution_hash"].as_str().is_some(),
        "Fix: plan must carry the selected shard execution hash."
    );
    assert!(
        matches!(parsed["backends"].as_array(), Some(backends) if !backends.is_empty()),
        "Fix: plan must name every backend selected for this shard."
    );
    assert!(
        matches!(parsed["ops"].as_array(), Some(ops) if !ops.is_empty()),
        "Fix: plan must name every op selected for this shard."
    );
}

#[test]
fn merge_verifies_and_resigns_disjoint_certificate_shards() {
    let dir = tempfile::tempdir().expect("tempdir");
    let shard_a = dir.path().join("shard-a.json");
    let shard_b = dir.path().join("shard-b.json");
    let merged = dir.path().join("merged.json");
    write_signed_shard(
        &shard_a,
        "catalog-hash",
        "execution-a",
        "program-a",
        serde_json::json!([
            {
                "op_id": "vyre-test::a",
                "backend_id": "cuda",
                "passed": true,
                "message": "a matched"
            }
        ]),
    );
    write_signed_shard(
        &shard_b,
        "catalog-hash",
        "execution-b",
        "program-b",
        serde_json::json!([
            {
                "op_id": "vyre-test::b",
                "backend_id": "cuda",
                "passed": true,
                "message": "b matched"
            }
        ]),
    );

    let status = Command::new("cargo")
        .args([
            "run",
            "-p",
            "vyre-conform-runner",
            "--no-default-features",
            "--quiet",
            "--",
            "merge",
            "--out",
        ])
        .arg(&merged)
        .arg(&shard_a)
        .arg(&shard_b)
        .status()
        .expect("Fix: cargo must be available in PATH");
    assert!(
        status.success(),
        "Fix: merge must accept signed disjoint shards"
    );

    let merged_json =
        std::fs::read_to_string(&merged).expect("Fix: merge must write a readable artifact");
    let parsed: Value =
        serde_json::from_str(&merged_json).expect("Fix: merged artifact must be valid JSON");
    assert_eq!(
        parsed["backend_id"].as_str(),
        Some("merged"),
        "Fix: aggregate certificate must name the merged backend set."
    );
    assert_eq!(
        parsed["plan"]["pair_count"].as_u64(),
        Some(2),
        "Fix: merged plan must count all shard pairs."
    );
    assert_eq!(
        parsed["plan"]["selection"]["shard_count"].as_u64(),
        Some(2),
        "Fix: merged plan must preserve source shard count."
    );
    assert_eq!(
        parsed["pairs"].as_array().map(Vec::len),
        Some(2),
        "Fix: merged certificate must carry all disjoint pairs."
    );
    verify_certificate_signature(&parsed);
}

#[test]

fn merge_rejects_tampered_certificate_shard() {
    let dir = tempfile::tempdir().expect("tempdir");
    let shard = dir.path().join("tampered.json");
    let merged = dir.path().join("merged.json");
    write_signed_shard(
        &shard,
        "catalog-hash",
        "execution-a",
        "program-a",
        serde_json::json!([
            {
                "op_id": "vyre-test::a",
                "backend_id": "cuda",
                "passed": true,
                "message": "a matched"
            }
        ]),
    );
    let mut parsed: Value = serde_json::from_str(
        &std::fs::read_to_string(&shard).expect("Fix: shard should be readable"),
    )
    .expect("Fix: shard should parse");
    parsed["pairs"][0]["message"] = Value::String("tampered after signing".to_string());
    std::fs::write(
        &shard,
        serde_json::to_string_pretty(&parsed).expect("Fix: tampered shard should serialize"),
    )
    .expect("Fix: tampered shard should be writable");

    let output = Command::new("cargo")
        .args([
            "run",
            "-p",
            "vyre-conform-runner",
            "--no-default-features",
            "--quiet",
            "--",
            "merge",
            "--out",
        ])
        .arg(&merged)
        .arg(&shard)
        .output()
        .expect("Fix: cargo must be available in PATH");
    assert!(
        !output.status.success(),
        "Fix: merge must reject a shard whose signed body was tampered."
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("signature verification failed"),
        "Fix: merge must report signature verification failure; stderr={stderr}"
    );
    assert!(
        !merged.exists(),
        "Fix: merge must not emit an aggregate from a tampered shard."
    );
}

fn write_signed_shard(
    path: &std::path::Path,
    catalog_hash: &str,
    execution_hash: &str,
    program_hash: &str,
    pairs: Value,
) {
    let pairs_array = pairs
        .as_array()
        .expect("Fix: synthetic test pairs must be an array");
    let plan = serde_json::json!({
        "backend_count": 1,
        "op_count": pairs_array.len(),
        "pair_count": pairs_array.len(),
        "witness_case_count": pairs_array.len(),
        "catalog_hash": catalog_hash,
        "execution_hash": execution_hash,
        "selection": {
            "backend_filter": "cuda",
            "ops_filter": "all",
            "shard_index": 0,
            "shard_count": 2,
            "universe_backend_count": 3,
            "universe_op_count": 2,
            "selected_backend_count": 1,
            "selected_op_count": pairs_array.len()
        }
    });
    let key = SigningKey::from_bytes(&[7u8; 32]);
    let signable = serde_json::json!({
        "wire_format_version": 1u32,
        "program_hash": program_hash,
        "backend_id": "all",
        "plan": plan,
        "pairs": pairs,
    });
    let signable_bytes =
        serde_json::to_vec(&signable).expect("Fix: synthetic shard should serialize");
    let signature = key.sign(&signable_bytes);
    let artifact = serde_json::json!({
        "wire_format_version": 1u32,
        "program_hash": program_hash,
        "backend_id": "all",
        "plan": signable["plan"].clone(),
        "signature": hex::encode(signature.to_bytes()),
        "public_key": hex::encode(key.verifying_key().to_bytes()),
        "pairs": signable["pairs"].clone(),
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
    let signable = serde_json::json!({
        "wire_format_version": parsed["wire_format_version"].clone(),
        "program_hash": parsed["program_hash"].clone(),
        "backend_id": parsed["backend_id"].clone(),
        "plan": parsed["plan"].clone(),
        "pairs": parsed["pairs"].clone(),
    });
    let signable_bytes =
        serde_json::to_vec(&signable).expect("Fix: certificate signable body must serialize");
    verifying_key
        .verify(&signable_bytes, &signature)
        .expect("Fix: certificate Ed25519 signature must verify over the canonical body");
}

#[test]
fn prove_precomputes_reference_witnesses_once_per_entry_not_once_per_backend() {
    let source = include_str!("../src/main.rs");
    let prepare_start = source
        .find("fn prepare_reference_cases(")
        .expect("Fix: prove must keep a dedicated reference-preparation function.");
    let compare_start = source
        .find("fn compare_backend_against_reference(")
        .expect("Fix: prove must keep backend comparison isolated.");
    let compare_end = source[compare_start..]
        .find("#[derive(Clone)]\nenum BackendInputSource")
        .map(|offset| compare_start + offset)
        .expect("Fix: backend comparison boundary must remain discoverable.");
    let prepare = &source[prepare_start..compare_start];
    let compare = &source[compare_start..compare_end];

    assert!(
        prepare.contains("vyre_reference::reference_eval")
            && prepare.contains("run_cpu_fixpoint_to_convergence")
            && prepare.contains("backend_dispatch_inputs_with_plan_into(inputs, input_plan")
            && prepare.contains("reference_cases.push"),
        "Fix: prove must build reference witness outputs once during entry preparation using the same planned witness stream as backend dispatch."
    );
    let convergence_pos = prepare
        .find("if let Some(max_iterations) = convergence_max_iterations")
        .expect("Fix: convergence contracts must be handled in reference preparation.");
    let expected_pos = prepare
        .find("if let Some(expected_cases) = expected_cases")
        .expect("Fix: non-convergence ops may use declared expected_output fixtures.");
    assert!(
        convergence_pos < expected_pos,
        "Fix: convergence-contract ops must compare CUDA against CPU fixpoint witnesses, not one-step expected_output fixtures."
    );
    assert!(
        !compare.contains("vyre_reference::reference_eval")
            && !compare.contains("run_cpu_fixpoint_to_convergence"),
        "Fix: backend comparison must reuse prepared reference witness outputs instead of recomputing them for every backend."
    );
}

#[test]
fn prove_runs_selected_backends_in_parallel_workers() {
    let source = include_str!("../src/main.rs");
    let prove_start = source
        .find("fn prove(")
        .expect("Fix: prove entry point must remain discoverable.");
    let prove_region = &source[prove_start..];

    assert!(
        prove_region.contains("prove_backends_in_parallel(&backends, &prepared_entries)"),
        "Fix: prove must dispatch selected backend comparisons through the parallel backend runner instead of serializing all backend work."
    );
    assert!(
        prove_region.contains("prepare_entries_in_parallel(entries, &backends)"),
        "Fix: prove must prepare catalog witness entries through the bounded worker pool instead of serializing reference preparation."
    );
    assert!(
        source.contains("std::thread::scope"),
        "Fix: backend proof workers must use scoped threads so prepared witness data is shared without cloning the catalog-scale proof inputs."
    );
    assert!(
        source.contains("std::thread::available_parallelism()")
            && source.contains("VYRE_CONFORM_PROOF_WORKERS")
            && source.contains(".max(8)")
            && source.contains("buckets[index % worker_count].push((index, entry))"),
        "Fix: proof preparation must use bounded CPU workers with an explicit proof-worker floor/override, not one unbounded thread per op or a cgroup-collapsed serial worker."
    );
    assert!(
        source.contains("scope.spawn(move || prove_one_backend(backend, prepared_entries))"),
        "Fix: every selected backend must run in its own proof worker."
    );
    assert!(
        source.contains("let instance = match backend.acquire()")
            && source.contains("let instance = instance.as_ref();")
            && source.contains("compare_backend_against_reference(instance, &backend.id, entry)"),
        "Fix: each backend proof must acquire one backend instance and share it across shard workers so WGPU/CUDA caches are reused instead of rebuilding per shard."
    );
    assert!(
        source.contains("backend `{} proof shard worker panicked")
            || source.contains("proof shard worker panicked"),
        "Fix: each backend proof must shard catalog op comparisons across bounded workers, not serialize one slow backend over the full registry."
    );
    assert!(
        source.contains("handle.join()") && source.contains("proof worker panicked"),
        "Fix: proof worker panics must be converted into failing pair results instead of losing certificate diagnostics."
    );
    assert!(
        source.contains("VYRE_CONFORM_PROOF_TIMING")
            && source.contains("VYRE_CONFORM_PROOF_PAIR_TIMING_MS")
            && source.contains("VYRE_CONFORM_PROOF_PAIR_START")
            && source.contains("vyre-conform proof timing:")
            && source.contains("vyre-conform proof backend timing:")
            && source.contains("vyre-conform proof pair timing:")
            && source.contains("vyre-conform proof pair start:")
            && source.contains("std::time::Instant::now()"),
        "Fix: release proof runs must expose opt-in phase/backend timing so host-bound CUDA certificate regressions are diagnosable."
    );
}
#[test]
fn release_scripts_make_sharded_conformance_certificate_load_bearing() {
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("Fix: test manifest must live under conform/vyre-conform-runner");
    let prove = std::fs::read_to_string(repo.join("scripts/prove-release-shards.sh"))
        .expect("Fix: sharded release proof helper must be readable");
    assert!(
        prove.contains("vyre_select_cargo_runner"),
        "Fix: sharded release proof must use the shared OOM-safe cargo runner selector."
    );
    assert!(
        prove.contains("metadata --no-deps --format-version 1")
            && prove.contains("target_directory"),
        "Fix: release proof must discover Cargo's configured target directory instead of assuming ./target."
    );
    assert!(
        prove.contains("VYRE_RELEASE_SHARD_WORKERS") && prove.contains("wait -n"),
        "Fix: release proof shards must run through a bounded parallel worker pool."
    );
    assert!(
        prove.contains("\"$RUNNER_BIN\" \"${prove_args[@]}\"")
            && prove.contains("\"$RUNNER_BIN\" \"${merge_args[@]}\""),
        "Fix: release proof must build vyre-conform-runner once, then use the binary for prove and merge."
    );

    let signoff =
        std::fs::read_to_string(repo.join("scripts/check_signed_conformance_certificate.sh"))
            .expect("Fix: signed conformance gate must be readable");
    assert!(
        signoff.contains("scripts/prove-release-shards.sh")
            && signoff.contains("VYRE_RELEASE_BACKEND")
            && signoff.contains("VYRE_RELEASE_SHARDS"),
        "Fix: signed conformance gate must execute sharded all-backend proof, not a narrow one-off test."
    );

    let final_launch = std::fs::read_to_string(repo.join("scripts/final-launch.sh"))
        .expect("Fix: final launch script must be readable");
    assert!(
        final_launch.contains("scripts/prove-release-shards.sh")
            && final_launch.contains("release/evidence/conformance/release-all-backends-certificate.json")
            && final_launch.contains("prove sharded all-backend conformance certificate"),
        "Fix: final launch must make the merged sharded certificate load-bearing release evidence before publish."
    );
}
