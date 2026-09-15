use super::*;

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
                "executor_id": "cuda",
                "passed": true,
                "message": "a matched"
            }
        ]),
        serde_json::json!([
            {
                "op_id": "vyre-test::a",
                "law": "Commutative",
                "witness": "ExchangedInputs",
                "cases": 4
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
                "executor_id": "cuda",
                "passed": true,
                "message": "b matched"
            }
        ]),
        serde_json::json!([
            {
                "op_id": "vyre-test::b",
                "law": "Idempotent",
                "witness": "Reapplied",
                "cases": 3
            }
        ]),
    );

    let parsed = merge_shards(&merged, &shard_a, &shard_b);
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
    let laws = parsed["laws"]
        .as_array()
        .expect("Fix: merged certificate must carry the proven-law roster.");
    let proven: Vec<(&str, &str)> = laws
        .iter()
        .map(|law| {
            (
                law["op_id"].as_str().unwrap_or_default(),
                law["law"].as_str().unwrap_or_default(),
            )
        })
        .collect();
    assert_eq!(
        proven,
        vec![
            ("vyre-test::a", "Commutative"),
            ("vyre-test::b", "Idempotent")
        ],
        "Fix: merged certificate must carry every law each shard proved."
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
                "executor_id": "cuda",
                "passed": true,
                "message": "a matched"
            }
        ]),
        serde_json::json!([]),
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

    let output = Command::new(conform_binary())
        .args(["merge", "--out"])
        .arg(&merged)
        .arg(&shard)
        .output()
        .expect("Fix: the built vyre-conform binary must launch");
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

/// Every merge refusal these three cases exercise, driven through the built
/// binary so the refusal wording is the one an operator reads.
fn merge_refusal(shards: &[&std::path::Path], merged: &std::path::Path) -> String {
    let mut command = Command::new(conform_binary());
    command.args(["merge", "--out"]).arg(merged);
    for shard in shards {
        command.arg(shard);
    }
    let output = command
        .output()
        .expect("Fix: the built vyre-conform binary must launch");
    assert!(
        !output.status.success(),
        "Fix: merge must refuse this shard set."
    );
    assert!(
        !merged.exists(),
        "Fix: merge must not emit an aggregate it refused."
    );
    String::from_utf8_lossy(&output.stderr).to_string()
}

/// WHY: the proven-law roster is the only record that a declared law was
/// executed against the oracle. Left out of the signed body, an attacker could
/// append or delete law rows on a valid certificate at no cost.
#[test]
fn merge_rejects_a_shard_whose_law_roster_was_edited_after_signing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let shard = dir.path().join("law-tampered.json");
    let merged = dir.path().join("merged.json");
    write_signed_shard(
        &shard,
        "catalog-hash",
        "execution-a",
        "program-a",
        serde_json::json!([
            {
                "op_id": "vyre-test::a",
                "executor_id": "cuda",
                "passed": true,
                "message": "a matched"
            }
        ]),
        serde_json::json!([
            {
                "op_id": "vyre-test::a",
                "law": "Commutative",
                "witness": "ExchangedInputs",
                "cases": 4
            }
        ]),
    );
    let mut parsed: Value = serde_json::from_str(
        &std::fs::read_to_string(&shard).expect("Fix: shard should be readable"),
    )
    .expect("Fix: shard should parse");
    parsed["laws"][0]["law"] = Value::String("Associative".to_string());
    std::fs::write(
        &shard,
        serde_json::to_string_pretty(&parsed).expect("Fix: tampered shard should serialize"),
    )
    .expect("Fix: tampered shard should be writable");

    let stderr = merge_refusal(&[shard.as_path()], &merged);
    assert!(
        stderr.contains("signature verification failed"),
        "Fix: an edited law roster must fail signature verification; stderr={stderr}"
    );
}

/// WHY: two shards proving the same law against the same registry must agree.
/// A disagreement means one of them ran a different oracle, and silently
/// keeping either row would publish a proof nothing produced.
#[test]
fn merge_rejects_shards_that_disagree_about_one_law_proof() {
    let dir = tempfile::tempdir().expect("tempdir");
    let shard_a = dir.path().join("agree-a.json");
    let shard_b = dir.path().join("agree-b.json");
    let merged = dir.path().join("merged.json");
    for (path, backend, witness) in [
        (&shard_a, "cuda", "ExchangedInputs"),
        (&shard_b, "metal", "PermutedElements"),
    ] {
        write_signed_shard(
            path,
            "catalog-hash",
            "execution-a",
            "program-a",
            serde_json::json!([
                {
                    "op_id": "vyre-test::a",
                    "executor_id": backend,
                    "passed": true,
                    "message": "a matched"
                }
            ]),
            serde_json::json!([
                {
                    "op_id": "vyre-test::a",
                    "law": "Commutative",
                    "witness": witness,
                    "cases": 4
                }
            ]),
        );
    }

    let stderr = merge_refusal(&[shard_a.as_path(), shard_b.as_path()], &merged);
    assert!(
        stderr.contains("shards disagree about its proof"),
        "Fix: merge must name the disagreeing law pair; stderr={stderr}"
    );
}

/// WHY: a v1 certificate predates the proven-law roster, so merging one would
/// emit an aggregate whose law set is silently empty rather than proven.
#[test]
fn merge_rejects_a_certificate_that_predates_the_proven_law_roster() {
    let dir = tempfile::tempdir().expect("tempdir");
    let shard = dir.path().join("v1.json");
    let merged = dir.path().join("merged.json");
    write_signed_shard(
        &shard,
        "catalog-hash",
        "execution-a",
        "program-a",
        serde_json::json!([
            {
                "op_id": "vyre-test::a",
                "executor_id": "cuda",
                "passed": true,
                "message": "a matched"
            }
        ]),
        serde_json::json!([]),
    );
    let mut parsed: Value = serde_json::from_str(
        &std::fs::read_to_string(&shard).expect("Fix: shard should be readable"),
    )
    .expect("Fix: shard should parse");
    parsed
        .as_object_mut()
        .expect("Fix: a certificate is a JSON object")
        .remove("laws");
    parsed["wire_format_version"] = Value::from(1u32);
    std::fs::write(
        &shard,
        serde_json::to_string_pretty(&parsed).expect("Fix: downgraded shard should serialize"),
    )
    .expect("Fix: downgraded shard should be writable");

    let stderr = merge_refusal(&[shard.as_path()], &merged);
    assert!(
        stderr.contains("wire_format_version 1"),
        "Fix: merge must name the refused wire format; stderr={stderr}"
    );
}
