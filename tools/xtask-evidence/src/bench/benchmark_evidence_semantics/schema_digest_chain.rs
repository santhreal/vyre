//! The digest chain that binds a benchmark artifact to everything it was
//! produced from.
//!
//! One artifact is only reproducible if the schema, command line, config,
//! hardware, dataset and comparator that produced it are all named, so those
//! six digests are hashed into one chain value. The chain is built and checked
//! by the same material function, which is what stops a writer and a reader
//! from disagreeing about what was covered.

use serde_json::Value;

use super::data::BENCHMARK_SCHEMA_DIGEST_CHAIN_PREFIX;

pub(crate) fn benchmark_schema_digest_chain_value(
    artifact_kind: &str,
    artifact_schema_version: u32,
    source_digest: &str,
    command_digest: &str,
    config_digest: &str,
    hardware_digest: &str,
    dataset_digest: &str,
    comparator_version: &str,
) -> Value {
    let material = benchmark_schema_digest_chain_material(
        artifact_kind,
        artifact_schema_version,
        source_digest,
        command_digest,
        config_digest,
        hardware_digest,
        dataset_digest,
        comparator_version,
    );
    let digest = format!(
        "{BENCHMARK_SCHEMA_DIGEST_CHAIN_PREFIX}{}",
        xtask::hash::sha256_hex(material.as_bytes())
    );
    serde_json::json!({
        "schema": "benchmark-schema-digest-chain",
        "schema_version": 1,
        "artifact_kind": artifact_kind,
        "artifact_schema_version": artifact_schema_version,
        "source_digest": source_digest,
        "command_digest": command_digest,
        "config_digest": config_digest,
        "hardware_digest": hardware_digest,
        "dataset_digest": dataset_digest,
        "comparator_version": comparator_version,
        "digest": digest,
    })
}

pub(crate) fn benchmark_schema_digest_chain_issues(
    evidence: &str,
    artifact: &Value,
    expected_artifact_kind: &str,
) -> Vec<String> {
    let mut issues = Vec::new();
    let Some(chain) = artifact.get("schema_digest_chain") else {
        issues.push(format!("{evidence}: missing schema_digest_chain"));
        return issues;
    };
    if chain.get("schema").and_then(Value::as_str) != Some("benchmark-schema-digest-chain") {
        issues.push(format!(
            "{evidence}: schema_digest_chain.schema must be benchmark-schema-digest-chain"
        ));
    }
    if chain.get("schema_version").and_then(Value::as_u64) != Some(1) {
        issues.push(format!(
            "{evidence}: schema_digest_chain.schema_version must be 1"
        ));
    }
    if chain.get("artifact_kind").and_then(Value::as_str) != Some(expected_artifact_kind) {
        issues.push(format!(
            "{evidence}: schema_digest_chain.artifact_kind must be `{expected_artifact_kind}`"
        ));
    }
    let artifact_schema_version = artifact
        .get("schema_version")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if chain.get("artifact_schema_version").and_then(Value::as_u64) != Some(artifact_schema_version)
    {
        issues.push(format!(
            "{evidence}: schema_digest_chain.artifact_schema_version must match artifact schema_version={artifact_schema_version}"
        ));
    }
    for field in [
        "source_digest",
        "command_digest",
        "config_digest",
        "hardware_digest",
        "dataset_digest",
        "comparator_version",
    ] {
        if chain
            .get(field)
            .and_then(Value::as_str)
            .is_none_or(|value| value.trim().is_empty())
        {
            issues.push(format!(
                "{evidence}: schema_digest_chain.{field} is blank or missing"
            ));
        }
    }
    let digest = chain.get("digest").and_then(Value::as_str).unwrap_or("");
    if !digest.starts_with(BENCHMARK_SCHEMA_DIGEST_CHAIN_PREFIX) {
        issues.push(format!(
            "{evidence}: schema_digest_chain.digest must start with {BENCHMARK_SCHEMA_DIGEST_CHAIN_PREFIX}"
        ));
    }
    if let (
        Some(artifact_kind),
        Some(source_digest),
        Some(command_digest),
        Some(config_digest),
        Some(hardware_digest),
        Some(dataset_digest),
        Some(comparator_version),
    ) = (
        chain.get("artifact_kind").and_then(Value::as_str),
        chain.get("source_digest").and_then(Value::as_str),
        chain.get("command_digest").and_then(Value::as_str),
        chain.get("config_digest").and_then(Value::as_str),
        chain.get("hardware_digest").and_then(Value::as_str),
        chain.get("dataset_digest").and_then(Value::as_str),
        chain.get("comparator_version").and_then(Value::as_str),
    ) {
        let recomputed = benchmark_schema_digest_chain_value(
            artifact_kind,
            artifact_schema_version as u32,
            source_digest,
            command_digest,
            config_digest,
            hardware_digest,
            dataset_digest,
            comparator_version,
        );
        if recomputed
            .get("digest")
            .and_then(Value::as_str)
            .is_some_and(|expected| expected != digest)
        {
            issues.push(format!(
                "{evidence}: schema_digest_chain.digest does not match chained benchmark provenance fields"
            ));
        }
    }
    issues
}

fn benchmark_schema_digest_chain_material(
    artifact_kind: &str,
    artifact_schema_version: u32,
    source_digest: &str,
    command_digest: &str,
    config_digest: &str,
    hardware_digest: &str,
    dataset_digest: &str,
    comparator_version: &str,
) -> String {
    format!(
        "benchmark-schema-digest-chain:v1\nartifact_kind={artifact_kind}\nartifact_schema_version={artifact_schema_version}\nsource_digest={source_digest}\ncommand_digest={command_digest}\nconfig_digest={config_digest}\nhardware_digest={hardware_digest}\ndataset_digest={dataset_digest}\ncomparator_version={comparator_version}\n"
    )
}
