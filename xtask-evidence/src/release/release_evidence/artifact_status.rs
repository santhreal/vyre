use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::Serialize;

use xtask::artifact_paths::FRONTIER_LEADERBOARD_ARTIFACT;
use xtask::gates::dedup_report::validate_duplicate_family_report_artifact;
use xtask::hash::sha256_hex;

use super::expected_artifacts::{
    expected_artifact_registry_blockers, COMMAND_MODE_EXTERNAL_ARTIFACTS_ONLY,
    COMMAND_MODE_SPAWNED, EXPECTED_ARTIFACT_REGISTRY,
};

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ReleaseEvidenceArtifactStatus {
    pub(crate) path: String,
    pub(crate) exists: bool,
    pub(crate) bytes: u64,
    pub(crate) read_error: Option<String>,
    pub(crate) owner_lane: &'static str,
    pub(crate) generator_command: String,
    pub(crate) command_mode: &'static str,
    pub(crate) content_sha256: Option<String>,
    pub(crate) source_fingerprint: Option<String>,
    pub(crate) freshness_fingerprint: Option<String>,
    pub(crate) blockers: Vec<String>,
}

/// Statuses for artifacts this same run is about to write, taken from the bytes
/// it will write rather than from the copy still on disk.
///
/// `release-evidence` owns two artifacts and one of them records the other's
/// digest, so reading the tree recorded the digest of the previous run. One
/// `--write` then left the artifact disagreeing with the tree it had just
/// written, and the gate only went quiet after a second and third write. A
/// generator that cannot settle in one pass reports a stale tree as a defect
/// and its own output as clean.
pub(crate) fn inspect_pending_artifacts(
    workspace_root: &Path,
    command_args: &[&'static str],
    expected_artifacts: &[&'static str],
    pending: &BTreeMap<&str, &str>,
) -> Vec<ReleaseEvidenceArtifactStatus> {
    inspect(
        workspace_root,
        command_args,
        expected_artifacts,
        COMMAND_MODE_SPAWNED,
        pending,
    )
}

pub(crate) fn inspect_expected_artifacts_with_mode(
    workspace_root: &Path,
    command_args: &[&'static str],
    expected_artifacts: &[&'static str],
    command_mode: &'static str,
) -> Vec<ReleaseEvidenceArtifactStatus> {
    inspect(
        workspace_root,
        command_args,
        expected_artifacts,
        command_mode,
        &BTreeMap::new(),
    )
}

fn inspect(
    workspace_root: &Path,
    command_args: &[&'static str],
    expected_artifacts: &[&'static str],
    command_mode: &'static str,
    pending: &BTreeMap<&str, &str>,
) -> Vec<ReleaseEvidenceArtifactStatus> {
    let owner_lane = owner_lane_for_command(command_args);
    let generator_command = generator_command(command_args);
    expected_artifacts
        .iter()
        .map(|artifact| {
            if let Some(bytes) = pending.get(artifact) {
                return pending_status(
                    artifact,
                    bytes.as_bytes(),
                    owner_lane,
                    &generator_command,
                    command_mode,
                );
            }
            let path = workspace_root.join(artifact);
            match fs::metadata(&path) {
                Ok(metadata) => {
                    let (content_sha256, read_error, semantic_blockers) = if metadata.is_file() {
                        match fs::read(&path) {
                            Ok(bytes) => (
                                Some(sha256_hex(&bytes)),
                                None,
                                artifact_semantic_blockers(
                                    artifact,
                                    &bytes,
                                    &generator_command,
                                    command_mode,
                                ),
                            ),
                            Err(error) => (None, Some(error.to_string()), Vec::new()),
                        }
                    } else {
                        (
                            None,
                            Some("expected artifact path is not a file".to_string()),
                            Vec::new(),
                        )
                    };
                    let (source_fingerprint, freshness_fingerprint) =
                        artifact_provenance_fingerprints(
                            artifact,
                            &generator_command,
                            metadata.len(),
                            content_sha256.as_deref(),
                        );
                    let mut blockers = artifact_provenance_blockers(
                        metadata.is_file(),
                        metadata.len(),
                        read_error.as_deref(),
                        source_fingerprint.as_deref(),
                        freshness_fingerprint.as_deref(),
                    );
                    blockers.extend(semantic_blockers);
                    ReleaseEvidenceArtifactStatus {
                        path: (*artifact).to_string(),
                        exists: metadata.is_file(),
                        bytes: metadata.len(),
                        read_error,
                        owner_lane,
                        generator_command: generator_command.clone(),
                        command_mode,
                        content_sha256,
                        source_fingerprint,
                        freshness_fingerprint,
                        blockers,
                    }
                }
                Err(error) => ReleaseEvidenceArtifactStatus {
                    path: (*artifact).to_string(),
                    exists: false,
                    bytes: 0,
                    read_error: Some(error.to_string()),
                    owner_lane,
                    generator_command: generator_command.clone(),
                    command_mode,
                    content_sha256: None,
                    source_fingerprint: None,
                    freshness_fingerprint: None,
                    blockers: artifact_provenance_blockers(
                        false,
                        0,
                        Some(&error.to_string()),
                        None,
                        None,
                    ),
                },
            }
        })
        .collect()
}

/// The status an artifact will have once `bytes` are on disk.
///
/// Every field the on-disk arm derives from the file is derived from the bytes
/// instead, so the record states the artifact this run produces. The file need
/// not exist yet, which is the case this exists for: the first run in a fresh
/// checkout writes both artifacts and neither can be read before the other.
fn pending_status(
    artifact: &str,
    bytes: &[u8],
    owner_lane: &'static str,
    generator_command: &str,
    command_mode: &'static str,
) -> ReleaseEvidenceArtifactStatus {
    let length = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    let content_sha256 = sha256_hex(bytes);
    let (source_fingerprint, freshness_fingerprint) = artifact_provenance_fingerprints(
        artifact,
        generator_command,
        length,
        Some(content_sha256.as_str()),
    );
    let mut blockers = artifact_provenance_blockers(
        true,
        length,
        None,
        source_fingerprint.as_deref(),
        freshness_fingerprint.as_deref(),
    );
    blockers.extend(artifact_semantic_blockers(
        artifact,
        bytes,
        generator_command,
        command_mode,
    ));
    ReleaseEvidenceArtifactStatus {
        path: artifact.to_string(),
        exists: true,
        bytes: length,
        read_error: None,
        owner_lane,
        generator_command: generator_command.to_string(),
        command_mode,
        content_sha256: Some(content_sha256),
        source_fingerprint,
        freshness_fingerprint,
        blockers,
    }
}

pub(crate) fn artifact_semantic_blockers(
    artifact: &str,
    bytes: &[u8],
    expected_generator_command: &str,
    command_mode: &str,
) -> Vec<String> {
    let mut blockers =
        xtask::release::repo_boundary::public_artifact_boundary_blockers(artifact, bytes);
    if artifact.starts_with("release/evidence/dedup/") {
        blockers.extend(validate_duplicate_family_report_artifact(
            bytes,
            expected_generator_command,
        ));
        return blockers;
    }
    if artifact == EXPECTED_ARTIFACT_REGISTRY {
        blockers.extend(expected_artifact_registry_blockers(bytes));
        return blockers;
    }
    if artifact == FRONTIER_LEADERBOARD_ARTIFACT {
        match serde_json::from_slice::<serde_json::Value>(bytes) {
            Ok(value) => blockers.extend(external_benchmark_artifact_freshness_blockers(
                artifact,
                &value,
                expected_generator_command,
                command_mode,
            )),
            Err(error) => blockers.push(format!(
                "external benchmark artifact `{artifact}` is not valid JSON: {error}"
            )),
        }
        blockers.extend(
            crate::bench::release_benchmarks::validate_frontier_leaderboard_artifact_bytes(bytes),
        );
        return blockers;
    }
    if is_release_benchmark_semantic_artifact(artifact) {
        let value = match serde_json::from_slice::<serde_json::Value>(bytes) {
            Ok(value) => value,
            Err(error) => {
                blockers.push(format!(
                    "benchmark artifact `{artifact}` is not valid JSON: {error}"
                ));
                return blockers;
            }
        };
        blockers.extend(external_benchmark_artifact_freshness_blockers(
            artifact,
            &value,
            expected_generator_command,
            command_mode,
        ));
        blockers.extend(
            crate::bench::benchmark_evidence_semantics::benchmark_evidence_blocker_issues(
                artifact, &value,
            ),
        );
        return blockers;
    }
    blockers
}

fn external_benchmark_artifact_freshness_blockers(
    artifact: &str,
    value: &serde_json::Value,
    expected_generator_command: &str,
    command_mode: &str,
) -> Vec<String> {
    let mut blockers = Vec::new();
    if !expected_generator_command.starts_with("xtask release-benchmarks") {
        blockers.push(format!(
            "external benchmark artifact `{artifact}` generator_command `{expected_generator_command}` must start with `xtask release-benchmarks`"
        ));
    }
    if command_mode != COMMAND_MODE_EXTERNAL_ARTIFACTS_ONLY {
        blockers.push(format!(
            "external benchmark artifact `{artifact}` command_mode `{command_mode}` must be `{COMMAND_MODE_EXTERNAL_ARTIFACTS_ONLY}` so release-evidence inspects existing benchmark artifacts without spawning release-benchmarks"
        ));
    }
    if value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0)
        == 0
    {
        blockers.push(format!(
            "external benchmark artifact `{artifact}` must declare positive schema_version"
        ));
    }
    if artifact.ends_with("cuda-release-suite.json")
        || artifact.ends_with("wgpu-fallback-suite.json")
    {
        blockers.extend(
            crate::bench::benchmark_evidence_semantics::benchmark_schema_digest_chain_issues(
                artifact,
                value,
                "backend-suite",
            ),
        );
        let chain = value.get("schema_digest_chain");
        for field in ["source_digest", "command_digest", "hardware_digest"] {
            if chain
                .and_then(|chain| chain.get(field))
                .and_then(serde_json::Value::as_str)
                .is_none_or(|digest| digest.trim().is_empty())
            {
                blockers.push(format!(
                    "external benchmark artifact `{artifact}` schema_digest_chain.{field} is blank or missing"
                ));
            }
        }
        let hardware_digest = value
            .get("hardware_digest")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if !hardware_digest.starts_with("benchmark-hardware-digest:v1:") {
            blockers.push(format!(
                "external benchmark artifact `{artifact}` hardware_digest must be a benchmark-hardware-digest:v1 digest"
            ));
        }
    }
    blockers
}

fn is_release_benchmark_semantic_artifact(artifact: &str) -> bool {
    matches!(
        artifact,
        "release/evidence/benchmarks/cuda-release-suite.json"
            | "release/evidence/benchmarks/wgpu-fallback-suite.json"
            | "release/evidence/benchmarks/bench-release-axes.json"
            | "release/evidence/benchmarks/cpu-only-100x-proof.json"
            | "release/evidence/optimization/pass-family-benchmark-manifest.json"
    ) || artifact == FRONTIER_LEADERBOARD_ARTIFACT
}

pub(crate) fn generator_command(command_args: &[&str]) -> String {
    let mut command = String::from("xtask");
    for arg in command_args {
        command.push(' ');
        command.push_str(arg);
    }
    command
}

fn owner_lane_for_command(command_args: &[&str]) -> &'static str {
    match command_args.first().copied().unwrap_or_default() {
        "backend-matrix" | "conformance-matrix" => "driver_shared",
        "release-workload-matrix" | "release-benchmarks" => "bench_harness",
        "optimization-corpus" | "optimization-matrix" => "foundation_optimizer",
        "whats-similar" | "lego-audit" => "testing_evidence",
        "docs-check" | "hygiene-matrix" | "release-evidence" => "testing_evidence",
        "version-matrix" | "metadata-matrix" | "feature-matrix" => "coordination",
        _ => "coordination",
    }
}

fn artifact_provenance_fingerprints(
    artifact: &str,
    generator_command: &str,
    bytes: u64,
    content_sha256: Option<&str>,
) -> (Option<String>, Option<String>) {
    let Some(content_sha256) = content_sha256 else {
        return (None, None);
    };
    let source_material = format!(
        "release-evidence-source:v1\ngenerator={generator_command}\nartifact={artifact}\nbytes={bytes}\ncontent_sha256={content_sha256}\n"
    );
    let freshness_material = format!(
        "release-evidence-freshness:v1\nartifact={artifact}\ngenerator={generator_command}\nsource={}\n",
        sha256_hex(source_material.as_bytes())
    );
    (
        Some(format!(
            "release-evidence-source:v1:{}",
            sha256_hex(source_material.as_bytes())
        )),
        Some(format!(
            "release-evidence-freshness:v1:{}",
            sha256_hex(freshness_material.as_bytes())
        )),
    )
}

fn artifact_provenance_blockers(
    exists: bool,
    bytes: u64,
    read_error: Option<&str>,
    source_fingerprint: Option<&str>,
    freshness_fingerprint: Option<&str>,
) -> Vec<String> {
    let mut blockers = Vec::new();
    if !exists {
        blockers.push("artifact is missing or not a file".to_string());
    }
    if bytes == 0 {
        blockers.push("artifact is empty".to_string());
    }
    if let Some(error) = read_error {
        blockers.push(format!("artifact is unreadable: {error}"));
    }
    if source_fingerprint.is_none() {
        blockers.push("artifact is missing source_fingerprint".to_string());
    }
    if freshness_fingerprint.is_none() {
        blockers.push("artifact is missing freshness_fingerprint".to_string());
    }
    blockers
}

pub(crate) fn release_artifact_status_has_failure(status: &ReleaseEvidenceArtifactStatus) -> bool {
    !status.exists
        || status.bytes == 0
        || status.read_error.is_some()
        || status.source_fingerprint.is_none()
        || status.freshness_fingerprint.is_none()
        || !status.blockers.is_empty()
}

pub(crate) fn artifact_blocker_suffix(status: &ReleaseEvidenceArtifactStatus) -> String {
    if status.blockers.is_empty() {
        return status
            .read_error
            .as_ref()
            .map(|error| format!(": {error}"))
            .unwrap_or_default();
    }
    format!(": {}", status.blockers.join("; "))
}
