use std::collections::BTreeMap;

use serde::Serialize;

use xtask::artifact_paths::{
    FRONTIER_LEADERBOARD_ARTIFACT, LEGO_AUDIT_DUPLICATES_ARTIFACT,
    REGISTERED_OP_DUPLICATES_ARTIFACT,
};

pub(crate) const EXPECTED_ARTIFACT_REGISTRY: &str =
    "release/evidence/final/expected-artifacts.json";
const EXPECTED_ARTIFACT_REGISTRY_SCHEMA_VERSION: u32 = 3;
pub(crate) const RELEASE_EVIDENCE_GENERATOR_COMMAND: &str = "xtask release-evidence";
pub(crate) const RELEASE_EVIDENCE_RUN_ARTIFACT: &str =
    "release/evidence/final/release-evidence-run.json";
pub(crate) const RELEASE_EVIDENCE_EXPECTED_ARTIFACTS: &[&str] =
    &[RELEASE_EVIDENCE_RUN_ARTIFACT, EXPECTED_ARTIFACT_REGISTRY];
pub(crate) const COMMAND_MODE_SPAWNED: &str = "spawned";
pub(crate) const COMMAND_MODE_EXTERNAL_ARTIFACTS_ONLY: &str = "external-artifacts-only";

pub(crate) fn expected_artifacts_for_args(args: &[&str]) -> Vec<&'static str> {
    let Some(&subcommand) = args.first() else {
        return Vec::new();
    };
    if subcommand == "release-benchmarks" {
        let mut backends = args
            .windows(2)
            .filter_map(|pair| (pair[0] == "--backend").then_some(pair[1]));
        let Some(backend) = backends.next() else {
            return Vec::new();
        };
        if backends.next().is_some() {
            return Vec::new();
        }
        return xtask::artifact_paths::RELEASE_BENCHMARKS_ARTIFACTS
            .iter()
            .copied()
            .filter(|artifact| {
                let is_wgpu = artifact
                    .strip_prefix("release/evidence/benchmarks/")
                    .is_some_and(|name| name.starts_with("wgpu-"));
                match backend {
                    "cuda" => !is_wgpu,
                    "wgpu" => is_wgpu,
                    _ => false,
                }
            })
            .collect();
    }
    let duplicate_report = match subcommand {
        "whats-similar" => Some(REGISTERED_OP_DUPLICATES_ARTIFACT),
        "lego-audit" => Some(LEGO_AUDIT_DUPLICATES_ARTIFACT),
        _ => None,
    };
    if let Some(expected_report) = duplicate_report {
        let mut reports = args
            .windows(2)
            .filter_map(|pair| (pair[0] == "--duplicate-report-json").then_some(pair[1]));
        return match (reports.next(), reports.next()) {
            (Some(actual), None) if actual == expected_report => vec![expected_report],
            _ => Vec::new(),
        };
    }
    expected_artifacts_for_subcommand(subcommand).to_vec()
}

fn expected_artifacts_for_subcommand(command: &str) -> &'static [&'static str] {
    xtask::gate_metadata::descriptor(command).map_or(&[], |descriptor| descriptor.artifacts)
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ReleaseExpectedArtifactRegistry {
    pub(crate) schema_version: u32,
    pub(crate) command_count: usize,
    pub(crate) artifact_count: usize,
    pub(crate) commands: Vec<ReleaseExpectedArtifactCommand>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ReleaseExpectedArtifactCommand {
    pub(crate) generator_command: String,
    pub(crate) command_mode: String,
    pub(crate) required: bool,
    pub(crate) expected_artifacts: Vec<String>,
    pub(crate) artifact_contracts: Vec<ReleaseExpectedArtifactContract>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ReleaseExpectedArtifactContract {
    pub(crate) artifact: String,
    pub(crate) generator_command: String,
    pub(crate) command_mode: String,
    pub(crate) command_required: bool,
    pub(crate) schema_version: u32,
    pub(crate) semantic_validator: String,
    pub(crate) required_fields: Vec<String>,
}

impl ReleaseExpectedArtifactCommand {
    pub(crate) fn new(
        generator_command: String,
        required: bool,
        expected_artifacts: Vec<String>,
    ) -> Self {
        Self::new_with_mode(
            generator_command,
            COMMAND_MODE_SPAWNED.to_string(),
            required,
            expected_artifacts,
        )
    }

    pub(crate) fn new_with_mode(
        generator_command: String,
        command_mode: String,
        required: bool,
        expected_artifacts: Vec<String>,
    ) -> Self {
        let artifact_contracts = artifact_contracts_for_command(
            generator_command.as_str(),
            command_mode.as_str(),
            required,
            &expected_artifacts,
        );
        Self {
            generator_command,
            command_mode,
            required,
            expected_artifacts,
            artifact_contracts,
        }
    }
}

fn artifact_contracts_for_command(
    generator_command: &str,
    command_mode: &str,
    command_required: bool,
    expected_artifacts: &[String],
) -> Vec<ReleaseExpectedArtifactContract> {
    let mut contracts = Vec::new();
    if generator_command.starts_with("xtask release-benchmarks")
        && expected_artifacts
            .iter()
            .any(|artifact| artifact == FRONTIER_LEADERBOARD_ARTIFACT)
    {
        contracts.push(ReleaseExpectedArtifactContract {
            artifact: FRONTIER_LEADERBOARD_ARTIFACT.to_string(),
            generator_command: generator_command.to_string(),
            command_mode: command_mode.to_string(),
            command_required,
            schema_version: crate::bench::release_benchmarks::FRONTIER_LEADERBOARD_SCHEMA_VERSION,
            semantic_validator:
                crate::bench::release_benchmarks::FRONTIER_LEADERBOARD_SEMANTIC_VALIDATOR
                    .to_string(),
            required_fields:
                crate::bench::release_benchmarks::frontier_leaderboard_required_artifact_fields()
                    .iter()
                    .map(|field| (*field).to_string())
                    .collect(),
        });
    }
    contracts
}

pub(crate) fn build_expected_artifact_registry(
    mut commands: Vec<ReleaseExpectedArtifactCommand>,
) -> ReleaseExpectedArtifactRegistry {
    commands.push(ReleaseExpectedArtifactCommand::new(
        RELEASE_EVIDENCE_GENERATOR_COMMAND.to_string(),
        true,
        RELEASE_EVIDENCE_EXPECTED_ARTIFACTS
            .iter()
            .map(|artifact| (*artifact).to_string())
            .collect(),
    ));
    let artifact_count = commands
        .iter()
        .map(|command| command.expected_artifacts.len())
        .sum();
    ReleaseExpectedArtifactRegistry {
        schema_version: EXPECTED_ARTIFACT_REGISTRY_SCHEMA_VERSION,
        command_count: commands.len(),
        artifact_count,
        commands,
    }
}

pub(crate) fn expected_artifact_registry_blockers(bytes: &[u8]) -> Vec<String> {
    let mut blockers = Vec::new();
    let value = match serde_json::from_slice::<serde_json::Value>(bytes) {
        Ok(value) => value,
        Err(error) => {
            return vec![format!(
                "expected artifact registry is not valid JSON: {error}"
            )];
        }
    };
    if value.get("schema_version").and_then(|raw| raw.as_u64())
        != Some(EXPECTED_ARTIFACT_REGISTRY_SCHEMA_VERSION.into())
    {
        blockers.push(format!(
            "expected artifact registry must use schema_version={EXPECTED_ARTIFACT_REGISTRY_SCHEMA_VERSION}"
        ));
    }
    let Some(commands) = value.get("commands").and_then(|raw| raw.as_array()) else {
        blockers.push("expected artifact registry must contain a commands array".to_string());
        return blockers;
    };
    if value.get("command_count").and_then(|raw| raw.as_u64()) != Some(commands.len() as u64) {
        blockers.push(
            "expected artifact registry command_count must match commands length".to_string(),
        );
    }
    let mut artifact_count = 0usize;
    let mut artifact_owners = BTreeMap::new();
    for (index, command) in commands.iter().enumerate() {
        if command
            .get("generator_command")
            .and_then(|raw| raw.as_str())
            .unwrap_or_default()
            .is_empty()
        {
            blockers.push(format!(
                "expected artifact registry command[{index}].generator_command is missing"
            ));
        }
        let generator_command = command
            .get("generator_command")
            .and_then(|raw| raw.as_str())
            .unwrap_or_default();
        let command_mode = command
            .get("command_mode")
            .and_then(|raw| raw.as_str())
            .unwrap_or_default();
        if !allowed_command_mode(command_mode) {
            blockers.push(format!(
                "expected artifact registry command[{index}].command_mode must be `{COMMAND_MODE_SPAWNED}` or `{COMMAND_MODE_EXTERNAL_ARTIFACTS_ONLY}`"
            ));
        }
        if !command
            .get("required")
            .is_some_and(serde_json::Value::is_boolean)
        {
            blockers.push(format!(
                "expected artifact registry command[{index}].required must be a boolean"
            ));
        }
        let command_required = command
            .get("required")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let Some(artifacts) = command
            .get("expected_artifacts")
            .and_then(|raw| raw.as_array())
        else {
            blockers.push(format!(
                "expected artifact registry command[{index}].expected_artifacts is missing"
            ));
            continue;
        };
        artifact_count += artifacts.len();
        for (artifact_index, artifact) in artifacts.iter().enumerate() {
            let Some(artifact) = artifact.as_str().filter(|artifact| !artifact.is_empty()) else {
                blockers.push(format!(
                    "expected artifact registry command[{index}].expected_artifacts[{artifact_index}] must be a non-empty string"
                ));
                continue;
            };
            if let Some(previous_owner) = artifact_owners.insert(artifact, generator_command) {
                blockers.push(format!(
                    "expected artifact registry artifact `{artifact}` has multiple owners: `{previous_owner}` and `{generator_command}`"
                ));
            }
        }
        let Some(contracts) = command
            .get("artifact_contracts")
            .and_then(|raw| raw.as_array())
        else {
            blockers.push(format!(
                "expected artifact registry command[{index}].artifact_contracts is missing"
            ));
            continue;
        };
        if command
            .get("generator_command")
            .and_then(|raw| raw.as_str())
            .is_some_and(|command| command.starts_with("xtask release-benchmarks"))
            && artifacts
                .iter()
                .any(|artifact| artifact.as_str() == Some(FRONTIER_LEADERBOARD_ARTIFACT))
            && !contracts.iter().any(is_frontier_leaderboard_contract)
        {
            blockers.push(format!(
                "expected artifact registry command[{index}] must declare frontier-leaderboard schema v{} semantic contract",
                crate::bench::release_benchmarks::FRONTIER_LEADERBOARD_SCHEMA_VERSION
            ));
        }
        if generator_command.starts_with("xtask release-benchmarks")
            && artifacts
                .iter()
                .any(|artifact| artifact.as_str() == Some(FRONTIER_LEADERBOARD_ARTIFACT))
            && command_mode != COMMAND_MODE_EXTERNAL_ARTIFACTS_ONLY
        {
            blockers.push(format!(
                "expected artifact registry command[{index}] release-benchmarks frontier artifacts must be `{COMMAND_MODE_EXTERNAL_ARTIFACTS_ONLY}` so release-evidence does not spawn long-running benchmarks"
            ));
        }
        for (contract_index, contract) in contracts.iter().enumerate() {
            let artifact = contract
                .get("artifact")
                .and_then(|raw| raw.as_str())
                .unwrap_or_default();
            if artifact.is_empty() {
                blockers.push(format!(
                    "expected artifact registry command[{index}].artifact_contracts[{contract_index}].artifact is missing"
                ));
            } else if !artifacts
                .iter()
                .any(|listed| listed.as_str() == Some(artifact))
            {
                blockers.push(format!(
                    "expected artifact registry command[{index}].artifact_contracts[{contract_index}].artifact `{artifact}` is not listed in expected_artifacts"
                ));
            }
            if contract
                .get("generator_command")
                .and_then(|raw| raw.as_str())
                != Some(generator_command)
            {
                blockers.push(format!(
                    "expected artifact registry command[{index}].artifact_contracts[{contract_index}].generator_command must match command generator"
                ));
            }
            if contract.get("command_mode").and_then(|raw| raw.as_str()) != Some(command_mode) {
                blockers.push(format!(
                    "expected artifact registry command[{index}].artifact_contracts[{contract_index}].command_mode must match command mode"
                ));
            }
            if contract
                .get("command_required")
                .and_then(serde_json::Value::as_bool)
                != Some(command_required)
            {
                blockers.push(format!(
                    "expected artifact registry command[{index}].artifact_contracts[{contract_index}].command_required must match command required state"
                ));
            }
            if contract
                .get("schema_version")
                .and_then(|raw| raw.as_u64())
                .unwrap_or_default()
                == 0
            {
                blockers.push(format!(
                    "expected artifact registry command[{index}].artifact_contracts[{contract_index}].schema_version must be positive"
                ));
            }
            if contract
                .get("semantic_validator")
                .and_then(|raw| raw.as_str())
                .unwrap_or_default()
                .is_empty()
            {
                blockers.push(format!(
                    "expected artifact registry command[{index}].artifact_contracts[{contract_index}].semantic_validator is missing"
                ));
            }
            if contract
                .get("required_fields")
                .and_then(serde_json::Value::as_array)
                .is_none_or(Vec::is_empty)
            {
                blockers.push(format!(
                    "expected artifact registry command[{index}].artifact_contracts[{contract_index}].required_fields must be a non-empty array"
                ));
            }
        }
    }
    if value.get("artifact_count").and_then(|raw| raw.as_u64()) != Some(artifact_count as u64) {
        blockers.push(
            "expected artifact registry artifact_count must match listed artifacts".to_string(),
        );
    }
    let has_release_evidence = commands.iter().any(|command| {
        command
            .get("generator_command")
            .and_then(|raw| raw.as_str())
            == Some(RELEASE_EVIDENCE_GENERATOR_COMMAND)
    });
    if !has_release_evidence {
        blockers.push(format!(
            "expected artifact registry must include {RELEASE_EVIDENCE_GENERATOR_COMMAND}"
        ));
    }
    blockers
}

fn allowed_command_mode(command_mode: &str) -> bool {
    matches!(
        command_mode,
        COMMAND_MODE_SPAWNED | COMMAND_MODE_EXTERNAL_ARTIFACTS_ONLY
    )
}

fn is_frontier_leaderboard_contract(contract: &serde_json::Value) -> bool {
    contract.get("artifact").and_then(|raw| raw.as_str()) == Some(FRONTIER_LEADERBOARD_ARTIFACT)
        && contract.get("schema_version").and_then(|raw| raw.as_u64())
            == Some(crate::bench::release_benchmarks::FRONTIER_LEADERBOARD_SCHEMA_VERSION.into())
        && contract
            .get("semantic_validator")
            .and_then(|raw| raw.as_str())
            == Some(crate::bench::release_benchmarks::FRONTIER_LEADERBOARD_SEMANTIC_VALIDATOR)
        && contract
            .get("required_fields")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|fields| {
                crate::bench::release_benchmarks::frontier_leaderboard_required_artifact_fields()
                    .iter()
                    .all(|required| fields.iter().any(|field| field.as_str() == Some(*required)))
            })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: one subcommand produces two disjoint backend-owned artifact sets.
    /// New artifacts are derived from the canonical producer list, so they
    /// cannot enter either set without this partition remaining exhaustive.
    #[test]
    fn release_benchmark_artifacts_have_one_backend_owner() {
        let cuda = expected_artifacts_for_args(&["release-benchmarks", "--backend", "cuda"]);
        let wgpu = expected_artifacts_for_args(&["release-benchmarks", "--backend", "wgpu"]);
        assert!(!cuda.is_empty());
        assert!(!wgpu.is_empty());
        assert!(cuda.iter().all(|artifact| !artifact
            .strip_prefix("release/evidence/benchmarks/")
            .is_some_and(|name| name.starts_with("wgpu-"))));
        assert!(wgpu.iter().all(|artifact| artifact
            .strip_prefix("release/evidence/benchmarks/")
            .is_some_and(|name| name.starts_with("wgpu-"))));
        assert!(cuda.iter().all(|artifact| !wgpu.contains(artifact)));
        assert_eq!(
            cuda.len() + wgpu.len(),
            xtask::artifact_paths::RELEASE_BENCHMARKS_ARTIFACTS.len()
        );
        assert!(expected_artifacts_for_args(&["release-benchmarks"]).is_empty());
        assert!(
            expected_artifacts_for_args(&["release-benchmarks", "--backend", "unknown"]).is_empty()
        );
        assert!(expected_artifacts_for_args(&[
            "release-benchmarks",
            "--backend",
            "cuda",
            "--backend",
            "wgpu"
        ])
        .is_empty());
    }

    #[test]
    fn release_benchmarks_expected_artifact_declares_frontier_leaderboard_contract() {
        let command = ReleaseExpectedArtifactCommand::new_with_mode(
            "xtask release-benchmarks --backend cuda".to_string(),
            COMMAND_MODE_EXTERNAL_ARTIFACTS_ONLY.to_string(),
            true,
            expected_artifacts_for_args(&["release-benchmarks", "--backend", "cuda"])
                .into_iter()
                .map(str::to_string)
                .collect(),
        );

        let contract = command
            .artifact_contracts
            .iter()
            .find(|contract| contract.artifact == FRONTIER_LEADERBOARD_ARTIFACT)
            .expect(
                "Fix: release-benchmarks must declare the frontier leaderboard semantic contract.",
            );
        assert_eq!(
            contract.schema_version,
            crate::bench::release_benchmarks::FRONTIER_LEADERBOARD_SCHEMA_VERSION
        );
        assert_eq!(
            contract.generator_command,
            "xtask release-benchmarks --backend cuda"
        );
        assert_eq!(contract.command_mode, COMMAND_MODE_EXTERNAL_ARTIFACTS_ONLY);
        assert!(contract.command_required);
        assert_eq!(
            contract.semantic_validator,
            crate::bench::release_benchmarks::FRONTIER_LEADERBOARD_SEMANTIC_VALIDATOR
        );
        assert!(contract
            .required_fields
            .iter()
            .any(|field| field == "source_suite"));
        assert!(contract.required_fields.iter().any(|field| field == "rows"));
    }

    #[test]
    fn expected_artifact_registry_rejects_release_benchmarks_without_frontier_contract() {
        let registry = br#"{
  "schema_version": 3,
  "command_count": 2,
  "artifact_count": 3,
  "commands": [
    {
      "generator_command": "xtask release-benchmarks --backend cuda",
      "command_mode": "external-artifacts-only",
      "required": true,
      "expected_artifacts": [
        "release/evidence/benchmarks/cuda-release-suite.json",
        "release/evidence/benchmarks/frontier-leaderboard.json"
      ],
      "artifact_contracts": []
    },
    {
      "generator_command": "xtask release-evidence",
      "command_mode": "spawned",
      "required": true,
      "expected_artifacts": [
        "release/evidence/final/release-evidence-run.json"
      ],
      "artifact_contracts": []
    }
  ]
}"#;

        let blockers = expected_artifact_registry_blockers(registry);

        assert!(blockers
            .iter()
            .any(|blocker| blocker.contains("frontier-leaderboard schema v1 semantic contract")));
    }

    #[test]
    fn expected_artifact_registry_rejects_malformed_contract_rows() {
        let registry = br#"{
  "schema_version": 3,
  "command_count": 2,
      "artifact_count": 3,
  "commands": [
    {
      "generator_command": "xtask version-matrix",
      "command_mode": "spawned",
      "required": true,
      "expected_artifacts": ["release/evidence/metadata/version-matrix.json"],
      "artifact_contracts": [
        {
          "artifact": "release/evidence/docs/not-listed.json",
          "schema_version": 0,
          "semantic_validator": "",
          "required_fields": []
        }
      ]
    },
    {
      "generator_command": "xtask release-evidence",
      "command_mode": "spawned",
      "required": true,
      "expected_artifacts": [
        "release/evidence/final/release-evidence-run.json",
        "release/evidence/final/expected-artifacts.json"
      ],
      "artifact_contracts": []
    }
  ]
}"#;

        let blockers = expected_artifact_registry_blockers(registry);

        assert!(blockers
            .iter()
            .any(|blocker| blocker.contains("not listed in expected_artifacts")));
        assert!(blockers
            .iter()
            .any(|blocker| blocker.contains("schema_version must be positive")));
        assert!(blockers
            .iter()
            .any(|blocker| blocker.contains("semantic_validator")));
        assert!(blockers
            .iter()
            .any(|blocker| blocker.contains("required_fields")));
    }

    #[test]
    fn expected_artifact_registry_rejects_spawned_release_benchmark_frontier_contract() {
        let registry = br#"{
  "schema_version": 3,
  "command_count": 2,
  "artifact_count": 3,
  "commands": [
    {
      "generator_command": "xtask release-benchmarks --backend cuda",
      "command_mode": "spawned",
      "required": true,
      "expected_artifacts": [
        "release/evidence/benchmarks/cuda-release-suite.json",
        "release/evidence/benchmarks/frontier-leaderboard.json"
      ],
      "artifact_contracts": [
        {
          "artifact": "release/evidence/benchmarks/frontier-leaderboard.json",
          "generator_command": "xtask release-benchmarks --backend cuda",
          "command_mode": "spawned",
          "command_required": true,
          "schema_version": 1,
          "semantic_validator": "release_benchmarks::validate_frontier_leaderboard_artifact_bytes",
          "required_fields": ["schema_version", "generator_command", "source_suite", "rows"]
        }
      ]
    },
    {
      "generator_command": "xtask release-evidence",
      "command_mode": "spawned",
      "required": true,
      "expected_artifacts": ["release/evidence/final/release-evidence-run.json"],
      "artifact_contracts": []
    }
  ]
}"#;

        let blockers = expected_artifact_registry_blockers(registry);

        assert!(blockers
            .iter()
            .any(|blocker| blocker.contains("external-artifacts-only")));
    }

    #[test]
    fn expected_artifact_registry_rejects_contract_command_provenance_drift() {
        let registry = br#"{
  "schema_version": 3,
  "command_count": 2,
  "artifact_count": 2,
  "commands": [
    {
      "generator_command": "xtask version-matrix",
      "command_mode": "spawned",
      "required": true,
      "expected_artifacts": ["release/evidence/metadata/version-matrix.json"],
      "artifact_contracts": [
        {
          "artifact": "release/evidence/metadata/version-matrix.json",
          "generator_command": "xtask other",
          "command_mode": "external-artifacts-only",
          "command_required": false,
          "schema_version": 1,
          "semantic_validator": "version_matrix::validate",
          "required_fields": ["schema_version"]
        }
      ]
    },
    {
      "generator_command": "xtask release-evidence",
      "command_mode": "spawned",
      "required": true,
      "expected_artifacts": ["release/evidence/final/release-evidence-run.json"],
      "artifact_contracts": []
    }
  ]
}"#;

        let blockers = expected_artifact_registry_blockers(registry);

        assert!(blockers
            .iter()
            .any(|blocker| blocker.contains("generator_command must match")));
        assert!(blockers
            .iter()
            .any(|blocker| blocker.contains("command_mode must match")));
        assert!(blockers
            .iter()
            .any(|blocker| blocker.contains("command_required must match")));
    }

    #[test]
    fn expected_artifact_registry_rejects_multiple_artifact_owners() {
        let registry = br#"{
  "schema_version": 3,
  "command_count": 2,
  "artifact_count": 2,
  "commands": [
    {
      "generator_command": "xtask first",
      "command_mode": "spawned",
      "required": true,
      "expected_artifacts": ["release/evidence/shared.json"],
      "artifact_contracts": []
    },
    {
      "generator_command": "xtask release-evidence",
      "command_mode": "spawned",
      "required": true,
      "expected_artifacts": ["release/evidence/shared.json"],
      "artifact_contracts": []
    }
  ]
}"#;

        let blockers = expected_artifact_registry_blockers(registry);

        assert!(blockers.iter().any(|blocker| {
            blocker.contains("multiple owners")
                && blocker.contains("xtask first")
                && blocker.contains("xtask release-evidence")
        }));
    }
}
