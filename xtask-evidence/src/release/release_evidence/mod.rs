//! Hold the release evidence set to the artifacts every generator owes it.
//!
//! This is the census, not the runner. Every generator named in `COMMANDS` is
//! a registered gate, so the sweep runs each one and fails on it directly. What
//! lives here is the record of which generator owes which artifact and whether
//! the artifact is on disk, readable and carrying provenance.

use std::path::Path;

use serde::Serialize;
use xtask::artifact_gate::Inspection;

mod artifact_status;
mod evidence_index;
pub(crate) mod expected_artifacts;

use artifact_status::{
    artifact_blocker_suffix, generator_command, inspect_expected_artifacts,
    inspect_expected_artifacts_with_mode, release_artifact_status_has_failure,
    ReleaseEvidenceArtifactStatus,
};
use evidence_index::{build_evidence_index, ReleaseEvidenceIndex};
pub(crate) use expected_artifacts::expected_artifacts_for_command;
use expected_artifacts::{
    build_expected_artifact_registry, ReleaseExpectedArtifactCommand,
    ReleaseExpectedArtifactRegistry, COMMAND_MODE_EXTERNAL_ARTIFACTS_ONLY, COMMAND_MODE_SPAWNED,
    EXPECTED_ARTIFACT_REGISTRY, RELEASE_EVIDENCE_GENERATOR_COMMAND, RELEASE_EVIDENCE_RUN_ARTIFACT,
};
use xtask::artifact_paths::{LEGO_AUDIT_DUPLICATES_ARTIFACT, REGISTERED_OP_DUPLICATES_ARTIFACT};

/// Bumped from 4: the record no longer carries spawn outcomes, because nothing
/// is spawned. Each generator is a registered gate the sweep runs on its own.
const RELEASE_EVIDENCE_RUN_SCHEMA_VERSION: u32 = 5;

const COMMANDS: &[EvidenceCommand] = &[
    EvidenceCommand::required(&["docs-check"]),
    EvidenceCommand::required(&["version-matrix"]),
    EvidenceCommand::required(&["backend-matrix"]),
    EvidenceCommand::required(&["conformance-matrix"]),
    EvidenceCommand::required(&["release-workload-matrix", "--enforce"]),
    EvidenceCommand::external_required(&["release-benchmarks", "--backend", "cuda"]),
    EvidenceCommand::required(&["hygiene-matrix"]),
    EvidenceCommand::required(&["metadata-matrix"]),
    EvidenceCommand::required(&["feature-matrix"]),
    EvidenceCommand::required(&["package-readiness"]),
    EvidenceCommand::required(&["optimization-corpus"]),
    EvidenceCommand::required(&["optimization-matrix"]),
    EvidenceCommand::required(&[
        "whats-similar",
        "--all",
        "--duplicate-report-json",
        REGISTERED_OP_DUPLICATES_ARTIFACT,
    ]),
    EvidenceCommand::required(&[
        "lego-audit",
        "--report-only",
        "--duplicate-report-json",
        LEGO_AUDIT_DUPLICATES_ARTIFACT,
    ]),
];

/// Every subcommand this table names, in table order.
///
/// Read by the dispatch contract: a name this table still spells after the
/// subcommand was renamed or moved would otherwise fail only when a release
/// ran. Every entry is checked now, not only the ones that used to be spawned.
#[must_use]
pub fn covered_subcommands() -> Vec<&'static str> {
    COMMANDS
        .iter()
        .filter_map(|command| command.args.first().copied())
        .collect()
}

struct EvidenceCommand {
    args: &'static [&'static str],
    required: bool,
    /// True when the sweep runs this gate itself, false when the artifact comes
    /// from an out-of-band measured run such as `release-benchmarks --backend cuda`.
    in_sweep: bool,
}

#[derive(Debug, Serialize)]
struct ReleaseEvidenceRun {
    schema_version: u32,
    total_commands: usize,
    artifact_failures: usize,
    command_count: usize,
    required_command_count: usize,
    report_only_command_count: usize,
    commands: Vec<ReleaseEvidenceCommandRecord>,
    final_artifacts: Vec<ReleaseEvidenceArtifactStatus>,
    evidence_index: ReleaseEvidenceIndex,
    expected_artifact_registry: ReleaseExpectedArtifactRegistry,
    blockers: Vec<String>,
    reports: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ReleaseEvidenceCommandRecord {
    args: Vec<&'static str>,
    required: bool,
    expected_artifacts: Vec<&'static str>,
    mode: &'static str,
    artifact_statuses: Vec<ReleaseEvidenceArtifactStatus>,
}

impl EvidenceCommand {
    const fn required(args: &'static [&'static str]) -> Self {
        Self {
            args,
            required: true,
            in_sweep: true,
        }
    }

    const fn external_required(args: &'static [&'static str]) -> Self {
        Self {
            args,
            required: true,
            in_sweep: false,
        }
    }

    /// The subcommand this row names, which is what an artifact list is keyed on.
    const fn subcommand(&self) -> &'static str {
        match self.args.first() {
            Some(first) => first,
            None => "",
        }
    }

    const fn mode(&self) -> &'static str {
        if self.in_sweep {
            COMMAND_MODE_SPAWNED
        } else {
            COMMAND_MODE_EXTERNAL_ARTIFACTS_ONLY
        }
    }
}

xtask::artifact_gate! {
    /// Holds the release evidence set to the artifacts every generator owes it.
    ReleaseEvidenceGate,
    name: "release-evidence",
    help: "Regenerate release/evidence/final/release-evidence-run.json and expected-artifacts.json \
       and report each line the committed copies disagree on. Proves every required generator \
       declares at least one expected artifact, and that every declared artifact exists, is \
       non-empty, is readable and carries provenance. Proves nothing about whether those \
       generators pass: it no longer runs them. Each one is a registered gate, so the sweep \
       runs it and fails on it directly rather than through a spawn this gate reports \
       second-hand.",
    inspect: |ctx| inspect(&ctx.root),
}

/// The state of the release evidence set, and the two artifacts recording it.
///
/// This used to spawn twelve subcommands as child processes and report their
/// exit codes. Every one of them is a registered gate now, so the sweep runs
/// each directly: a failure arrives with its own findings instead of as one
/// line saying a child exited non-zero. What remains here, and exists nowhere
/// else, is the census: which generator owes which artifact, and whether the
/// artifact is there.
fn inspect(workspace_root: &Path) -> Inspection {
    let mut inspection = Inspection::new();
    let mut failures = Vec::new();
    let mut reports = Vec::new();
    let mut records = Vec::new();
    for command in COMMANDS {
        let label = format!("xtask {}", command.args.join(" "));
        let expected = expected_artifacts_for_command(command.subcommand());
        if command.required && expected.is_empty() {
            failures.push(format!(
                "`{label}` is required but declares no expected artifacts"
            ));
        }
        let artifact_statuses = inspect_expected_artifacts_with_mode(
            workspace_root,
            command.args,
            expected,
            command.mode(),
        );
        for artifact in &artifact_statuses {
            if release_artifact_status_has_failure(artifact) {
                let finding = format!(
                    "`{label}` expected `{}` but it was missing, empty, unreadable, or missing provenance{}",
                    artifact.path,
                    artifact_blocker_suffix(artifact)
                );
                if command.required {
                    failures.push(finding);
                } else {
                    reports.push(finding);
                }
            }
        }
        records.push(ReleaseEvidenceCommandRecord {
            args: command.args.to_vec(),
            required: command.required,
            expected_artifacts: expected.to_vec(),
            mode: command.mode(),
            artifact_statuses,
        });
    }
    let run = release_evidence_run(workspace_root, records, &failures, &reports, &mut inspection);
    for blocker in &run.blockers {
        inspection.blocked(
            RELEASE_EVIDENCE_RUN_ARTIFACT,
            blocker.clone(),
            "Run the gate that owns the named artifact with --write and commit it. A declared \
             artifact that is absent is a generator nobody has run.",
        );
    }
    for report in &run.reports {
        inspection.notes.push(report.clone());
    }
    inspection.generates(RELEASE_EVIDENCE_RUN_ARTIFACT, &run);
    inspection
}

/// Assemble the run record and hand both final artifacts to the inspection.
///
/// Both artifacts used to be written straight to disk by this function, one of
/// them from inside the other, so nothing ever compared either against the
/// tree. They are rendered here and settled by the caller.
fn release_evidence_run(
    workspace_root: &Path,
    commands: Vec<ReleaseEvidenceCommandRecord>,
    blockers: &[String],
    reports: &[String],
    inspection: &mut Inspection,
) -> ReleaseEvidenceRun {
    let required_command_count = commands.iter().filter(|command| command.required).count();
    let report_only_command_count = commands.len().saturating_sub(required_command_count);
    let expected_artifact_registry = build_expected_artifact_registry(
        commands
            .iter()
            .map(|command| {
                ReleaseExpectedArtifactCommand::new_with_mode(
                    generator_command(&command.args),
                    command.mode.to_string(),
                    command.required,
                    command
                        .expected_artifacts
                        .iter()
                        .map(|artifact| (*artifact).to_string())
                        .collect(),
                )
            })
            .collect(),
    );
    inspection.generates(EXPECTED_ARTIFACT_REGISTRY, &expected_artifact_registry);
    let final_artifacts = inspect_expected_artifacts(
        workspace_root,
        &["release-evidence"],
        &[EXPECTED_ARTIFACT_REGISTRY],
    );
    let artifact_failures = commands
        .iter()
        .flat_map(|command| &command.artifact_statuses)
        .chain(final_artifacts.iter())
        .filter(|artifact| release_artifact_status_has_failure(artifact))
        .count();
    let evidence_index = build_evidence_index(
        commands
            .iter()
            .flat_map(|command| command.artifact_statuses.iter())
            .chain(final_artifacts.iter()),
    );
    let final_artifact_failures = final_artifacts
        .iter()
        .filter(|artifact| release_artifact_status_has_failure(artifact))
        .map(|artifact| {
            format!(
                "`{RELEASE_EVIDENCE_GENERATOR_COMMAND}` expected final artifact `{}` but it was missing, empty, unreadable, or missing provenance{}",
                artifact.path,
                artifact_blocker_suffix(artifact)
            )
        })
        .collect::<Vec<_>>();
    let mut combined_blockers = blockers.to_vec();
    combined_blockers.extend(final_artifact_failures);
    ReleaseEvidenceRun {
        schema_version: RELEASE_EVIDENCE_RUN_SCHEMA_VERSION,
        total_commands: commands.len(),
        artifact_failures,
        command_count: commands.len(),
        required_command_count,
        report_only_command_count,
        commands,
        final_artifacts,
        evidence_index,
        expected_artifact_registry,
        blockers: combined_blockers,
        reports: reports.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::expected_artifacts::expected_artifact_registry_blockers;
    use super::*;

    #[test]
    fn artifact_status_records_generator_owner_and_fingerprints() {
        let tmp = tempfile::tempdir().unwrap();
        let artifact = tmp
            .path()
            .join("release/evidence/metadata/version-matrix.json");
        std::fs::create_dir_all(artifact.parent().unwrap()).unwrap();
        std::fs::write(&artifact, b"{\"blockers\":[]}\n").unwrap();

        let statuses = inspect_expected_artifacts(
            tmp.path(),
            &["version-matrix"],
            &["release/evidence/metadata/version-matrix.json"],
        );

        assert_eq!(statuses.len(), 1);
        let status = &statuses[0];
        assert_eq!(status.owner_lane, "coordination");
        assert_eq!(status.generator_command, "xtask version-matrix");
        assert_eq!(status.command_mode, COMMAND_MODE_SPAWNED);
        assert_eq!(status.content_sha256.as_deref().map(str::len), Some(64));
        assert!(status
            .source_fingerprint
            .as_deref()
            .is_some_and(|value| value.starts_with("release-evidence-source:v1:")));
        assert!(status
            .freshness_fingerprint
            .as_deref()
            .is_some_and(|value| value.starts_with("release-evidence-freshness:v1:")));
        assert!(status.blockers.is_empty(), "{:?}", status.blockers);
    }

    /// The orchestrator must preserve every public-boundary failure reported
    /// for generated evidence instead of treating readable JSON as sufficient.
    #[test]
    fn artifact_status_rejects_public_boundary_leaks() {
        let tmp = tempfile::tempdir().unwrap();
        let artifact = tmp
            .path()
            .join("release/evidence/metadata/version-matrix.json");
        std::fs::create_dir_all(artifact.parent().unwrap()).unwrap();
        std::fs::write(
            &artifact,
            br#"{"blockers":[],"repositories_public":["santhreal/vyre"],"public_repository":"santhreal/Santh","path":"/mnt/shared/SanthData/Santh/private.json","command":"gh repo edit Santh --visibility public","env":"VYRE_RELEASE_REPOS=santhreal/vyre","provenance":"token=abc"}"#,
        )
        .unwrap();

        let statuses = inspect_expected_artifacts(
            tmp.path(),
            &["version-matrix"],
            &["release/evidence/metadata/version-matrix.json"],
        );

        let blockers = &statuses[0].blockers;
        assert_eq!(
            xtask::release::repo_boundary::missing_public_artifact_boundary_markers(blockers),
            Vec::<&str>::new(),
            "Fix: the inspector must preserve every boundary finding the primitive reports; got {blockers:?}"
        );
    }

    #[test]
    fn duplicate_family_reports_are_release_indexed() {
        let tmp = tempfile::tempdir().unwrap();
        for artifact in expected_artifacts_for_command("whats-similar") {
            let artifact_path = tmp.path().join(artifact);
            std::fs::create_dir_all(artifact_path.parent().unwrap()).unwrap();
            std::fs::write(
                &artifact_path,
                b"{\"schema_version\":2,\"generator_command\":\"xtask whats-similar --all --duplicate-report-json release/evidence/dedup/registered-op-duplicates.json\",\"family_count\":0,\"families\":[]}\n",
            )
            .unwrap();
        }

        let statuses = inspect_expected_artifacts(
            tmp.path(),
            &[
                "whats-similar",
                "--all",
                "--duplicate-report-json",
                "release/evidence/dedup/registered-op-duplicates.json",
            ],
            expected_artifacts_for_command("whats-similar"),
        );

        assert_eq!(
            expected_artifacts_for_command("lego-audit"),
            &["release/evidence/dedup/lego-audit-duplicates.json"]
        );
        assert_eq!(statuses.len(), 1);
        let status = &statuses[0];
        assert_eq!(status.owner_lane, "testing_evidence");
        assert_eq!(
            status.generator_command,
            "xtask whats-similar --all --duplicate-report-json release/evidence/dedup/registered-op-duplicates.json"
        );
        assert_eq!(status.command_mode, COMMAND_MODE_SPAWNED);
        assert!(status.source_fingerprint.is_some());
        assert!(status.freshness_fingerprint.is_some());
        assert!(status.blockers.is_empty(), "{:?}", status.blockers);
    }

    /// WHY: the lookup is keyed on the subcommand, and the census asked it for a
    /// rendered command line ("xtask release-workload-matrix --enforce"), so every
    /// row missed and all fourteen required generators read as declaring no
    /// artifact at all. The table is enumerated here at run time, so adding a
    /// generator without listing what it owes turns this red instead of producing
    /// a census that reports fourteen defects nobody can act on.
    #[test]
    fn every_required_generator_declares_the_artifacts_it_owes() {
        let undeclared: Vec<&str> = COMMANDS
            .iter()
            .filter(|command| {
                command.required && expected_artifacts_for_command(command.subcommand()).is_empty()
            })
            .map(EvidenceCommand::subcommand)
            .collect();
        assert_eq!(
            undeclared,
            Vec::<&str>::new(),
            "Fix: list the artifacts each of these generators owes in expected_artifacts_for_command"
        );
    }

    #[test]
    fn duplicate_family_reports_require_schema_v2_and_subject_fingerprints() {
        let tmp = tempfile::tempdir().unwrap();
        let artifact = tmp
            .path()
            .join("release/evidence/dedup/registered-op-duplicates.json");
        std::fs::create_dir_all(artifact.parent().unwrap()).unwrap();
        std::fs::write(
            &artifact,
            br#"{
  "schema_version": 1,
  "family_count": 1,
  "families": [
    {
      "left": {"id": "left"},
      "right": {"id": "right"}
    }
  ]
}
"#,
        )
        .unwrap();

        let statuses = inspect_expected_artifacts(
            tmp.path(),
            &["whats-similar"],
            &["release/evidence/dedup/registered-op-duplicates.json"],
        );

        let blockers = &statuses[0].blockers;
        assert_eq!(
            xtask::gates::dedup_report::missing_duplicate_family_blocker_markers(blockers),
            Vec::<&str>::new(),
            "Fix: the inspector must preserve every drift blocker the validator reports; got {blockers:?}"
        );
    }

    #[test]
    fn external_release_benchmark_status_requires_external_mode_and_digest_chain() {
        let tmp = tempfile::tempdir().unwrap();
        let artifact_rel = "release/evidence/benchmarks/cuda-release-suite.json";
        let artifact = tmp.path().join(artifact_rel);
        std::fs::create_dir_all(artifact.parent().unwrap()).unwrap();
        std::fs::write(
            &artifact,
            br#"{
  "schema_version": 3,
  "blockers": ["stale source fingerprint"],
  "artifact_statuses": []
}
"#,
        )
        .unwrap();

        let spawned_statuses = inspect_expected_artifacts(
            tmp.path(),
            &["release-benchmarks", "--backend", "cuda"],
            &[artifact_rel],
        );
        let spawned_blockers = &spawned_statuses[0].blockers;
        assert!(spawned_blockers
            .iter()
            .any(|blocker| blocker.contains("command_mode `spawned`")));

        let external_statuses = inspect_expected_artifacts_with_mode(
            tmp.path(),
            &["release-benchmarks", "--backend", "cuda"],
            &[artifact_rel],
            COMMAND_MODE_EXTERNAL_ARTIFACTS_ONLY,
        );
        let external_blockers = &external_statuses[0].blockers;
        assert_eq!(
            external_statuses[0].command_mode,
            COMMAND_MODE_EXTERNAL_ARTIFACTS_ONLY
        );
        assert!(external_blockers
            .iter()
            .all(|blocker| !blocker.contains("command_mode `spawned`")));
        assert!(external_blockers
            .iter()
            .any(|blocker| blocker.contains("stale source fingerprint")));
        assert!(external_blockers
            .iter()
            .any(|blocker| blocker.contains("schema_digest_chain.source_digest")));
        assert!(external_blockers
            .iter()
            .any(|blocker| blocker.contains("schema_digest_chain.command_digest")));
        assert!(external_blockers
            .iter()
            .any(|blocker| blocker.contains("schema_digest_chain.hardware_digest")));
        assert!(external_blockers
            .iter()
            .any(|blocker| blocker.contains("hardware_digest")));
    }

    #[test]
    fn expected_artifact_registry_counts_commands_and_artifacts() {
        let registry = build_expected_artifact_registry(vec![ReleaseExpectedArtifactCommand::new(
            "xtask version-matrix".to_string(),
            true,
            vec![
                "release/evidence/metadata/version-matrix.json",
                "release/evidence/docs/vyre-readme-contracts.json",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
        )]);

        assert_eq!(registry.schema_version, 2);
        assert_eq!(registry.command_count, 2);
        assert_eq!(registry.artifact_count, 4);
        assert_eq!(
            registry.commands[0].generator_command,
            "xtask version-matrix"
        );
        assert!(registry.commands[0].required);
        assert_eq!(
            registry.commands[1].expected_artifacts,
            vec![
                RELEASE_EVIDENCE_RUN_ARTIFACT.to_string(),
                EXPECTED_ARTIFACT_REGISTRY.to_string()
            ]
        );
    }

    #[test]
    fn expected_artifact_registry_validation_rejects_drift() {
        let blockers = expected_artifact_registry_blockers(
            b"{\"schema_version\":0,\"command_count\":2,\"artifact_count\":9,\"commands\":[{\"generator_command\":\"xtask version-matrix\",\"expected_artifacts\":[]}]}\n",
        );

        assert!(blockers
            .iter()
            .any(|blocker| blocker.contains("schema_version=2")));
        assert!(blockers
            .iter()
            .any(|blocker| blocker.contains("command_count")));
        assert!(blockers.iter().any(|blocker| blocker.contains("required")));
        assert!(blockers
            .iter()
            .any(|blocker| blocker.contains("artifact_contracts")));
        assert!(blockers
            .iter()
            .any(|blocker| blocker.contains("artifact_count")));
        assert!(blockers
            .iter()
            .any(|blocker| blocker.contains(RELEASE_EVIDENCE_GENERATOR_COMMAND)));
    }

    #[test]
    fn evidence_index_includes_final_release_evidence_artifacts() {
        let final_artifact = ReleaseEvidenceArtifactStatus {
            path: EXPECTED_ARTIFACT_REGISTRY.to_string(),
            exists: true,
            bytes: 64,
            read_error: None,
            owner_lane: "testing_evidence",
            generator_command: RELEASE_EVIDENCE_GENERATOR_COMMAND.to_string(),
            command_mode: COMMAND_MODE_SPAWNED,
            content_sha256: Some("a".repeat(64)),
            source_fingerprint: Some("release-evidence-source:v1:abc".to_string()),
            freshness_fingerprint: Some("release-evidence-freshness:v1:def".to_string()),
            blockers: Vec::new(),
        };

        let final_artifacts = [final_artifact];
        let index = build_evidence_index(final_artifacts.iter());

        assert_eq!(index.artifact_count, 1);
        assert_eq!(index.artifacts[0].path, EXPECTED_ARTIFACT_REGISTRY);
    }

    #[test]
    fn evidence_index_surfaces_missing_provenance_blockers() {
        let record = ReleaseEvidenceCommandRecord {
            args: vec!["version-matrix"],
            required: true,
            expected_artifacts: vec!["release/evidence/metadata/version-matrix.json"],
            mode: COMMAND_MODE_SPAWNED,
            artifact_statuses: vec![ReleaseEvidenceArtifactStatus {
                path: "release/evidence/metadata/version-matrix.json".to_string(),
                exists: true,
                bytes: 12,
                read_error: None,
                owner_lane: "testing_evidence",
                generator_command: "xtask version-matrix".to_string(),
                command_mode: COMMAND_MODE_SPAWNED,
                content_sha256: None,
                source_fingerprint: None,
                freshness_fingerprint: None,
                blockers: vec!["artifact is missing source_fingerprint".to_string()],
            }],
        };

        let records = [record];
        let index = build_evidence_index(
            records
                .iter()
                .flat_map(|record| record.artifact_statuses.iter()),
        );

        assert_eq!(index.artifact_count, 1);
        assert_eq!(index.blockers.len(), 1);
        assert!(index.blockers[0].contains("source_fingerprint"));
    }
}
