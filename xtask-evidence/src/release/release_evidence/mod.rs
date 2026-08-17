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
pub(crate) use expected_artifacts::expected_artifacts_for_args;
use expected_artifacts::{
    build_expected_artifact_registry, ReleaseExpectedArtifactCommand,
    ReleaseExpectedArtifactRegistry, COMMAND_MODE_EXTERNAL_ARTIFACTS_ONLY, COMMAND_MODE_SPAWNED,
    EXPECTED_ARTIFACT_REGISTRY, RELEASE_EVIDENCE_GENERATOR_COMMAND, RELEASE_EVIDENCE_RUN_ARTIFACT,
};
#[cfg(test)]
use xtask::artifact_paths::FRONTIER_LEADERBOARD_ARTIFACT;
use xtask::artifact_paths::{LEGO_AUDIT_DUPLICATES_ARTIFACT, REGISTERED_OP_DUPLICATES_ARTIFACT};

/// Bumped from 5: command identity and artifact ownership now use the complete
/// argument vector, including backend-specific measured-evidence invocations.
const RELEASE_EVIDENCE_RUN_SCHEMA_VERSION: u32 = 6;

const COMMANDS: &[EvidenceCommand] = &[
    EvidenceCommand::required(&["docs-check"]),
    EvidenceCommand::required(&["version-matrix"]),
    EvidenceCommand::required(&["backend-matrix"]),
    EvidenceCommand::required(&["conformance-matrix"]),
    EvidenceCommand::required(&["release-workload-matrix", "--enforce"]),
    EvidenceCommand::external_required(&[
        "release-benchmarks",
        "--backend",
        "cuda",
        "--measured-samples",
        "30",
        "--write",
    ]),
    EvidenceCommand::external_required(&[
        "release-benchmarks",
        "--backend",
        "wgpu",
        "--measured-samples",
        "30",
        "--write",
    ]),
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
        let expected = expected_artifacts_for_args(command.args);
        if command.required && expected.is_empty() {
            failures.push(format!(
                "`{label}` is required but declares no expected artifacts"
            ));
        }
        let artifact_statuses = inspect_expected_artifacts_with_mode(
            workspace_root,
            command.args,
            &expected,
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
            expected_artifacts: expected,
            mode: command.mode(),
            artifact_statuses,
        });
    }
    let run = release_evidence_run(
        workspace_root,
        records,
        &failures,
        &reports,
        &mut inspection,
    );
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

/// Judge a committed `release-evidence-run.json` against the schema this module
/// writes and the command table it writes it from.
///
/// The census has one owner, so the field names it records and the rows it owes
/// are spelled here and nowhere else. A second reader spelled them again and
/// asked for `successful_commands`, `command_failures` and a per-command exit
/// `status`. All three were retired when the sweep stopped spawning generators,
/// so every generator read as failed and an absent counter printed as
/// `u64::MAX`. The required rows are derived from `COMMANDS`, so a new required
/// generator cannot leave a reader behind.
pub fn judge_committed_run(run: &serde_json::Value, subject: &str, failures: &mut Vec<String>) {
    let schema_version = run
        .get("schema_version")
        .and_then(serde_json::Value::as_u64);
    if schema_version != Some(u64::from(RELEASE_EVIDENCE_RUN_SCHEMA_VERSION)) {
        failures.push(format!(
            "{subject} release-evidence-run declares schema_version {} but this tree writes {RELEASE_EVIDENCE_RUN_SCHEMA_VERSION}. Regenerate it with `xtask release-evidence --write`",
            schema_version.map_or_else(|| "<missing>".to_string(), |version| version.to_string())
        ));
        return;
    }
    let required_rows = COMMANDS.iter().filter(|command| command.required).count();
    for (field, expected) in [
        ("total_commands", COMMANDS.len()),
        ("command_count", COMMANDS.len()),
        ("required_command_count", required_rows),
        ("artifact_failures", 0),
    ] {
        match count_field(run, field) {
            Some(count) if count == expected => {}
            Some(count) => failures.push(format!(
                "{subject} release-evidence-run reports {field} {count}, expected {expected}"
            )),
            None => failures.push(format!(
                "{subject} release-evidence-run carries no {field}. Regenerate it with `xtask release-evidence --write`"
            )),
        }
    }
    let blockers = run
        .get("blockers")
        .and_then(serde_json::Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    for blocker in blockers {
        failures.push(format!(
            "{subject} release-evidence-run records blocker {}",
            blocker.as_str().unwrap_or("<not a string>")
        ));
    }
    let records = run
        .get("commands")
        .and_then(serde_json::Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    for command in COMMANDS.iter().filter(|command| command.required) {
        let label = format!("xtask {}", command.args.join(" "));
        let Some(record) = records
            .iter()
            .find(|record| record_matches_args(record, command.args))
        else {
            failures.push(format!(
                "{subject} release-evidence-run is missing required generator `{label}`"
            ));
            continue;
        };
        judge_committed_command(
            record,
            subject,
            command.args,
            command.required,
            command.mode(),
            failures,
        );
    }
}

fn record_matches_args(record: &serde_json::Value, expected: &[&str]) -> bool {
    let Some(actual) = record.get("args").and_then(serde_json::Value::as_array) else {
        return false;
    };
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual.as_str() == Some(*expected))
}

/// Judge one committed command record against the artifacts its generator owes.
fn judge_committed_command(
    record: &serde_json::Value,
    subject: &str,
    command_args: &[&str],
    command_required: bool,
    command_mode: &str,
    failures: &mut Vec<String>,
) {
    let label = format!("xtask {}", command_args.join(" "));
    let expected_artifacts = expected_artifacts_for_args(command_args);
    if record.get("required").and_then(serde_json::Value::as_bool) != Some(command_required) {
        failures.push(format!(
            "{subject} release-evidence-run generator `{label}` required state does not match {command_required}"
        ));
    }
    if record.get("mode").and_then(serde_json::Value::as_str) != Some(command_mode) {
        failures.push(format!(
            "{subject} release-evidence-run generator `{label}` mode does not match `{command_mode}`"
        ));
    }
    let declared = record
        .get("expected_artifacts")
        .and_then(serde_json::Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    let statuses = record
        .get("artifact_statuses")
        .and_then(serde_json::Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    if declared.len() != expected_artifacts.len() {
        failures.push(format!(
            "{subject} release-evidence-run generator `{label}` declares {} artifacts, expected exactly {}",
            declared.len(),
            expected_artifacts.len()
        ));
    }
    for artifact in declared {
        match artifact.as_str() {
            Some(artifact) if expected_artifacts.contains(&artifact) => {}
            Some(artifact) => failures.push(format!(
                "{subject} release-evidence-run generator `{label}` declares unowned artifact `{artifact}`"
            )),
            None => failures.push(format!(
                "{subject} release-evidence-run generator `{label}` declares a non-string artifact"
            )),
        }
    }
    if statuses.len() != expected_artifacts.len() {
        failures.push(format!(
            "{subject} release-evidence-run generator `{label}` carries {} artifact statuses, expected exactly {}",
            statuses.len(),
            expected_artifacts.len()
        ));
    }
    for status in statuses {
        let Some(path) = status.get("path").and_then(serde_json::Value::as_str) else {
            failures.push(format!(
                "{subject} release-evidence-run generator `{label}` carries an artifact status without a string path"
            ));
            continue;
        };
        if !expected_artifacts
            .iter()
            .any(|expected| Path::new(path).ends_with(Path::new(expected)))
        {
            failures.push(format!(
                "{subject} release-evidence-run generator `{label}` carries status for unowned artifact `{path}`"
            ));
        }
    }
    for expected in &expected_artifacts {
        if !declared
            .iter()
            .any(|artifact| artifact.as_str() == Some(*expected))
        {
            failures.push(format!(
                "{subject} release-evidence-run generator `{label}` does not declare expected artifact `{expected}`"
            ));
        }
        let Some(status) = statuses.iter().find(|status| {
            status
                .get("path")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|path| Path::new(path).ends_with(Path::new(expected)))
        }) else {
            failures.push(format!(
                "{subject} release-evidence-run generator `{label}` has no artifact status for `{expected}`"
            ));
            continue;
        };
        if status
            .get("generator_command")
            .and_then(serde_json::Value::as_str)
            != Some(label.as_str())
        {
            failures.push(format!(
                "{subject} release-evidence-run artifact `{expected}` generator provenance does not match `{label}`"
            ));
        }
        if status
            .get("command_mode")
            .and_then(serde_json::Value::as_str)
            != Some(command_mode)
        {
            failures.push(format!(
                "{subject} release-evidence-run artifact `{expected}` command mode does not match `{command_mode}`"
            ));
        }
        let exists = status
            .get("exists")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let bytes = count_field(status, "bytes").unwrap_or(0);
        let read_error = status.get("read_error").and_then(serde_json::Value::as_str);
        let missing_provenance = ["source_fingerprint", "freshness_fingerprint"]
            .into_iter()
            .filter(|field| {
                status
                    .get(field)
                    .and_then(serde_json::Value::as_str)
                    .is_none()
            })
            .collect::<Vec<_>>();
        if !exists || bytes == 0 || read_error.is_some() || !missing_provenance.is_empty() {
            failures.push(format!(
                "{subject} release-evidence-run generator `{label}` artifact `{expected}` exists={exists} bytes={bytes} read_error={} missing={}",
                read_error.unwrap_or("none"),
                if missing_provenance.is_empty() {
                    "none".to_string()
                } else {
                    missing_provenance.join(", ")
                }
            ));
        }
    }
}

/// Read one non-negative count, absent or non-numeric reported as `None` rather
/// than defaulted, because a defaulted counter is what let a retired field name
/// pass as a clean run.
fn count_field(value: &serde_json::Value, field: &str) -> Option<usize> {
    value
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .and_then(|count| usize::try_from(count).ok())
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
        for artifact in expected_artifacts_for_args(&[
            "whats-similar",
            "--all",
            "--duplicate-report-json",
            REGISTERED_OP_DUPLICATES_ARTIFACT,
        ]) {
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
            &expected_artifacts_for_args(&[
                "whats-similar",
                "--all",
                "--duplicate-report-json",
                REGISTERED_OP_DUPLICATES_ARTIFACT,
            ]),
        );

        assert_eq!(
            expected_artifacts_for_args(&[
                "lego-audit",
                "--report-only",
                "--duplicate-report-json",
                LEGO_AUDIT_DUPLICATES_ARTIFACT,
            ]),
            vec![LEGO_AUDIT_DUPLICATES_ARTIFACT]
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

    /// WHY: artifact ownership is keyed on the complete generator arguments.
    /// Backend-specific producers share a subcommand but own disjoint outputs,
    /// so a new command or argument variant must turn this test red until its
    /// exact artifact set is declared.
    #[test]
    fn every_required_generator_declares_the_artifacts_it_owes() {
        let undeclared: Vec<&str> = COMMANDS
            .iter()
            .filter(|command| {
                command.required && expected_artifacts_for_args(command.args).is_empty()
            })
            .map(EvidenceCommand::subcommand)
            .collect();
        assert_eq!(
            undeclared,
            Vec::<&str>::new(),
            "Fix: list the artifacts each exact generator invocation owes in expected_artifacts_for_args"
        );
    }

    /// WHY: measured evidence provenance names the invocation that produced the
    /// bytes, including the release sample floor and write mode. Backend-only
    /// identity would accept a comparison run or a differently sampled suite.
    #[test]
    fn measured_evidence_census_uses_canonical_complete_arguments() {
        let external = COMMANDS
            .iter()
            .filter(|command| !command.in_sweep)
            .map(|command| command.args)
            .collect::<Vec<_>>();
        let expected: Vec<&[&str]> = vec![
            &[
                "release-benchmarks",
                "--backend",
                "cuda",
                "--measured-samples",
                "30",
                "--write",
            ],
            &[
                "release-benchmarks",
                "--backend",
                "wgpu",
                "--measured-samples",
                "30",
                "--write",
            ],
        ];

        assert_eq!(external, expected);
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

        assert_eq!(registry.schema_version, 3);
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
            .any(|blocker| blocker.contains("schema_version=3")));
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

    /// A committed run in the shape this module writes, every artifact clean.
    ///
    /// Derived from `COMMANDS` at run time, so a new required generator changes
    /// what every test below judges instead of leaving a written list behind.
    fn clean_committed_run() -> serde_json::Value {
        let commands = COMMANDS
            .iter()
            .map(|command| {
                let expected = expected_artifacts_for_args(command.args);
                let generator_command = format!("xtask {}", command.args.join(" "));
                serde_json::json!({
                    "args": command.args,
                    "required": command.required,
                    "expected_artifacts": expected,
                    "mode": command.mode(),
                    "artifact_statuses": expected
                        .iter()
                        .map(|path| serde_json::json!({
                            "path": path,
                            "exists": true,
                            "bytes": 1,
                            "read_error": serde_json::Value::Null,
                            "generator_command": generator_command.as_str(),
                            "command_mode": command.mode(),
                            "source_fingerprint": "git:0000000000000000000000000000000000000000",
                            "freshness_fingerprint": "source-tree-v1:0000",
                            "blockers": [],
                        }))
                        .collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "schema_version": RELEASE_EVIDENCE_RUN_SCHEMA_VERSION,
            "total_commands": COMMANDS.len(),
            "command_count": COMMANDS.len(),
            "required_command_count": COMMANDS.iter().filter(|command| command.required).count(),
            "artifact_failures": 0,
            "commands": commands,
            "blockers": [],
        })
    }

    fn judge(run: &serde_json::Value) -> Vec<String> {
        let mut failures = Vec::new();
        judge_committed_run(run, "subject", &mut failures);
        failures
    }

    #[test]
    fn a_run_in_the_shape_this_module_writes_is_clean() {
        assert_eq!(judge(&clean_committed_run()), Vec::<String>::new());
    }

    /// The defect this judgment replaced: the reader asked for
    /// `successful_commands`, `command_failures` and a per-command spawn
    /// `status`, none of which schema 5 records. Absent counters defaulted, so
    /// the reader reported eleven failures against a clean artifact and would
    /// have reported none against a broken one.
    #[test]
    fn a_retired_schema_is_named_rather_than_judged_field_by_field() {
        let mut run = clean_committed_run();
        run["schema_version"] = serde_json::json!(4);
        run["successful_commands"] = serde_json::json!(14);
        run["command_failures"] = serde_json::json!(0);

        let failures = judge(&run);

        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(failures[0].contains("schema_version 4"), "{failures:?}");
        assert!(
            failures[0].contains(&RELEASE_EVIDENCE_RUN_SCHEMA_VERSION.to_string()),
            "{failures:?}"
        );
    }

    #[test]
    fn an_absent_counter_is_reported_rather_than_defaulted() {
        let mut run = clean_committed_run();
        let object = run.as_object_mut().unwrap();
        object.remove("artifact_failures");

        let failures = judge(&run);

        assert_eq!(failures.len(), 1, "{failures:?}");
        assert!(
            failures[0].contains("carries no artifact_failures"),
            "{failures:?}"
        );
    }

    #[test]
    fn a_missing_required_generator_is_reported_by_name() {
        let required = COMMANDS
            .iter()
            .find(|command| command.required)
            .map(EvidenceCommand::subcommand)
            .unwrap();
        let mut run = clean_committed_run();
        let commands = run["commands"].as_array_mut().unwrap();
        commands.retain(|record| record["args"][0].as_str() != Some(required));

        let failures = judge(&run);

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("missing required generator")
                    && failure.contains(required)),
            "{failures:?}"
        );
    }

    #[test]
    fn an_artifact_the_generator_never_produced_is_reported() {
        let mut run = clean_committed_run();
        run["commands"][0]["artifact_statuses"][0]["exists"] = serde_json::json!(false);

        let failures = judge(&run);

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("exists=false")),
            "{failures:?}"
        );
    }

    #[test]
    fn an_artifact_without_provenance_is_reported() {
        let mut run = clean_committed_run();
        run["commands"][0]["artifact_statuses"][0]["freshness_fingerprint"] =
            serde_json::Value::Null;

        let failures = judge(&run);

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("missing=freshness_fingerprint")),
            "{failures:?}"
        );
    }

    /// WHY: CUDA and WGPU share a subcommand but are distinct required
    /// generators. Matching only the first argument silently accepted a second
    /// CUDA record in place of the WGPU evidence.
    #[test]
    fn backend_specific_generator_identity_uses_all_arguments() {
        let mut run = clean_committed_run();
        let records = run["commands"].as_array_mut().unwrap();
        let wgpu = records
            .iter_mut()
            .find(|record| {
                record_matches_args(
                    record,
                    &[
                        "release-benchmarks",
                        "--backend",
                        "wgpu",
                        "--measured-samples",
                        "30",
                        "--write",
                    ],
                )
            })
            .expect("clean fixture must contain WGPU measured evidence");
        wgpu["args"][2] = serde_json::json!("cuda");

        let failures = judge(&run);

        assert!(failures.iter().any(|failure| {
            failure.contains("missing required generator")
                && failure.contains("release-benchmarks --backend wgpu")
        }));
    }

    #[test]
    fn backend_specific_generator_rejects_unowned_artifacts() {
        let mut run = clean_committed_run();
        let records = run["commands"].as_array_mut().unwrap();
        let wgpu = records
            .iter_mut()
            .find(|record| {
                record_matches_args(
                    record,
                    &[
                        "release-benchmarks",
                        "--backend",
                        "wgpu",
                        "--measured-samples",
                        "30",
                        "--write",
                    ],
                )
            })
            .expect("clean fixture must contain WGPU measured evidence");
        wgpu["expected_artifacts"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!(FRONTIER_LEADERBOARD_ARTIFACT));
        wgpu["artifact_statuses"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "path": FRONTIER_LEADERBOARD_ARTIFACT,
                "exists": true,
                "bytes": 1,
                "read_error": serde_json::Value::Null,
                "source_fingerprint": "git:0000000000000000000000000000000000000000",
                "freshness_fingerprint": "source-tree-v1:0000",
                "blockers": [],
            }));

        let failures = judge(&run);

        assert!(failures.iter().any(|failure| {
            failure.contains("unowned artifact") && failure.contains(FRONTIER_LEADERBOARD_ARTIFACT)
        }));
    }

    /// WHY: artifact ownership is path-component exact. A filename prefix that
    /// merely ends with the expected bytes is not the producer-owned artifact.
    #[test]
    fn artifact_status_suffix_spoof_is_rejected() {
        let mut run = clean_committed_run();
        let expected = run["commands"][0]["expected_artifacts"][0]
            .as_str()
            .expect("clean fixture artifact must be a string")
            .to_string();
        run["commands"][0]["artifact_statuses"][0]["path"] =
            serde_json::json!(format!("spoof-{expected}"));

        let failures = judge(&run);

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("unowned artifact")),
            "{failures:?}"
        );
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("has no artifact status")),
            "{failures:?}"
        );
    }

    #[test]
    fn committed_generator_and_status_provenance_must_match_the_command() {
        let mut run = clean_committed_run();
        let records = run["commands"].as_array_mut().unwrap();
        let wgpu = records
            .iter_mut()
            .find(|record| {
                record_matches_args(
                    record,
                    &[
                        "release-benchmarks",
                        "--backend",
                        "wgpu",
                        "--measured-samples",
                        "30",
                        "--write",
                    ],
                )
            })
            .expect("clean fixture must contain WGPU measured evidence");
        wgpu["required"] = serde_json::json!(false);
        wgpu["mode"] = serde_json::json!(COMMAND_MODE_SPAWNED);
        wgpu["artifact_statuses"][0]["generator_command"] =
            serde_json::json!("xtask release-benchmarks --backend cuda");
        wgpu["artifact_statuses"][0]["command_mode"] = serde_json::json!(COMMAND_MODE_SPAWNED);

        let failures = judge(&run);

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("required state")),
            "{failures:?}"
        );
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("mode does not match")),
            "{failures:?}"
        );
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("generator provenance")),
            "{failures:?}"
        );
    }

    #[test]
    fn a_recorded_blocker_is_reported() {
        let mut run = clean_committed_run();
        run["blockers"] = serde_json::json!(["version-matrix owes version-matrix.json"]);

        let failures = judge(&run);

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("records blocker")
                    && failure.contains("version-matrix.json")),
            "{failures:?}"
        );
    }

    /// WHY: The authoritative descriptor and release-evidence census producer must agree on
    /// the exact output paths so comparison is immutable and write mutations
    /// are never undeclared.
    #[test]
    fn authoritative_descriptor_declares_exact_release_evidence_artifacts() {
        let descriptor = xtask::gate_metadata::descriptor_by_name("release-evidence");
        let mut expected: Vec<&str> =
            super::expected_artifacts::RELEASE_EVIDENCE_EXPECTED_ARTIFACTS.to_vec();
        expected.sort_unstable();
        let mut actual: Vec<&str> = descriptor.artifacts.to_vec();
        actual.sort_unstable();
        assert_eq!(
            actual, expected,
            "Fix: release-evidence gate descriptor must declare exactly the canonical expected artifact registry and release evidence run artifacts"
        );
    }
}
