//! Generate cheap structural release evidence artifacts.
//!
//! Long-running artifacts remain explicit: benchmark suites and full
//! Linux corpus parsing are not launched here.

use std::fs;
use std::path::Path;
use std::process::Command;

use serde::Serialize;

const COMMANDS: &[EvidenceCommand] = &[
    EvidenceCommand::required(&["docs-matrix"]),
    EvidenceCommand::required(&["version-matrix"]),
    EvidenceCommand::required(&["backend-matrix"]),
    EvidenceCommand::required(&["conformance-matrix"]),
    EvidenceCommand::required(&["release-workload-matrix", "--enforce"]),
    EvidenceCommand::required(&["hygiene-matrix"]),
    EvidenceCommand::required(&["test-matrix"]),
    EvidenceCommand::required(&["metadata-matrix"]),
    EvidenceCommand::required(&["feature-matrix"]),
    EvidenceCommand::required(&["optimization-corpus"]),
    EvidenceCommand::required(&["optimization-matrix"]),
    EvidenceCommand::required(&["parser-coherence"]),
    EvidenceCommand::required(&["weir-matrix"]),
];

struct EvidenceCommand {
    args: &'static [&'static str],
    required: bool,
}

#[derive(Debug, Serialize)]
struct ReleaseEvidenceRun {
    schema_version: u32,
    total_commands: usize,
    successful_commands: usize,
    command_failures: usize,
    artifact_failures: usize,
    command_count: usize,
    required_command_count: usize,
    report_only_command_count: usize,
    commands: Vec<ReleaseEvidenceCommandRecord>,
    blockers: Vec<String>,
    reports: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ReleaseEvidenceCommandRecord {
    args: Vec<&'static str>,
    required: bool,
    expected_artifacts: Vec<&'static str>,
    status: String,
    exit_code: Option<i32>,
    artifact_statuses: Vec<ReleaseEvidenceArtifactStatus>,
}

#[derive(Debug, Serialize)]
struct ReleaseEvidenceArtifactStatus {
    path: String,
    exists: bool,
    bytes: u64,
    read_error: Option<String>,
}

impl EvidenceCommand {
    const fn required(args: &'static [&'static str]) -> Self {
        Self {
            args,
            required: true,
        }
    }
}

pub(crate) fn run(_args: &[String]) {
    let workspace_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let mut failures = Vec::new();
    let xtask = match std::env::current_exe() {
        Ok(path) => path,
        Err(error) => {
            eprintln!("release-evidence: failed to locate current xtask binary: {error}");
            std::process::exit(1);
        }
    };
    let mut reports = Vec::new();
    let mut records = Vec::new();
    for command in COMMANDS {
        let status = Command::new(&xtask)
            .args(command.args)
            .current_dir(&workspace_root)
            .status();
        let expected = expected_artifacts(command.args);
        if command.required && expected.is_empty() {
            failures.push(format!(
                "`xtask {}` is required but declares no expected artifacts",
                command.args.join(" ")
            ));
        }
        let artifact_statuses = inspect_expected_artifacts(&workspace_root, expected);
        let status_text = command_status_text(&status);
        let exit_code = status
            .as_ref()
            .ok()
            .and_then(std::process::ExitStatus::code);
        for artifact in &artifact_statuses {
            if !artifact.exists || artifact.bytes == 0 || artifact.read_error.is_some() {
                let finding = format!(
                    "`xtask {}` expected `{}` but it was missing, empty, or unreadable{}",
                    command.args.join(" "),
                    artifact.path,
                    artifact
                        .read_error
                        .as_ref()
                        .map(|error| format!(": {error}"))
                        .unwrap_or_default()
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
            status: status_text,
            exit_code,
            artifact_statuses,
        });
        match status {
            Ok(status) if status.success() => {}
            Ok(status) if command.required => failures.push(format!(
                "`xtask {}` failed with {status}",
                command.args.join(" ")
            )),
            Ok(status) => reports.push(format!(
                "`xtask {}` reported {status}; artifact was still written for review",
                command.args.join(" ")
            )),
            Err(error) if command.required => failures.push(format!(
                "failed to run `xtask {}`: {error}",
                command.args.join(" ")
            )),
            Err(error) => reports.push(format!(
                "failed to run report-only `xtask {}`: {error}",
                command.args.join(" ")
            )),
        }
    }
    write_release_evidence_run(&workspace_root, records, &failures, &reports);
    if failures.is_empty() {
        for report in &reports {
            eprintln!("release-evidence: {report}");
        }
        println!(
            "release-evidence: structural evidence generated; run `cargo_full run --bin xtask -- release-benchmarks --backend cuda` separately for benchmark artifacts"
        );
    } else {
        eprintln!("release-evidence: {} blocker(s):", failures.len());
        for failure in &failures {
            eprintln!("  - {failure}");
        }
        std::process::exit(1);
    }
}

fn command_status_text(status: &std::io::Result<std::process::ExitStatus>) -> String {
    match status {
        Ok(status) if status.success() => "success".to_string(),
        Ok(status) => format!("failed: {status}"),
        Err(error) => format!("spawn error: {error}"),
    }
}

fn inspect_expected_artifacts(
    workspace_root: &Path,
    expected_artifacts: &[&'static str],
) -> Vec<ReleaseEvidenceArtifactStatus> {
    expected_artifacts
        .iter()
        .map(|artifact| {
            let path = workspace_root.join(artifact);
            match fs::metadata(&path) {
                Ok(metadata) => ReleaseEvidenceArtifactStatus {
                    path: (*artifact).to_string(),
                    exists: metadata.is_file(),
                    bytes: metadata.len(),
                    read_error: None,
                },
                Err(error) => ReleaseEvidenceArtifactStatus {
                    path: (*artifact).to_string(),
                    exists: false,
                    bytes: 0,
                    read_error: Some(error.to_string()),
                },
            }
        })
        .collect()
}

fn expected_artifacts(args: &[&str]) -> &'static [&'static str] {
    match args.first().copied().unwrap_or_default() {
        "version-matrix" => &[
            "release/evidence/version/version-matrix.json",
            "release/evidence/version/release-tag-plan.json",
        ],
        "backend-matrix" => &["release/evidence/backends/backend-matrix.json"],
        "conformance-matrix" => &["release/evidence/conformance/conformance-matrix.json"],
        "release-workload-matrix" => &["release/evidence/benchmarks/release-workload-matrix.json"],
        "hygiene-matrix" => &[
            "release/evidence/hygiene/hygiene-matrix.json",
            "release/evidence/hygiene/no-stubs-scan.json",
            "release/evidence/hygiene/no-hidden-fallback-scan.json",
            "release/evidence/hygiene/resource-bound-scan.json",
            "release/evidence/hygiene/error-surface-scan.json",
            "release/evidence/hygiene/cargo-wrapper-scan.json",
            "release/evidence/hygiene/audit-location-scan.json",
            "release/evidence/hygiene/public-doc-scan.json",
            "release/evidence/hygiene/test-hygiene-scan.json",
        ],
        "test-matrix" => &[
            "release/evidence/tests/test-matrix.json",
            "release/evidence/tests/modularization-map.json",
            "release/evidence/tests/oversized-test-closure.json",
            "release/evidence/tests/release-surface-suite-coverage.json",
            "release/evidence/tests/unit-suite.json",
            "release/evidence/tests/adversarial-suite.json",
            "release/evidence/tests/property-suite.json",
            "release/evidence/tests/conformance-suite.json",
            "release/evidence/tests/corpus-suite.json",
            "release/evidence/tests/benchmark-suite.json",
            "release/evidence/tests/gap-suite.json",
            "release/evidence/tests/fuzz-suite.json",
        ],
        "docs-matrix" => &[
            "release/evidence/docs/docs-matrix.json",
            "release/evidence/docs/vyre-readme-contracts.json",
            "release/evidence/docs/release-notes-version-story.md",
            "release/evidence/docs/cuda-release-path.md",
            "release/evidence/docs/wgpu-fallback-proof.md",
            "release/evidence/docs/megakernel-default-proof.md",
            "release/evidence/docs/optimization-proof.md",
            "release/evidence/docs/egraph-saturation.md",
            "release/evidence/docs/c-parser-linux-proof.md",
            "release/evidence/docs/distributed-parser-coherence.md",
            "release/evidence/docs/weir-integration.md",
            "release/evidence/docs/test-architecture.md",
            "release/evidence/docs/vyre-readme-proof.md",
            "release/evidence/docs/weir-readme-proof.md",
            "release/evidence/docs/parser-doc-proof.md",
            "release/evidence/docs/benchmark-doc-proof.md",
            "release/evidence/docs/conformance-doc-proof.md",
            "release/evidence/docs/release-notes.md",
            "release/evidence/docs/crate-metadata-proof.md",
            "release/evidence/docs/release-hygiene-proof.md",
            "release/evidence/docs/cpu-only-100x-proof.md",
        ],
        "metadata-matrix" => &["release/evidence/metadata/metadata-matrix.json"],
        "feature-matrix" => &["release/evidence/metadata/feature-matrix.json"],
        "optimization-corpus" => &[
            "release/evidence/optimization/optimization-corpus.json",
            "release/evidence/optimization/optimization-corpus-contracts.json",
            "release/evidence/optimization/optimization-family-manifest.json",
            "release/evidence/optimization/optimization-analysis-fixtures.json",
            "release/evidence/optimization/optimization-case-manifest.json",
        ],
        "optimization-matrix" => &[
            "release/evidence/optimization/optimization-integration-matrix.json",
            "release/evidence/optimization/alias-aware-dse.json",
            "release/evidence/optimization/alias-aware-stlf.json",
            "release/evidence/optimization/alias-aware-licm.json",
            "release/evidence/optimization/alias-aware-fusion-fission.json",
            "release/evidence/optimization/weir-facts-pass-firing.json",
            "release/evidence/optimization/egraph-saturation-matrix.json",
            "release/evidence/optimization/egraph-semantic-contracts.json",
        ],
        "parser-coherence" => &[
            "release/evidence/parser/distributed-parser-map.json",
            "release/evidence/parser/vyre-frontend-c-contracts.json",
            "release/evidence/parser/vyrec-cli-contracts.json",
            "release/evidence/parser/weir-contracts.json",
            "release/evidence/parser/security-analysis-consumer-contracts.json",
            "release/evidence/parser/security-grammar-gen-contracts.json",
        ],
        "weir-matrix" => &[
            "release/evidence/weir/weir-analysis-api-matrix.json",
            "release/evidence/weir/weir-vyre-integration-tests.json",
            "release/evidence/weir/weir-readme-contracts.json",
        ],
        "release-completion-audit" => &["release/evidence/final/completion-audit.json"],
        _ => &[],
    }
}

fn write_release_evidence_run(
    workspace_root: &Path,
    commands: Vec<ReleaseEvidenceCommandRecord>,
    blockers: &[String],
    reports: &[String],
) {
    let output = workspace_root.join("release/evidence/final/release-evidence-run.json");
    if let Some(parent) = output.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            eprintln!(
                "release-evidence: failed to create `{}`: {error}",
                parent.display()
            );
            std::process::exit(1);
        }
    }
    let required_command_count = commands.iter().filter(|command| command.required).count();
    let report_only_command_count = commands.len().saturating_sub(required_command_count);
    let successful_commands = commands
        .iter()
        .filter(|command| command.status == "success")
        .count();
    let artifact_failures = commands
        .iter()
        .flat_map(|command| &command.artifact_statuses)
        .filter(|artifact| !artifact.exists || artifact.bytes == 0 || artifact.read_error.is_some())
        .count();
    let run = ReleaseEvidenceRun {
        schema_version: 2,
        total_commands: commands.len(),
        successful_commands,
        command_failures: commands.len().saturating_sub(successful_commands),
        artifact_failures,
        command_count: commands.len(),
        required_command_count,
        report_only_command_count,
        commands,
        blockers: blockers.to_vec(),
        reports: reports.to_vec(),
    };
    let json = match serde_json::to_string_pretty(&run) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("release-evidence: failed to serialize run evidence: {error}");
            std::process::exit(1);
        }
    };
    if let Err(error) = fs::write(&output, format!("{json}\n")) {
        eprintln!(
            "release-evidence: failed to write `{}`: {error}",
            output.display()
        );
        std::process::exit(1);
    }
}
