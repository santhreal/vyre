//! Hold the recorded backend conformance evidence to the op matrix and to itself.

use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::path::Path;
use std::process::Command;

// The op matrix and everything derived from it have one owner, so a second
// copy here cannot drift from the registered-op matrix again.
use crate::artifact_gate::{self, Inspection};
use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::release::conformance_op_matrix::{
    evaluate_op_matrix_coverage, read_conformance_required_op_matrix,
};
use serde::{Deserialize, Serialize};

const MIN_RELEASE_OP_PAIRS: usize = 49;
const MAX_RELEASE_CONFORMANCE_TEXT_BYTES: u64 = 8_388_608;

/// Shape version of a per-backend conformance artifact.
///
/// A recorded artifact outlives the struct that wrote it, so a reader that only
/// deserializes cannot tell a stale shape from a corrupt file: three artifacts
/// carried a row-count field under its former name and reported as unparseable
/// JSON. Every rename or removal in `BackendConformanceArtifact` raises this,
/// and an artifact recorded under a lower version is reported as stale rather
/// than read.
const ARTIFACT_SCHEMA_VERSION: u32 = 4;

#[derive(Debug, Deserialize, Serialize)]
struct PairResult {
    op_id: String,
    backend_id: String,
    passed: bool,
    message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    input_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    output_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    timing_class: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    failure_class: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    replay_capsule: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
struct BackendDiffSummary {
    op_id: String,
    backend_id: String,
    input_digest: String,
    output_digest: String,
    timing_class: String,
    failure_class: String,
    passed: bool,
    source: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct BackendConformanceArtifact {
    schema_version: u32,
    backend_id: String,
    command: String,
    stdout_diagnostics: Vec<String>,
    total_pairs: usize,
    distinct_op_count: usize,
    catalog_required_op_count: usize,
    catalog_covered_op_count: usize,
    missing_catalog_ops: Vec<String>,
    release_backend_row_count: usize,
    supported_release_backend_row_count: usize,
    release_backend_rows: Vec<String>,
    missing_release_backend_rows: Vec<String>,
    op_matrix_blocked_release_count: usize,
    op_matrix_blocked_release_rows: Vec<String>,
    op_matrix_errors: Vec<String>,
    passed_pairs: usize,
    failed_pairs: usize,
    duplicate_op_ids: Vec<String>,
    diff_schema_version: u32,
    diff_summary_count: usize,
    diff_summary_errors: Vec<String>,
    diff_summaries: Vec<BackendDiffSummary>,
    pairs: Vec<PairResult>,
    blockers: Vec<String>,
}

/// Holds the recorded backend conformance evidence to the op matrix and to itself.
pub struct ReleaseConformanceGate;

impl Gate for ReleaseConformanceGate {
    fn name(&self) -> &'static str {
        "release-conformance"
    }

    fn help(&self) -> &'static str {
        "Judge the recorded backend conformance artifacts under \
         release/evidence/conformance. Proves each requested backend recorded an artifact, that \
         it covers every op id the op matrix requires, reaches the release op-pair floor, \
         repeats no op id, emits no empty op id, carries no failed pair, carries no pair from \
         another backend, and that its diff summaries are the ones its own pairs imply. Proves \
         nothing by itself about the hardware: no dispatch runs unless --write is passed, and a \
         recorded artifact is only as current as the run that wrote it. With --write it runs \
         vyre-conform against each requested backend and rewrites the artifacts."
    }

    fn generates(&self) -> bool {
        true
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let config = match parse_args(&ctx.args) {
            Ok(config) => config,
            Err(message) => {
                return Ok(Report::with_findings(vec![Finding::new(
                    message,
                    "Pass --backend with one of cuda, wgpu, metal, cpu-ref, reference, or all.",
                )]))
            }
        };
        let inspection = if ctx.write {
            measure(&ctx.root, &config)
        } else {
            audit(&ctx.root, &config)
        };
        Ok(artifact_gate::settle_inspection(
            ctx,
            self.name(),
            inspection,
        ))
    }
}

/// The artifact each backend records, keyed by its normalised backend id.
const BACKEND_ARTIFACTS: &[(&str, &str)] = &[
    ("cuda", "release/evidence/conformance/cuda-conformance.json"),
    ("wgpu", "release/evidence/conformance/wgpu-conformance.json"),
    (
        "metal",
        "release/evidence/conformance/metal-conformance.json",
    ),
    (
        "cpu-ref",
        "release/evidence/conformance/reference-conformance.json",
    ),
];

const RELEASE_LOG: &str = "release/evidence/conformance/release-gate-log.json";

/// `reference` is the caller's spelling of the backend the runner calls `cpu-ref`.
fn backend_id_of(backend: &str) -> &str {
    if backend == "reference" {
        "cpu-ref"
    } else {
        backend
    }
}

fn artifact_of(backend_id: &str) -> Option<&'static str> {
    BACKEND_ARTIFACTS
        .iter()
        .find(|(id, _)| *id == backend_id)
        .map(|(_, artifact)| *artifact)
}

/// Judge the artifacts already on disk, running no dispatch.
///
/// This is what the sweep does. Every assertion the generator made at write
/// time is made again here against the committed record, so an artifact that
/// was correct when written and is wrong against today's op matrix is caught by
/// a run that needs no device.
fn audit(workspace_root: &Path, config: &Config) -> Inspection {
    let mut inspection = Inspection::new();
    for backend in &config.backends {
        let backend_id = backend_id_of(backend);
        let Some(artifact) = artifact_of(backend_id) else {
            inspection.find(Finding::new(
                format!("unsupported release conformance backend `{backend}`"),
                "Pass one of cuda, wgpu, metal, cpu-ref, reference, or all.",
            ));
            continue;
        };
        let recorded = match read_text_bounded(&workspace_root.join(artifact)) {
            Ok(text) => text,
            Err(error) => {
                inspection.blocked(
                    artifact,
                    format!("{backend_id} conformance evidence could not be read: {error}"),
                    "Run `./cargo_full run --bin xtask -- release-conformance --backend \
                     <backend> --write` on a host with that device and commit the artifact.",
                );
                continue;
            }
        };
        let recorded = match recorded_artifact(&recorded) {
            Ok(value) => value,
            Err(reason) => {
                inspection.blocked(
                    artifact,
                    format!("{backend_id} conformance evidence {reason}"),
                    "Regenerate it with --write. A conformance artifact this reader cannot read \
                     records nothing.",
                );
                continue;
            }
        };
        if recorded.backend_id != backend_id {
            inspection.blocked(
                artifact,
                format!(
                    "{artifact} records backend `{}`, not `{backend_id}`",
                    recorded.backend_id
                ),
                "One artifact holds one backend. Regenerate it for the backend its path names.",
            );
        }
        let assessed = assess(
            workspace_root,
            backend_id,
            &recorded.pairs,
            &recorded.stdout_diagnostics,
            Vec::new(),
        );
        for blocker in &assessed.blockers {
            inspection.blocked(
                artifact,
                format!("{backend_id}: {blocker}"),
                "Fix the operation, the op matrix row, or the conformance runner the sentence \
                 names, then rerun with --write on a host with that device.",
            );
        }
        for divergence in recorded_summary_divergences(&recorded, &assessed) {
            inspection.blocked(
                artifact,
                divergence,
                "The artifact's own summary no longer follows from the pairs beside it, so one \
                 of the two was edited. Regenerate it with --write.",
            );
        }
    }
    audit_release_log(workspace_root, &mut inspection);
    inspection
}

/// Blockers the release log recorded, and whether it is there to read at all.
fn audit_release_log(workspace_root: &Path, inspection: &mut Inspection) {
    let text = match read_text_bounded(&workspace_root.join(RELEASE_LOG)) {
        Ok(text) => text,
        Err(error) => {
            inspection.blocked(
                RELEASE_LOG,
                format!("the release conformance log could not be read: {error}"),
                "Run `./cargo_full run --bin xtask -- release-conformance --backend all --write` \
                 and commit the log.",
            );
            return;
        }
    };
    let log: serde_json::Value = match serde_json::from_str(&text) {
        Ok(value) => value,
        Err(error) => {
            inspection.blocked(
                RELEASE_LOG,
                format!("the release conformance log is not readable as JSON: {error}"),
                "Regenerate it with --write.",
            );
            return;
        }
    };
    for status in log
        .get("artifact_statuses")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
    {
        let path = status
            .get("path")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("(unnamed)");
        if status.get("exists").and_then(serde_json::Value::as_bool) != Some(true) {
            inspection.blocked(
                RELEASE_LOG,
                format!("the release conformance log records `{path}` as absent"),
                "Run the sweep for that backend with --write on a host with the device.",
            );
        }
        if status.get("bytes").and_then(serde_json::Value::as_u64) == Some(0) {
            inspection.blocked(
                RELEASE_LOG,
                format!("the release conformance log records `{path}` as empty"),
                "An empty conformance artifact records nothing. Rerun with --write.",
            );
        }
    }
    for blocker in log
        .get("blockers")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
    {
        inspection.blocked(
            RELEASE_LOG,
            format!("the last recorded release conformance run was blocked: {blocker}"),
            "Resolve the blocker and rerun with --write so the log records a clean run.",
        );
    }
}

/// Run each requested backend and rewrite its artifact and the release log.
fn measure(workspace_root: &Path, config: &Config) -> Inspection {
    let mut inspection = Inspection::new();
    let mut failures = Vec::new();
    for backend in &config.backends {
        let backend_id = backend_id_of(backend);
        let Some(artifact) = artifact_of(backend_id) else {
            failures.push(format!(
                "unsupported release conformance backend `{backend}`"
            ));
            inspection.find(Finding::new(
                format!("unsupported release conformance backend `{backend}`"),
                "Pass one of cuda, wgpu, metal, cpu-ref, reference, or all.",
            ));
            continue;
        };
        let body = measure_backend(workspace_root, backend_id);
        for blocker in &body.blockers {
            failures.push(format!("{backend_id}: {blocker}"));
            inspection.blocked(
                artifact,
                format!("{backend_id}: {blocker}"),
                "Fix the operation, the op matrix row, or the conformance runner the sentence \
                 names, then rerun.",
            );
        }
        inspection.generates(artifact, &body);
    }
    inspection.generates(RELEASE_LOG, &release_log(workspace_root, config, &failures));
    inspection
}

/// Dispatch one backend and record what it produced.
fn measure_backend(workspace_root: &Path, backend_id: &str) -> BackendConformanceArtifact {
    let mut args = vec![
        "run".to_string(),
        "-p".to_string(),
        "vyre-conform".to_string(),
        "--release".to_string(),
    ];
    if matches!(backend_id, "cuda" | "wgpu" | "metal") {
        args.push("--features".to_string());
        args.push("gpu".to_string());
    }
    args.extend([
        "--bin".to_string(),
        "vyre-conform".to_string(),
        "--".to_string(),
        "dispatch".to_string(),
        "--backend".to_string(),
        backend_id.to_string(),
        "--ops".to_string(),
        "all".to_string(),
    ]);
    let runner = crate::cargo_runner::binary(workspace_root);
    let command = format!("{} {}", runner.display(), args.join(" "));
    let output = Command::new(&runner)
        .args(&args)
        .current_dir(workspace_root)
        .output();
    let (pairs, stdout_diagnostics, mut blockers) = match &output {
        Ok(output) => match parse_pairs(&output.stdout) {
            Ok(parsed) => (parsed.pairs, parsed.diagnostics, Vec::new()),
            Err(error) => (Vec::new(), Vec::new(), vec![error]),
        },
        Err(error) => (
            Vec::new(),
            Vec::new(),
            vec![format!(
                "failed to run `{command}`: {error}. Set VYRE_CARGO_RUNNER to the bounded workspace cargo wrapper if it is not named `cargo_full`."
            )],
        ),
    };
    if let Ok(output) = &output {
        if !output.status.success() {
            blockers.push(format!(
                "`{command}` exited with {}; stderr: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
    }
    let assessed = assess(
        workspace_root,
        backend_id,
        &pairs,
        &stdout_diagnostics,
        blockers,
    );
    artifact_body(backend_id, command, pairs, stdout_diagnostics, assessed)
}

/// Every judgement a set of recorded pairs supports, and the derived fields.
struct Assessment {
    blockers: Vec<String>,
    catalog: crate::release::conformance_op_matrix::OpMatrixCatalog,
    coverage: crate::release::conformance_op_matrix::OpMatrixCoverage,
    distinct_op_count: usize,
    duplicate_op_ids: Vec<String>,
    failed_pairs: usize,
    diff_summaries: Vec<BackendDiffSummary>,
    diff_summary_errors: Vec<String>,
}

/// Judge `pairs` against the release floors and the live op matrix.
///
/// Nothing here reads a device. The same function judges pairs that were just
/// measured and pairs read back out of a committed artifact, so a recorded run
/// answers to the current op matrix rather than to the one it was written
/// against.
fn assess(
    workspace_root: &Path,
    backend_id: &str,
    pairs: &[PairResult],
    stdout_diagnostics: &[String],
    mut blockers: Vec<String>,
) -> Assessment {
    let failed_pairs = pairs.iter().filter(|pair| !pair.passed).count();
    let mut seen_ops = BTreeSet::new();
    let mut duplicate_op_ids = BTreeSet::new();
    for pair in pairs {
        if pair.op_id.trim().is_empty() {
            blockers.push(format!("{backend_id} conformance emitted an empty op_id"));
        }
        if !seen_ops.insert(pair.op_id.clone()) {
            duplicate_op_ids.insert(pair.op_id.clone());
        }
    }
    if !stdout_diagnostics.is_empty() {
        blockers.push(format!(
            "{backend_id} conformance stdout contained {} non-evidence line(s); fix the runner to emit JSONL evidence on stdout and diagnostics on stderr",
            stdout_diagnostics.len()
        ));
    }
    if pairs.is_empty() {
        blockers.push(format!("{backend_id} conformance produced zero op pairs"));
    }
    if pairs.len() < MIN_RELEASE_OP_PAIRS {
        blockers.push(format!(
            "{backend_id} conformance produced {} op pair(s), below release floor {MIN_RELEASE_OP_PAIRS}",
            pairs.len()
        ));
    }
    if seen_ops.len() < MIN_RELEASE_OP_PAIRS {
        blockers.push(format!(
            "{backend_id} conformance covered {} distinct op id(s), below release floor {MIN_RELEASE_OP_PAIRS}",
            seen_ops.len()
        ));
    }
    if !duplicate_op_ids.is_empty() {
        blockers.push(format!(
            "{backend_id} conformance emitted {} duplicate op id(s)",
            duplicate_op_ids.len()
        ));
    }
    let catalog = read_conformance_required_op_matrix(workspace_root);
    for error in &catalog.errors {
        blockers.push(error.clone());
    }
    let coverage = evaluate_op_matrix_coverage(
        &catalog,
        |op| seen_ops.contains(op),
        |missing| {
            format!("{backend_id} conformance is missing {missing} OP_MATRIX-required op id(s)")
        },
        &mut blockers,
    );
    if failed_pairs != 0 {
        blockers.push(format!(
            "{backend_id} conformance reported {failed_pairs} failed pair(s)"
        ));
    }
    let wrong_backend_pairs = pairs
        .iter()
        .filter(|pair| pair.backend_id != backend_id)
        .count();
    if wrong_backend_pairs != 0 {
        blockers.push(format!(
            "{backend_id} conformance artifact contains {wrong_backend_pairs} pair(s) with a different backend_id"
        ));
    }
    let diff_summaries = backend_diff_summaries(pairs);
    let diff_summary_errors = validate_backend_diff_summaries(backend_id, pairs, &diff_summaries);
    for error in &diff_summary_errors {
        blockers.push(error.clone());
    }
    Assessment {
        blockers,
        catalog,
        coverage,
        distinct_op_count: seen_ops.len(),
        duplicate_op_ids: duplicate_op_ids.into_iter().collect(),
        failed_pairs,
        diff_summaries,
        diff_summary_errors,
    }
}

/// Assemble the artifact from the pairs and the judgement they earned.
fn artifact_body(
    backend_id: &str,
    command: String,
    pairs: Vec<PairResult>,
    stdout_diagnostics: Vec<String>,
    assessed: Assessment,
) -> BackendConformanceArtifact {
    BackendConformanceArtifact {
        schema_version: ARTIFACT_SCHEMA_VERSION,
        backend_id: backend_id.to_string(),
        command,
        stdout_diagnostics,
        total_pairs: pairs.len(),
        distinct_op_count: assessed.distinct_op_count,
        catalog_required_op_count: assessed.coverage.catalog_required_op_count,
        catalog_covered_op_count: assessed.coverage.catalog_covered_op_count,
        missing_catalog_ops: assessed.coverage.missing_catalog_ops,
        release_backend_row_count: assessed.coverage.release_backend_row_count,
        supported_release_backend_row_count: assessed.coverage.supported_release_backend_row_count,
        release_backend_rows: assessed.catalog.release_backend_rows,
        missing_release_backend_rows: assessed.catalog.missing_release_backend_rows,
        op_matrix_blocked_release_count: assessed.coverage.op_matrix_blocked_release_count,
        op_matrix_blocked_release_rows: assessed.catalog.blocked_release_rows,
        op_matrix_errors: assessed.catalog.errors,
        passed_pairs: pairs.len().saturating_sub(assessed.failed_pairs),
        failed_pairs: assessed.failed_pairs,
        duplicate_op_ids: assessed.duplicate_op_ids,
        diff_schema_version: 1,
        diff_summary_count: assessed.diff_summaries.len(),
        diff_summary_errors: assessed.diff_summary_errors,
        diff_summaries: assessed.diff_summaries,
        pairs,
        blockers: assessed.blockers,
    }
}

/// Every summary field a recorded artifact states that its own pairs deny.
///
/// A conformance artifact carries both the pairs and a summary of them, so the
/// two can be edited apart. Recomputing the summary from the recorded pairs is
/// the only way a reader learns which of the two is lying.
fn recorded_summary_divergences(
    recorded: &BackendConformanceArtifact,
    assessed: &Assessment,
) -> Vec<String> {
    let mut divergences = Vec::new();
    let mut compare = |field: &str, recorded_value: usize, derived: usize| {
        if recorded_value != derived {
            divergences.push(format!(
                "records {field} as {recorded_value}; its own pairs give {derived}"
            ));
        }
    };
    compare("total_pairs", recorded.total_pairs, recorded.pairs.len());
    compare(
        "distinct_op_count",
        recorded.distinct_op_count,
        assessed.distinct_op_count,
    );
    compare("failed_pairs", recorded.failed_pairs, assessed.failed_pairs);
    compare(
        "passed_pairs",
        recorded.passed_pairs,
        recorded.pairs.len().saturating_sub(assessed.failed_pairs),
    );
    compare(
        "catalog_required_op_count",
        recorded.catalog_required_op_count,
        assessed.coverage.catalog_required_op_count,
    );
    compare(
        "catalog_covered_op_count",
        recorded.catalog_covered_op_count,
        assessed.coverage.catalog_covered_op_count,
    );
    compare(
        "release_backend_row_count",
        recorded.release_backend_row_count,
        assessed.coverage.release_backend_row_count,
    );
    compare(
        "op_matrix_blocked_release_count",
        recorded.op_matrix_blocked_release_count,
        assessed.coverage.op_matrix_blocked_release_count,
    );
    compare(
        "diff_summary_count",
        recorded.diff_summary_count,
        assessed.diff_summaries.len(),
    );
    if recorded.diff_summaries != assessed.diff_summaries {
        divergences.push("records diff summaries its own pairs do not produce".to_string());
    }
    if recorded.missing_catalog_ops != assessed.coverage.missing_catalog_ops {
        divergences.push(
            "records a missing-op list the current op matrix and its own pairs do not produce"
                .to_string(),
        );
    }
    if recorded.release_backend_rows != assessed.catalog.release_backend_rows {
        divergences.push(
            "records release backend rows the current op matrix does not declare".to_string(),
        );
    }
    divergences
}

struct ParsedPairs {
    pairs: Vec<PairResult>,
    diagnostics: Vec<String>,
}

fn parse_pairs(stdout: &[u8]) -> Result<ParsedPairs, String> {
    let text = String::from_utf8_lossy(stdout);
    let trimmed_text = text.trim();
    if trimmed_text.starts_with('[') || trimmed_text.starts_with('{') {
        if let Ok(parsed) = parse_json_conformance_payload(trimmed_text) {
            return Ok(parsed);
        }
    }
    let mut pairs = Vec::new();
    let mut diagnostics = Vec::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let trimmed = line.trim();
        if !trimmed.starts_with('{') {
            diagnostics.push(trimmed.to_string());
            continue;
        }
        let pair = serde_json::from_str::<PairResult>(trimmed).map_err(|error| {
            format!(
                "conformance runner emitted invalid JSON evidence line: {error}; line={trimmed}"
            )
        })?;
        pairs.push(pair);
    }
    Ok(ParsedPairs { pairs, diagnostics })
}

fn parse_json_conformance_payload(text: &str) -> Result<ParsedPairs, String> {
    let value = serde_json::from_str::<serde_json::Value>(text)
        .map_err(|error| format!("conformance runner emitted invalid JSON payload: {error}"))?;
    let values = if let Some(array) = value.as_array() {
        array.clone()
    } else if let Some(array) = value.get("pairs").and_then(serde_json::Value::as_array) {
        array.clone()
    } else if value.get("op_id").is_some() && value.get("backend_id").is_some() {
        vec![value]
    } else {
        return Err(
            "conformance runner JSON payload did not contain a pair object or pairs array"
                .to_string(),
        );
    };
    let mut pairs = Vec::with_capacity(values.len());
    for value in values {
        let pair = serde_json::from_value::<PairResult>(value)
            .map_err(|error| format!("conformance JSON pair failed schema validation: {error}"))?;
        pairs.push(pair);
    }
    Ok(ParsedPairs {
        pairs,
        diagnostics: Vec::new(),
    })
}

fn backend_diff_summaries(pairs: &[PairResult]) -> Vec<BackendDiffSummary> {
    pairs.iter().map(backend_diff_summary).collect()
}

fn backend_diff_summary(pair: &PairResult) -> BackendDiffSummary {
    let (input_digest, input_source) = pair_input_digest(pair);
    let (output_digest, output_source) = pair_output_digest(pair);
    BackendDiffSummary {
        op_id: pair.op_id.clone(),
        backend_id: pair.backend_id.clone(),
        input_digest,
        output_digest,
        timing_class: pair
            .timing_class
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or("not_reported")
            .to_string(),
        failure_class: pair
            .failure_class
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| classify_failure_class(pair)),
        passed: pair.passed,
        source: format!("input={input_source};output={output_source}"),
    }
}

fn pair_input_digest(pair: &PairResult) -> (String, &'static str) {
    if let Some(digest) = pair
        .input_digest
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        return (digest.to_string(), "runner_pair_field");
    }
    if let Some(digest) = replay_capsule_string(pair, "witness_input_blake3") {
        return (digest.to_string(), "replay_capsule");
    }
    let witness_case_count = witness_case_count_from_message(&pair.message);
    (
        pair_envelope_digest(
            "vyre-conform-input-envelope-v1",
            &[pair.op_id.as_str(), witness_case_count.as_str()],
        ),
        "derived_pair_envelope",
    )
}

fn pair_output_digest(pair: &PairResult) -> (String, &'static str) {
    if let Some(digest) = pair
        .output_digest
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        return (digest.to_string(), "runner_pair_field");
    }
    if let Some(digest) = replay_capsule_string(pair, "backend_output_blake3") {
        return (digest.to_string(), "replay_capsule");
    }
    let witness_case_count = witness_case_count_from_message(&pair.message);
    let failure_class = classify_failure_class(pair);
    (
        pair_envelope_digest(
            "vyre-conform-output-envelope-v1",
            &[
                pair.op_id.as_str(),
                witness_case_count.as_str(),
                if pair.passed { "passed" } else { "failed" },
                failure_class.as_str(),
            ],
        ),
        "derived_pair_envelope",
    )
}

fn replay_capsule_string<'a>(pair: &'a PairResult, field: &str) -> Option<&'a str> {
    pair.replay_capsule
        .as_ref()
        .and_then(|capsule| capsule.get(field))
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
}

fn pair_envelope_digest(domain: &str, fields: &[&str]) -> String {
    let mut hasher = blake3::Hasher::new();
    update_digest_str(&mut hasher, domain);
    for field in fields {
        update_digest_str(&mut hasher, field);
    }
    format!("{domain}:{}", hasher.finalize().to_hex())
}

fn update_digest_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn witness_case_count_from_message(message: &str) -> String {
    let Some((count, _)) = message.split_once(" witness case") else {
        return "unknown".to_string();
    };
    let count = count.trim();
    if count.is_empty() || !count.bytes().all(|byte| byte.is_ascii_digit()) {
        return "unknown".to_string();
    }
    count.to_string()
}

fn classify_failure_class(pair: &PairResult) -> String {
    if pair.passed {
        return "passed".to_string();
    }
    let lowered = pair.message.to_ascii_lowercase();
    if lowered.contains("panicked") || lowered.contains("panic") {
        "panic".to_string()
    } else if lowered.contains("acquisition") {
        "backend_acquisition_error".to_string()
    } else if lowered.contains("diverged")
        || lowered.contains("mismatch")
        || lowered.contains("different")
    {
        "output_mismatch".to_string()
    } else if lowered.contains("dispatch failed") || lowered.contains("backend dispatch failed") {
        "dispatch_error".to_string()
    } else if lowered.contains("witness")
        || lowered.contains("fixture")
        || lowered.contains("expected_output")
    {
        "fixture_error".to_string()
    } else {
        "other_failure".to_string()
    }
}

fn validate_backend_diff_summaries(
    backend_id: &str,
    pairs: &[PairResult],
    summaries: &[BackendDiffSummary],
) -> Vec<String> {
    let mut errors = Vec::new();
    if summaries.len() != pairs.len() {
        errors.push(format!(
            "{backend_id} conformance diff_summary_count={} must equal pair count {}",
            summaries.len(),
            pairs.len()
        ));
    }
    let pair_ops = pairs
        .iter()
        .filter(|pair| !pair.op_id.trim().is_empty())
        .map(|pair| pair.op_id.as_str())
        .collect::<BTreeSet<_>>();
    let summary_ops = summaries
        .iter()
        .filter(|summary| !summary.op_id.trim().is_empty())
        .map(|summary| summary.op_id.as_str())
        .collect::<BTreeSet<_>>();
    for op in pair_ops.difference(&summary_ops) {
        errors.push(format!(
            "{backend_id} conformance diff_summaries missing op `{op}`"
        ));
    }
    for op in summary_ops.difference(&pair_ops) {
        errors.push(format!(
            "{backend_id} conformance diff_summaries contain non-pair op `{op}`"
        ));
    }
    for summary in summaries {
        if summary.op_id.trim().is_empty() {
            errors.push(format!(
                "{backend_id} conformance diff summary has empty op_id"
            ));
        }
        if summary.backend_id != backend_id {
            errors.push(format!(
                "{backend_id} conformance diff summary for `{}` reports backend `{}`",
                summary.op_id, summary.backend_id
            ));
        }
        if summary.input_digest.trim().is_empty() {
            errors.push(format!(
                "{backend_id} conformance diff summary for `{}` has empty input_digest",
                summary.op_id
            ));
        }
        if summary.output_digest.trim().is_empty() {
            errors.push(format!(
                "{backend_id} conformance diff summary for `{}` has empty output_digest",
                summary.op_id
            ));
        }
        if summary.timing_class.trim().is_empty() {
            errors.push(format!(
                "{backend_id} conformance diff summary for `{}` has empty timing_class",
                summary.op_id
            ));
        }
        if summary.failure_class.trim().is_empty() {
            errors.push(format!(
                "{backend_id} conformance diff summary for `{}` has empty failure_class",
                summary.op_id
            ));
        }
    }
    errors
}

#[derive(Serialize)]
struct ReleaseLog {
    schema_version: u32,
    command: &'static str,
    requested_backends: Vec<String>,
    required_artifacts: Vec<&'static str>,
    artifact_statuses: Vec<ReleaseArtifactStatus>,
    blockers: Vec<String>,
}

#[derive(Serialize)]
struct ReleaseArtifactStatus {
    path: &'static str,
    exists: bool,
    bytes: u64,
    read_error: Option<String>,
}

/// What the requested sweep left on disk, recorded beside the artifacts.
fn release_log(workspace_root: &Path, config: &Config, failures: &[String]) -> ReleaseLog {
    let mut required_artifacts = vec![
        "cuda-conformance.json",
        "wgpu-conformance.json",
        "reference-conformance.json",
    ];
    if config.backends.iter().any(|backend| backend == "metal") {
        required_artifacts.push("metal-conformance.json");
    }
    let artifact_statuses = required_artifacts
        .iter()
        .map(|artifact| {
            let path = workspace_root
                .join("release/evidence/conformance")
                .join(artifact);
            match fs::metadata(&path) {
                Ok(metadata) => ReleaseArtifactStatus {
                    path: artifact,
                    exists: metadata.is_file(),
                    bytes: metadata.len(),
                    read_error: None,
                },
                Err(error) => ReleaseArtifactStatus {
                    path: artifact,
                    exists: false,
                    bytes: 0,
                    read_error: Some(error.to_string()),
                },
            }
        })
        .collect();
    ReleaseLog {
        schema_version: 2,
        command: "cargo_full run --bin xtask -- release-conformance",
        requested_backends: config.backends.clone(),
        required_artifacts,
        artifact_statuses,
        blockers: failures.to_vec(),
    }
}

struct Config {
    backends: Vec<String>,
}

/// The backends the caller asked for.
///
/// `args` holds the flags after the subcommand name, so the scan starts at
/// zero. `--write` is read by the runner before the gate sees it and is skipped
/// here rather than rejected as unknown.
fn parse_args(args: &[String]) -> Result<Config, String> {
    let mut backends = vec![
        "cuda".to_string(),
        "wgpu".to_string(),
        "cpu-ref".to_string(),
    ];
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--write" => index += 1,
            "--backend" => {
                let Some(value) = args.get(index + 1) else {
                    return Err(
                        "Fix: --backend requires cuda, wgpu, metal, cpu-ref, reference, or all."
                            .to_string(),
                    );
                };
                backends = if value == "all" {
                    vec![
                        "cuda".to_string(),
                        "wgpu".to_string(),
                        "cpu-ref".to_string(),
                    ]
                } else if matches!(
                    value.as_str(),
                    "cuda" | "wgpu" | "metal" | "cpu-ref" | "reference"
                ) {
                    vec![value.clone()]
                } else {
                    return Err(
                        "Fix: --backend requires cuda, wgpu, metal, cpu-ref, reference, or all."
                            .to_string(),
                    );
                };
                index += 2;
            }
            other => {
                return Err(format!(
                    "Fix: unknown release-conformance option `{other}`."
                ))
            }
        }
    }
    Ok(Config { backends })
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    crate::output_arg::read_text_bounded(
        path,
        MAX_RELEASE_CONFORMANCE_TEXT_BYTES,
        "release conformance",
    )
}

/// Read a recorded artifact, or say which of the two ways it is unreadable.
///
/// The version is read before the shape so that an artifact written by an older
/// producer is reported as stale, with the version it carries, instead of as a
/// missing field on a struct the reader happens to hold today.
fn recorded_artifact(text: &str) -> Result<BackendConformanceArtifact, String> {
    let version = serde_json::from_str::<serde_json::Value>(text)
        .ok()
        .and_then(|value| value.get("schema_version")?.as_u64());
    match version {
        Some(version) if version as u32 != ARTIFACT_SCHEMA_VERSION => {
            return Err(format!(
                "was recorded under artifact schema {version}, and this reader holds schema \
                 {ARTIFACT_SCHEMA_VERSION}"
            ))
        }
        _ => {}
    }
    serde_json::from_str(text)
        .map_err(|error| format!("is not readable as a conformance artifact: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(op_id: &str, backend_id: &str, passed: bool, message: &str) -> PairResult {
        PairResult {
            op_id: op_id.to_string(),
            backend_id: backend_id.to_string(),
            passed,
            message: message.to_string(),
            input_digest: None,
            output_digest: None,
            timing_class: None,
            failure_class: None,
            replay_capsule: None,
        }
    }

    #[test]
    fn diff_summary_derives_stable_success_digests_for_external_flow_across_backends() {
        let cuda = pair(
            "external.flow.alias_ifds",
            "cuda",
            true,
            "3 witness case(s) matched vyre-reference byte-for-byte via backend.dispatch",
        );
        let wgpu = pair(
            "external.flow.alias_ifds",
            "wgpu",
            true,
            "3 witness case(s) matched vyre-reference byte-for-byte via backend.dispatch",
        );

        let summaries = backend_diff_summaries(&[cuda, wgpu]);

        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].input_digest, summaries[1].input_digest);
        assert_eq!(summaries[0].output_digest, summaries[1].output_digest);
        assert_eq!(summaries[0].timing_class, "not_reported");
        assert_eq!(summaries[0].failure_class, "passed");
        assert!(summaries[0].source.contains("derived_pair_envelope"));
    }

    #[test]
    fn diff_summary_uses_replay_capsule_digests_for_output_mismatch() {
        let mut failure = pair(
            "vyre-primitives::math::tensor_network_pair_contract",
            "cuda",
            false,
            "backend output diverged from vyre-reference on case 0",
        );
        failure.replay_capsule = Some(serde_json::json!({
            "witness_input_blake3": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "backend_output_blake3": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        }));

        let summary = backend_diff_summary(&failure);

        assert_eq!(
            summary.input_digest,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        );
        assert_eq!(
            summary.output_digest,
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        );
        assert_eq!(summary.failure_class, "output_mismatch");
        assert!(summary.source.contains("replay_capsule"));
    }

    #[test]
    fn diff_summary_validation_rejects_missing_and_wrong_backend_fields() {
        let pair = pair("vyre.add", "cuda", true, "1 witness case(s) matched");
        let bad = BackendDiffSummary {
            op_id: String::new(),
            backend_id: "wgpu".to_string(),
            input_digest: String::new(),
            output_digest: String::new(),
            timing_class: String::new(),
            failure_class: String::new(),
            passed: true,
            source: String::new(),
        };

        let errors = validate_backend_diff_summaries("cuda", &[pair], &[bad]);

        assert!(
            errors
                .iter()
                .any(|error| error.contains("diff_summaries missing op `vyre.add`")),
            "Fix: validation must reject pair rows without a matching diff summary; errors={errors:?}"
        );
        assert!(
            errors
                .iter()
                .any(|error| error.contains("reports backend `wgpu`")),
            "Fix: validation must reject cross-backend mislabeled diff summaries; errors={errors:?}"
        );
        assert!(
            errors
                .iter()
                .any(|error| error.contains("empty input_digest")),
            "Fix: validation must reject summaries without input_digest; errors={errors:?}"
        );
        assert!(
            errors
                .iter()
                .any(|error| error.contains("empty output_digest")),
            "Fix: validation must reject summaries without output_digest; errors={errors:?}"
        );
    }

    /// Every field name a current artifact carries, sorted.
    ///
    /// WHY: a recorded artifact is read back by a later build, so a rename here
    /// silently turns committed evidence into an unparseable file. This list and
    /// `ARTIFACT_SCHEMA_VERSION` move together: change the shape and the suite is
    /// red until the version records that it changed.
    const RECORDED_FIELDS: &[&str] = &[
        "backend_id",
        "blockers",
        "catalog_covered_op_count",
        "catalog_required_op_count",
        "command",
        "diff_schema_version",
        "diff_summaries",
        "diff_summary_count",
        "diff_summary_errors",
        "distinct_op_count",
        "duplicate_op_ids",
        "failed_pairs",
        "missing_catalog_ops",
        "missing_release_backend_rows",
        "op_matrix_blocked_release_count",
        "op_matrix_blocked_release_rows",
        "op_matrix_errors",
        "pairs",
        "passed_pairs",
        "release_backend_row_count",
        "release_backend_rows",
        "schema_version",
        "stdout_diagnostics",
        "supported_release_backend_row_count",
        "total_pairs",
    ];

    fn recorded_shape() -> serde_json::Value {
        let assessed = Assessment {
            blockers: Vec::new(),
            catalog: crate::release::conformance_op_matrix::OpMatrixCatalog::default(),
            coverage: crate::release::conformance_op_matrix::OpMatrixCoverage::default(),
            distinct_op_count: 0,
            duplicate_op_ids: Vec::new(),
            failed_pairs: 0,
            diff_summaries: Vec::new(),
            diff_summary_errors: Vec::new(),
        };
        serde_json::to_value(artifact_body(
            "reference",
            "vyre-conform".to_string(),
            Vec::new(),
            Vec::new(),
            assessed,
        ))
        .expect("an artifact serializes")
    }

    #[test]
    fn the_recorded_shape_is_the_one_the_version_names() {
        let shape = recorded_shape();
        let mut fields: Vec<&str> = shape
            .as_object()
            .expect("an artifact is a JSON object")
            .keys()
            .map(String::as_str)
            .collect();
        fields.sort_unstable();

        assert_eq!(
            fields, RECORDED_FIELDS,
            "the conformance artifact shape changed. Raise ARTIFACT_SCHEMA_VERSION so a recorded \
             artifact of the old shape is reported as stale, then record the new field set here."
        );
        assert_eq!(
            shape["schema_version"].as_u64(),
            Some(u64::from(ARTIFACT_SCHEMA_VERSION)),
            "a written artifact must carry the version this reader requires"
        );
    }

    #[test]
    fn a_current_artifact_reads_back() {
        let text = recorded_shape().to_string();

        let read = recorded_artifact(&text).expect("a current artifact reads back");

        assert_eq!(read.schema_version, ARTIFACT_SCHEMA_VERSION);
        assert_eq!(read.backend_id, "reference");
    }

    #[test]
    fn an_artifact_of_an_older_shape_is_reported_as_stale() {
        let mut shape = recorded_shape();
        let object = shape.as_object_mut().expect("an artifact is a JSON object");
        object.insert("schema_version".to_string(), serde_json::json!(3));
        let renamed = object
            .remove("supported_release_backend_row_count")
            .expect("the row count is recorded");
        object.insert(
            "non_runtime_supported_release_backend_row_count".to_string(),
            renamed,
        );

        let reason = recorded_artifact(&shape.to_string()).expect_err("a stale shape is rejected");

        assert!(
            reason.contains("schema 3") && reason.contains(&ARTIFACT_SCHEMA_VERSION.to_string()),
            "a stale artifact must name both versions, got: {reason}"
        );
    }

    #[test]
    fn a_file_that_is_not_json_reports_the_parse_error() {
        let reason = recorded_artifact("{").expect_err("truncated JSON is rejected");

        assert!(
            reason.contains("not readable as a conformance artifact"),
            "got: {reason}"
        );
    }
}
