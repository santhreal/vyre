//! Generate real backend conformance evidence artifacts.

use std::collections::BTreeSet;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

const MIN_RELEASE_OP_PAIRS: usize = 49;
const MAX_RELEASE_CONFORMANCE_TEXT_BYTES: u64 = 8_388_608;
const RUNTIME_DIALECT_CONTRACT_OPS: &[&str] = &[
    "core.indirect_dispatch",
    "io.dma_from_nvme",
    "io.write_back_to_nvme",
    "mem.unmap",
    "mem.zerocopy_map",
];

#[derive(Debug, Deserialize, Serialize)]
struct PairResult {
    op_id: String,
    backend_id: String,
    passed: bool,
    message: String,
}

#[derive(Debug, Serialize)]
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
    non_runtime_supported_release_backend_row_count: usize,
    runtime_dialect_contract_row_count: usize,
    runtime_dialect_contract_ops: Vec<&'static str>,
    release_backend_rows: Vec<String>,
    missing_release_backend_rows: Vec<String>,
    op_matrix_blocked_release_count: usize,
    op_matrix_blocked_release_rows: Vec<String>,
    op_matrix_errors: Vec<String>,
    passed_pairs: usize,
    failed_pairs: usize,
    duplicate_op_ids: Vec<String>,
    pairs: Vec<PairResult>,
    blockers: Vec<String>,
}

pub(crate) fn run(args: &[String]) {
    let config = match parse_args(args) {
        Ok(config) => config,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let mut failures = Vec::new();
    for backend in &config.backends {
        let artifact = match backend.as_str() {
            "cuda" => "release/evidence/conformance/cuda-conformance.json",
            "wgpu" => "release/evidence/conformance/wgpu-conformance.json",
            "cpu-ref" | "reference" => "release/evidence/conformance/reference-conformance.json",
            other => {
                failures.push(format!("unsupported release conformance backend `{other}`"));
                continue;
            }
        };
        if let Err(error) = run_backend_conformance(&workspace_root, backend, artifact) {
            failures.push(error);
        }
    }
    write_release_log(&workspace_root, &config.backends, &failures);
    if !failures.is_empty() {
        eprintln!("release-conformance: {} blocker(s):", failures.len());
        for failure in failures {
            eprintln!("  - {failure}");
        }
        std::process::exit(1);
    }
    println!("release-conformance: wrote backend conformance artifacts");
}

fn run_backend_conformance(
    workspace_root: &Path,
    backend: &str,
    artifact: &str,
) -> Result<(), String> {
    let backend_id = if backend == "reference" {
        "cpu-ref"
    } else {
        backend
    };
    let mut args = vec![
        "run".to_string(),
        "-p".to_string(),
        "vyre-conform-runner".to_string(),
        "--release".to_string(),
    ];
    if matches!(backend_id, "cuda" | "wgpu") {
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
    let runner = cargo_runner(workspace_root);
    let output = Command::new(&runner)
        .args(&args)
        .current_dir(workspace_root)
        .output()
        .map_err(|error| {
            format!(
                "failed to run `{} {}`: {error}. Set VYRE_CARGO_RUNNER to the bounded workspace cargo wrapper if it is not named `cargo_full`.",
                runner.display(),
                args.join(" ")
            )
        })?;
    let command = format!("{} {}", runner.display(), args.join(" "));
    let (pairs, stdout_diagnostics, mut blockers) = match parse_pairs(&output.stdout) {
        Ok(parsed) => (parsed.pairs, parsed.diagnostics, Vec::new()),
        Err(error) => (Vec::new(), Vec::new(), vec![error]),
    };
    let failed_pairs = pairs.iter().filter(|pair| !pair.passed).count();
    let mut seen_ops = BTreeSet::new();
    let mut duplicate_op_ids = BTreeSet::new();
    for pair in &pairs {
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
    if !output.status.success() {
        blockers.push(format!(
            "`{command}` exited with {}; stderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
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
    let missing_catalog_ops = catalog
        .required_ops
        .iter()
        .filter(|op| !seen_ops.contains(op.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let catalog_covered_op_count = catalog
        .required_ops
        .len()
        .saturating_sub(missing_catalog_ops.len());
    if catalog.required_ops.is_empty() {
        blockers.push("OP_MATRIX contributed zero conformance-required op ids".to_string());
    }
    if !missing_catalog_ops.is_empty() {
        blockers.push(format!(
            "{backend_id} conformance is missing {} OP_MATRIX-required op id(s)",
            missing_catalog_ops.len()
        ));
    }
    if !catalog.blocked_release_rows.is_empty() {
        blockers.push(format!(
            "OP_MATRIX contains {} release backend row(s) marked blocked_release",
            catalog.blocked_release_rows.len()
        ));
    }
    if !catalog.missing_release_backend_rows.is_empty() {
        blockers.push(format!(
            "OP_MATRIX is missing {} release backend row(s)",
            catalog.missing_release_backend_rows.len()
        ));
    }
    let runtime_dialect_contract_row_count =
        count_runtime_dialect_contract_rows(&catalog.release_backend_rows);
    let non_runtime_supported_release_backend_row_count =
        count_non_runtime_supported_release_backend_rows(&catalog.release_backend_rows);
    let expected_runtime_rows = RUNTIME_DIALECT_CONTRACT_OPS.len().saturating_mul(3);
    if runtime_dialect_contract_row_count != expected_runtime_rows {
        blockers.push(format!(
            "OP_MATRIX declares {runtime_dialect_contract_row_count} Category C runtime dialect contract row(s), expected {expected_runtime_rows}"
        ));
    }
    let expected_non_runtime_supported_rows = catalog
        .required_ops
        .len()
        .saturating_sub(RUNTIME_DIALECT_CONTRACT_OPS.len())
        .saturating_mul(3);
    if non_runtime_supported_release_backend_row_count != expected_non_runtime_supported_rows {
        blockers.push(format!(
            "OP_MATRIX declares {non_runtime_supported_release_backend_row_count} supported non-runtime release backend row(s), expected {expected_non_runtime_supported_rows}"
        ));
    }
    let expected_release_backend_rows = catalog.required_ops.len().saturating_mul(3);
    if catalog.release_backend_rows.len() < expected_release_backend_rows {
        blockers.push(format!(
            "OP_MATRIX declares {} release backend row(s), expected {expected_release_backend_rows} for reference/cuda/wgpu coverage",
            catalog.release_backend_rows.len()
        ));
    }
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
    let artifact_body = BackendConformanceArtifact {
        schema_version: 2,
        backend_id: backend_id.to_string(),
        command,
        stdout_diagnostics,
        total_pairs: pairs.len(),
        distinct_op_count: seen_ops.len(),
        catalog_required_op_count: catalog.required_ops.len(),
        catalog_covered_op_count,
        missing_catalog_ops,
        release_backend_row_count: catalog.release_backend_rows.len(),
        non_runtime_supported_release_backend_row_count,
        runtime_dialect_contract_row_count,
        runtime_dialect_contract_ops: RUNTIME_DIALECT_CONTRACT_OPS.to_vec(),
        release_backend_rows: catalog.release_backend_rows,
        missing_release_backend_rows: catalog.missing_release_backend_rows,
        op_matrix_blocked_release_count: catalog.blocked_release_rows.len(),
        op_matrix_blocked_release_rows: catalog.blocked_release_rows,
        op_matrix_errors: catalog.errors,
        passed_pairs: pairs.len().saturating_sub(failed_pairs),
        failed_pairs,
        duplicate_op_ids: duplicate_op_ids.into_iter().collect(),
        pairs,
        blockers,
    };
    write_json(&workspace_root.join(artifact), &artifact_body)?;
    if artifact_body.blockers.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} conformance artifact reports {} blocker(s)",
            artifact_body.backend_id,
            artifact_body.blockers.len()
        ))
    }
}

fn count_runtime_dialect_contract_rows(rows: &[String]) -> usize {
    rows.iter()
        .filter(|row| {
            let Some((op, backend, status)) = parse_release_backend_row(row) else {
                return false;
            };
            RUNTIME_DIALECT_CONTRACT_OPS.contains(&op)
                && ((backend == "reference" && status == "not_applicable")
                    || (matches!(backend, "cuda" | "wgpu") && status == "experimental"))
        })
        .count()
}

fn count_non_runtime_supported_release_backend_rows(rows: &[String]) -> usize {
    rows.iter()
        .filter(|row| {
            let Some((op, _backend, status)) = parse_release_backend_row(row) else {
                return false;
            };
            !RUNTIME_DIALECT_CONTRACT_OPS.contains(&op) && status == "supported"
        })
        .count()
}

fn parse_release_backend_row(row: &str) -> Option<(&str, &str, &str)> {
    let (prefix, status) = row.rsplit_once(':')?;
    let (op, backend) = prefix.rsplit_once(':')?;
    Some((op, backend, status))
}

struct OpMatrixCatalog {
    required_ops: BTreeSet<String>,
    release_backend_rows: Vec<String>,
    missing_release_backend_rows: Vec<String>,
    blocked_release_rows: Vec<String>,
    errors: Vec<String>,
}

fn read_conformance_required_op_matrix(vyre_root: &Path) -> OpMatrixCatalog {
    let matrix_path = vyre_root.join("docs/optimization/OP_MATRIX.toml");
    let text = match read_text_bounded(&matrix_path) {
        Ok(text) => text,
        Err(error) => {
            return OpMatrixCatalog {
                required_ops: BTreeSet::new(),
                release_backend_rows: Vec::new(),
                missing_release_backend_rows: Vec::new(),
                blocked_release_rows: Vec::new(),
                errors: vec![format!(
                    "could not read OP_MATRIX at {}: {error}",
                    matrix_path.display()
                )],
            };
        }
    };
    let value = match text.parse::<toml::Value>() {
        Ok(value) => value,
        Err(error) => {
            return OpMatrixCatalog {
                required_ops: BTreeSet::new(),
                release_backend_rows: Vec::new(),
                missing_release_backend_rows: Vec::new(),
                blocked_release_rows: Vec::new(),
                errors: vec![format!(
                    "could not parse OP_MATRIX at {}: {error}",
                    matrix_path.display()
                )],
            };
        }
    };
    let rows = match value.get("op").and_then(toml::Value::as_array) {
        Some(rows) => rows,
        None => {
            return OpMatrixCatalog {
                required_ops: BTreeSet::new(),
                release_backend_rows: Vec::new(),
                missing_release_backend_rows: Vec::new(),
                blocked_release_rows: Vec::new(),
                errors: vec![format!(
                    "OP_MATRIX at {} has no [[op]] array",
                    matrix_path.display()
                )],
            };
        }
    };
    if rows.is_empty() {
        return OpMatrixCatalog {
            required_ops: BTreeSet::new(),
            release_backend_rows: Vec::new(),
            missing_release_backend_rows: Vec::new(),
            blocked_release_rows: Vec::new(),
            errors: vec![format!(
                "OP_MATRIX at {} has zero op rows",
                matrix_path.display()
            )],
        };
    }
    let mut required_ops = BTreeSet::new();
    let mut release_backend_rows = Vec::new();
    let mut missing_release_backend_rows = Vec::new();
    let mut blocked_release_rows = Vec::new();
    for row in rows {
        let tier = row.get("tier").and_then(toml::Value::as_str).unwrap_or("");
        if tier == "foundation_ir" {
            continue;
        }
        let family = row
            .get("family")
            .and_then(toml::Value::as_str)
            .unwrap_or("<unknown>");
        for backend in ["reference", "cuda", "wgpu"] {
            if row.get(backend).and_then(toml::Value::as_str) == Some("blocked_release") {
                blocked_release_rows.push(format!("{family}:{backend}"));
            }
        }
        let Some(row_ops) = row.get("ops").and_then(toml::Value::as_array) else {
            continue;
        };
        for op in row_ops {
            if let Some(op) = op.as_str() {
                required_ops.insert(op.to_string());
                for backend in ["reference", "cuda", "wgpu"] {
                    match row.get(backend).and_then(toml::Value::as_str) {
                        Some("blocked_release") => {}
                        Some(status) if !status.trim().is_empty() => {
                            release_backend_rows.push(format!("{op}:{backend}:{status}"));
                        }
                        _ => missing_release_backend_rows.push(format!("{op}:{backend}")),
                    }
                }
            }
        }
    }
    OpMatrixCatalog {
        required_ops,
        release_backend_rows,
        missing_release_backend_rows,
        blocked_release_rows,
        errors: Vec::new(),
    }
}

struct ParsedPairs {
    pairs: Vec<PairResult>,
    diagnostics: Vec<String>,
}

fn cargo_runner(workspace_root: &Path) -> PathBuf {
    if let Some(runner) = std::env::var_os("VYRE_CARGO_RUNNER") {
        return PathBuf::from(runner);
    }
    let local = workspace_root.join("cargo_full");
    if local.is_file() {
        return local;
    }
    PathBuf::from("cargo_full")
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

fn write_release_log(workspace_root: &Path, requested_backends: &[String], failures: &[String]) {
    #[derive(Serialize)]
    struct ReleaseLog<'a> {
        schema_version: u32,
        command: &'static str,
        requested_backends: &'a [String],
        required_artifacts: Vec<&'static str>,
        artifact_statuses: Vec<ReleaseArtifactStatus>,
        blockers: &'a [String],
    }
    #[derive(Serialize)]
    struct ReleaseArtifactStatus {
        path: &'static str,
        exists: bool,
        bytes: u64,
        read_error: Option<String>,
    }
    let required_artifacts = vec![
        "cuda-conformance.json",
        "wgpu-conformance.json",
        "reference-conformance.json",
    ];
    let artifact_statuses = required_artifacts
        .iter()
        .map(|artifact| {
            let path = workspace_root
                .join("release/evidence/conformance")
                .join(artifact);
            match fs::metadata(&path) {
                Ok(metadata) => ReleaseArtifactStatus {
                    path: *artifact,
                    exists: metadata.is_file(),
                    bytes: metadata.len(),
                    read_error: None,
                },
                Err(error) => ReleaseArtifactStatus {
                    path: *artifact,
                    exists: false,
                    bytes: 0,
                    read_error: Some(error.to_string()),
                },
            }
        })
        .collect();
    let log = ReleaseLog {
        schema_version: 2,
        command: "cargo_full run --bin xtask -- release-conformance",
        requested_backends,
        required_artifacts,
        artifact_statuses,
        blockers: failures,
    };
    if let Err(error) = write_json(
        &workspace_root.join("release/evidence/conformance/release-gate-log.json"),
        &log,
    ) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create `{}`: {error}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(value)
        .map_err(|error| format!("failed to serialize `{}`: {error}", path.display()))?;
    fs::write(path, format!("{json}\n"))
        .map_err(|error| format!("failed to write `{}`: {error}", path.display()))
}

struct Config {
    backends: Vec<String>,
}

fn parse_args(args: &[String]) -> Result<Config, String> {
    let mut backends = vec![
        "cuda".to_string(),
        "wgpu".to_string(),
        "cpu-ref".to_string(),
    ];
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--backend" => {
                let Some(value) = args.get(index + 1) else {
                    return Err("Fix: --backend requires cuda, wgpu, cpu-ref, or all.".to_string());
                };
                backends = if value == "all" {
                    vec![
                        "cuda".to_string(),
                        "wgpu".to_string(),
                        "cpu-ref".to_string(),
                    ]
                } else if matches!(value.as_str(), "cuda" | "wgpu" | "cpu-ref" | "reference") {
                    vec![value.clone()]
                } else {
                    return Err(
                        "Fix: --backend requires cuda, wgpu, cpu-ref, reference, or all."
                            .to_string(),
                    );
                };
                index += 2;
            }
            "--help" | "-h" => {
                println!(
                    "USAGE:\n  cargo_full run --bin xtask -- release-conformance [--backend all|cuda|wgpu|cpu-ref]\n\n\
                     Runs real vyre-conform dispatch for release conformance artifacts."
                );
                std::process::exit(0);
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
    let mut reader =
        fs::File::open(path)?.take(MAX_RELEASE_CONFORMANCE_TEXT_BYTES.saturating_add(1));
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    if text.len() as u64 > MAX_RELEASE_CONFORMANCE_TEXT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} exceeds {MAX_RELEASE_CONFORMANCE_TEXT_BYTES} byte release conformance read cap",
                path.display()
            ),
        ));
    }
    Ok(text)
}
