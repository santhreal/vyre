//! Release conformance matrix evidence.

use std::collections::BTreeSet;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;
use vyre_driver::backend::{backend_dispatches, registered_backends_by_precedence_slice};
use walkdir::WalkDir;

use vyre_driver_cuda as _;
use vyre_driver_reference as _;
use vyre_driver_spirv as _;
use vyre_driver_wgpu as _;
use vyre_intrinsics as _;
use vyre_libs as _;
use vyre_primitives as _;

const MIN_RELEASE_OP_COUNT: usize = 49;
const MAX_CONFORMANCE_EVIDENCE_TEXT_BYTES: u64 = 8_388_608;
const INT4_CONFORMANCE_OPS: &[&str] = &[
    "vyre-libs::quant::int4_dot_i32",
    "vyre-libs::quant::int4_dot_f32_scaled",
    "vyre-libs::quant::int4_matvec_f32_scaled",
    "vyre-libs::quant::int4_batched_matvec_f32_scaled",
    "vyre-libs::quant::int4_batched_matmul_f32_scaled",
    "vyre-libs::quant::int4_batched_matmul_top1_f32_scaled",
];
const RUNTIME_DIALECT_CONTRACT_OPS: &[&str] = &[
    "core.indirect_dispatch",
    "io.dma_from_nvme",
    "io.write_back_to_nvme",
    "mem.unmap",
    "mem.zerocopy_map",
];

#[derive(Debug, Serialize)]
struct ConformanceMatrix {
    schema_version: u32,
    op_count: usize,
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
    duplicate_op_ids: Vec<String>,
    fixture_input_count: usize,
    expected_output_count: usize,
    dispatch_backends: Vec<String>,
    ci_blocking_gate_count: usize,
    ci_gates: Vec<CiConformanceGate>,
    required_ci_statuses: Vec<String>,
    missing_required_ci_statuses: Vec<String>,
    ci_status_scan_errors: Vec<String>,
    path_filtered_required_workflows: Vec<String>,
    missing_required_workflow_triggers: Vec<String>,
    missing_fail_closed_fanins: Vec<String>,
    entries: Vec<ConformanceEntry>,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct ConformanceEntry {
    id: String,
    has_test_inputs: bool,
    has_expected_output: bool,
    tolerance_ulp: u32,
}

#[derive(Debug, Clone, Serialize)]
struct CiConformanceGate {
    workflow: String,
    read_error: Option<String>,
    gate: String,
    present: bool,
    command_present: bool,
    artifact_check_present: bool,
}

pub(crate) fn run(args: &[String]) {
    let (output, check) = match parse_args(args) {
        Ok(parsed) => parsed,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let mut entries = Vec::new();
    let mut ids = BTreeSet::new();
    let mut duplicate_op_ids = BTreeSet::new();
    for entry in vyre_harness::all_entries() {
        if !ids.insert(entry.id) {
            duplicate_op_ids.insert(entry.id.to_string());
        }
        entries.push(ConformanceEntry {
            id: entry.id.to_string(),
            has_test_inputs: entry.test_inputs.is_some(),
            has_expected_output: entry.expected_output.is_some(),
            tolerance_ulp: vyre_harness::OpEntry::tolerance_for_id(entry.id),
        });
    }
    entries.sort_by(|left, right| left.id.cmp(&right.id));
    let dispatch_backends: Vec<String> = registered_backends_by_precedence_slice()
        .iter()
        .filter(|backend| backend_dispatches(backend.id))
        .map(|backend| backend.id.to_string())
        .collect();
    let fixture_input_count = entries.iter().filter(|entry| entry.has_test_inputs).count();
    let expected_output_count = entries
        .iter()
        .filter(|entry| entry.has_expected_output)
        .count();
    let vyre_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let santh_root = vyre_root
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| vyre_root.clone());
    let ci_gates = inspect_ci_conformance_gates(&vyre_root);
    let (required_ci_statuses, mut ci_status_scan_errors) = parse_required_ci_statuses(&santh_root);
    let mut missing_required_ci_statuses = Vec::new();
    for status in &required_ci_statuses {
        if !ci_status_defined(&santh_root, status, &mut ci_status_scan_errors) {
            missing_required_ci_statuses.push(status.clone());
        }
    }
    let path_filtered_required_workflows = inspect_path_filtered_required_workflows(&santh_root);
    let missing_required_workflow_triggers = inspect_required_workflow_triggers(&santh_root);
    let missing_fail_closed_fanins = inspect_fail_closed_fanins(&santh_root);
    let mut blockers = Vec::new();
    let catalog = read_conformance_required_op_matrix(&vyre_root);
    for error in &catalog.errors {
        blockers.push(error.clone());
    }
    let missing_catalog_ops = catalog
        .required_ops
        .iter()
        .filter(|op| !ids.contains(op.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let catalog_covered_op_count = catalog
        .required_ops
        .len()
        .saturating_sub(missing_catalog_ops.len());
    let ci_blocking_gate_count = ci_gates
        .iter()
        .filter(|gate| gate.present && gate.command_present && gate.artifact_check_present)
        .count();
    if entries.is_empty() {
        blockers.push("no registered conformance op entries found".to_string());
    }
    if entries.len() < MIN_RELEASE_OP_COUNT {
        blockers.push(format!(
            "registered conformance op count {} is below release floor {MIN_RELEASE_OP_COUNT}",
            entries.len()
        ));
    }
    if ids.len() < MIN_RELEASE_OP_COUNT {
        blockers.push(format!(
            "registered distinct conformance op count {} is below release floor {MIN_RELEASE_OP_COUNT}",
            ids.len()
        ));
    }
    if !duplicate_op_ids.is_empty() {
        blockers.push(format!(
            "registered conformance matrix contains {} duplicate op id(s)",
            duplicate_op_ids.len()
        ));
    }
    if catalog.required_ops.is_empty() {
        blockers.push("OP_MATRIX contributed zero conformance-required op ids".to_string());
    }
    if !missing_catalog_ops.is_empty() {
        blockers.push(format!(
            "{} OP_MATRIX op id(s) are missing registered conformance entries",
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
    for required in ["cuda", "wgpu", "cpu-ref"] {
        if !dispatch_backends.iter().any(|backend| backend == required) {
            blockers.push(format!("required dispatch backend `{required}` is missing"));
        }
    }
    if fixture_input_count != entries.len() {
        blockers.push(format!(
            "only {fixture_input_count}/{} op entries have fixture inputs",
            entries.len()
        ));
    }
    if expected_output_count != entries.len() {
        blockers.push(format!(
            "only {expected_output_count}/{} op entries have expected outputs",
            entries.len()
        ));
    }
    if ci_blocking_gate_count < 3 {
        blockers.push(format!(
            "only {ci_blocking_gate_count}/{} conformance CI gate(s) are fully wired",
            ci_gates.len()
        ));
    }
    for gate in &ci_gates {
        if let Some(error) = &gate.read_error {
            blockers.push(format!(
                "conformance CI gate `{}` in `{}` could not read workflow: {error}",
                gate.gate, gate.workflow
            ));
        } else if !gate.present || !gate.command_present || !gate.artifact_check_present {
            blockers.push(format!(
                "conformance CI gate `{}` in `{}` is incomplete: present={}, command_present={}, artifact_check_present={}",
                gate.gate, gate.workflow, gate.present, gate.command_present, gate.artifact_check_present
            ));
        }
    }
    if !missing_required_ci_statuses.is_empty() {
        blockers.push(format!(
            "{} required branch-protection status context(s) are not defined by any workflow",
            missing_required_ci_statuses.len()
        ));
    }
    if !ci_status_scan_errors.is_empty() {
        blockers.push(format!(
            "{} CI status scan error(s) make branch-protection status evidence incomplete",
            ci_status_scan_errors.len()
        ));
    }
    if !path_filtered_required_workflows.is_empty() {
        blockers.push(format!(
            "{} required workflow(s) still use path filters",
            path_filtered_required_workflows.len()
        ));
    }
    if !missing_required_workflow_triggers.is_empty() {
        blockers.push(format!(
            "{} required workflow(s) are missing pull_request + push main trigger coverage",
            missing_required_workflow_triggers.len()
        ));
    }
    if !missing_fail_closed_fanins.is_empty() {
        blockers.push(format!(
            "{} required fan-in job(s) are missing fail-closed dependency checks",
            missing_fail_closed_fanins.len()
        ));
    }
    let matrix = ConformanceMatrix {
        schema_version: 2,
        op_count: entries.len(),
        distinct_op_count: ids.len(),
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
        duplicate_op_ids: duplicate_op_ids.into_iter().collect(),
        fixture_input_count,
        expected_output_count,
        dispatch_backends,
        ci_blocking_gate_count,
        ci_gates,
        required_ci_statuses,
        missing_required_ci_statuses,
        ci_status_scan_errors,
        path_filtered_required_workflows,
        missing_required_workflow_triggers,
        missing_fail_closed_fanins,
        entries,
        blockers,
    };
    if check {
        check_against_disk(&matrix, &output);
        return;
    }

    let json = match serde_json::to_string_pretty(&matrix) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("Fix: failed to serialize conformance matrix: {error}");
            std::process::exit(1);
        }
    };
    if let Some(parent) = output.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            eprintln!("Fix: failed to create `{}`: {error}", parent.display());
            std::process::exit(1);
        }
    }
    if let Err(error) = fs::write(&output, format!("{json}\n")) {
        eprintln!("Fix: failed to write `{}`: {error}", output.display());
        std::process::exit(1);
    }
    println!("conformance-matrix: wrote {}", output.display());
    if !matrix.blockers.is_empty() {
        std::process::exit(1);
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
    let value = match strip_toml_comment_lines(&text).parse::<toml::Value>() {
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

fn inspect_ci_conformance_gates(vyre_root: &Path) -> Vec<CiConformanceGate> {
    let santh_root = vyre_root
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .and_then(Path::parent)
        .unwrap_or(vyre_root);
    vec![
        inspect_ci_gate(
            santh_root,
            ".github/workflows/conform.yml",
            "conformance matrix release blocker",
            "cargo_full run --bin xtask -- conformance-matrix",
            "release/evidence/conformance/conformance-matrix.json",
        ),
        inspect_ci_gate(
            santh_root,
            ".github/workflows/gpu-parity.yml",
            "gpu-release-gate",
            "cargo_full run --release --bin xtask -- release-conformance --backend all",
            "release/evidence/conformance",
        ),
        inspect_ci_gate(
            santh_root,
            ".github/workflows/conform.yml",
            "conform-release-gate",
            "cargo_full run --bin xtask -- conformance-matrix",
            "vyre-conformance-release-gate",
        ),
        inspect_ci_gate(
            santh_root,
            ".github/workflows/santh-ci.yml",
            "Vyre structural release evidence",
            "cargo_full run --bin xtask -- release-evidence",
            "release/evidence/**/*.json",
        ),
        inspect_ci_gate(
            santh_root,
            ".github/workflows/architectural-invariants.yml",
            "architectural-invariants",
            "scripts/architectural_invariants.sh",
            "lego-audit",
        ),
        inspect_ci_gate(
            santh_root,
            ".github/CI_REQUIRED.md",
            "Vyre/Weir final release gate",
            "GPU release gate",
            "scripts/apply-branch-protection.sh",
        ),
        inspect_ci_gate(
            santh_root,
            "scripts/apply-branch-protection.sh",
            "required_status_checks",
            ".github/CI_REQUIRED.md",
            "gh api",
        ),
        inspect_ci_gate(
            santh_root,
            ".github/workflows/gpu-parity.yml",
            "Vyre/Weir final release gate",
            "cargo_full run --bin xtask -- vyre-release-gate",
            "release/evidence/optimization",
        ),
        inspect_ci_gate(
            santh_root,
            ".github/workflows/gpu-parity.yml",
            "Vyre/Weir final conformance artifact download",
            "vyre-release-conformance-evidence",
            "actions/download-artifact@v4",
        ),
        inspect_ci_gate(
            santh_root,
            ".github/workflows/gpu-parity.yml",
            "Vyre/Weir final benchmark artifact download",
            "vyre-release-benchmark-evidence",
            "actions/download-artifact@v4",
        ),
        inspect_ci_gate(
            santh_root,
            ".github/workflows/gpu-parity.yml",
            "Vyre/Weir final conformance staging",
            "Stage GPU release evidence into release tree",
            "release/evidence/conformance",
        ),
        inspect_ci_gate(
            santh_root,
            ".github/workflows/gpu-parity.yml",
            "Vyre/Weir final benchmark staging",
            "Stage GPU release evidence into release tree",
            "release/evidence/benchmarks",
        ),
        inspect_ci_gate(
            santh_root,
            ".github/workflows/gpu-parity.yml",
            "Vyre/Weir final optimization staging",
            "Stage GPU release evidence into release tree",
            "release/evidence/optimization",
        ),
        inspect_ci_gate(
            santh_root,
            ".github/workflows/gpu-parity.yml",
            "Vyre/Weir final structural evidence",
            "cargo_full run --bin xtask -- release-evidence",
            "release/evidence/**/*.json",
        ),
        inspect_ci_gate(
            santh_root,
            ".github/workflows/gpu-parity.yml",
            "Vyre/Weir final completion audit",
            "cargo_full run --bin xtask -- release-completion-audit --output release/evidence/final/completion-audit.json",
            "release/evidence/final/completion-audit.json",
        ),
        inspect_ci_gate(
            santh_root,
            ".github/workflows/gpu-parity.yml",
            "vyre-weir-final-release-evidence",
            "cargo_full run --bin xtask -- release-completion-audit",
            "vyre-weir-final-release-evidence",
        ),
        inspect_ci_gate(
            vyre_root,
            ".github/workflows/conform.yml",
            "conformance matrix release blocker",
            "cargo_full run --bin xtask -- conformance-matrix",
            "release/evidence/conformance/conformance-matrix.json",
        ),
        inspect_ci_gate(
            vyre_root,
            ".github/workflows/gpu-parity.yml",
            "GPU release gate",
            "cargo_full run --release --bin xtask -- release-conformance --backend all",
            "vyre-release-benchmark-evidence",
        ),
    ]
}

fn inspect_ci_gate(
    vyre_root: &Path,
    workflow: &str,
    gate: &str,
    command: &str,
    artifact_marker: &str,
) -> CiConformanceGate {
    let workflow_path = vyre_root.join(workflow);
    let (text, read_error) = match read_text_bounded(&workflow_path) {
        Ok(text) => (text, None),
        Err(error) => (String::new(), Some(error.to_string())),
    };
    CiConformanceGate {
        workflow: workflow_path.display().to_string(),
        read_error,
        gate: gate.to_string(),
        present: text.contains(gate),
        command_present: text.contains(command),
        artifact_check_present: text.contains(artifact_marker),
    }
}

fn parse_required_ci_statuses(santh_root: &Path) -> (Vec<String>, Vec<String>) {
    let path = santh_root.join(".github/CI_REQUIRED.md");
    let text = match read_text_bounded(&path) {
        Ok(text) => text,
        Err(error) => {
            return (
                Vec::new(),
                vec![format!(
                    "could not read required CI status manifest `{}`: {error}",
                    path.display()
                )],
            );
        }
    };
    let mut statuses = BTreeSet::new();
    let mut skip_rest = false;
    for line in text.lines() {
        if line.starts_with("## Scheduled or Manual Deep Gates") {
            skip_rest = true;
        }
        if skip_rest {
            continue;
        }
        let Some(stripped) = line.strip_prefix("- `") else {
            continue;
        };
        let Some((status, _)) = stripped.split_once('`') else {
            continue;
        };
        if status == "reproducible" {
            continue;
        }
        statuses.insert(status.to_string());
    }
    (statuses.into_iter().collect(), Vec::new())
}

fn ci_status_defined(santh_root: &Path, status: &str, scan_errors: &mut Vec<String>) -> bool {
    let workflow_root = santh_root.join(".github/workflows");
    if !workflow_root.is_dir() {
        scan_errors.push(format!(
            "workflow root `{}` is not a directory while searching status `{status}`",
            workflow_root.display()
        ));
        return false;
    }
    for entry in WalkDir::new(&workflow_root)
        .into_iter()
        .filter_entry(|entry| {
            let name = entry.file_name().to_string_lossy();
            !matches!(name.as_ref(), "target" | ".git")
        })
    {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                scan_errors.push(format!(
                    "could not walk workflow tree `{}` while searching status `{status}`: {error}",
                    workflow_root.display()
                ));
                continue;
            }
        };
        let path = entry.path();
        let Some(extension) = path.extension().and_then(|ext| ext.to_str()) else {
            continue;
        };
        if !matches!(extension, "yml" | "yaml") {
            continue;
        }
        let text = match read_text_bounded(path) {
            Ok(text) => text,
            Err(error) => {
                scan_errors.push(format!(
                    "could not read workflow `{}` while searching status `{status}`: {error}",
                    path.display()
                ));
                continue;
            }
        };
        if text.contains(&format!("name: {status}"))
            || text.contains(&format!("  {status}:"))
            || text.contains(&format!("    name: {status}"))
        {
            return true;
        }
    }
    false
}

fn inspect_path_filtered_required_workflows(santh_root: &Path) -> Vec<String> {
    let mut findings = Vec::new();
    for workflow in REQUIRED_WORKFLOWS {
        let path = santh_root.join(workflow);
        let Ok(text) = read_text_bounded(&path) else {
            continue;
        };
        let trigger_prefix = text
            .split_once("\njobs:")
            .map_or(text.as_str(), |(prefix, _)| prefix);
        if trigger_prefix.lines().any(|line| {
            let trimmed = line.trim_start();
            trimmed.starts_with("paths:") || trimmed.starts_with("paths-ignore:")
        }) {
            findings.push(path.display().to_string());
        }
    }
    findings
}

fn inspect_required_workflow_triggers(santh_root: &Path) -> Vec<String> {
    let mut missing = Vec::new();
    for workflow in REQUIRED_WORKFLOWS {
        let path = santh_root.join(workflow);
        let Ok(text) = read_text_bounded(&path) else {
            missing.push(format!("{}:unreadable", path.display()));
            continue;
        };
        let trigger_prefix = text
            .split_once("\njobs:")
            .map_or(text.as_str(), |(prefix, _)| prefix);
        let has_pull_request = trigger_prefix.lines().any(|line| {
            let trimmed = line.trim();
            trimmed == "pull_request:" || trimmed.starts_with("pull_request:")
        });
        let has_push = trigger_prefix.lines().any(|line| {
            let trimmed = line.trim();
            trimmed == "push:" || trimmed.starts_with("push:")
        });
        let has_main_branch = trigger_prefix.lines().any(|line| {
            let trimmed = line.trim();
            trimmed.starts_with("branches:")
                && (trimmed.contains("[main]")
                    || trimmed.contains("[\"main\"]")
                    || trimmed.contains("[ 'main' ]")
                    || trimmed.contains("[ \"main\" ]")
                    || trimmed == "branches: main"
                    || trimmed == "branches: [ main ]")
        });
        if !(has_pull_request && has_push && has_main_branch) {
            missing.push(format!(
                "{}:pull_request={has_pull_request},push={has_push},main_branch={has_main_branch}",
                path.display()
            ));
        }
    }
    missing
}

fn inspect_fail_closed_fanins(santh_root: &Path) -> Vec<String> {
    let mut missing = Vec::new();
    for (workflow, job_name) in [
        (".github/workflows/santh-ci.yml", "crate-checks"),
        (".github/workflows/conform.yml", "Conform release gate"),
        (".github/workflows/gpu-parity.yml", "GPU release gate"),
        (
            ".github/workflows/gpu-parity.yml",
            "Vyre/Weir final release gate",
        ),
    ] {
        let path = santh_root.join(workflow);
        let Ok(text) = read_text_bounded(&path) else {
            missing.push(format!("{}:{job_name}", path.display()));
            continue;
        };
        let Some(section) = workflow_job_section(&text, job_name) else {
            missing.push(format!("{}:{job_name}", path.display()));
            continue;
        };
        if !(section.contains("if: ${{ always() }}")
            && section.contains(".result")
            && section.contains("exit 1"))
        {
            missing.push(format!("{}:{job_name}", path.display()));
        }
    }
    missing
}

const REQUIRED_WORKFLOWS: &[&str] = &[
    ".github/workflows/santh-ci.yml",
    ".github/workflows/conform.yml",
    ".github/workflows/gpu-parity.yml",
    ".github/workflows/bench-regression.yml",
    ".github/workflows/architectural-invariants.yml",
    ".github/workflows/vyre-matrix.yml",
    ".github/workflows/vyre-core.yml",
    ".github/workflows/vyre-rewrite-proofs.yml",
    ".github/workflows/vyre-lego-audit.yml",
    "libs/performance/matching/vyre/.github/workflows/conform.yml",
    "libs/performance/matching/vyre/.github/workflows/gpu-parity.yml",
    "libs/performance/matching/vyre/.github/workflows/ci.yml",
    "libs/performance/matching/vyre/.github/workflows/bench.yml",
    "libs/performance/matching/vyre/.github/workflows/fuzz.yml",
    "libs/performance/matching/vyre/.github/workflows/architectural-invariants.yml",
];

fn workflow_job_section<'a>(workflow: &'a str, job_name: &str) -> Option<&'a str> {
    let marker = format!("name: {job_name}");
    let name_index = workflow.find(&marker)?;
    let job_start = workflow[..name_index]
        .rfind("\n  ")
        .map_or(0, |index| index + 1);
    let rest = &workflow[job_start..];
    let mut section_end = rest.len();
    for (offset, _) in rest.match_indices("\n  ") {
        if offset == 0 {
            continue;
        }
        let candidate = &rest[offset + 3..];
        let Some(first) = candidate.chars().next() else {
            continue;
        };
        if first.is_whitespace() {
            continue;
        }
        let first_line = candidate.lines().next().unwrap_or_default();
        if first_line.contains(':') {
            section_end = offset;
            break;
        }
    }
    Some(&rest[..section_end])
}

fn strip_toml_comment_lines(text: &str) -> String {
    text.lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
}

fn check_against_disk(matrix: &ConformanceMatrix, output: &Path) {
    for op in INT4_CONFORMANCE_OPS {
        if !matrix
            .entries
            .iter()
            .any(|entry| entry.id == *op && entry.has_test_inputs && entry.has_expected_output)
        {
            eprintln!(
                "Fix: INT4 conformance op `{op}` must be registered with fixture inputs and expected outputs."
            );
            std::process::exit(1);
        }
    }
    if !matrix.missing_catalog_ops.is_empty() {
        for op in INT4_CONFORMANCE_OPS {
            if matrix.missing_catalog_ops.iter().any(|missing| missing == *op) {
                eprintln!(
                    "Fix: INT4 conformance op `{op}` is listed in missing_catalog_ops."
                );
                std::process::exit(1);
            }
        }
    }

    let disk_text = match read_text_bounded(output) {
        Ok(text) => text,
        Err(error) => {
            eprintln!(
                "Fix: conformance-matrix --check requires `{}`: {error}",
                output.display()
            );
            std::process::exit(1);
        }
    };
    let disk: Value = match serde_json::from_str(&disk_text) {
        Ok(value) => value,
        Err(error) => {
            eprintln!(
                "Fix: `{}` is not valid conformance matrix JSON: {error}",
                output.display()
            );
            std::process::exit(1);
        }
    };
    let live = match serde_json::to_value(matrix) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("Fix: failed to serialize live conformance matrix: {error}");
            std::process::exit(1);
        }
    };

    let mut drift = Vec::new();
    for field in [
        "schema_version",
        "op_count",
        "distinct_op_count",
        "catalog_required_op_count",
        "catalog_covered_op_count",
        "missing_catalog_ops",
        "release_backend_row_count",
        "release_backend_rows",
        "entries",
    ] {
        if live.get(field) != disk.get(field) {
            drift.push(format!("`{field}` diverges from committed evidence"));
        }
    }

    if drift.is_empty() {
        println!(
            "conformance-matrix: live inventory matches {} ({} ops, {} INT4 rows)",
            output.display(),
            matrix.op_count,
            INT4_CONFORMANCE_OPS.len()
        );
        return;
    }

    eprintln!("conformance-matrix drift detected against `{}`:", output.display());
    for line in &drift {
        eprintln!("  - {line}");
    }
    eprintln!(
        "Fix: run `cargo_full run --bin xtask -- conformance-matrix --output {}`, commit, then re-run --check.",
        output.display()
    );
    std::process::exit(1);
}

fn parse_args(args: &[String]) -> Result<(PathBuf, bool), String> {
    let mut output = None;
    let mut check = false;
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--output" => {
                let Some(path) = args.get(index + 1) else {
                    return Err("Fix: --output requires a path.".to_string());
                };
                output = Some(PathBuf::from(path));
                index += 2;
            }
            "--check" => {
                check = true;
                index += 1;
            }
            "--help" | "-h" => {
                println!(
                    "USAGE:\n  cargo_full run --bin xtask -- conformance-matrix [--check] [--output PATH]\n\n\
                     Writes or checks registered-op and release-backend conformance coverage evidence."
                );
                std::process::exit(0);
            }
            other => return Err(format!("Fix: unknown conformance-matrix option `{other}`.")),
        }
    }
    Ok((output.unwrap_or_else(default_output), check))
}

fn default_output() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|path| path.join("release/evidence/conformance/conformance-matrix.json"))
        .unwrap_or_else(|| PathBuf::from("release/evidence/conformance/conformance-matrix.json"))
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    let mut reader =
        fs::File::open(path)?.take(MAX_CONFORMANCE_EVIDENCE_TEXT_BYTES.saturating_add(1));
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    if text.len() as u64 > MAX_CONFORMANCE_EVIDENCE_TEXT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} exceeds {MAX_CONFORMANCE_EVIDENCE_TEXT_BYTES} byte conformance evidence read cap",
                path.display()
            ),
        ));
    }
    Ok(text)
}
