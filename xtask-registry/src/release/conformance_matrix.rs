//! Release conformance matrix evidence.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use vyre_driver::backend::{backend_dispatches, registered_backends_by_precedence_slice};
use xtask::release::conformance_op_matrix::{
    read_conformance_required_op_matrix, OpMatrixReleaseBackendSpec,
};
use xtask::release::conformance_workflows::{
    ci_status_defined, inspect_ci_conformance_gates, inspect_fail_closed_fanins,
    inspect_path_filtered_required_workflows, inspect_required_workflow_triggers,
    parse_required_ci_statuses, CiConformanceGate,
};
use xtask::release::release_backend_rows::count_supported_release_backend_rows;

use vyre_driver_cuda as _;
use vyre_driver_reference as _;
use vyre_driver_spirv as _;
use vyre_driver_wgpu as _;
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
const REPORTED_CONFORMANCE_CASE_CLASSES: &[&str] = &[
    "positive",
    "negative",
    "boundary",
    "adversarial",
    "byte_output",
    "unsupported_diagnostic",
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
    supported_release_backend_row_count: usize,
    pub(crate) release_backend_rows: Vec<String>,
    release_backend_case_rows: Vec<ReleaseBackendCaseRow>,
    required_case_classes: Vec<&'static str>,
    case_class_blocker_count: usize,
    pub(crate) missing_release_backend_rows: Vec<String>,
    op_matrix_blocked_release_count: usize,
    op_matrix_blocked_release_rows: Vec<String>,
    op_matrix_errors: Vec<String>,
    duplicate_op_ids: Vec<String>,
    fixture_required_count: usize,
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
    scan_conformance_rows: Vec<ScanConformanceRowEvidence>,
    scan_conformance_findings: Vec<ScanConformanceFinding>,
    entries: Vec<ConformanceEntry>,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct ConformanceEntry {
    id: String,
    requires_fixture: bool,
    has_test_inputs: bool,
    has_expected_output: bool,
    tolerance_ulp: u32,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct ReleaseBackendCaseRow {
    op_id: String,
    backend: String,
    status: String,
    test_paths: Vec<String>,
    case_classes: Vec<ConformanceCaseClassEvidence>,
    required_case_classes: Vec<&'static str>,
    missing_required_case_classes: Vec<&'static str>,
    blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct ConformanceCaseClassEvidence {
    class: &'static str,
    covered: bool,
    source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct ScanConformanceRowEvidence {
    semantics: String,
    engine_support: BTreeMap<String, String>,
    evidence_path: String,
    expected_output_hex: String,
    unsupported_diagnostic_code: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct ScanConformanceFinding {
    semantics: String,
    engine: Option<String>,
    issue: String,
}

#[derive(Debug, Deserialize)]
struct ScanConformanceMatrixToml {
    schema_version: u32,
    row: Vec<ScanConformanceRowEvidence>,
}

const SCAN_CONFORMANCE_MATRIX: &str = "docs/optimization/SCAN_CONFORMANCE_MATRIX.toml";
const REQUIRED_SCAN_CONFORMANCE_SEMANTICS: &[&str] = &[
    "leftmost_semantics",
    "overlapping_matches",
    "capture_groups",
    "byte_mode",
    "unicode_mode",
    "streaming_chunks",
    "unsupported_constructs",
];
const REQUIRED_SCAN_CONFORMANCE_ENGINES: &[&str] = &[
    "cpu_ref",
    "cuda",
    "wgpu",
    "metal",
    "rust_regex",
    "hyperscan",
    "vectorscan",
];
const ALLOWED_SCAN_ENGINE_SUPPORT: &[&str] =
    &["supported", "unsupported", "not_applicable", "experimental"];

pub(crate) fn run(args: &[String]) {
    let (output, check) = match parse_args(args) {
        Ok(parsed) => parsed,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let operations = crate::live_registry::live_operation_registry()
        .iter()
        .collect::<Vec<_>>();
    let mut entries = Vec::with_capacity(operations.len());
    let mut ids = BTreeSet::new();
    let mut duplicate_op_ids = BTreeSet::new();
    for entry in operations {
        if !ids.insert(entry.id) {
            duplicate_op_ids.insert(entry.id.to_string());
        }
        entries.push(ConformanceEntry {
            id: entry.id.to_string(),
            requires_fixture: entry.program().is_some(),
            has_test_inputs: entry.test_inputs.is_some(),
            has_expected_output: entry.expected_output.is_some(),
            tolerance_ulp: entry.tolerance(),
        });
    }
    entries.sort_by(|left, right| left.id.cmp(&right.id));
    let registered_backends = registered_backends_by_precedence_slice()
        .unwrap_or_else(|error| panic!("backend registry startup failed: {error}"));
    let mut dispatch_backends = Vec::new();
    for backend in registered_backends {
        if backend_dispatches(backend.id)
            .unwrap_or_else(|error| panic!("backend registry startup failed: {error}"))
        {
            dispatch_backends.push(backend.id.to_string());
        }
    }
    let fixture_required_count = entries
        .iter()
        .filter(|entry| entry.requires_fixture)
        .count();
    let fixture_input_count = entries
        .iter()
        .filter(|entry| entry.requires_fixture && entry.has_test_inputs)
        .count();
    let expected_output_count = entries
        .iter()
        .filter(|entry| entry.requires_fixture && entry.has_expected_output)
        .count();
    let vyre_root = xtask::checkout::checkout_root();
    let ci_gates = inspect_ci_conformance_gates(&vyre_root);
    let (required_ci_statuses, mut ci_status_scan_errors) = parse_required_ci_statuses(&vyre_root);
    let mut missing_required_ci_statuses = Vec::new();
    for status in &required_ci_statuses {
        if !ci_status_defined(&vyre_root, status, &mut ci_status_scan_errors) {
            missing_required_ci_statuses.push(status.clone());
        }
    }
    let path_filtered_required_workflows = inspect_path_filtered_required_workflows(&vyre_root);
    let missing_required_workflow_triggers = inspect_required_workflow_triggers(&vyre_root);
    let missing_fail_closed_fanins = inspect_fail_closed_fanins(&vyre_root);
    let mut blockers = Vec::new();
    let catalog = read_conformance_required_op_matrix(&vyre_root);
    let (scan_conformance_rows, scan_conformance_findings) =
        read_scan_conformance_matrix(&vyre_root);
    let entry_by_id = entries
        .iter()
        .map(|entry| (entry.id.as_str(), entry))
        .collect::<BTreeMap<_, _>>();
    let release_backend_case_rows =
        release_backend_case_rows(&catalog.release_backend_specs, &entry_by_id);
    for error in &catalog.errors {
        blockers.push(error.clone());
    }
    for row in &release_backend_case_rows {
        for blocker in &row.blockers {
            blockers.push(blocker.clone());
        }
    }
    for finding in &scan_conformance_findings {
        blockers.push(format!(
            "scan conformance row `{}` engine {:?} is invalid: {}",
            finding.semantics, finding.engine, finding.issue
        ));
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
    let supported_release_backend_row_count =
        count_supported_release_backend_rows(&catalog.release_backend_rows);
    let expected_supported_rows = catalog.required_ops.len().saturating_mul(3);
    if supported_release_backend_row_count != expected_supported_rows {
        blockers.push(format!(
            "OP_MATRIX declares {supported_release_backend_row_count} supported release backend row(s), expected {expected_supported_rows}"
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
    if fixture_input_count != fixture_required_count {
        blockers.push(format!(
            "only {fixture_input_count}/{fixture_required_count} executable op entries have fixture inputs"
        ));
    }
    if expected_output_count != fixture_required_count {
        blockers.push(format!(
            "only {expected_output_count}/{fixture_required_count} executable op entries have expected outputs"
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
        schema_version: 5,
        op_count: entries.len(),
        distinct_op_count: ids.len(),
        catalog_required_op_count: catalog.required_ops.len(),
        catalog_covered_op_count,
        missing_catalog_ops,
        release_backend_row_count: catalog.release_backend_rows.len(),
        supported_release_backend_row_count,
        release_backend_rows: catalog.release_backend_rows,
        case_class_blocker_count: release_backend_case_rows
            .iter()
            .map(|row| row.blockers.len())
            .sum(),
        release_backend_case_rows,
        required_case_classes: REPORTED_CONFORMANCE_CASE_CLASSES.to_vec(),
        missing_release_backend_rows: catalog.missing_release_backend_rows,
        op_matrix_blocked_release_count: catalog.blocked_release_rows.len(),
        op_matrix_blocked_release_rows: catalog.blocked_release_rows,
        op_matrix_errors: catalog.errors,
        duplicate_op_ids: duplicate_op_ids.into_iter().collect(),
        fixture_required_count,
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
        scan_conformance_rows,
        scan_conformance_findings,
        entries,
        blockers,
    };
    if check {
        check_against_disk(&matrix, &output);
        return;
    }

    if let Some(parent) = output.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            eprintln!("Fix: failed to create `{}`: {error}", parent.display());
            std::process::exit(1);
        }
    }
    xtask::output_arg::write_json(&output, &matrix);
    println!("conformance-matrix: wrote {}", output.display());
    if !matrix.blockers.is_empty() {
        std::process::exit(1);
    }
}

fn release_backend_case_rows(
    specs: &[OpMatrixReleaseBackendSpec],
    entries: &BTreeMap<&str, &ConformanceEntry>,
) -> Vec<ReleaseBackendCaseRow> {
    specs
        .iter()
        .map(|spec| release_backend_case_row(spec, entries.get(spec.op_id.as_str()).copied()))
        .collect()
}

fn release_backend_case_row(
    spec: &OpMatrixReleaseBackendSpec,
    entry: Option<&ConformanceEntry>,
) -> ReleaseBackendCaseRow {
    let positive = spec.status == "supported"
        && entry.is_some_and(|entry| entry.has_test_inputs && entry.has_expected_output);
    let byte_output = entry.is_some_and(|entry| entry.has_expected_output);
    let unsupported_diagnostic =
        spec.status != "supported" || spec.test_case_classes.contains("unsupported_diagnostic");
    let class_sources = [
        (
            "positive",
            positive,
            if positive {
                "registered fixture inputs and expected output"
            } else {
                "missing supported-row fixture input/output pair"
            },
        ),
        (
            "negative",
            spec.test_case_classes.contains("negative"),
            "OP_MATRIX referenced tests contain reject/error/invalid/unsupported evidence",
        ),
        (
            "boundary",
            spec.test_case_classes.contains("boundary"),
            "OP_MATRIX referenced tests contain boundary/zero/limit/overflow evidence",
        ),
        (
            "adversarial",
            spec.test_case_classes.contains("adversarial"),
            "OP_MATRIX referenced tests contain adversarial/hostile/malformed evidence",
        ),
        (
            "byte_output",
            byte_output,
            if byte_output {
                "registered expected byte output"
            } else {
                "missing registered expected byte output"
            },
        ),
        (
            "unsupported_diagnostic",
            unsupported_diagnostic,
            if spec.status == "supported" {
                "supported row; unsupported diagnostics are only covered when referenced tests exercise them"
            } else {
                "OP_MATRIX backend status is non-supported and therefore carries explicit unsupported/not-applicable diagnostics"
            },
        ),
    ];
    let case_classes = class_sources
        .into_iter()
        .map(|(class, covered, source)| ConformanceCaseClassEvidence {
            class,
            covered,
            source: source.to_string(),
        })
        .collect::<Vec<_>>();
    let required_case_classes = required_case_classes_for_status(&spec.status);
    let missing_required_case_classes = required_case_classes
        .iter()
        .copied()
        .filter(|class| {
            !case_classes
                .iter()
                .any(|evidence| evidence.class == *class && evidence.covered)
        })
        .collect::<Vec<_>>();
    let blockers = missing_required_case_classes
        .iter()
        .map(|class| {
            format!(
                "conformance op/backend row `{}:{}` status `{}` is missing required `{class}` case-class evidence",
                spec.op_id, spec.backend, spec.status
            )
        })
        .collect::<Vec<_>>();
    ReleaseBackendCaseRow {
        op_id: spec.op_id.clone(),
        backend: spec.backend.clone(),
        status: spec.status.clone(),
        test_paths: spec.test_paths.clone(),
        case_classes,
        required_case_classes,
        missing_required_case_classes,
        blockers,
    }
}

fn required_case_classes_for_status(status: &str) -> Vec<&'static str> {
    if status == "supported" {
        vec!["positive", "byte_output"]
    } else {
        vec!["unsupported_diagnostic"]
    }
}

fn read_scan_conformance_matrix(
    vyre_root: &Path,
) -> (Vec<ScanConformanceRowEvidence>, Vec<ScanConformanceFinding>) {
    let path = vyre_root.join(SCAN_CONFORMANCE_MATRIX);
    let text = match read_text_bounded(&path) {
        Ok(text) => text,
        Err(error) => {
            return (
                Vec::new(),
                vec![ScanConformanceFinding {
                    semantics: "<matrix>".to_string(),
                    engine: None,
                    issue: format!(
                        "could not read `{SCAN_CONFORMANCE_MATRIX}`: {error}. Fix: keep scan compatibility rows in the canonical conformance matrix."
                    ),
                }],
            );
        }
    };
    let matrix = match toml::from_str::<ScanConformanceMatrixToml>(&text) {
        Ok(matrix) => matrix,
        Err(error) => {
            return (
                Vec::new(),
                vec![ScanConformanceFinding {
                    semantics: "<matrix>".to_string(),
                    engine: None,
                    issue: format!(
                        "could not parse `{SCAN_CONFORMANCE_MATRIX}`: {error}. Fix: use [[row]] entries with semantics, engine_support, evidence_path, expected_output_hex, and unsupported_diagnostic_code."
                    ),
                }],
            );
        }
    };

    let mut findings = Vec::new();
    if matrix.schema_version != 1 {
        findings.push(ScanConformanceFinding {
            semantics: "<matrix>".to_string(),
            engine: None,
            issue: format!(
                "schema_version {} is unsupported; expected 1",
                matrix.schema_version
            ),
        });
    }

    let mut seen_semantics = BTreeSet::new();
    let mut rows = Vec::new();
    for row in matrix.row {
        let semantics = row.semantics.trim().to_string();
        if !REQUIRED_SCAN_CONFORMANCE_SEMANTICS.contains(&semantics.as_str()) {
            findings.push(ScanConformanceFinding {
                semantics: row.semantics.clone(),
                engine: None,
                issue: "unknown scan conformance semantics. Fix: use a required scan semantics id."
                    .to_string(),
            });
        } else if !seen_semantics.insert(semantics.clone()) {
            findings.push(ScanConformanceFinding {
                semantics: semantics.clone(),
                engine: None,
                issue: "duplicate scan conformance semantics row. Fix: keep one row per semantics."
                    .to_string(),
            });
        }

        let evidence_path = row.evidence_path.trim();
        if evidence_path.is_empty() {
            findings.push(ScanConformanceFinding {
                semantics: semantics.clone(),
                engine: None,
                issue:
                    "missing evidence_path. Fix: point at the conformance test source for this row."
                        .to_string(),
            });
        } else if !vyre_root.join(evidence_path).is_file() {
            findings.push(ScanConformanceFinding {
                semantics: semantics.clone(),
                engine: None,
                issue: format!(
                    "evidence_path `{evidence_path}` does not exist. Fix: point at a committed conformance test."
                ),
            });
        }

        let output_hex = row.expected_output_hex.trim();
        if output_hex.is_empty() || !is_even_hex(output_hex) {
            findings.push(ScanConformanceFinding {
                semantics: semantics.clone(),
                engine: None,
                issue: "expected_output_hex must be non-empty even-length hex bytes. Fix: record exact expected output bytes."
                    .to_string(),
            });
        }
        if row.unsupported_diagnostic_code.trim().is_empty() {
            findings.push(ScanConformanceFinding {
                semantics: semantics.clone(),
                engine: None,
                issue: "unsupported_diagnostic_code is missing. Fix: record the exact diagnostic code or explicit not-applicable code."
                    .to_string(),
            });
        }

        for engine in REQUIRED_SCAN_CONFORMANCE_ENGINES {
            match row.engine_support.get(*engine).map(String::as_str) {
                Some(status) if ALLOWED_SCAN_ENGINE_SUPPORT.contains(&status) => {}
                Some(status) => findings.push(ScanConformanceFinding {
                    semantics: semantics.clone(),
                    engine: Some((*engine).to_string()),
                    issue: format!(
                        "engine support status `{status}` is invalid. Fix: use supported, unsupported, not_applicable, or experimental."
                    ),
                }),
                None => findings.push(ScanConformanceFinding {
                    semantics: semantics.clone(),
                    engine: Some((*engine).to_string()),
                    issue: "missing engine support status. Fix: every scan row must report every required engine."
                        .to_string(),
                }),
            }
        }
        for engine in row.engine_support.keys() {
            if !REQUIRED_SCAN_CONFORMANCE_ENGINES.contains(&engine.as_str()) {
                findings.push(ScanConformanceFinding {
                    semantics: semantics.clone(),
                    engine: Some(engine.clone()),
                    issue: "unknown engine in scan conformance matrix. Fix: dedup through the required engine set."
                        .to_string(),
                });
            }
        }
        rows.push(row);
    }

    for required in REQUIRED_SCAN_CONFORMANCE_SEMANTICS {
        if !seen_semantics.contains(*required) {
            findings.push(ScanConformanceFinding {
                semantics: (*required).to_string(),
                engine: None,
                issue: "missing required scan conformance semantics row".to_string(),
            });
        }
    }

    (rows, findings)
}

fn is_even_hex(value: &str) -> bool {
    !value.is_empty() && value.len() % 2 == 0 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn strip_toml_comment_lines(text: &str) -> String {
    text.lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
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
            if matrix
                .missing_catalog_ops
                .iter()
                .any(|missing| missing == *op)
            {
                eprintln!("Fix: INT4 conformance op `{op}` is listed in missing_catalog_ops.");
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

    eprintln!(
        "conformance-matrix drift detected against `{}`:",
        output.display()
    );
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
                    "USAGE:\n  cargo xtask conformance-matrix [--check] [--output PATH]\n\n\
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
    xtask::checkout::checkout_root().join("release/evidence/conformance/conformance-matrix.json")
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    xtask::output_arg::read_text_bounded(
        path,
        MAX_CONFORMANCE_EVIDENCE_TEXT_BYTES,
        "conformance evidence",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence<'a>(
        row: &'a ReleaseBackendCaseRow,
        class: &str,
    ) -> &'a ConformanceCaseClassEvidence {
        row.case_classes
            .iter()
            .find(|evidence| evidence.class == class)
            .expect("case class evidence must be reported")
    }

    fn assert_all_case_classes_reported(row: &ReleaseBackendCaseRow) {
        assert_eq!(
            row.case_classes.len(),
            REPORTED_CONFORMANCE_CASE_CLASSES.len()
        );
        for class in REPORTED_CONFORMANCE_CASE_CLASSES {
            assert!(
                row.case_classes
                    .iter()
                    .any(|evidence| evidence.class == *class),
                "missing reported case class {class}"
            );
        }
    }

    #[test]
    fn release_backend_case_rows_report_all_case_classes_for_external_flow_row() {
        let entry = ConformanceEntry {
            id: "external.flow.alias_ifds".to_string(),
            requires_fixture: true,
            has_test_inputs: true,
            has_expected_output: true,
            tolerance_ulp: 0,
        };
        let spec = OpMatrixReleaseBackendSpec {
            op_id: entry.id.clone(),
            backend: "cuda".to_string(),
            status: "supported".to_string(),
            test_paths: vec!["tests/external_flow_boundary_adversarial.rs".to_string()],
            test_case_classes: BTreeSet::from(["negative", "boundary", "adversarial"]),
        };
        let entries = BTreeMap::from([(entry.id.as_str(), &entry)]);

        let rows = release_backend_case_rows(&[spec], &entries);
        let row = &rows[0];

        assert_all_case_classes_reported(row);
        assert_eq!(row.required_case_classes, vec!["positive", "byte_output"]);
        assert!(row.missing_required_case_classes.is_empty());
        assert!(row.blockers.is_empty());
        assert!(evidence(row, "positive").covered);
        assert!(evidence(row, "negative").covered);
        assert!(evidence(row, "boundary").covered);
        assert!(evidence(row, "adversarial").covered);
        assert!(evidence(row, "byte_output").covered);
        assert!(!evidence(row, "unsupported_diagnostic").covered);
    }

    #[test]
    fn release_backend_case_rows_accept_non_supported_rows_with_unsupported_diagnostic() {
        let spec = OpMatrixReleaseBackendSpec {
            op_id: "vyre-libs::scan::prefix_sum_u32".to_string(),
            backend: "cuda".to_string(),
            status: "not_applicable".to_string(),
            test_paths: Vec::new(),
            test_case_classes: BTreeSet::new(),
        };
        let entries = BTreeMap::new();

        let rows = release_backend_case_rows(&[spec], &entries);
        let row = &rows[0];

        assert_all_case_classes_reported(row);
        assert_eq!(row.required_case_classes, vec!["unsupported_diagnostic"]);
        assert!(row.missing_required_case_classes.is_empty());
        assert!(row.blockers.is_empty());
        assert!(!evidence(row, "positive").covered);
        assert!(!evidence(row, "byte_output").covered);
        assert!(evidence(row, "unsupported_diagnostic").covered);
    }

    #[test]
    fn release_backend_case_rows_block_supported_rows_missing_byte_output() {
        let entry = ConformanceEntry {
            id: "vyre-libs::scan::prefix_sum_u32".to_string(),
            requires_fixture: true,
            has_test_inputs: true,
            has_expected_output: false,
            tolerance_ulp: 0,
        };
        let spec = OpMatrixReleaseBackendSpec {
            op_id: entry.id.clone(),
            backend: "wgpu".to_string(),
            status: "supported".to_string(),
            test_paths: Vec::new(),
            test_case_classes: BTreeSet::new(),
        };
        let entries = BTreeMap::from([(entry.id.as_str(), &entry)]);

        let rows = release_backend_case_rows(&[spec], &entries);
        let row = &rows[0];

        assert_all_case_classes_reported(row);
        assert_eq!(row.required_case_classes, vec!["positive", "byte_output"]);
        assert_eq!(
            row.missing_required_case_classes,
            vec!["positive", "byte_output"]
        );
        assert_eq!(row.blockers.len(), 2);
        assert!(!evidence(row, "positive").covered);
        assert!(!evidence(row, "byte_output").covered);
    }
}
