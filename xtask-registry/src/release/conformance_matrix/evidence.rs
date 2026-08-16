//! The shape of `release/evidence/conformance/conformance-matrix.json`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use xtask::release::conformance_workflows::CiConformanceGate;

/// Every case class a row reports on, covered or not. A row that reports fewer
/// is a row that hides a gap.
pub(super) const REPORTED_CONFORMANCE_CASE_CLASSES: &[&str] = &[
    "positive",
    "negative",
    "boundary",
    "adversarial",
    "byte_output",
    "unsupported_diagnostic",
];

#[derive(Debug, Serialize)]
pub(super) struct ConformanceMatrix {
    pub(super) schema_version: u32,
    pub(super) op_count: usize,
    pub(super) distinct_op_count: usize,
    pub(super) catalog_required_op_count: usize,
    pub(super) catalog_covered_op_count: usize,
    pub(super) missing_catalog_ops: Vec<String>,
    pub(super) release_backend_row_count: usize,
    pub(super) supported_release_backend_row_count: usize,
    pub(super) release_backend_rows: Vec<String>,
    pub(super) release_backend_case_rows: Vec<ReleaseBackendCaseRow>,
    pub(super) required_case_classes: Vec<&'static str>,
    pub(super) case_class_blocker_count: usize,
    pub(super) missing_release_backend_rows: Vec<String>,
    pub(super) op_matrix_blocked_release_count: usize,
    pub(super) op_matrix_blocked_release_rows: Vec<String>,
    pub(super) op_matrix_errors: Vec<String>,
    pub(super) duplicate_op_ids: Vec<String>,
    pub(super) fixture_required_count: usize,
    pub(super) fixture_input_count: usize,
    pub(super) expected_output_count: usize,
    pub(super) dispatch_backends: Vec<String>,
    pub(super) ci_blocking_gate_count: usize,
    pub(super) ci_gates: Vec<CiConformanceGate>,
    pub(super) required_ci_statuses: Vec<String>,
    pub(super) missing_required_ci_statuses: Vec<String>,
    pub(super) ci_status_scan_errors: Vec<String>,
    pub(super) path_filtered_required_workflows: Vec<String>,
    pub(super) missing_required_workflow_triggers: Vec<String>,
    pub(super) missing_fail_closed_fanins: Vec<String>,
    pub(super) scan_conformance_rows: Vec<ScanConformanceRowEvidence>,
    pub(super) scan_conformance_findings: Vec<ScanConformanceFinding>,
    pub(super) entries: Vec<ConformanceEntry>,
    pub(super) blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(super) struct ConformanceEntry {
    pub(super) id: String,
    pub(super) requires_fixture: bool,
    pub(super) has_test_inputs: bool,
    pub(super) has_expected_output: bool,
    pub(super) tolerance_ulp: u32,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(super) struct ReleaseBackendCaseRow {
    pub(super) op_id: String,
    pub(super) backend: String,
    pub(super) status: String,
    pub(super) test_paths: Vec<String>,
    pub(super) case_classes: Vec<ConformanceCaseClassEvidence>,
    pub(super) required_case_classes: Vec<&'static str>,
    pub(super) missing_required_case_classes: Vec<&'static str>,
    pub(super) blockers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(super) struct ConformanceCaseClassEvidence {
    pub(super) class: &'static str,
    pub(super) covered: bool,
    pub(super) source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct ScanConformanceRowEvidence {
    pub(super) semantics: String,
    pub(super) engine_support: BTreeMap<String, String>,
    pub(super) evidence_path: String,
    pub(super) unsupported_diagnostic_code: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(super) struct ScanConformanceFinding {
    pub(super) semantics: String,
    pub(super) engine: Option<String>,
    pub(super) issue: String,
}

#[derive(Debug, Deserialize)]
pub(super) struct ScanConformanceMatrixToml {
    pub(super) schema_version: u32,
    /// The code a row carries when the semantics has no refusal path at all.
    /// Declared once here so no row invents a spelling for "not applicable".
    pub(super) no_refusal_code: String,
    pub(super) row: Vec<ScanConformanceRowEvidence>,
}
