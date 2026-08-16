//! Release conformance matrix evidence.
//!
//! ## Layout
//!
//! - `evidence` the shape of the document this gate writes
//! - `case_classes` per op and backend, which test-case classes are covered
//! - `scan_matrix` the scan compatibility rows and what invalidates one

use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::{Path, PathBuf};

use vyre_driver::backend_dispatches;
use xtask::gate::{Finding, Gate, GateCtx, GateError, Report};
use xtask::release::conformance_op_matrix::{
    evaluate_op_matrix_coverage, read_conformance_required_op_matrix,
};
use xtask::release::conformance_workflows::{
    ci_status_defined, inspect_ci_conformance_gates, inspect_fail_closed_fanins,
    inspect_path_filtered_required_workflows, inspect_required_workflow_triggers,
    parse_required_ci_statuses,
};

use self::case_classes::release_backend_case_rows;
use self::evidence::{ConformanceEntry, ConformanceMatrix, REPORTED_CONFORMANCE_CASE_CLASSES};
use self::scan_matrix::read_scan_conformance_matrix;

mod case_classes;
mod evidence;
mod scan_matrix;

const MIN_RELEASE_OP_COUNT: usize = 49;
const MAX_CONFORMANCE_EVIDENCE_TEXT_BYTES: u64 = 8_388_608;

/// Operations whose INT4 conformance coverage blocks a release outright. They
/// carry the quantized kernels a release is judged on, so a missing fixture or
/// expected output here is a blocker rather than a gap someone files.
const INT4_CONFORMANCE_OPS: &[&str] = &[
    "vyre-libs::quant::int4_dot_i32",
    "vyre-libs::quant::int4_dot_f32_scaled",
    "vyre-libs::quant::int4_matvec_f32_scaled",
    "vyre-libs::quant::int4_batched_matvec_f32_scaled",
    "vyre-libs::quant::int4_batched_matmul_f32_scaled",
    "vyre-libs::quant::int4_batched_matmul_top1_f32_scaled",
];

/// Holds release op and backend conformance coverage to the recorded matrix.
pub struct ConformanceMatrixGate;

impl Gate for ConformanceMatrixGate {
    fn name(&self) -> &'static str {
        "conformance-matrix"
    }

    fn help(&self) -> &'static str {
        "Hold release op and backend conformance coverage to the recorded matrix; --write regenerates it"
    }

    fn generates(&self) -> bool {
        true
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let operations = vyre_registry_link::operation::live_operation_registry()
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
        let registered_backends =
            vyre_registry_link::backend::live_backend_registry_by_precedence().map_err(|error| {
                GateError::new(
                    format!("the backend registry did not start: {error}"),
                    "repair the backend registration this error names; conformance coverage cannot be measured without the live backend list",
                )
            })?;
        let mut dispatch_backends = Vec::new();
        for backend in registered_backends {
            if backend_dispatches(backend.id).map_err(|error| {
                GateError::new(
                    format!(
                        "the backend registry did not start while asking whether `{}` dispatches: {error}",
                        backend.id
                    ),
                    "repair the backend registration this error names",
                )
            })? {
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
        let (required_ci_statuses, mut ci_status_scan_errors) =
            parse_required_ci_statuses(&vyre_root);
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
        let mut catalog_blockers = Vec::new();
        let coverage = evaluate_op_matrix_coverage(
            &catalog,
            |op| ids.contains(op),
            |missing| {
                format!("{missing} OP_MATRIX op id(s) are missing registered conformance entries")
            },
            &mut catalog_blockers,
        );
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
        blockers.append(&mut catalog_blockers);
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
        for op in INT4_CONFORMANCE_OPS {
            if !entries
                .iter()
                .any(|entry| entry.id == *op && entry.has_test_inputs && entry.has_expected_output)
            {
                blockers.push(format!(
                    "INT4 conformance op `{op}` is not registered with fixture inputs and expected outputs"
                ));
            }
            if coverage
                .missing_catalog_ops
                .iter()
                .any(|missing| missing == *op)
            {
                blockers.push(format!(
                    "INT4 conformance op `{op}` is missing from the op matrix catalog"
                ));
            }
        }
        let matrix = ConformanceMatrix {
            schema_version: 5,
            op_count: entries.len(),
            distinct_op_count: ids.len(),
            catalog_required_op_count: coverage.catalog_required_op_count,
            catalog_covered_op_count: coverage.catalog_covered_op_count,
            missing_catalog_ops: coverage.missing_catalog_ops,
            release_backend_row_count: coverage.release_backend_row_count,
            supported_release_backend_row_count: coverage.supported_release_backend_row_count,
            release_backend_rows: catalog.release_backend_rows,
            case_class_blocker_count: release_backend_case_rows
                .iter()
                .map(|row| row.blockers.len())
                .sum(),
            release_backend_case_rows,
            required_case_classes: REPORTED_CONFORMANCE_CASE_CLASSES.to_vec(),
            missing_release_backend_rows: catalog.missing_release_backend_rows,
            op_matrix_blocked_release_count: coverage.op_matrix_blocked_release_count,
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

        let output = default_output();
        let relative = output
            .strip_prefix(&ctx.root)
            .unwrap_or(output.as_path())
            .display()
            .to_string();
        let mut inspection = xtask::artifact_gate::Inspection::new();
        for blocker in &matrix.blockers {
            inspection.find(Finding::new(
                blocker.clone(),
                "close the conformance gap this blocker names, then run the gate again",
            ));
        }
        inspection.generates(&relative, &matrix);
        let mut report = xtask::artifact_gate::settle_inspection(ctx, self.name(), inspection);
        report.note(format!(
            "{} registered conformance op entry(ies)",
            matrix.op_count
        ));
        Ok(report)
    }
}

fn default_output() -> PathBuf {
    xtask::checkout::checkout_root().join("release/evidence/conformance/conformance-matrix.json")
}

pub(super) fn read_text_bounded(path: &Path) -> io::Result<String> {
    xtask::output_arg::read_text_bounded(
        path,
        MAX_CONFORMANCE_EVIDENCE_TEXT_BYTES,
        "conformance evidence",
    )
}
