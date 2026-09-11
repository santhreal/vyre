//! What a recorded conformance matrix artifact has to say to count, and how the
//! conformance surface reads text off disk.
//!
//! The artifact is read back as JSON and checked against the backends, gates and
//! workflows a release requires, without consulting the registry that produced
//! it.

use std::io;
use std::path::Path;

use serde_json::Value;

/// Largest conformance evidence or workflow file this surface will read.
const MAX_TEXT_BYTES: u64 = 8_388_608;

/// Read a file on the conformance surface, bounded.
///
/// The bound and the error context are one decision about one surface. Three
/// modules carried their own copy of both, so a change to either had to be made
/// in three files and agreed by eye.
pub fn read_conformance_text(path: &Path) -> io::Result<String> {
    crate::output_arg::read_text_bounded(path, MAX_TEXT_BYTES, "conformance evidence")
}

/// Device backends a release must dispatch. One release decision, read by the
/// gate that records the matrix and by the check that reads the recorded
/// matrix back.
///
/// Every name here is a registered `BackendRegistration`. This crate links no
/// driver, so the list is pinned to the live registry by a contract in
/// `xtask-registry`, which does: a required name the registry stops reporting
/// turns that contract red rather than drifting.
pub const REQUIRED_DISPATCH_BACKENDS: &[&str] = &["cuda", "wgpu"];

/// The oracle every dispatch is compared against, and the id its record writes.
///
/// The reference interpreter is host code reached through a named API. It
/// submits no `BackendRegistration` and never appears in discovery,
/// precedence, autoroute or a backend count, so a release requires its record
/// as an oracle record and requires the dispatch roster not to name it. The
/// required dispatch list held it as a third backend, which asked the backend
/// registry for something the registry is built to refuse, so the requirement
/// failed on every run and named the wrong repair.
///
/// This crate links no driver, so the value is a copy of the id
/// `vyre-driver-reference` owns. A contract in `xtask-registry`, which links
/// both, holds the two equal. A record written before the oracle stopped
/// being spelled as a backend carries `cpu-ref`, which is converted where a
/// recorded report is read rather than accepted as a second spelling.
pub const ORACLE_RECORD_ID: &str = "reference-oracle";
const REQUIRED_WORKFLOWS: &[&str] = &[
    ".github/workflows/conform.yml",
    ".github/workflows/gpu-parity.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/architectural-invariants.yml",
    "scripts/apply-branch-protection.sh",
];
const REQUIRED_GATES: &[&str] = &[
    "GPU release gate",
    "Conform release gate",
    "CI release gate",
    "Architecture release gate",
    "required_status_checks",
];

fn u64_field(value: &Value, field: &str, missing: u64) -> u64 {
    value.get(field).and_then(Value::as_u64).unwrap_or(missing)
}

fn array_len(value: &Value, field: &str, missing: usize) -> usize {
    value
        .get(field)
        .and_then(Value::as_array)
        .map_or(missing, Vec::len)
}

fn complete_ci_entry(entry: &Value) -> bool {
    entry.get("present").and_then(Value::as_bool) == Some(true)
        && entry.get("command_present").and_then(Value::as_bool) == Some(true)
        && entry.get("artifact_check_present").and_then(Value::as_bool) == Some(true)
}

/// Check a recorded conformance matrix against what a release requires.
pub fn inspect_conformance_matrix(context: &str, matrix: &Value, failures: &mut Vec<String>) {
    let op_count = u64_field(matrix, "op_count", 0);
    let distinct_op_count = u64_field(matrix, "distinct_op_count", 0);
    let catalog_required = u64_field(matrix, "catalog_required_op_count", 0);
    let catalog_covered = u64_field(matrix, "catalog_covered_op_count", 0);
    let missing_catalog_ops = array_len(matrix, "missing_catalog_ops", usize::MAX);
    let op_matrix_errors = array_len(matrix, "op_matrix_errors", usize::MAX);
    if op_matrix_errors != 0 {
        failures.push(format!(
            "{context}: reports {op_matrix_errors} OP_MATRIX read/parse error(s)"
        ));
    }
    if let Some(blockers) = matrix.get("blockers").and_then(Value::as_array) {
        for blocker in blockers {
            failures.push(format!(
                "{context}: conformance matrix blocker: {}",
                blocker.as_str().unwrap_or("<non-string blocker>")
            ));
        }
    }
    if op_count < 49 {
        failures.push(format!(
            "{context}: op_count is {op_count}, below release floor 49"
        ));
    }
    if distinct_op_count < 49 {
        failures.push(format!(
            "{context}: distinct_op_count is {distinct_op_count}, below release floor 49"
        ));
    }
    if catalog_required == 0 || catalog_covered != catalog_required || missing_catalog_ops != 0 {
        failures.push(format!(
            "{context}: covers {catalog_covered}/{catalog_required} OP_MATRIX-required op id(s), missing_catalog_ops={missing_catalog_ops}"
        ));
    }
    let blocked_release = u64_field(matrix, "op_matrix_blocked_release_count", u64::MAX);
    if blocked_release != 0 {
        failures.push(format!(
            "{context}: op_matrix_blocked_release_count must be zero, got {blocked_release}"
        ));
    }
    let release_backend_rows = u64_field(matrix, "release_backend_row_count", 0);
    let expected_release_backend_rows = catalog_required.saturating_mul(3);
    let missing_release_backend_rows =
        array_len(matrix, "missing_release_backend_rows", usize::MAX);
    if release_backend_rows < expected_release_backend_rows || missing_release_backend_rows != 0 {
        failures.push(format!(
            "{context}: release_backend_row_count={release_backend_rows}, expected {expected_release_backend_rows}, missing_release_backend_rows={missing_release_backend_rows}"
        ));
    }
    let fixture_required = u64_field(matrix, "fixture_required_count", u64::MAX);
    let fixture_inputs = u64_field(matrix, "fixture_input_count", 0);
    let expected_outputs = u64_field(matrix, "expected_output_count", 0);
    if fixture_required > op_count {
        failures.push(format!(
            "{context}: fixture_required_count {fixture_required} exceeds op_count {op_count}"
        ));
    }
    if fixture_inputs != fixture_required {
        failures.push(format!(
            "{context}: fixture_input_count {fixture_inputs} must equal fixture_required_count {fixture_required}"
        ));
    }
    if expected_outputs != fixture_required {
        failures.push(format!(
            "{context}: expected_output_count {expected_outputs} must equal fixture_required_count {fixture_required}"
        ));
    }
    if array_len(matrix, "duplicate_op_ids", 0) != 0 {
        failures.push(format!("{context}: duplicate_op_ids must be empty"));
    }
    let backends = matrix
        .get("dispatch_backends")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    for required in REQUIRED_DISPATCH_BACKENDS {
        if !backends
            .iter()
            .any(|backend| backend.as_str() == Some(required))
        {
            failures.push(format!(
                "{context}: dispatch_backends must include `{required}`"
            ));
        }
    }
    if backends
        .iter()
        .any(|backend| backend.as_str() == Some(ORACLE_RECORD_ID))
    {
        failures.push(format!(
            "{context}: dispatch_backends names the oracle `{ORACLE_RECORD_ID}`, which executes \
             on the host and is reached through a named API rather than the backend registry"
        ));
    }
    let schema_version = u64_field(matrix, "schema_version", 0);
    if schema_version < 2 {
        failures.push(format!(
            "{context}: schema_version is {schema_version}, expected >= 2"
        ));
    }
    let ci_gate_count = u64_field(matrix, "ci_blocking_gate_count", 0);
    if ci_gate_count < 3 {
        failures.push(format!(
            "{context}: ci_blocking_gate_count is {ci_gate_count}, needs at least 3"
        ));
    }
    if array_len(matrix, "required_ci_statuses", 0) == 0 {
        failures.push(format!(
            "{context}: parsed zero required CI status context(s)"
        ));
    }
    for (field, description) in [
        (
            "missing_required_ci_statuses",
            "required CI status context(s) are missing from workflows",
        ),
        (
            "ci_status_scan_errors",
            "CI status scan error(s) make workflow status evidence incomplete",
        ),
        (
            "path_filtered_required_workflows",
            "required workflow(s) still use path filters",
        ),
        (
            "missing_required_workflow_triggers",
            "required workflow(s) are missing pull_request + push main trigger coverage",
        ),
        (
            "missing_fail_closed_fanins",
            "required fan-in job(s) are missing fail-closed dependency checks",
        ),
    ] {
        let count = array_len(matrix, field, usize::MAX);
        if count != 0 {
            failures.push(format!("{context}: {count} {description}"));
        }
    }
    let ci_gates = matrix
        .get("ci_gates")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    for required in REQUIRED_WORKFLOWS {
        if !ci_gates.iter().any(|entry| {
            entry
                .get("workflow")
                .and_then(Value::as_str)
                .is_some_and(|workflow| workflow == *required)
                && complete_ci_entry(entry)
        }) {
            failures.push(format!(
                "{context}: missing complete CI conformance workflow `{required}`"
            ));
        }
    }
    for required in REQUIRED_GATES {
        if !ci_gates.iter().any(|entry| {
            entry.get("gate").and_then(Value::as_str) == Some(required) && complete_ci_entry(entry)
        }) {
            failures.push(format!(
                "{context}: missing complete CI conformance gate `{required}`"
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::release::conformance_workflows::inspect_ci_conformance_gates;
    use std::path::Path;

    fn complete_matrix(workflows: impl Iterator<Item = &'static str>) -> Value {
        let mut ci_gates = workflows
            .map(|workflow| {
                serde_json::json!({
                    "workflow": workflow,
                    "present": true,
                    "command_present": true,
                    "artifact_check_present": true
                })
            })
            .collect::<Vec<_>>();
        ci_gates.extend(REQUIRED_GATES.iter().map(|gate| {
            serde_json::json!({
                "gate": gate,
                "present": true,
                "command_present": true,
                "artifact_check_present": true
            })
        }));
        serde_json::json!({
            "schema_version": 2,
            "op_count": 49,
            "distinct_op_count": 49,
            "catalog_required_op_count": 49,
            "catalog_covered_op_count": 49,
            "missing_catalog_ops": [],
            "op_matrix_errors": [],
            "blockers": [],
            "op_matrix_blocked_release_count": 0,
            "release_backend_row_count": 147,
            "missing_release_backend_rows": [],
            "fixture_required_count": 49,
            "fixture_input_count": 49,
            "expected_output_count": 49,
            "duplicate_op_ids": [],
            "dispatch_backends": REQUIRED_DISPATCH_BACKENDS,
            "ci_blocking_gate_count": 3,
            "required_ci_statuses": ["conform"],
            "missing_required_ci_statuses": [],
            "ci_status_scan_errors": [],
            "path_filtered_required_workflows": [],
            "missing_required_workflow_triggers": [],
            "missing_fail_closed_fanins": [],
            "ci_gates": ci_gates
        })
    }

    /// Public evidence stores root and repository workflows in stable relative
    /// form, and each distinct workflow must satisfy the conformance contract.
    #[test]
    fn normalized_relative_workflow_paths_are_complete() {
        let matrix = complete_matrix(REQUIRED_WORKFLOWS.iter().copied());
        let mut failures = Vec::new();

        inspect_conformance_matrix("matrix.json", &matrix, &mut failures);

        assert_eq!(failures, Vec::<String>::new());
    }

    /// A root workflow cannot satisfy the separate repository-local workflow
    /// requirement merely because both file names share the same suffix.
    #[test]
    fn root_workflow_does_not_alias_repository_workflow() {
        let matrix = complete_matrix(
            REQUIRED_WORKFLOWS
                .iter()
                .copied()
                .filter(|workflow| *workflow != ".github/workflows/conform.yml"),
        );
        let mut failures = Vec::new();

        inspect_conformance_matrix("matrix.json", &matrix, &mut failures);

        assert_eq!(
            failures,
            vec![
                "matrix.json: missing complete CI conformance workflow `.github/workflows/conform.yml`"
                    .to_string()
            ]
        );
    }

    /// Every name this module requires must be a name the matrix generator can
    /// actually emit, so that a requirement is a statement about the repository
    /// rather than one the producer is structurally unable to satisfy. Both
    /// sides are enumerated from source at run time: adding a required workflow
    /// or gate without the matching `inspect_ci_gate` row turns this red.
    ///
    /// The class it closes: `.github/CI_REQUIRED.md` was listed as a required
    /// workflow while the generator only ever named it as the *command* of the
    /// `scripts/apply-branch-protection.sh` row, so the blocker fired on every
    /// run regardless of repository state. It does not catch a required name
    /// the generator emits with the wrong predicate wired to it; the per-gate
    /// `command_present` / `artifact_check_present` assertions cover that.
    #[test]
    fn every_required_name_is_one_the_matrix_generator_emits() {
        let emitted = inspect_ci_conformance_gates(Path::new(""));

        let missing_workflows = REQUIRED_WORKFLOWS
            .iter()
            .filter(|required| !emitted.iter().any(|gate| gate.workflow == **required))
            .collect::<Vec<_>>();
        let missing_gates = REQUIRED_GATES
            .iter()
            .filter(|required| !emitted.iter().any(|gate| gate.gate == **required))
            .collect::<Vec<_>>();

        assert_eq!(missing_workflows, Vec::<&&str>::new());
        assert_eq!(missing_gates, Vec::<&&str>::new());
    }

    /// WHY: the requirement this replaced could not pass. `cpu-ref` was listed
    /// as a third dispatch backend, and the matrix generator reads the backend
    /// registry, which the reference interpreter is built never to enter. The
    /// inverse is the contract that can fail: a roster naming the oracle means
    /// a host execution route reached backend discovery, and that has to be a
    /// blocker rather than the state the requirement asks for.
    ///
    /// What it does not catch: an oracle registered under some other id. The
    /// execution-domain closure in `vyre-driver-reference` decides every
    /// registry entry by exhaustive match and owns that.
    #[test]
    fn a_roster_naming_the_oracle_is_a_failure_rather_than_the_requirement() {
        let mut matrix = complete_matrix(REQUIRED_WORKFLOWS.iter().copied());
        let mut roster = REQUIRED_DISPATCH_BACKENDS
            .iter()
            .map(|backend| Value::from(*backend))
            .collect::<Vec<_>>();
        roster.push(Value::from(ORACLE_RECORD_ID));
        matrix["dispatch_backends"] = Value::from(roster);
        let mut failures = Vec::new();

        inspect_conformance_matrix("matrix.json", &matrix, &mut failures);

        assert_eq!(
            failures,
            vec![format!(
                "matrix.json: dispatch_backends names the oracle `{ORACLE_RECORD_ID}`, which \
                 executes on the host and is reached through a named API rather than the \
                 backend registry"
            )]
        );
    }

    /// WHY: the oracle spelling and the dispatch roster are two lists, and the
    /// defect was that one held a member of the other. A required dispatch
    /// backend that is also the oracle id would restore it in a shape the
    /// assertion above cannot see, because the roster would then be required
    /// to hold the name it is required to omit.
    #[test]
    fn the_oracle_is_not_a_required_dispatch_backend() {
        assert!(
            !REQUIRED_DISPATCH_BACKENDS.contains(&ORACLE_RECORD_ID),
            "Fix: the oracle is compared against, never dispatched to."
        );
    }
}
