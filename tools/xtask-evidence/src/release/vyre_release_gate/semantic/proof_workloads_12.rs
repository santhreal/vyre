use std::path::Path;

use super::super::checks::*;
use super::super::gate_inputs::Requirement;

pub(super) fn check(requirement: &Requirement, base_dir: &Path, failures: &mut Vec<String>) {
    let Some(matrix) = release_workload_matrix(requirement, base_dir, failures) else {
        return;
    };
    let required = u64_field(&matrix, "required_closed_families", 0);
    let matched = u64_field(&matrix, "matched_required_families", 0);
    let release_cases = u64_field(&matrix, "release_suite_case_count", 0);
    let blockers = array_len(&matrix, "blockers");
    if required < 12 {
        failures.push(format!(
            "requirement `proof-workloads-12` matrix requires only {required} workload families; needs at least 12"
        ));
    }
    if matched < 12 {
        failures.push(format!(
            "requirement `proof-workloads-12` matrix covers {matched} workload families; needs at least 12"
        ));
    }
    if matched < required {
        failures.push(format!(
            "requirement `proof-workloads-12` matrix covers {matched} of {required} required workload families"
        ));
    }
    if release_cases < matched {
        failures.push(format!(
            "requirement `proof-workloads-12` matrix reports {release_cases} release cases for {matched} matched workload families"
        ));
    }
    if blockers != 0 {
        failures.push(format!(
            "requirement `proof-workloads-12` matrix still reports {blockers} blocker(s)"
        ));
    }
    check_release_bench_targets(requirement, base_dir, failures);
    check_workload_matrix_artifact_coverage(requirement, base_dir, &matrix, failures);
    check_benchmark_evidence_reports(requirement, base_dir, "workload-", true, None, failures);
}
