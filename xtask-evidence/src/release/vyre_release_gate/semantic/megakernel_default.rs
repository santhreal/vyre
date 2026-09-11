use std::path::Path;

use super::super::checks::*;
use super::super::gate_inputs::Requirement;

pub(super) fn check(requirement: &Requirement, base_dir: &Path, failures: &mut Vec<String>) {
    let Some(backend_matrix) =
        first_json_evidence(requirement, base_dir, "backend-matrix.json", failures)
    else {
        return;
    };
    check_backend_feature_marker_id(
        "megakernel-default",
        &backend_matrix,
        "cuda_feature_markers",
        "megakernel-paired-speculation",
        failures,
    );
    check_backend_feature_marker_id(
        "megakernel-default",
        &backend_matrix,
        "wgpu_feature_markers",
        "megakernel-paired-speculation",
        failures,
    );
    let Some(matrix) = release_workload_matrix(requirement, base_dir, failures) else {
        return;
    };
    let has_megakernel = matrix
        .get("families")
        .and_then(serde_json::Value::as_array)
        .and_then(|families| {
            families.iter().find(|family| {
                family.get("id").and_then(serde_json::Value::as_str)
                    == Some("megakernel-queued-batches")
            })
        })
        .and_then(|family| family.get("matched_cases"))
        .and_then(serde_json::Value::as_array)
        .is_some_and(|cases| !cases.is_empty());
    let blockers = array_len(&matrix, "blockers");
    if !has_megakernel {
        failures.push(
            "requirement `megakernel-default` has no active megakernel-queued-batches workload in the release matrix"
                .to_string(),
        );
    }
    if blockers != 0 {
        failures.push(format!(
            "requirement `megakernel-default` workload matrix still reports {blockers} blocker(s)"
        ));
    }
    check_named_cuda_benchmark_report(
        requirement,
        base_dir,
        "megakernel-condition-cuda.json",
        failures,
    );
    check_named_cuda_benchmark_report(
        requirement,
        base_dir,
        "megakernel-latency-cuda.json",
        failures,
    );
}
