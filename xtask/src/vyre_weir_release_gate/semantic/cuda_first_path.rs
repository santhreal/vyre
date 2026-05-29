use std::path::Path;

use super::super::types::Requirement;
use super::super::checks::*;

pub(super) fn check(
    requirement: &Requirement,
    base_dir: &Path,
    failures: &mut Vec<String>,
) {
    let Some(matrix) =
        first_json_evidence(requirement, base_dir, "backend-matrix.json", failures)
    else {
        return;
    };
    let cuda_first = matrix
        .get("cuda_first")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let blockers = matrix
        .get("blockers")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if !cuda_first {
        failures.push(
            "requirement `cuda-first-path` backend matrix does not prove CUDA-first dispatch"
                .to_string(),
        );
    }
    check_backend_matrix_schema("cuda-first-path", &matrix, failures);
    if blockers != 0 {
        failures.push(format!(
            "requirement `cuda-first-path` backend matrix still reports {blockers} blocker(s)"
        ));
    }
    require_no_hidden_backend_fallback_findings("cuda-first-path", &matrix, failures);
    check_backend_gpu_probe("cuda-first-path", &matrix, failures);
    check_preferred_backend_gpu_only("cuda-first-path", &matrix, failures);
    check_backend_acquire_entry("cuda-first-path", &matrix, "cuda", failures);
    check_backend_feature_markers(
        "cuda-first-path",
        &matrix,
        "cuda_feature_markers",
        12,
        failures,
    );
    check_json_evidence_has_no_blockers(
        requirement,
        base_dir,
        "cuda-release-suite.json",
        failures,
    );
    check_backend_suite_report(requirement, base_dir, "cuda-release-suite.json", failures);
    check_benchmark_report_has_cases(
        requirement,
        base_dir,
        "cuda-ptx-patterns.json",
        failures,
    );
    check_json_evidence_has_no_blockers(
        requirement,
        base_dir,
        "bench-release-axes.json",
        failures,
    );
    if let Some(axes) =
        first_json_evidence(requirement, base_dir, "bench-release-axes.json", failures)
    {
        let source_artifacts = axes
            .get("source_artifacts")
            .and_then(serde_json::Value::as_array)
            .map_or(0, Vec::len);
        if source_artifacts < 12 {
            failures.push(format!(
                "requirement `cuda-first-path` bench-release-axes has {source_artifacts} source artifact(s), needs at least 12"
            ));
        }
    }
}
