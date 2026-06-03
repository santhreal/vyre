use std::path::Path;

use crate::benchmark_evidence_semantics::benchmark_source_artifact_count;

use super::super::checks::*;
use super::super::types::Requirement;

pub(super) fn check(requirement: &Requirement, base_dir: &Path, failures: &mut Vec<String>) {
    let Some(matrix) = first_json_evidence(requirement, base_dir, "backend-matrix.json", failures)
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
    check_json_evidence_has_no_blockers(requirement, base_dir, "cuda-release-suite.json", failures);
    check_backend_suite_report(requirement, base_dir, "cuda-release-suite.json", failures);
    check_benchmark_report_has_cases(requirement, base_dir, "cuda-ptx-patterns.json", failures);
    check_json_evidence_has_no_blockers(requirement, base_dir, "bench-release-axes.json", failures);
    if let Some(axes) =
        first_json_evidence(requirement, base_dir, "bench-release-axes.json", failures)
    {
        check_release_axes_source_artifacts(&axes, failures);
    }
}

fn check_release_axes_source_artifacts(axes: &serde_json::Value, failures: &mut Vec<String>) {
    let source_artifacts = benchmark_source_artifact_count(axes);
    if source_artifacts < 12 {
        failures.push(format!(
            "requirement `cuda-first-path` bench-release-axes has {source_artifacts} source artifact(s), needs at least 12"
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_first_axes_counts_only_usable_source_artifacts() {
        let axes = serde_json::json!({
            "source_artifacts": [
                "",
                null,
                "release/evidence/benchmarks/workload-01-condition-eval.json"
            ]
        });
        let mut failures = Vec::new();

        check_release_axes_source_artifacts(&axes, &mut failures);

        assert!(
            failures.iter().any(|failure| failure.contains(
                "bench-release-axes has 1 source artifact(s), needs at least 12"
            )),
            "Fix: CUDA-first release axes must not count blank/non-string source_artifacts as evidence; failures={failures:?}"
        );
    }
}
