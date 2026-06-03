use std::path::Path;

use crate::benchmark_evidence_semantics::benchmark_source_artifact_paths;

use super::super::checks::*;
use super::super::paths::resolve_artifact_path;
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
        check_release_axes_source_artifacts(base_dir, &axes, failures);
    }
}

fn check_release_axes_source_artifacts(
    base_dir: &Path,
    axes: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    let source_artifact_paths = benchmark_source_artifact_paths(axes);
    let source_artifacts = source_artifact_paths.len();
    if source_artifacts < 12 {
        failures.push(format!(
            "requirement `cuda-first-path` bench-release-axes has {source_artifacts} source artifact(s), needs at least 12"
        ));
    }
    for artifact in source_artifact_paths {
        let path = resolve_artifact_path(base_dir, &artifact);
        if !path.is_file() {
            failures.push(format!(
                "requirement `cuda-first-path` bench-release-axes source artifact `{artifact}` is not a readable file at {}",
                path.display()
            ));
        }
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
                "release/evidence/benchmarks/workload-01-condition-eval.json",
                "release/evidence/benchmarks/workload-01-condition-eval.json"
            ]
        });
        let mut failures = Vec::new();

        check_release_axes_source_artifacts(Path::new("."), &axes, &mut failures);

        assert!(
            failures.iter().any(|failure| failure.contains(
                "bench-release-axes has 1 source artifact(s), needs at least 12"
            )),
            "Fix: CUDA-first release axes must not count blank/non-string source_artifacts as evidence; failures={failures:?}"
        );
    }

    #[test]
    fn cuda_first_axes_rejects_missing_source_artifact_files() {
        let dir = tempfile::TempDir::new()
            .expect("Fix: create temporary workspace for source artifact existence test.");
        let release_dir = dir.path().join("release");
        std::fs::create_dir_all(release_dir.join("evidence/benchmarks"))
            .expect("Fix: create temporary benchmark evidence directory.");
        let mut source_artifacts = Vec::new();
        for index in 0..12 {
            let artifact = format!("release/evidence/benchmarks/workload-{index:02}.json");
            if index != 11 {
                std::fs::write(dir.path().join(&artifact), "{}")
                    .expect("Fix: write temporary source artifact.");
            }
            source_artifacts.push(artifact);
        }
        let axes = serde_json::json!({
            "source_artifacts": source_artifacts
        });
        let mut failures = Vec::new();

        check_release_axes_source_artifacts(&release_dir, &axes, &mut failures);

        assert!(
            failures.iter().any(|failure| failure.contains(
                "bench-release-axes source artifact `release/evidence/benchmarks/workload-11.json` is not a readable file"
            )),
            "Fix: CUDA-first release axes must reject source_artifacts that do not resolve to files; failures={failures:?}"
        );
    }
}
