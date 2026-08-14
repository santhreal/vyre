//! Whether one cited artifact carries the metrics a release axis is computed
//! from.
//!
//! An axis aggregated over twelve artifacts is only as strong as the weakest
//! one, so each artifact is required to carry a positive wall time, a cold build
//! time, a scan throughput and the device memory the memory axis reads. An
//! artifact that omits one cannot contribute to the axis, which would otherwise
//! be silently computed over fewer artifacts than were cited.

use serde_json::Value;

use super::artifact_reader::{
    artifact_environment_first_gpu_u64, artifact_positive_metric_percentile,
    first_positive_artifact_metric_percentile,
};
use super::data::{COLD_PIPELINE_BUILD_METRICS, SCAN_THROUGHPUT_METRICS};

pub(crate) fn inspect_release_axis_source_artifact_metrics(
    artifact: &str,
    report: &Value,
    issues: &mut Vec<String>,
) {
    if artifact_positive_metric_percentile(report, "wall_ns", "p50").is_none() {
        issues.push(format!(
            "source_artifact `{artifact}` has no positive p50 wall_ns metric for warm_us_per_file"
        ));
    }
    if first_positive_artifact_metric_percentile(report, COLD_PIPELINE_BUILD_METRICS, "p50")
        .is_none()
    {
        issues.push(format!(
            "source_artifact `{artifact}` has no positive p50 cold/compile metric for cold_pipeline_build_ms"
        ));
    }
    if first_positive_artifact_metric_percentile(report, SCAN_THROUGHPUT_METRICS, "p50").is_none() {
        issues.push(format!(
            "source_artifact `{artifact}` has no positive p50 throughput metric for gbs_scan_throughput"
        ));
    }
    if artifact_environment_first_gpu_u64(report, "memory_total_mib")
        .filter(|value| *value > 0)
        .or_else(|| artifact_positive_metric_percentile(report, "memory_total_mib", "p50"))
        .is_none()
    {
        issues.push(format!(
            "source_artifact `{artifact}` has no GPU memory_total_mib evidence for max_vram_mib"
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::super::release_axes_cuda::cuda_release_axes_source_artifact_issues;
    use crate::report_fixture::EvidenceWorkspace;

    #[test]
    fn cuda_release_axes_reject_source_artifacts_missing_axis_metrics() {
        let workspace = EvidenceWorkspace::new();
        let artifact = workspace.write_cuda_release_artifact(
            "workload-missing-axis-metrics.json",
            serde_json::json!([
                {
                    "id": "release.missing-axis-metrics",
                    "backend_id": "cuda",
                    "status": "pass",
                    "metrics": {
                        "wall_ns": {"p50": 10}
                    }
                }
            ]),
        );
        let axes = serde_json::json!({
            "source_artifacts": [artifact]
        });
        let cuda_suite = EvidenceWorkspace::cuda_release_suite(&[&artifact]);

        let issues = cuda_release_axes_source_artifact_issues(workspace.path(), &axes, &cuda_suite);

        assert!(
            issues.iter().any(|issue| issue.contains(
                "source_artifact `release/evidence/benchmarks/workload-missing-axis-metrics.json` has no positive p50 cold/compile metric for cold_pipeline_build_ms"
            )),
            "Fix: release-axis source artifacts must individually prove cold/compile metrics, not rely on another artifact; issues={issues:?}"
        );
        assert!(
            issues.iter().any(|issue| issue.contains(
                "source_artifact `release/evidence/benchmarks/workload-missing-axis-metrics.json` has no positive p50 throughput metric for gbs_scan_throughput"
            )),
            "Fix: release-axis source artifacts must individually prove throughput metrics, not rely on another artifact; issues={issues:?}"
        );
        assert!(
            issues.iter().any(|issue| issue.contains(
                "source_artifact `release/evidence/benchmarks/workload-missing-axis-metrics.json` has no GPU memory_total_mib evidence for max_vram_mib"
            )),
            "Fix: release-axis source artifacts must individually prove GPU memory evidence, not rely on another artifact; issues={issues:?}"
        );
    }
}
