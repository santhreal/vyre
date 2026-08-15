//! The scalar release axes, recomputed from the artifacts that are supposed to
//! prove them.
//!
//! `bench-release` prints one number per axis, and each of those numbers is an
//! aggregate over the cited artifacts: the fastest warm case, the fastest cold
//! build, the best scan throughput, the worst observed ULP drift, the largest
//! device memory. Recomputing them here and comparing means a stale axes file
//! cannot carry a number its own evidence no longer supports.

use serde_json::Value;

use super::artifact_reader::{
    artifact_environment_first_gpu_u64, artifact_positive_metric_percentile,
};
use super::data::{COLD_PIPELINE_BUILD_METRICS, SCAN_THROUGHPUT_METRICS};

pub(crate) fn inspect_release_axes_scalar_values(
    axes: &Value,
    source_reports: &[Value],
    issues: &mut Vec<String>,
) {
    if source_reports.is_empty() {
        return;
    }
    if let Some(expected) = min_positive_metric_percentile(source_reports, "wall_ns", "p50") {
        inspect_release_axis_f64(axes, "warm_us_per_file", expected as f64 / 1_000.0, issues);
    }
    if let Some(expected) =
        first_min_positive_metric_percentile(source_reports, COLD_PIPELINE_BUILD_METRICS, "p50")
    {
        inspect_release_axis_f64(
            axes,
            "cold_pipeline_build_ms",
            expected as f64 / 1_000_000.0,
            issues,
        );
    }
    if let Some(expected) =
        first_max_positive_metric_percentile(source_reports, SCAN_THROUGHPUT_METRICS, "p50")
    {
        inspect_release_axis_f64(
            axes,
            "gbs_scan_throughput",
            expected as f64 / 1_000.0,
            issues,
        );
    }
    inspect_release_axis_u64(
        axes,
        "ulp_drift_max",
        max_observed_ulp(source_reports),
        issues,
    );
    if let Some(expected) = max_release_axis_vram_mib(source_reports) {
        inspect_release_axis_u64(axes, "max_vram_mib", expected, issues);
    }
}

fn inspect_release_axis_f64(axes: &Value, axis: &str, expected: f64, issues: &mut Vec<String>) {
    let Some(actual) = axes_number_f64(axes, axis) else {
        issues.push(format!(
            "bench-release-axes {axis} is missing or not numeric; expected {expected}"
        ));
        return;
    };
    if (actual - expected).abs() > 0.000_001 {
        issues.push(format!(
            "bench-release-axes {axis}={actual} does not match source artifacts {expected}"
        ));
    }
}

fn inspect_release_axis_u64(axes: &Value, axis: &str, expected: u64, issues: &mut Vec<String>) {
    let Some(actual) = axes_number_u64(axes, axis) else {
        issues.push(format!(
            "bench-release-axes {axis} is missing or not numeric; expected {expected}"
        ));
        return;
    };
    if actual != expected {
        issues.push(format!(
            "bench-release-axes {axis}={actual} does not match source artifacts {expected}"
        ));
    }
}

fn axes_number_f64(axes: &Value, axis: &str) -> Option<f64> {
    axes.get(axis).and_then(|value| {
        value
            .as_f64()
            .or_else(|| value.as_str()?.parse::<f64>().ok())
    })
}

fn axes_number_u64(axes: &Value, axis: &str) -> Option<u64> {
    axes.get(axis).and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_str()?.parse::<u64>().ok())
    })
}

fn min_positive_metric_percentile(
    reports: &[Value],
    metric_name: &str,
    percentile: &str,
) -> Option<u64> {
    reports
        .iter()
        .filter_map(|report| artifact_positive_metric_percentile(report, metric_name, percentile))
        .min()
}

fn max_positive_metric_percentile(
    reports: &[Value],
    metric_name: &str,
    percentile: &str,
) -> Option<u64> {
    reports
        .iter()
        .filter_map(|report| artifact_positive_metric_percentile(report, metric_name, percentile))
        .max()
}

fn first_min_positive_metric_percentile(
    reports: &[Value],
    metric_names: &[&str],
    percentile: &str,
) -> Option<u64> {
    metric_names
        .iter()
        .find_map(|metric_name| min_positive_metric_percentile(reports, metric_name, percentile))
}

fn first_max_positive_metric_percentile(
    reports: &[Value],
    metric_names: &[&str],
    percentile: &str,
) -> Option<u64> {
    metric_names
        .iter()
        .find_map(|metric_name| max_positive_metric_percentile(reports, metric_name, percentile))
}

fn max_observed_ulp(reports: &[Value]) -> u64 {
    reports
        .iter()
        .filter_map(|report| report.get("cases").and_then(Value::as_array))
        .flat_map(|cases| cases.iter())
        .filter_map(|case| {
            case.get("correctness")
                .and_then(|correctness| correctness.get("Toleranced"))
                .and_then(|toleranced| toleranced.get("max_observed_ulp"))
                .and_then(Value::as_u64)
        })
        .max()
        .unwrap_or(0)
}

fn max_release_axis_vram_mib(reports: &[Value]) -> Option<u64> {
    let environment_values = reports
        .iter()
        .filter_map(|report| artifact_environment_first_gpu_u64(report, "memory_total_mib"))
        .filter(|value| *value > 0);
    let metric_values = reports.iter().filter_map(|report| {
        artifact_positive_metric_percentile(report, "memory_total_mib", "p50")
    });
    environment_values.chain(metric_values).max()
}

#[cfg(test)]
mod tests {
    use super::super::release_axes_cuda::cuda_release_axes_source_artifact_issues;
    use crate::report_fixture::EvidenceWorkspace;

    #[test]
    fn cuda_release_axes_require_scalar_axes_from_source_artifacts() {
        let workspace = EvidenceWorkspace::new();
        let artifacts = workspace.cuda_release_axis_artifacts("release.scalar-required", 3);
        let axes = EvidenceWorkspace::cuda_release_axes(&artifacts);
        let cuda_suite = EvidenceWorkspace::cuda_release_suite(&artifacts);

        let issues = cuda_release_axes_source_artifact_issues(workspace.path(), &axes, &cuda_suite);

        for axis in [
            "warm_us_per_file",
            "cold_pipeline_build_ms",
            "gbs_scan_throughput",
            "ulp_drift_max",
            "max_vram_mib",
        ] {
            assert!(
                issues.iter().any(|issue| issue.contains(&format!(
                    "bench-release-axes {axis} is missing or not numeric"
                ))),
                "Fix: release axes must require scalar `{axis}` once source artifacts prove it; issues={issues:?}"
            );
        }
    }

    #[test]
    fn cuda_release_axes_reject_axis_values_that_drift_from_source_artifacts() {
        let workspace = EvidenceWorkspace::new();
        let artifacts = workspace.cuda_release_axis_artifacts("release.scalar-drift", 0);
        let axes = serde_json::json!({
            "warm_us_per_file": 17.0,
            "cold_pipeline_build_ms": 2.0,
            "gbs_scan_throughput": 999.0,
            "ulp_drift_max": 0,
            "max_vram_mib": 24576,
            "source_artifacts": artifacts
        });
        let cuda_suite = EvidenceWorkspace::cuda_release_suite(&artifacts);

        let issues = cuda_release_axes_source_artifact_issues(workspace.path(), &axes, &cuda_suite);

        assert!(
            issues.iter().any(|issue| issue.contains(
                "bench-release-axes gbs_scan_throughput=999 does not match source artifacts 4"
            )),
            "Fix: release-axis scalar values must be recomputed from source artifacts instead of trusting stale axes JSON; issues={issues:?}"
        );
    }
}
