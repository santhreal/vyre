//! Reading the fields of one benchmark artifact report that a suite status row
//! claims to summarize.
//!
//! Every number a suite status carries is derivable from the artifact it names:
//! the host and device attribution out of `environment`, the metric minima out
//! of the case metrics, and the pass, fail and backend-drift counts out of the
//! cases themselves. Deriving them here is what makes the status row provable
//! rather than declared.

use serde_json::Value;

use super::case_summary::benchmark_case_passes_summary_evidence;
use super::json_reader::{non_empty_str, nonnegative_json_number_as_u64};

fn artifact_environment<'a>(artifact_report: &'a Value) -> Option<&'a Value> {
    artifact_report.get("environment")
}

pub(crate) fn artifact_environment_str(
    artifact_report: &Value,
    field: &str,
) -> Option<Option<String>> {
    let value = artifact_environment(artifact_report)?.get(field)?;
    Some(non_empty_str(value).map(str::to_string))
}

pub(crate) fn artifact_environment_host_cpu_model(
    artifact_report: &Value,
) -> Option<Option<String>> {
    let environment = artifact_environment(artifact_report)?;
    let value = environment
        .get("host_cpu_model")
        .or_else(|| environment.get("cpu_model"))
        .or_else(|| environment.get("host_cpu"))?;
    Some(non_empty_str(value).map(str::to_string))
}

fn artifact_environment_first_gpu<'a>(artifact_report: &'a Value) -> Option<&'a Value> {
    artifact_environment(artifact_report)?
        .get("gpu_devices")
        .and_then(Value::as_array)
        .and_then(|devices| devices.first())
}

pub(crate) fn artifact_environment_first_gpu_str(
    artifact_report: &Value,
    field: &str,
) -> Option<Option<String>> {
    let value = artifact_environment_first_gpu(artifact_report)?.get(field)?;
    Some(non_empty_str(value).map(str::to_string))
}

pub(crate) fn artifact_environment_first_gpu_u64(
    artifact_report: &Value,
    field: &str,
) -> Option<u64> {
    artifact_environment_first_gpu(artifact_report)?
        .get(field)
        .and_then(Value::as_u64)
}

pub(crate) fn artifact_min_metric_samples(
    artifact_report: &Value,
    metric_name: &str,
) -> Option<u64> {
    let cases = artifact_report.get("cases").and_then(Value::as_array)?;
    if cases.is_empty() {
        return None;
    }
    let mut seen_metric = false;
    let min = cases
        .iter()
        .map(|case| {
            let metric = case
                .get("metrics")
                .and_then(|metrics| metrics.get(metric_name));
            if metric.is_some() {
                seen_metric = true;
            }
            metric
                .and_then(|metric| metric.get("samples"))
                .and_then(Value::as_u64)
                .unwrap_or(0)
        })
        .min()
        .unwrap_or(0);
    seen_metric.then_some(min)
}

pub(crate) fn artifact_min_metric_percentile(
    artifact_report: &Value,
    metric_name: &str,
    percentile: &str,
) -> Option<u64> {
    let cases = artifact_report.get("cases").and_then(Value::as_array)?;
    if cases.is_empty() {
        return None;
    }
    let mut seen_metric = false;
    let min = cases
        .iter()
        .map(|case| {
            let metric = case
                .get("metrics")
                .and_then(|metrics| metrics.get(metric_name));
            if metric.is_some() {
                seen_metric = true;
            }
            metric
                .and_then(|metric| metric.get(percentile))
                .and_then(nonnegative_json_number_as_u64)
                .unwrap_or(0)
        })
        .min()
        .unwrap_or(0);
    seen_metric.then_some(min)
}

pub(crate) fn artifact_positive_metric_percentile(
    report: &Value,
    metric_name: &str,
    percentile: &str,
) -> Option<u64> {
    artifact_min_metric_percentile(report, metric_name, percentile).filter(|value| *value > 0)
}

pub(crate) fn first_positive_artifact_metric_percentile(
    report: &Value,
    metric_names: &[&str],
    percentile: &str,
) -> Option<u64> {
    metric_names.iter().find_map(|metric_name| {
        artifact_positive_metric_percentile(report, metric_name, percentile)
    })
}

pub(crate) fn artifact_nonmatching_case_backend_count(
    status: &Value,
    artifact_report: &Value,
) -> Option<u64> {
    let expected_backend = status
        .get("selected_backend")
        .and_then(non_empty_str)
        .or_else(|| {
            artifact_report
                .get("selected_backend")
                .and_then(non_empty_str)
        })?;
    let cases = artifact_report.get("cases").and_then(Value::as_array)?;
    Some(
        cases
            .iter()
            .filter(|case| case.get("backend_id").and_then(Value::as_str) != Some(expected_backend))
            .count() as u64,
    )
}

pub(crate) fn artifact_case_failed_count(artifact_report: &Value) -> Option<u64> {
    let cases = artifact_report.get("cases").and_then(Value::as_array)?;
    Some(
        cases
            .iter()
            .filter(|case| !benchmark_case_passes_summary_evidence(case))
            .count() as u64,
    )
}

pub(crate) fn artifact_case_passed_count(artifact_report: &Value) -> Option<u64> {
    let cases = artifact_report.get("cases").and_then(Value::as_array)?;
    Some(
        cases
            .iter()
            .filter(|case| benchmark_case_passes_summary_evidence(case))
            .count() as u64,
    )
}
