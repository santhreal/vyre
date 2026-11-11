//! Metric normalization for a measured case.
//!
//! A backend reports the counters it has, under the names it has. These
//! functions restate one case's metrics in the schema the benchmark and
//! release evidence records declare, so a reader comparing two backends is
//! comparing the same quantity rather than two spellings of it.

use std::collections::BTreeMap;

use crate::api::metric::MetricStats;

pub(super) fn thermal_status_applies(
    metrics: &BTreeMap<String, MetricStats>,
    workload_needs_gpu: bool,
) -> bool {
    workload_needs_gpu
        && metrics
            .get("thermal_unstable")
            .is_some_and(|stats| stats.max > 0)
}

pub(super) fn normalize_benchmark_evidence_metrics(
    metrics: &mut BTreeMap<String, MetricStats>,
    cpu_digest: Option<u64>,
    gpu_digest: Option<u64>,
) {
    if let Some(cpu_digest) = cpu_digest {
        metrics
            .entry("cpu_digest".to_string())
            .or_insert_with(|| single_sample_stats(cpu_digest));
    }
    if let Some(gpu_digest) = gpu_digest {
        metrics
            .entry("gpu_digest".to_string())
            .or_insert_with(|| single_sample_stats(gpu_digest));
    }
    if let Some(active_time) = metrics
        .get("kernel_execute_ns")
        .or_else(|| metrics.get("dispatch_ns"))
        .or_else(|| metrics.get("wall_ns"))
        .cloned()
    {
        metrics
            .entry("active_time_ns".to_string())
            .or_insert(active_time);
    }
    if let (Some(input), Some(output)) = (
        metrics.get("host_to_device_bytes").cloned(),
        metrics.get("device_to_host_bytes").cloned(),
    ) {
        metrics
            .entry("transfer_bytes".to_string())
            .or_insert_with(|| sum_metric_stats(&input, &output));
    } else if let Some(bytes) = metrics
        .get("bytes_touched")
        .or_else(|| metrics.get("bytes_read"))
        .or_else(|| metrics.get("bytes_written"))
        .cloned()
    {
        metrics.entry("transfer_bytes".to_string()).or_insert(bytes);
    }
}

pub(super) fn sum_metric_stats(left: &MetricStats, right: &MetricStats) -> MetricStats {
    MetricStats {
        min: left.min.saturating_add(right.min),
        p50: left.p50.saturating_add(right.p50),
        p90: left.p90.saturating_add(right.p90),
        p95: left.p95.saturating_add(right.p95),
        p99: left.p99.saturating_add(right.p99),
        p999: left.p999.saturating_add(right.p999),
        p9999: left.p9999.saturating_add(right.p9999),
        max: left.max.saturating_add(right.max),
        mean: left.mean + right.mean,
        stddev: (left
            .stddev
            .mul_add(left.stddev, right.stddev * right.stddev))
        .sqrt(),
        samples: left.samples.min(right.samples),
        determinism_cv: None,
    }
}

/// Produce a degenerate `MetricStats` for a single
/// observation. Used to surface the cold (first-warmup) sample
/// alongside the warm-batch stats without inventing a separate
/// schema. min == p50 == max, samples == 1, stddev == 0.
pub(super) fn single_sample_stats(value: u64) -> MetricStats {
    MetricStats::single(value)
}

pub(super) fn normalize_release_evidence_metrics(
    metrics: &mut BTreeMap<String, MetricStats>,
    backend_id: &str,
) {
    if backend_id == "cuda" {
        if let Some(input) = metrics
            .get("cuda_host_to_device_bytes")
            .filter(|stats| stats.max > 0)
            .cloned()
        {
            metrics.insert("host_to_device_bytes".to_string(), input);
        }
        if let Some(output) = metrics
            .get("cuda_device_to_host_bytes")
            .filter(|stats| stats.max > 0)
            .cloned()
        {
            metrics.insert("device_to_host_bytes".to_string(), output);
        }
    }
    if let Some(input) = metrics
        .get("input_bytes")
        .or_else(|| metrics.get("bytes_read"))
        .or_else(|| metrics.get("bytes_touched"))
        .cloned()
    {
        metrics
            .entry("host_to_device_bytes".to_string())
            .or_insert(input);
    }
    metrics
        .entry("host_to_device_bytes".to_string())
        .or_insert_with(|| single_sample_stats(0));
    if let Some(output) = metrics
        .get("output_bytes")
        .or_else(|| metrics.get("bytes_written"))
        .or_else(|| metrics.get("bytes_touched"))
        .cloned()
    {
        metrics
            .entry("device_to_host_bytes".to_string())
            .or_insert(output);
    }
    metrics
        .entry("device_to_host_bytes".to_string())
        .or_insert_with(|| single_sample_stats(0));
    if backend_id == "cuda" {
        if let Some(launches) = metrics
            .get("cuda_kernel_launches")
            .filter(|stats| stats.max > 0)
            .cloned()
        {
            metrics
                .entry("kernel_launches".to_string())
                .or_insert(launches);
        }
    }
    if backend_id != "cpu-ref" {
        metrics
            .entry("kernel_launches".to_string())
            .or_insert_with(|| single_sample_stats(1));
    }
}

pub(super) fn infer_optimization_passes_applied(
    metrics: &BTreeMap<String, MetricStats>,
    backend_id: &str,
) -> Vec<String> {
    let mut passes = Vec::new();
    let metric_positive = |name: &str| metrics.get(name).is_some_and(|stats| stats.max > 0);
    if backend_id == "cuda" {
        passes.push("cuda-explicit-backend-selection".to_string());
    }
    if metrics.contains_key("cache_hit") || metrics.contains_key("cold_cache_lookup_ns") {
        passes.push("pipeline-cache-lookup".to_string());
    }
    if metric_positive("cuda_ptx_source_cache_entries")
        || metric_positive("cuda_ptx_source_cache_hits")
        || metric_positive("cuda_ptx_source_cache_misses")
    {
        passes.push("cuda-ptx-source-cache".to_string());
    }
    if metric_positive("cuda_graph_launches") {
        passes.push("cuda-graph-replay".to_string());
    }
    if metric_positive("cuda_graph_materialized_cache_hits") {
        passes.push("cuda-graph-materialized-output-cache".to_string());
    }
    if metric_positive("cuda_host_upload_operations")
        || metric_positive("cuda_device_readback_operations")
    {
        passes.push("cuda-transfer-operation-telemetry".to_string());
    }
    if metrics.contains_key("optimize_ns") || metrics.contains_key("cold_optimize_ns") {
        passes.push("optimizer-pipeline".to_string());
    }
    if metrics.contains_key("lower_ns") || metrics.contains_key("cold_lower_ns") {
        passes.push("backend-lowering".to_string());
    }
    if metrics
        .get("kernel_launches")
        .is_some_and(|stats| stats.max == 1)
    {
        passes.push("single-dispatch-launch-plan".to_string());
    } else if metrics
        .get("kernel_launches")
        .is_some_and(|stats| stats.max > 1)
    {
        passes.push("multi-dispatch-launch-plan".to_string());
    }
    if metrics.keys().any(|key| {
        key.starts_with("lower_") || key.starts_with("alias_") || key.starts_with("egraph_")
    }) {
        passes.push("measured-lower-optimization-family".to_string());
    }
    passes.sort();
    passes.dedup();
    passes
}

pub(super) fn workload_fingerprint(case_id: &str, program_fingerprint: Option<[u8; 32]>) -> String {
    let Some(fingerprint) = program_fingerprint else {
        return format!("bench-case:{case_id}");
    };
    let mut encoded = String::with_capacity("program:".len() + 64);
    encoded.push_str("program:");
    for byte in fingerprint {
        use std::fmt::Write as _;
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(value: u64) -> MetricStats {
        single_sample_stats(value)
    }

    #[test]
    fn cuda_graph_backend_metrics_are_reported_as_release_path_passes() {
        let mut metrics = BTreeMap::new();
        metrics.insert("cuda_ptx_source_cache_misses".to_string(), stats(1));
        metrics.insert("cuda_graph_launches".to_string(), stats(3));
        metrics.insert("cuda_graph_materialized_cache_hits".to_string(), stats(2));
        metrics.insert("cuda_host_upload_operations".to_string(), stats(4));
        metrics.insert("cuda_device_readback_operations".to_string(), stats(1));

        let passes = infer_optimization_passes_applied(&metrics, "cuda");

        for expected in [
            "cuda-explicit-backend-selection",
            "cuda-graph-replay",
            "cuda-graph-materialized-output-cache",
            "cuda-ptx-source-cache",
            "cuda-transfer-operation-telemetry",
        ] {
            assert!(
                passes.iter().any(|pass| pass == expected),
                "Fix: CUDA benchmark reports must label `{expected}` when backend telemetry exposes the release-path metric."
            );
        }
    }

    #[test]
    fn cuda_graph_backend_passes_require_positive_release_path_counters() {
        let mut metrics = BTreeMap::new();
        metrics.insert("cuda_ptx_source_cache_hits".to_string(), stats(0));
        metrics.insert("cuda_graph_launches".to_string(), stats(0));
        metrics.insert("cuda_graph_materialized_cache_hits".to_string(), stats(0));
        metrics.insert("cuda_host_upload_operations".to_string(), stats(0));
        metrics.insert("cuda_device_readback_operations".to_string(), stats(0));

        let passes = infer_optimization_passes_applied(&metrics, "cuda");

        for absent in [
            "cuda-graph-replay",
            "cuda-graph-materialized-output-cache",
            "cuda-ptx-source-cache",
            "cuda-transfer-operation-telemetry",
        ] {
            assert!(
                !passes.iter().any(|pass| pass == absent),
                "Fix: CUDA benchmark reports must not label `{absent}` when telemetry exposes only zero observations."
            );
        }
        assert!(
            passes
                .iter()
                .any(|pass| pass == "cuda-explicit-backend-selection"),
            "Fix: explicit CUDA backend selection is independent of per-counter activity."
        );
    }

    #[test]
    fn release_metrics_use_cuda_launch_counter_before_single_launch_fallback() {
        let mut metrics = BTreeMap::new();
        metrics.insert("cuda_kernel_launches".to_string(), stats(4));

        normalize_release_evidence_metrics(&mut metrics, "cuda");

        let launch_stats = metrics
            .get("kernel_launches")
            .expect("Fix: CUDA release reports must expose canonical kernel_launches.");
        assert_eq!(
            launch_stats.p50, 4,
            "Fix: canonical kernel_launches must preserve CUDA telemetry instead of reporting the synthetic single-launch fallback."
        );
    }

    /// WHY: artifact submissions may bypass the lower-level CUDA telemetry
    /// object. A zero observation is unavailable telemetry, while a successful
    /// measured GPU sample proves that at least one kernel was launched.
    #[test]
    fn zero_cuda_launch_counter_uses_single_submission_fallback() {
        let mut metrics = BTreeMap::new();
        metrics.insert("cuda_kernel_launches".to_string(), stats(0));

        normalize_release_evidence_metrics(&mut metrics, "cuda");

        assert_eq!(metrics["kernel_launches"].p50, 1);
    }

    /// WHY: an explicit zero identifies compiler-only evidence. It is a real
    /// observation, unlike a zero backend counter that means telemetry is absent.
    #[test]
    fn explicit_zero_launch_metric_bypasses_single_submission_fallback() {
        let mut metrics = BTreeMap::new();
        metrics.insert("kernel_launches".to_string(), stats(0));

        normalize_release_evidence_metrics(&mut metrics, "cuda");

        assert_eq!(metrics["kernel_launches"].p50, 0);
    }

    #[test]
    fn release_metrics_keep_single_launch_fallback_when_backend_has_no_counter() {
        let mut metrics = BTreeMap::new();

        normalize_release_evidence_metrics(&mut metrics, "wgpu");

        let launch_stats = metrics.get("kernel_launches").expect(
            "Fix: non-CPU release reports without backend counters still need launch evidence.",
        );
        assert_eq!(
            launch_stats.p50, 1,
            "Fix: launch fallback must remain for backends that do not expose a backend-specific launch counter."
        );
    }

    #[test]
    fn launch_plan_labels_match_measured_kernel_launch_count() {
        let mut single = BTreeMap::new();
        single.insert("kernel_launches".to_string(), stats(1));

        let single_passes = infer_optimization_passes_applied(&single, "wgpu");
        assert!(
            single_passes
                .iter()
                .any(|pass| pass == "single-dispatch-launch-plan"),
            "Fix: one measured kernel launch must keep the single-dispatch launch-plan label."
        );
        assert!(
            !single_passes
                .iter()
                .any(|pass| pass == "multi-dispatch-launch-plan"),
            "Fix: one measured kernel launch must not be reported as a multi-dispatch plan."
        );

        let mut multi = BTreeMap::new();
        multi.insert("kernel_launches".to_string(), stats(4));

        let multi_passes = infer_optimization_passes_applied(&multi, "cuda");
        assert!(
            multi_passes
                .iter()
                .any(|pass| pass == "multi-dispatch-launch-plan"),
            "Fix: more than one measured kernel launch must be labeled as a multi-dispatch launch plan."
        );
        assert!(
            !multi_passes
                .iter()
                .any(|pass| pass == "single-dispatch-launch-plan"),
            "Fix: multi-launch CUDA evidence must not claim the single-dispatch launch-plan label."
        );
    }

    #[test]
    fn release_metrics_use_cuda_transfer_counters_before_logical_byte_fallbacks() {
        let mut metrics = BTreeMap::new();
        metrics.insert("bytes_read".to_string(), stats(12));
        metrics.insert("bytes_written".to_string(), stats(4));
        metrics.insert("cuda_host_to_device_bytes".to_string(), stats(48));
        metrics.insert("cuda_device_to_host_bytes".to_string(), stats(16));

        normalize_release_evidence_metrics(&mut metrics, "cuda");

        let host_to_device = metrics
            .get("host_to_device_bytes")
            .expect("Fix: CUDA release reports must expose canonical host_to_device_bytes.");
        assert_eq!(
            host_to_device.p50, 48,
            "Fix: canonical host_to_device_bytes must preserve CUDA transfer telemetry instead of logical input bytes."
        );
        let device_to_host = metrics
            .get("device_to_host_bytes")
            .expect("Fix: CUDA release reports must expose canonical device_to_host_bytes.");
        assert_eq!(
            device_to_host.p50, 16,
            "Fix: canonical device_to_host_bytes must preserve CUDA transfer telemetry instead of logical output bytes."
        );
    }

    /// WHY: artifact materializers may not expose backend telemetry through the
    /// lower-level dispatch object. Zero counters must not erase measured byte
    /// accounting and produce a false missing-transfer release blocker.
    #[test]
    fn zero_cuda_transfer_counters_fall_back_to_measured_byte_accounting() {
        let mut metrics = BTreeMap::new();
        metrics.insert("bytes_read".to_string(), stats(12));
        metrics.insert("bytes_written".to_string(), stats(4));
        metrics.insert("cuda_host_to_device_bytes".to_string(), stats(0));
        metrics.insert("cuda_device_to_host_bytes".to_string(), stats(0));

        normalize_release_evidence_metrics(&mut metrics, "cuda");
        normalize_benchmark_evidence_metrics(&mut metrics, None, None);

        assert_eq!(metrics["host_to_device_bytes"].p50, 12);
        assert_eq!(metrics["device_to_host_bytes"].p50, 4);
        assert_eq!(metrics["transfer_bytes"].p50, 16);
    }

    #[test]
    fn release_metrics_keep_logical_transfer_fallback_when_backend_has_no_transfer_counter() {
        let mut metrics = BTreeMap::new();
        metrics.insert("bytes_read".to_string(), stats(12));
        metrics.insert("bytes_written".to_string(), stats(4));

        normalize_release_evidence_metrics(&mut metrics, "wgpu");

        let host_to_device = metrics
            .get("host_to_device_bytes")
            .expect("Fix: non-CPU release reports still need host_to_device_bytes.");
        assert_eq!(
            host_to_device.p50, 12,
            "Fix: logical input-byte fallback must remain for backends without transfer telemetry."
        );
        let device_to_host = metrics
            .get("device_to_host_bytes")
            .expect("Fix: non-CPU release reports still need device_to_host_bytes.");
        assert_eq!(
            device_to_host.p50, 4,
            "Fix: logical output-byte fallback must remain for backends without transfer telemetry."
        );
    }

    /// CPU-only optimizer benchmarks ignore idle GPU clocks even under a CUDA release run.
    #[test]
    fn cpu_only_workloads_ignore_gpu_thermal_status() {
        let metrics = BTreeMap::from([("thermal_unstable".to_string(), stats(1))]);

        assert!(!thermal_status_applies(&metrics, false));
    }

    /// GPU workloads retain the fail-closed thermal stability gate.
    #[test]
    fn gpu_workloads_preserve_thermal_status() {
        let metrics = BTreeMap::from([("thermal_unstable".to_string(), stats(1))]);

        assert!(thermal_status_applies(&metrics, true));
    }

    /// Stable or absent GPU telemetry never produces a thermal failure.
    #[test]
    fn stable_and_absent_gpu_telemetry_passes() {
        assert!(!thermal_status_applies(&BTreeMap::new(), true));
        assert!(!thermal_status_applies(
            &BTreeMap::from([("thermal_unstable".to_string(), stats(0))]),
            true,
        ));
    }
}
