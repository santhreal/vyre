use super::*;

use crate::bench::benchmark_evidence_semantics::{
    current_freshness_fingerprint_for_report, report_freshness_fingerprint,
};

pub(crate) fn check_release_bench_targets(
    requirement: &Requirement,
    base_dir: &Path,
    failures: &mut Vec<String>,
) {
    let path = base_dir.join("../docs/optimization/BENCH_TARGETS.toml");
    let text = match read_text_bounded(&path) {
        Ok(text) => text,
        Err(error) => {
            failures.push(format!(
                "requirement `{}` failed to read canonical benchmark targets `{}`: {error}",
                requirement.id,
                path.display()
            ));
            return;
        }
    };
    let target_count = text.matches("[[target]]").count();
    if target_count < 15 {
        failures.push(format!(
            "requirement `{}` benchmark target table contains {target_count} target(s); needs at least 15 including release workloads and the canonical semantic optimizer target",
            requirement.id
        ));
    }
    for required in [
        "release.workload.condition_eval",
        "release.workload.string_bitmap_scatter",
        "release.workload.offset_count_aggregation",
        "release.workload.pe_metadata",
        "release.workload.entropy_window",
        "release.workload.for_any_all_n",
        "release.workload.alias_reaching_def",
        "release.workload.ifds_witness",
        "release.workload.callgraph_reachability",
        "release.workload.ast_motif_traversal",
        "release.workload.megakernel_stream",
        "release.workload.conformance_sparse_readback",
        "release.optimization.foundation_optimizer_impact",
    ] {
        if !text.contains(required) {
            failures.push(format!(
                "requirement `{}` benchmark target table is missing release target `{required}`",
                requirement.id
            ));
        }
    }
    if text.matches("baseline_class_values").count() != 1
        || !text.contains("\"cpu_sota\"")
        || !text.contains("min_speedup_over_cpu_sota")
    {
        failures.push(format!(
            "requirement `{}` benchmark target table must declare CPU-SOTA baseline classes and speedup thresholds",
            requirement.id
        ));
    }
}
pub(crate) fn check_benchmark_evidence_reports(
    requirement: &Requirement,
    base_dir: &Path,
    name_fragment: &str,
    require_cuda: bool,
    min_speedup_x: Option<f64>,
    failures: &mut Vec<String>,
) {
    let mut matched = 0usize;
    for evidence in &requirement.evidence {
        if !evidence.ends_with(".json") || !evidence.contains(name_fragment) {
            continue;
        }
        if evidence.ends_with("release-workload-matrix.json") {
            continue;
        }
        matched += 1;
        let path = resolve_manifest_path(base_dir, evidence);
        let text = match read_text_bounded(&path) {
            Ok(text) => text,
            Err(error) => {
                failures.push(format!(
                    "requirement `{}` failed to read benchmark evidence `{}`: {error}",
                    requirement.id,
                    path.display()
                ));
                continue;
            }
        };
        let report = match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(report) => report,
            Err(error) => {
                failures.push(format!(
                    "requirement `{}` benchmark evidence `{}` is invalid JSON: {error}",
                    requirement.id,
                    path.display()
                ));
                continue;
            }
        };
        check_single_benchmark_report(
            requirement,
            base_dir,
            &path,
            &report,
            require_cuda,
            min_speedup_x,
            failures,
        );
    }
    if matched == 0 {
        failures.push(format!(
            "requirement `{}` has no benchmark evidence JSON matching `{name_fragment}`",
            requirement.id
        ));
    }
}
pub(crate) fn check_single_benchmark_report(
    requirement: &Requirement,
    base_dir: &Path,
    path: &Path,
    report: &serde_json::Value,
    require_cuda: bool,
    min_speedup_x: Option<f64>,
    failures: &mut Vec<String>,
) {
    check_benchmark_report_summary(requirement, &path.display().to_string(), report, failures);
    let selected_backend = report
        .get("selected_backend")
        .and_then(serde_json::Value::as_str);
    if require_cuda && selected_backend != Some("cuda") {
        failures.push(format!(
            "requirement `{}` benchmark `{}` selected backend `{:?}`, expected cuda",
            requirement.id,
            path.display(),
            selected_backend
        ));
    }
    if require_cuda {
        check_benchmark_cuda_environment_provenance(
            requirement,
            &path.display().to_string(),
            report,
            failures,
        );
    }
    check_benchmark_reproducibility_provenance(
        requirement,
        &path.display().to_string(),
        base_dir,
        report,
        failures,
    );
    if let (Some((field, source_fingerprint)), Some(current_source_fingerprint)) = (
        report_freshness_fingerprint(report),
        current_freshness_fingerprint_for_report(path, report),
    ) {
        check_source_fingerprint_freshness(
            requirement,
            &path.display().to_string(),
            field,
            source_fingerprint,
            &current_source_fingerprint,
            failures,
        );
    }
    let Some(cases) = report.get("cases").and_then(serde_json::Value::as_array) else {
        failures.push(format!(
            "requirement `{}` benchmark `{}` has no cases array",
            requirement.id,
            path.display()
        ));
        return;
    };
    if cases.is_empty() {
        failures.push(format!(
            "requirement `{}` benchmark `{}` has zero cases",
            requirement.id,
            path.display()
        ));
    }
    for case in cases {
        let id = case
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        if require_cuda
            && case.get("backend_id").and_then(serde_json::Value::as_str) != Some("cuda")
        {
            failures.push(format!(
                "requirement `{}` benchmark `{}` case `{id}` did not run on cuda",
                requirement.id,
                path.display()
            ));
        }
        if case.get("contract").is_none_or(serde_json::Value::is_null) {
            failures.push(format!(
                "requirement `{}` benchmark `{}` case `{id}` has no performance contract",
                requirement.id,
                path.display()
            ));
        }
        let metrics = case.get("metrics").and_then(serde_json::Value::as_object);
        let wall_samples = metrics
            .and_then(|metrics| metric_samples(metrics.get("wall_ns")))
            .unwrap_or(0);
        if wall_samples < 30 {
            failures.push(format!(
                "requirement `{}` benchmark `{}` case `{id}` has {wall_samples} wall_ns sample(s), needs at least 30",
                requirement.id,
                path.display()
            ));
        }
        require_benchmark_metric_percentiles(
            &requirement.id,
            &path.display().to_string(),
            id,
            metrics,
            "wall_ns",
            failures,
        );
        let contract_passed = case
            .get("performance")
            .and_then(|performance| performance.get("contract_passed"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        if !contract_passed {
            let reason =
                crate::bench::benchmark_evidence_semantics::benchmark_case_failure_reason(case)
                    .map(|reason| format!(": {reason}"))
                    .unwrap_or_default();
            failures.push(format!(
                "requirement `{}` benchmark `{}` case `{id}` did not pass its performance contract{reason}",
                requirement.id,
                path.display()
            ));
        }
        if let Some(required_speedup) = min_speedup_x {
            let case_backend = case
                .get("backend_id")
                .and_then(serde_json::Value::as_str)
                .or(selected_backend);
            if !crate::bench::benchmark_evidence_semantics::benchmark_case_has_cpu_sota_contract(
                case,
                case_backend,
                required_speedup,
            ) {
                failures.push(format!(
                    "requirement `{}` benchmark `{}` case `{id}` must carry an applicable CPU-SOTA performance contract with min_speedup_x >= {required_speedup:.2}",
                    requirement.id,
                    path.display()
                ));
            }
            let metrics = case.get("metrics").and_then(serde_json::Value::as_object);
            let wall_p50 = metrics.and_then(|metrics| metric_p50(metrics.get("wall_ns")));
            let baseline_p50 =
                metrics.and_then(|metrics| metric_p50(metrics.get("baseline_wall_ns")));
            require_benchmark_metric_percentiles(
                &requirement.id,
                &path.display().to_string(),
                id,
                metrics,
                "baseline_wall_ns",
                failures,
            );
            let wall_samples = metrics
                .and_then(|metrics| metric_samples(metrics.get("wall_ns")))
                .unwrap_or(0);
            if wall_samples < 30 {
                failures.push(format!(
                    "requirement `{}` benchmark `{}` case `{id}` has {wall_samples} wall_ns sample(s), needs at least 30",
                    requirement.id,
                    path.display()
                ));
            }
            match (wall_p50, baseline_p50) {
                (Some(wall), Some(baseline)) if wall > 0.0 => {
                    let measured_speedup = baseline / wall;
                    if measured_speedup < required_speedup {
                        failures.push(format!(
                            "requirement `{}` benchmark `{}` case `{id}` end-to-end p50 speedup was {measured_speedup:.2}x, needs at least {required_speedup:.2}x",
                            requirement.id,
                            path.display()
                        ));
                    }
                }
                _ => failures.push(format!(
                    "requirement `{}` benchmark `{}` case `{id}` must include p50 wall_ns and baseline_wall_ns metrics for the 100x proof",
                    requirement.id,
                    path.display()
                )),
            }
            let speedup = case
                .get("performance")
                .and_then(|performance| performance.get("speedup_x"))
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            if speedup < required_speedup {
                failures.push(format!(
                    "requirement `{}` benchmark `{}` case `{id}` observed {speedup:.2}x, needs at least {required_speedup:.2}x",
                    requirement.id,
                    path.display()
                ));
            }
        }
    }
}

#[cfg(test)]
mod benchmark_backend_tests {
    use super::*;
    use crate::report_fixture::{
        cpu_sota_baseline, hidden_invalid_measured_case, percentile_metrics,
    };

    #[test]
    fn single_benchmark_report_rejects_explicit_blockers() {
        let requirement = Requirement {
            id: "cuda-first-path".to_string(),
            title: "cuda first".to_string(),
            status: "required".to_string(),
            evidence: Vec::new(),
        };
        let report = serde_json::json!({
            "blockers": ["benchmark runner reused stale CUDA evidence"],
            "selected_backend": "cuda",
            "summary": {"failed": 0},
            "cases": [
                {
                    "id": "release.condition_eval.1m",
                    "backend_id": "cuda",
                    "status": "pass",
                    "contract": {"baselines": []},
                    "metrics": {"wall_ns": {"samples": 30, "p50": 1, "p95": 2, "p99": 3}},
                    "performance": {"contract_passed": true}
                }
            ]
        });
        let mut failures = Vec::new();

        check_single_benchmark_report(
            &requirement,
            Path::new("."),
            Path::new("cuda-workload.json"),
            &report,
            true,
            None,
            &mut failures,
        );

        assert!(
            failures.iter().any(|failure| failure.contains(
                "requirement `cuda-first-path` benchmark `cuda-workload.json` reports 1 blocker(s)"
            )),
            "Fix: direct benchmark release gate must reject explicit benchmark blockers; failures={failures:?}"
        );
    }

    #[test]
    fn single_benchmark_report_rejects_wrong_backend_cpu_sota_contract() {
        let requirement = Requirement::required("proof-workloads-12", "proof workloads");
        let report = serde_json::json!({
            "selected_backend": "wgpu",
            "summary": {"failed": 0},
            "cases": [
                {
                    "id": "release.condition_eval.1m",
                    "backend_id": "wgpu",
                    "status": "pass",
                    "contract": cpu_sota_baseline(&["cuda"], 100.0),
                    "metrics": percentile_metrics([10, 11, 12], [2000, 2100, 2200]),
                    "performance": {"contract_passed": true, "speedup_x": 200.0}
                }
            ]
        });
        let mut failures = Vec::new();

        check_single_benchmark_report(
            &requirement,
            Path::new("."),
            Path::new("wgpu-workload.json"),
            &report,
            false,
            Some(100.0),
            &mut failures,
        );

        assert!(
            failures.iter().any(|failure| failure
                .contains("must carry an applicable CPU-SOTA performance contract")),
            "Fix: release gate must expose wrong-backend CPU-SOTA contracts; failures={failures:?}"
        );
    }

    #[test]
    fn single_benchmark_report_preserves_failed_case_reason() {
        let requirement = Requirement::required("wgpu-fallback", "wgpu fallback");
        let report = serde_json::json!({
            "selected_backend": "wgpu",
            "summary": {"failed": 1},
            "cases": [
                {
                    "id": "sparse.compaction.count.1m",
                    "backend_id": "wgpu",
                    "status": "failed",
                    "correctness": {
                        "Invalid": {
                            "reason": "Performance contract failed: sparse output compaction count requires 100.00x over optimized CPU fired-rule collection over predicate masks, observed 86.90x"
                        }
                    },
                    "contract": cpu_sota_baseline(&["cuda", "wgpu"], 100.0),
                    "metrics": percentile_metrics([10, 11, 12], [1000, 1001, 1002]),
                    "performance": null
                }
            ]
        });
        let mut failures = Vec::new();

        check_single_benchmark_report(
            &requirement,
            Path::new("."),
            Path::new("wgpu-workload-12-sparse-output-compaction.json"),
            &report,
            false,
            None,
            &mut failures,
        );

        assert!(
            failures.iter().any(|failure| failure.contains(
                "case `sparse.compaction.count.1m` did not pass its performance contract: Performance contract failed"
            ) && failure.contains("observed 86.90x")),
            "Fix: direct benchmark gate failures must carry the failed benchmark case reason; failures={failures:?}"
        );
        assert!(
            failures.iter().any(|failure| failure.contains(
                "reports 1 failed case(s): `sparse.compaction.count.1m`: Performance contract failed"
            ) && failure.contains("observed 86.90x")),
            "Fix: direct benchmark summary failures must include failed case identity and reason; failures={failures:?}"
        );
    }

    #[test]
    fn single_benchmark_report_rejects_hidden_failed_case_summary_zero() {
        let requirement = Requirement::required("wgpu-fallback", "wgpu fallback");
        let report = serde_json::json!({
            "selected_backend": "wgpu",
            "summary": {"failed": 0},
            "cases": [hidden_invalid_measured_case(
                "release.condition_eval.1m",
                "wgpu",
                cpu_sota_baseline(&["wgpu"], 100.0),
                percentile_metrics([10, 11, 12], [2000, 2001, 2002]),
            )]
        });
        let mut failures = Vec::new();

        check_single_benchmark_report(
            &requirement,
            Path::new("."),
            Path::new("wgpu-hidden-invalid.json"),
            &report,
            false,
            None,
            &mut failures,
        );

        assert!(
            failures.iter().any(|failure| failure.contains(
                "reports 0 failed case(s); case evidence reports 1 failed case(s): `release.condition_eval.1m`: CUDA/WGPU output mismatch at row 17"
            )),
            "Fix: direct benchmark gate must reject hidden case failures even when summary.failed is zero; failures={failures:?}"
        );
    }

    #[test]
    fn single_benchmark_report_rejects_stale_summary_passed_count() {
        let requirement = Requirement::required("wgpu-fallback", "wgpu fallback");
        let report = serde_json::json!({
            "selected_backend": "wgpu",
            "summary": {"total_cases": 1, "passed": 0, "failed": 0},
            "cases": [
                {
                    "id": "release.condition_eval.1m",
                    "backend_id": "wgpu",
                    "status": "pass",
                    "contract": cpu_sota_baseline(&["wgpu"], 100.0),
                    "metrics": percentile_metrics([10, 11, 12], [2000, 2001, 2002]),
                    "performance": {"contract_passed": true, "speedup_x": 200.0}
                }
            ]
        });
        let mut failures = Vec::new();

        check_single_benchmark_report(
            &requirement,
            Path::new("."),
            Path::new("wgpu-stale-passed.json"),
            &report,
            false,
            None,
            &mut failures,
        );

        assert!(
            failures.iter().any(|failure| failure.contains(
                "has invalid summary: summary total/pass/fail (Some(1)/Some(0)/Some(0)) contradicts case evidence (1/1/0)"
            )),
            "Fix: direct benchmark gate must reject stale summary.passed even when summary.failed is zero; failures={failures:?}"
        );
    }

    #[test]
    fn backend_gpu_probe_accepts_qualifying_device_in_multi_gpu_setup() {
        let matrix = serde_json::json!({
            "gpu_probe": {
                "nvidia_smi_ok": true,
                "nvidia_smi_devices": ["GPU 0: sub-floor-device", "GPU 1: qualifying-device"],
                "nvidia_driver_version": "580.0",
                "nvidia_cuda_version": "13.0",
                "nvidia_smi_device_details": [
                    mock_sub_floor_gpu_device(),
                    mock_qualifying_gpu_device(),
                ]
            }
        });
        let mut failures = Vec::new();
        check_backend_gpu_probe("cuda-first-path", &matrix, &mut failures);
        assert!(
            failures.is_empty(),
            "Fix: GPU probe must accept setup where at least one device meets release floors; failures={failures:?}"
        );
    }

    #[test]
    fn backend_gpu_probe_rejects_sub_floor_gpu() {
        let matrix = serde_json::json!({
            "gpu_probe": {
                "nvidia_smi_ok": true,
                "nvidia_smi_devices": ["GPU 0: sub-floor-device"],
                "nvidia_driver_version": "580.0",
                "nvidia_cuda_version": "13.0",
                "nvidia_smi_device_details": [
                    mock_sub_floor_gpu_device(),
                ]
            }
        });
        let mut failures = Vec::new();
        check_backend_gpu_probe("cuda-first-path", &matrix, &mut failures);
        assert!(
            failures.iter().any(|failure| failure.contains(
                "must include a CUDA GPU with >=16384 MiB VRAM and compute capability >=8.0"
            )),
            "Fix: GPU probe must reject setups where no device meets release floors; failures={failures:?}"
        );
    }

    #[test]
    fn preferred_backend_gpu_only_rejects_cpu_ref() {
        let matrix = serde_json::json!({
            "preferred_backend_gpu_only": false,
            "preferred_backend_id": "cpu-ref"
        });
        let mut failures = Vec::new();
        check_preferred_backend_gpu_only("cuda-first-path", &matrix, &mut failures);
        assert!(
            failures
                .iter()
                .any(|failure| failure
                    .contains("must prove preferred runtime acquisition is GPU-only")),
            "Fix: release gate must reject preferred_backend_gpu_only=false; failures={failures:?}"
        );
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("must be cuda or wgpu, never cpu-ref/reference")),
            "Fix: release gate must reject cpu-ref as preferred backend; failures={failures:?}"
        );
    }
}
pub(crate) fn require_no_hidden_backend_fallback_findings(
    requirement_id: &str,
    matrix: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    let Some(scan_errors) = matrix
        .get("hidden_fallback_scan_errors")
        .and_then(serde_json::Value::as_array)
    else {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix is missing hidden_fallback_scan_errors"
        ));
        return;
    };
    if !scan_errors.is_empty() {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix reports {} hidden fallback scan error(s)",
            scan_errors.len()
        ));
    }
    let Some(findings) = matrix
        .get("hidden_fallback_findings")
        .and_then(serde_json::Value::as_array)
    else {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix is missing hidden_fallback_findings"
        ));
        return;
    };
    if !findings.is_empty() {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix reports {} hidden fallback finding(s)",
            findings.len()
        ));
    }
}
pub(crate) fn check_backend_matrix_schema(
    requirement_id: &str,
    matrix: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    let schema_version = u64_field(matrix, "schema_version", 0);
    if schema_version < 2 {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix schema_version is {schema_version}, expected >= 2"
        ));
    }
}
pub(crate) fn check_backend_gpu_probe(
    requirement_id: &str,
    matrix: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    if matrix
        .get("gpu_probe")
        .and_then(|probe| probe.get("nvidia_smi_ok"))
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix did not prove nvidia-smi GPU visibility"
        ));
    }
    let devices = matrix
        .get("gpu_probe")
        .and_then(|probe| probe.get("nvidia_smi_devices"))
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if devices == 0 {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix lists zero nvidia-smi devices"
        ));
    }
    let release_floor_device = matrix
        .get("gpu_probe")
        .and_then(|probe| probe.get("nvidia_smi_device_details"))
        .and_then(serde_json::Value::as_array)
        .is_some_and(|devices| devices.iter().any(is_qualifying_gpu_device));
    if !release_floor_device {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix gpu_probe.nvidia_smi_device_details must include a CUDA GPU with >=16384 MiB VRAM and compute capability >=8.0"
        ));
    }
    for field in ["nvidia_driver_version", "nvidia_cuda_version"] {
        if matrix
            .get("gpu_probe")
            .and_then(|probe| probe.get(field))
            .and_then(serde_json::Value::as_str)
            .is_none_or(str::is_empty)
        {
            failures.push(format!(
                "requirement `{requirement_id}` backend matrix gpu_probe.{field} must be recorded"
            ));
        }
    }
}
pub(crate) fn check_backend_acquire_entry(
    requirement_id: &str,
    matrix: &serde_json::Value,
    backend_id: &str,
    failures: &mut Vec<String>,
) {
    let backends = matrix
        .get("backends")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    if !backends.iter().any(|backend| {
        backend.get("id").and_then(serde_json::Value::as_str) == Some(backend_id)
            && backend
                .get("dispatches")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
            && backend
                .get("acquire_ok")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
    }) {
        failures.push(format!(
            "requirement `{requirement_id}` backend `{backend_id}` must dispatch and acquire successfully"
        ));
    }
}
pub(crate) fn check_preferred_backend_gpu_only(
    requirement_id: &str,
    matrix: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    if matrix
        .get("preferred_backend_gpu_only")
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        failures.push(format!(
            "requirement `{requirement_id}` backend matrix must prove preferred runtime acquisition is GPU-only"
        ));
    }
    let preferred = matrix
        .get("preferred_backend_id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if !matches!(preferred, "cuda" | "wgpu") {
        failures.push(format!(
            "requirement `{requirement_id}` preferred_backend_id `{preferred}` must be cuda or wgpu, never cpu-ref/reference"
        ));
    }
}
