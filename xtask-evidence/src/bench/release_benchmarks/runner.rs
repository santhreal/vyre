use std::fs;
use std::path::Path;
use std::process::Command;

use serde_json::Value;

use super::artifact_metrics::{read_text_bounded, suite_metric_percentile, suite_metric_samples};
use super::release_thresholds::{
    MAX_RELEASE_BENCHMARK_TEXT_BYTES, MIN_CUDA_RELEASE_COMPUTE_CAPABILITY_MAJOR,
    MIN_CUDA_RELEASE_COMPUTE_CAPABILITY_MINOR, MIN_CUDA_RELEASE_MEMORY_MIB,
};

const RELEASE_WARMUP_SAMPLES: usize = 300;

pub(super) fn run_named_benchmark(
    workspace_root: &Path,
    case_id: &str,
    backend: &str,
    output: &str,
    measured_samples: Option<usize>,
    sample_timeout_secs: u64,
) -> Result<(), String> {
    let owned_args = benchmark_command_args(
        case_id,
        backend,
        output,
        measured_samples,
        sample_timeout_secs,
    );
    let borrowed = owned_args.iter().map(String::as_str).collect::<Vec<_>>();
    run_command_status(workspace_root, &borrowed)
}

/// The command that measures one release case in a child process.
///
/// `--release` is part of the measurement, not a convenience. A debug build of
/// the harness runs the CPU baseline scan tens of times slower than the release
/// build while device time barely moves, so a suite measured without it reports
/// a speedup that is mostly the missing optimizer. Measured 2026-08-15 on one
/// CUDA host for `release.condition_eval.1m`: the debug child reported a CPU p50
/// of 215874264 ns against a GPU p50 of 110240 ns, a claimed 1958.2x, where the
/// release build of the same case on the same device reported 4422427 ns against
/// 26688 ns, which is 165.7x. Every workflow step that measures a case directly
/// already passes it.
pub(super) fn benchmark_command_args(
    case_id: &str,
    backend: &str,
    output: &str,
    measured_samples: Option<usize>,
    sample_timeout_secs: u64,
) -> Vec<String> {
    let mut args = vec![
        "run".to_string(),
        "-p".to_string(),
        "vyre-bench".to_string(),
        "--release".to_string(),
        "--quiet".to_string(),
        "--".to_string(),
        "run".to_string(),
        "--suite".to_string(),
        "release".to_string(),
        "--case".to_string(),
        case_id.to_string(),
        "--backend".to_string(),
        backend.to_string(),
        "--enforce-budgets".to_string(),
        "--output".to_string(),
        output.to_string(),
        "--sample-timeout-secs".to_string(),
        sample_timeout_secs.to_string(),
        "--warmup-samples".to_string(),
        RELEASE_WARMUP_SAMPLES.to_string(),
    ];
    if let Some(samples) = measured_samples {
        args.push("--measured-samples".to_string());
        args.push(samples.to_string());
    }
    args
}

pub(super) fn run_named_benchmark_if_needed(
    workspace_root: &Path,
    case_id: &str,
    backend: &str,
    output: &str,
    measured_samples: Option<usize>,
    sample_timeout_secs: u64,
    reuse_existing: bool,
) -> Result<(), String> {
    if reuse_existing
        && benchmark_artifact_is_reusable(workspace_root, backend, case_id, case_id, output, None)
    {
        return Ok(());
    }
    run_named_benchmark(
        workspace_root,
        case_id,
        backend,
        output,
        measured_samples,
        sample_timeout_secs,
    )
}

pub(super) fn benchmark_artifact_is_reusable(
    workspace_root: &Path,
    backend: &str,
    family_id: &str,
    case_id: &str,
    output: &str,
    required_cpu_sota_min_speedup: Option<f64>,
) -> bool {
    let path = workspace_root.join(output);
    let text = match read_text_bounded(&path, MAX_RELEASE_BENCHMARK_TEXT_BYTES) {
        Ok(text) => text,
        Err(_) => return false,
    };
    let Ok(report) = serde_json::from_str::<Value>(&text) else {
        return false;
    };
    let Some(report_source_fingerprint) = report
        .get("source_fingerprint")
        .and_then(Value::as_str)
        .filter(|fingerprint| !fingerprint.trim().is_empty())
    else {
        return false;
    };
    if !xtask::source_provenance::issues(report_source_fingerprint).is_empty() {
        return false;
    }
    let Some(report_source_tree_fingerprint) = report
        .get("source_tree_fingerprint")
        .and_then(Value::as_str)
        .filter(|fingerprint| !fingerprint.trim().is_empty())
    else {
        let current_git = vyre_bench::probes::capture_git_info_at(workspace_root);
        let current_source_fingerprint = vyre_bench::probes::source_fingerprint(&current_git);
        if report_source_fingerprint != current_source_fingerprint {
            return false;
        }
        return benchmark_artifact_report_shape_is_reusable(
            &report,
            backend,
            family_id,
            case_id,
            required_cpu_sota_min_speedup,
        );
    };
    let current_source_tree_fingerprint =
        vyre_bench::probes::source_tree_fingerprint_at(workspace_root);
    if report_source_tree_fingerprint != current_source_tree_fingerprint {
        return false;
    }
    benchmark_artifact_report_shape_is_reusable(
        &report,
        backend,
        family_id,
        case_id,
        required_cpu_sota_min_speedup,
    )
}

fn benchmark_artifact_report_shape_is_reusable(
    report: &Value,
    backend: &str,
    family_id: &str,
    case_id: &str,
    required_cpu_sota_min_speedup: Option<f64>,
) -> bool {
    if report.get("selected_backend").and_then(Value::as_str) != Some(backend) {
        return false;
    }
    let environment = report.get("environment");
    if environment
        .and_then(|environment| environment.get("build_profile"))
        .and_then(Value::as_str)
        != Some("release")
    {
        return false;
    }
    if backend == "cuda" {
        let Some(devices) = environment
            .and_then(|environment| environment.get("gpu_devices"))
            .and_then(Value::as_array)
        else {
            return false;
        };
        let has_qualifying_device = devices.iter().any(|device| {
            device
                .get("memory_total_mib")
                .and_then(Value::as_u64)
                .is_some_and(|mib| mib >= MIN_CUDA_RELEASE_MEMORY_MIB)
                && matches!(
                    (
                        device
                            .get("compute_capability_major")
                            .and_then(Value::as_u64),
                        device
                            .get("compute_capability_minor")
                            .and_then(Value::as_u64),
                    ),
                    (Some(major), Some(minor))
                        if (major, minor)
                            >= (
                                MIN_CUDA_RELEASE_COMPUTE_CAPABILITY_MAJOR,
                                MIN_CUDA_RELEASE_COMPUTE_CAPABILITY_MINOR,
                            )
                )
        });
        if !has_qualifying_device {
            return false;
        }
    }
    if report
        .get("summary")
        .and_then(|summary| summary.get("failed"))
        .and_then(Value::as_u64)
        != Some(0)
    {
        return false;
    }
    if !crate::bench::benchmark_evidence_semantics::benchmark_report_summary_matches_case_evidence(
        report,
    ) {
        return false;
    }
    if !crate::bench::benchmark_evidence_semantics::benchmark_failed_case_summaries(report)
        .is_empty()
    {
        return false;
    }
    let Some(cases) = report.get("cases").and_then(Value::as_array) else {
        return false;
    };
    if cases.len() != 1 {
        return false;
    }
    let case = &cases[0];
    if case.get("id").and_then(Value::as_str) != Some(case_id) {
        return false;
    }
    if case.get("backend_id").and_then(Value::as_str) != Some(backend) {
        return false;
    }
    if case.get("status").and_then(Value::as_str) != Some("pass") {
        return false;
    }
    if !case_has_reusable_timing_metrics(case) {
        return false;
    }
    if let Some(required_speedup) = required_cpu_sota_min_speedup {
        if !case_has_reusable_cpu_sota_contract(case, backend, required_speedup) {
            return false;
        }
    }
    if (family_id == "compound-fused-filter" || case_id == "compound.pipeline.fused_filter.1m")
        && !crate::bench::benchmark_evidence_semantics::benchmark_fused_execution_dag_issues(
            case_id, report,
        )
        .is_empty()
    {
        return false;
    }
    true
}

fn case_has_reusable_cpu_sota_contract(case: &Value, backend: &str, required_speedup: f64) -> bool {
    crate::bench::benchmark_evidence_semantics::benchmark_case_has_cpu_sota_contract(
        case,
        Some(backend),
        required_speedup,
    ) && crate::bench::benchmark_evidence_semantics::benchmark_case_claims_contract_win(
        case,
        required_speedup,
    ) && measured_speedup(case).is_some_and(|speedup| speedup >= required_speedup)
}

fn measured_speedup(case: &Value) -> Option<f64> {
    let metrics = case.get("metrics").and_then(Value::as_object)?;
    let wall = metrics
        .get("wall_ns")
        .and_then(|metric| suite_metric_percentile(Some(metric), "p50"))? as f64;
    let baseline = metrics
        .get("baseline_wall_ns")
        .and_then(|metric| suite_metric_percentile(Some(metric), "p50"))? as f64;
    (wall > 0.0).then_some(baseline / wall)
}

fn case_has_reusable_timing_metrics(case: &Value) -> bool {
    let Some(metrics) = case.get("metrics").and_then(Value::as_object) else {
        return false;
    };
    for metric_name in ["wall_ns", "baseline_wall_ns"] {
        if !metrics
            .get(metric_name)
            .and_then(|metric| suite_metric_samples(Some(metric)))
            .is_some_and(|samples| samples >= 30)
        {
            return false;
        }
        for percentile in ["p50", "p95", "p99"] {
            if !metrics
                .get(metric_name)
                .and_then(|metric| suite_metric_percentile(Some(metric), percentile))
                .is_some_and(|value| value > 0)
            {
                return false;
            }
        }
    }
    true
}

pub(super) fn copy_artifact(
    workspace_root: &Path,
    source: &str,
    target: &str,
) -> Result<(), String> {
    let source = workspace_root.join(source);
    let target = workspace_root.join(target);
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create `{}`: {error}", parent.display()))?;
    }
    fs::copy(&source, &target).map(|_| ()).map_err(|error| {
        format!(
            "failed to copy `{}` to `{}`: {error}",
            source.display(),
            target.display()
        )
    })
}

/// Bytes of a failed child's output carried into the error.
const MAX_CHILD_OUTPUT_BYTES: usize = 4096;

/// Run one child command, keeping its output out of this process's stdout.
///
/// A delegated gate's stdout is the report the parent parses, so a child that
/// inherits it writes into the middle of that protocol: the harness prints a
/// formatted result table, the parent reads the first line of the table as JSON,
/// and the gate is reported as having returned no report at all. The output is
/// captured and only a failure carries its tail, which is the only case a reader
/// needs it for.
pub(super) fn run_command_status(workspace_root: &Path, args: &[&str]) -> Result<(), String> {
    let runner = xtask::cargo_runner::binary(workspace_root);
    let status = Command::new(&runner)
        .args(args)
        .current_dir(workspace_root)
        .output();
    let display = format!("{} {}", runner.display(), args.join(" "));
    match status {
        Ok(output) if output.status.success() => Ok(()),
        Ok(output) => Err(format!(
            "Fix: `{display}` failed with {}: {}",
            output.status,
            child_output_tail(&output.stdout, &output.stderr)
        )),
        Err(error) => Err(format!(
            "Fix: failed to run `{display}`: {error}. Set VYRE_CARGO_RUNNER to the bounded workspace cargo wrapper if it is not named `cargo_full`."
        )),
    }
}

/// The last `MAX_CHILD_OUTPUT_BYTES` of what a failed child said, stderr first.
///
/// The bound is on the text the report carries, so the streams are joined
/// before the cut. Cutting each stream to the bound and then joining them lets
/// a child that writes on both put twice the bound into the report.
fn child_output_tail(stdout: &[u8], stderr: &[u8]) -> String {
    let mut text = String::new();
    for stream in [stderr, stdout] {
        let said = String::from_utf8_lossy(stream);
        let said = said.trim();
        if said.is_empty() {
            continue;
        }
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(said);
    }
    if text.is_empty() {
        return "the child said nothing".to_string();
    }
    let start = text.len().saturating_sub(MAX_CHILD_OUTPUT_BYTES);
    let start = text
        .char_indices()
        .map(|(index, _)| index)
        .find(|index| *index >= start)
        .unwrap_or(text.len());
    text.split_off(start)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report_fixture::hidden_invalid_case;
    use tempfile::TempDir;

    /// Release evidence is measured by an optimized harness.
    ///
    /// The command used to omit `--release`, so every artifact under
    /// `release/evidence/benchmarks` was measured by a debug build: the recorded
    /// CPU baseline was dominated by missing optimization and the reported
    /// speedup was an artifact of the build, not of the device. The workflow
    /// steps that measure a case directly always passed the flag, so the two
    /// paths disagreed by a factor of twelve on the same case and host.
    #[test]
    fn release_benchmark_commands_measure_an_optimized_build() {
        let args = benchmark_command_args(
            "release.condition_eval.1m",
            "cuda",
            "release/evidence/benchmarks/workload.json",
            Some(30),
            30,
        );
        let separator = args
            .iter()
            .position(|arg| arg == "--")
            .expect("Fix: the command must separate cargo flags from harness flags.");

        assert!(
            args[..separator].iter().any(|arg| arg == "--release"),
            "Fix: measure release evidence with an optimized build, got `{args:?}`."
        );
    }

    #[test]
    fn release_benchmark_commands_precondition_accelerator_clocks() {
        let args = benchmark_command_args(
            "nn.linear_4bit_affine_grouped.1m",
            "cuda",
            "release/evidence/benchmarks/quantized.json",
            Some(30),
            30,
        );
        let warmup_flag = args
            .iter()
            .position(|arg| arg == "--warmup-samples")
            .expect("Fix: release benchmark commands must set an explicit warmup count.");

        assert_eq!(
            args.get(warmup_flag + 1).map(String::as_str),
            Some("300"),
            "Fix: release evidence must precondition accelerator clocks before measured samples."
        );
    }

    #[test]
    fn cuda_reuse_accepts_a_qualifying_device_in_multi_gpu_inventory() {
        let report = serde_json::json!({
            "selected_backend": "cuda",
            "environment": {
                "build_profile": "release",
                "gpu_devices": [
                    {
                        "memory_total_mib": 8192,
                        "compute_capability_major": 6,
                        "compute_capability_minor": 1
                    },
                    {
                        "memory_total_mib": 24576,
                        "compute_capability_major": 8,
                        "compute_capability_minor": 9
                    }
                ]
            },
            "summary": {"total_cases": 1, "passed": 1, "failed": 0},
            "cases": [{
                "id": "release.condition_eval.1m",
                "backend_id": "cuda",
                "status": "pass",
                "metrics": reusable_timing_metrics()
            }]
        });

        assert!(
            benchmark_artifact_report_shape_is_reusable(
                &report,
                "cuda",
                "condition-eval",
                "release.condition_eval.1m",
                None,
            ),
            "Fix: CUDA artifact reuse must inspect the complete device inventory."
        );

        let mut missing_profile = report;
        missing_profile["environment"]
            .as_object_mut()
            .expect("Fix: the test report must contain environment provenance.")
            .remove("build_profile");
        assert!(
            !benchmark_artifact_report_shape_is_reusable(
                &missing_profile,
                "cuda",
                "condition-eval",
                "release.condition_eval.1m",
                None,
            ),
            "Fix: CUDA artifact reuse must reject missing release-profile provenance."
        );
    }

    /// Reuse must not bypass release-only evidence enrichment. The previous
    /// predicate accepted a valid timing report that lacked the fused DAG, so
    /// `--reuse-existing` could never repair the gated artifact.
    #[test]
    fn compound_reuse_rejects_artifact_without_fused_execution_dag() {
        let report = serde_json::json!({
            "selected_backend": "cuda",
            "summary": {"total_cases": 1, "passed": 1, "failed": 0},
            "cases": [{
                "id": "compound.pipeline.fused_filter.1m",
                "backend_id": "cuda",
                "status": "pass",
                "metrics": reusable_timing_metrics()
            }]
        });

        assert!(!benchmark_artifact_report_shape_is_reusable(
            &report,
            "cuda",
            "compound-fused-filter",
            "compound.pipeline.fused_filter.1m",
            None,
        ));
    }

    #[test]
    fn wgpu_reuse_accepts_matching_passed_artifact() {
        let dir = TempDir::new().expect("Fix: create temp workspace for WGPU reuse test.");
        write_benchmark_artifact(
            dir.path(),
            "release/evidence/benchmarks/wgpu-condition.json",
            reusable_wgpu_condition_artifact(&current_test_source_fingerprint(dir.path()), None),
        );

        assert!(
            benchmark_artifact_is_reusable(
                dir.path(),
                "wgpu",
                "condition-eval",
                "release.condition_eval.1m",
                "release/evidence/benchmarks/wgpu-condition.json",
                None,
            ),
            "Fix: --reuse-existing should skip valid WGPU fallback artifacts instead of rerunning parity benchmarks."
        );
    }

    #[test]
    fn reuse_prefers_matching_source_tree_fingerprint() {
        let dir = TempDir::new().expect("Fix: create temp workspace for source-tree reuse test.");
        write_benchmark_artifact(
            dir.path(),
            "release/evidence/benchmarks/wgpu-source-tree.json",
            reusable_wgpu_condition_artifact(
                "git:stale:dirty=false",
                Some(&current_test_source_tree_fingerprint(dir.path())),
            ),
        );

        assert!(
            benchmark_artifact_is_reusable(
                dir.path(),
                "wgpu",
                "condition-eval",
                "release.condition_eval.1m",
                "release/evidence/benchmarks/wgpu-source-tree.json",
                None,
            ),
            "Fix: reusable benchmark evidence should survive evidence-only commit changes via source_tree_fingerprint."
        );
    }

    #[test]
    fn cuda_reuse_rejects_optional_cpu_sota_contract_failure() {
        let dir =
            TempDir::new().expect("Fix: create temp workspace for optional contract reuse test.");
        write_benchmark_artifact(
            dir.path(),
            "release/evidence/benchmarks/workload-16-quantized-linear.json",
            serde_json::json!({
                "selected_backend": "cuda",
                "source_fingerprint": current_test_source_fingerprint(dir.path()),
                "summary": {"total_cases": 1, "passed": 1, "failed": 0},
                "cases": [
                    {
                        "id": "nn.linear_4bit_affine_grouped.1m",
                        "backend_id": "cuda",
                        "status": "pass",
                        "contract": cpu_sota_contract_json("cuda", 100.0),
                        "performance": {"contract_passed": false, "speedup_x": 99.0},
                        "metrics": reusable_timing_metrics()
                    }
                ]
            }),
        );

        assert!(
            !benchmark_artifact_is_reusable(
                dir.path(),
                "cuda",
                "quantized-linear",
                "nn.linear_4bit_affine_grouped.1m",
                "release/evidence/benchmarks/workload-16-quantized-linear.json",
                Some(100.0),
            ),
            "Fix: --reuse-existing must rerun optional CUDA CPU-SOTA artifacts when their published contract failed."
        );
    }

    #[test]
    fn cuda_reuse_accepts_optional_cpu_sota_contract_with_measured_speedup() {
        let dir =
            TempDir::new().expect("Fix: create temp workspace for optional contract reuse test.");
        write_benchmark_artifact(
            dir.path(),
            "release/evidence/benchmarks/workload-16-quantized-linear.json",
            serde_json::json!({
                "selected_backend": "cuda",
                "source_fingerprint": current_test_source_fingerprint(dir.path()),
                "environment": {
                    "build_profile": "release",
                    "gpu_devices": [{
                        "memory_total_mib": 24576,
                        "compute_capability_major": 8,
                        "compute_capability_minor": 9
                    }]
                },
                "summary": {"total_cases": 1, "passed": 1, "failed": 0},
                "cases": [
                    {
                        "id": "nn.linear_4bit_affine_grouped.1m",
                        "backend_id": "cuda",
                        "status": "pass",
                        "contract": cpu_sota_contract_json("cuda", 100.0),
                        "performance": {"contract_passed": true, "speedup_x": 100.0},
                        "metrics": reusable_timing_metrics()
                    }
                ]
            }),
        );

        assert!(
            benchmark_artifact_is_reusable(
                dir.path(),
                "cuda",
                "quantized-linear",
                "nn.linear_4bit_affine_grouped.1m",
                "release/evidence/benchmarks/workload-16-quantized-linear.json",
                Some(100.0),
            ),
            "Fix: --reuse-existing should accept optional CUDA CPU-SOTA artifacts only when contract and measured timing evidence agree."
        );
    }

    #[test]
    fn reuse_rejects_matching_source_tree_with_legacy_dirty_source_fingerprint() {
        let dir =
            TempDir::new().expect("Fix: create temp workspace for dirty source-tree reuse test.");
        write_benchmark_artifact(
            dir.path(),
            "release/evidence/benchmarks/wgpu-legacy-dirty-source.json",
            serde_json::json!({
                "selected_backend": "wgpu",
                "source_fingerprint": "git:abc123:dirty=true",
                "source_tree_fingerprint": current_test_source_tree_fingerprint(dir.path()),
                "summary": {"total_cases": 1, "passed": 1, "failed": 0},
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "wgpu",
                        "status": "pass",
                        "metrics": reusable_timing_metrics()
                    }
                ]
            }),
        );

        assert!(
            !benchmark_artifact_is_reusable(
                dir.path(),
                "wgpu",
                "condition-eval",
                "release.condition_eval.1m",
                "release/evidence/benchmarks/wgpu-legacy-dirty-source.json",
                None,
            ),
            "Fix: --reuse-existing must rerun artifacts whose dirty source_fingerprint lacks a worktree digest even when source_tree_fingerprint matches."
        );
    }

    #[test]
    fn reuse_rejects_matching_source_tree_without_source_fingerprint() {
        let dir = TempDir::new()
            .expect("Fix: create temp workspace for missing source provenance reuse test.");
        write_benchmark_artifact(
            dir.path(),
            "release/evidence/benchmarks/wgpu-missing-source-fingerprint.json",
            serde_json::json!({
                "selected_backend": "wgpu",
                "source_tree_fingerprint": current_test_source_tree_fingerprint(dir.path()),
                "summary": {"total_cases": 1, "passed": 1, "failed": 0},
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "wgpu",
                        "status": "pass",
                        "metrics": reusable_timing_metrics()
                    }
                ]
            }),
        );

        assert!(
            !benchmark_artifact_is_reusable(
                dir.path(),
                "wgpu",
                "condition-eval",
                "release.condition_eval.1m",
                "release/evidence/benchmarks/wgpu-missing-source-fingerprint.json",
                None,
            ),
            "Fix: --reuse-existing must rerun artifacts that cannot satisfy backend suite source_fingerprint provenance."
        );
    }

    #[test]
    fn wgpu_reuse_rejects_backend_or_case_drift() {
        let dir = TempDir::new().expect("Fix: create temp workspace for WGPU reuse drift test.");
        write_benchmark_artifact(
            dir.path(),
            "release/evidence/benchmarks/wgpu-with-cuda-backend.json",
            serde_json::json!({
                "selected_backend": "cuda",
                "source_fingerprint": current_test_source_fingerprint(dir.path()),
                "summary": {"total_cases": 1, "passed": 1, "failed": 0},
                "cases": [
                    {"id": "release.condition_eval.1m", "backend_id": "cuda", "status": "pass"}
                ]
            }),
        );
        write_benchmark_artifact(
            dir.path(),
            "release/evidence/benchmarks/wgpu-wrong-case.json",
            serde_json::json!({
                "selected_backend": "wgpu",
                "source_fingerprint": current_test_source_fingerprint(dir.path()),
                "summary": {"total_cases": 1, "passed": 1, "failed": 0},
                "cases": [
                    {"id": "release.other.1m", "backend_id": "wgpu", "status": "pass"}
                ]
            }),
        );

        assert!(
            !benchmark_artifact_is_reusable(
                dir.path(),
                "wgpu",
                "condition-eval",
                "release.condition_eval.1m",
                "release/evidence/benchmarks/wgpu-with-cuda-backend.json",
                None,
            ),
            "Fix: WGPU reuse must reject artifacts whose selected backend drifted to CUDA."
        );
        assert!(
            !benchmark_artifact_is_reusable(
                dir.path(),
                "wgpu",
                "condition-eval",
                "release.condition_eval.1m",
                "release/evidence/benchmarks/wgpu-wrong-case.json",
                None,
            ),
            "Fix: WGPU reuse must reject artifacts that do not contain the requested release case."
        );
    }

    #[test]
    fn reuse_rejects_passed_artifact_without_timing_metrics() {
        let dir =
            TempDir::new().expect("Fix: create temp workspace for missing metrics reuse test.");
        write_benchmark_artifact(
            dir.path(),
            "release/evidence/benchmarks/wgpu-missing-metrics.json",
            serde_json::json!({
                "selected_backend": "wgpu",
                "source_fingerprint": current_test_source_fingerprint(dir.path()),
                "summary": {"total_cases": 1, "passed": 1, "failed": 0},
                "cases": [
                    {"id": "release.condition_eval.1m", "backend_id": "wgpu", "status": "pass"}
                ]
            }),
        );

        assert!(
            !benchmark_artifact_is_reusable(
                dir.path(),
                "wgpu",
                "condition-eval",
                "release.condition_eval.1m",
                "release/evidence/benchmarks/wgpu-missing-metrics.json",
                None,
            ),
            "Fix: --reuse-existing must rerun pass-only artifacts that lack release timing metrics."
        );
    }

    #[test]
    fn reuse_rejects_multi_case_artifact_contamination() {
        let dir = TempDir::new().expect("Fix: create temp workspace for multi-case reuse test.");
        write_benchmark_artifact(
            dir.path(),
            "release/evidence/benchmarks/wgpu-multi-case.json",
            serde_json::json!({
                "selected_backend": "wgpu",
                "source_fingerprint": current_test_source_fingerprint(dir.path()),
                "summary": {"total_cases": 2, "passed": 2, "failed": 0},
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "wgpu",
                        "status": "pass",
                        "metrics": reusable_timing_metrics()
                    },
                    {
                        "id": "release.entropy_window.1m",
                        "backend_id": "wgpu",
                        "status": "pass",
                        "metrics": reusable_timing_metrics()
                    }
                ]
            }),
        );

        assert!(
            !benchmark_artifact_is_reusable(
                dir.path(),
                "wgpu",
                "condition-eval",
                "release.condition_eval.1m",
                "release/evidence/benchmarks/wgpu-multi-case.json",
                None,
            ),
            "Fix: --reuse-existing must rerun multi-case artifacts instead of contaminating one-workload backend suite rows."
        );
    }

    #[test]
    fn reuse_rejects_case_failure_hidden_by_summary_zero() {
        let dir =
            TempDir::new().expect("Fix: create temp workspace for hidden failure reuse test.");
        write_benchmark_artifact(
            dir.path(),
            "release/evidence/benchmarks/wgpu-hidden-failure.json",
            serde_json::json!({
                "selected_backend": "wgpu",
                "source_fingerprint": current_test_source_fingerprint(dir.path()),
                "summary": {"total_cases": 1, "passed": 1, "failed": 0},
                "cases": [hidden_invalid_case("release.condition_eval.1m", "wgpu", [])]
            }),
        );

        assert!(
            !benchmark_artifact_is_reusable(
                dir.path(),
                "wgpu",
                "condition-eval",
                "release.condition_eval.1m",
                "release/evidence/benchmarks/wgpu-hidden-failure.json",
                None,
            ),
            "Fix: --reuse-existing must rerun artifacts whose case evidence contradicts summary.failed and pass status."
        );
    }

    #[test]
    fn reuse_rejects_stale_summary_passed_count() {
        let dir = TempDir::new()
            .expect("Fix: create temp workspace for stale summary passed reuse test.");
        write_benchmark_artifact(
            dir.path(),
            "release/evidence/benchmarks/wgpu-stale-passed.json",
            serde_json::json!({
                "selected_backend": "wgpu",
                "source_fingerprint": current_test_source_fingerprint(dir.path()),
                "summary": {"total_cases": 1, "passed": 0, "failed": 0},
                "cases": [
                    {"id": "release.condition_eval.1m", "backend_id": "wgpu", "status": "pass"}
                ]
            }),
        );

        assert!(
            !benchmark_artifact_is_reusable(
                dir.path(),
                "wgpu",
                "condition-eval",
                "release.condition_eval.1m",
                "release/evidence/benchmarks/wgpu-stale-passed.json",
                None,
            ),
            "Fix: --reuse-existing must rerun artifacts whose summary.passed contradicts pass-status case evidence."
        );
    }

    #[test]
    fn reuse_rejects_stale_summary_total_cases() {
        let dir = TempDir::new()
            .expect("Fix: create temp workspace for stale summary total_cases reuse test.");
        write_benchmark_artifact(
            dir.path(),
            "release/evidence/benchmarks/wgpu-stale-total-cases.json",
            serde_json::json!({
                "selected_backend": "wgpu",
                "source_fingerprint": current_test_source_fingerprint(dir.path()),
                "summary": {"total_cases": 2, "passed": 1, "failed": 0},
                "cases": [
                    {"id": "release.condition_eval.1m", "backend_id": "wgpu", "status": "pass"}
                ]
            }),
        );

        assert!(
            !benchmark_artifact_is_reusable(
                dir.path(),
                "wgpu",
                "condition-eval",
                "release.condition_eval.1m",
                "release/evidence/benchmarks/wgpu-stale-total-cases.json",
                None,
            ),
            "Fix: --reuse-existing must rerun artifacts whose summary.total_cases contradicts the cases array."
        );
    }

    #[test]
    fn reuse_rejects_stale_source_fingerprint() {
        let dir = TempDir::new().expect("Fix: create temp workspace for stale source test.");
        write_benchmark_artifact(
            dir.path(),
            "release/evidence/benchmarks/wgpu-stale-source.json",
            serde_json::json!({
                "selected_backend": "wgpu",
                "source_fingerprint": "git:stale:dirty=false",
                "summary": {"total_cases": 1, "passed": 1, "failed": 0},
                "cases": [
                    {"id": "release.condition_eval.1m", "backend_id": "wgpu", "status": "pass"}
                ]
            }),
        );

        assert!(
            !benchmark_artifact_is_reusable(
                dir.path(),
                "wgpu",
                "condition-eval",
                "release.condition_eval.1m",
                "release/evidence/benchmarks/wgpu-stale-source.json",
                None,
            ),
            "Fix: --reuse-existing must rerun benchmark artifacts captured from a different source fingerprint."
        );
    }

    #[test]
    fn reuse_rejects_stale_source_tree_fingerprint() {
        let dir = TempDir::new().expect("Fix: create temp workspace for stale source-tree test.");
        write_benchmark_artifact(
            dir.path(),
            "release/evidence/benchmarks/wgpu-stale-source-tree.json",
            serde_json::json!({
                "selected_backend": "wgpu",
                "source_fingerprint": current_test_source_fingerprint(dir.path()),
                "source_tree_fingerprint": "source-tree-v1:stale",
                "summary": {"total_cases": 1, "passed": 1, "failed": 0},
                "cases": [
                    {"id": "release.condition_eval.1m", "backend_id": "wgpu", "status": "pass"}
                ]
            }),
        );

        assert!(
            !benchmark_artifact_is_reusable(
                dir.path(),
                "wgpu",
                "condition-eval",
                "release.condition_eval.1m",
                "release/evidence/benchmarks/wgpu-stale-source-tree.json",
                None,
            ),
            "Fix: source_tree_fingerprint must remain a real freshness gate, not only an optional annotation."
        );
    }

    fn current_test_source_fingerprint(workspace_root: &Path) -> String {
        let git = vyre_bench::probes::capture_git_info_at(workspace_root);
        vyre_bench::probes::source_fingerprint(&git)
    }

    fn current_test_source_tree_fingerprint(workspace_root: &Path) -> String {
        vyre_bench::probes::source_tree_fingerprint_at(workspace_root)
    }

    fn reusable_timing_metrics() -> Value {
        serde_json::json!({
            "wall_ns": {"samples": 30, "p50": 10, "p95": 11, "p99": 12},
            "baseline_wall_ns": {"samples": 30, "p50": 1000, "p95": 1001, "p99": 1002}
        })
    }

    fn cpu_sota_contract_json(backend: &str, min_speedup_x: f64) -> Value {
        serde_json::json!({
            "primitive": "fused grouped INT4 linear",
            "baselines": [
                {
                    "name": "Rayon-parallel packed INT4 affine dequantization oracle",
                    "crate_name": "rayon",
                    "class": "CpuSota",
                    "min_speedup_x": min_speedup_x,
                    "backend_ids": [backend]
                }
            ]
        })
    }

    fn reusable_wgpu_condition_artifact(
        source_fingerprint: &str,
        source_tree_fingerprint: Option<&str>,
    ) -> Value {
        let mut artifact = serde_json::json!({
            "selected_backend": "wgpu",
            "source_fingerprint": source_fingerprint,
            "environment": {
                "build_profile": "release"
            },
            "summary": {"total_cases": 1, "passed": 1, "failed": 0},
            "cases": [
                {
                    "id": "release.condition_eval.1m",
                    "backend_id": "wgpu",
                    "status": "pass",
                    "metrics": reusable_timing_metrics()
                }
            ]
        });
        if let Some(source_tree_fingerprint) = source_tree_fingerprint {
            artifact["source_tree_fingerprint"] = serde_json::json!(source_tree_fingerprint);
        }
        artifact
    }

    fn write_benchmark_artifact(workspace_root: &Path, relative: &str, value: Value) {
        let path = workspace_root.join(relative);
        fs::create_dir_all(
            path.parent()
                .expect("Fix: benchmark artifact test path must have a parent directory."),
        )
        .expect("Fix: create benchmark artifact test directory.");
        fs::write(&path, format!("{value}\n")).expect("Fix: write benchmark artifact test JSON.");
    }

    /// WHY: the tail of a failed child is cut by byte count, and a child that
    /// prints UTF-8 puts characters across that cut. Slicing on the raw offset
    /// returned `None` there, and the fallback handed back the whole stream, so
    /// the one case the bound exists for was the case it did not bound.
    #[test]
    fn a_child_tail_is_cut_at_a_character_boundary() {
        let said = "\u{20ac}".repeat(2000);
        assert_eq!(said.len(), 6000);
        assert!(!said.is_char_boundary(said.len() - MAX_CHILD_OUTPUT_BYTES));

        let tail = child_output_tail(&[], said.as_bytes());

        assert!(
            tail.len() <= MAX_CHILD_OUTPUT_BYTES,
            "Fix: the child tail must stay within its byte bound, got {}",
            tail.len()
        );
        assert!(said.ends_with(&tail));
        assert_eq!(tail.chars().next(), Some('\u{20ac}'));
    }

    /// WHY: the bound is on the text the report carries. Cutting stdout and
    /// stderr to the bound one at a time and joining them afterwards let a
    /// child that writes on both streams put twice the bound into the report,
    /// which is what a noisy failing benchmark does.
    #[test]
    fn a_child_tail_bounds_both_streams_together() {
        let out = "o".repeat(MAX_CHILD_OUTPUT_BYTES);
        let err = "e".repeat(MAX_CHILD_OUTPUT_BYTES);

        let tail = child_output_tail(out.as_bytes(), err.as_bytes());

        assert!(
            tail.len() <= MAX_CHILD_OUTPUT_BYTES,
            "Fix: the child tail must stay within its byte bound, got {}",
            tail.len()
        );
        assert!(
            tail.ends_with('o'),
            "Fix: the tail keeps what was said last."
        );
    }
}
