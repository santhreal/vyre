//! Whether a case proves the CPU-SOTA 100x claim, and how many in a report do.
//!
//! The claim needs three things at once: a baseline contract that applies to the
//! case's backend and demands the speedup, a measured speedup that reaches it,
//! and a pass status the case summary evidence actually supports. Counting
//! contract cases and passing cases in one pass is what keeps a report's
//! declared totals from drifting from its cases.

use serde_json::Value;

use super::case_summary::benchmark_case_passes_summary_evidence;
use super::json_reader::metric_p50_f64;

pub(crate) fn cpu_sota_100x_case_counts(artifact_report: &Value) -> (u64, u64) {
    let report_backend = artifact_report
        .get("selected_backend")
        .and_then(Value::as_str);
    let Some(cases) = artifact_report.get("cases").and_then(Value::as_array) else {
        return (0, 0);
    };
    cases
        .iter()
        .fold((0, 0), |(contract_count, passing_count), case| {
            let case_backend = case
                .get("backend_id")
                .and_then(Value::as_str)
                .or(report_backend);
            if !benchmark_case_has_cpu_sota_contract(case, case_backend, 100.0) {
                return (contract_count, passing_count);
            }
            (
                contract_count + 1,
                passing_count + u64::from(benchmark_case_proves_cpu_sota_100x(case, case_backend)),
            )
        })
}

pub(crate) fn inspect_cpu_sota_100x_case_count_consistency(
    context: &str,
    report: &Value,
    findings: &mut Vec<String>,
) {
    let (derived_contract_cases, derived_passing_cases) = cpu_sota_100x_case_counts(report);
    for (field, derived) in [
        ("cpu_sota_100x_contract_case_count", derived_contract_cases),
        ("cpu_sota_100x_passing_case_count", derived_passing_cases),
    ] {
        let declared = report.get(field).and_then(Value::as_u64).unwrap_or(0);
        if declared != derived {
            findings.push(format!(
                "{context} {field}={declared}, but cases prove {derived}"
            ));
        }
    }
}

pub(crate) fn benchmark_case_proves_cpu_sota_100x(case: &Value, backend_id: Option<&str>) -> bool {
    benchmark_case_has_cpu_sota_contract(case, backend_id, 100.0)
        && benchmark_case_passes_summary_evidence(case)
        && case
            .get("performance")
            .and_then(|performance| performance.get("contract_passed"))
            .and_then(Value::as_bool)
            == Some(true)
        && case
            .get("performance")
            .and_then(|performance| performance.get("speedup_x"))
            .and_then(Value::as_f64)
            .is_some_and(|speedup| speedup >= 100.0)
        && cpu_sota_100x_measured_speedup(case)
            .is_some_and(|measured_speedup| measured_speedup >= 100.0)
}

fn cpu_sota_100x_measured_speedup(case: &Value) -> Option<f64> {
    let metrics = case.get("metrics").and_then(Value::as_object)?;
    let active_gpu = metrics
        .get("dispatch_ns")
        .or_else(|| metrics.get("kernel_execute_ns"))
        .or_else(|| metrics.get("wall_ns"));
    let wall = metric_p50_f64(active_gpu)?;
    let baseline = metric_p50_f64(metrics.get("baseline_wall_ns"))?;
    (wall > 0.0).then_some(baseline / wall)
}

pub(crate) fn benchmark_case_has_cpu_sota_contract(
    case: &Value,
    backend_id: Option<&str>,
    required_speedup: f64,
) -> bool {
    case.get("contract")
        .and_then(|contract| contract.get("baselines"))
        .and_then(Value::as_array)
        .is_some_and(|baselines| {
            baselines.iter().any(|baseline| {
                baseline.get("class").and_then(Value::as_str) == Some("CpuSota")
                    && baseline
                        .get("min_speedup_x")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0)
                        >= required_speedup
                    && baseline_applies_to_backend(baseline, backend_id)
            })
        })
}

pub(crate) fn baseline_applies_to_backend(baseline: &Value, backend_id: Option<&str>) -> bool {
    let Some(backend_ids) = baseline.get("backend_ids").and_then(Value::as_array) else {
        return true;
    };
    if backend_ids.is_empty() {
        return true;
    }
    let Some(backend_id) = backend_id else {
        return false;
    };
    backend_ids
        .iter()
        .any(|candidate| candidate.as_str() == Some(backend_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report_fixture::{cpu_sota_baseline, cpu_sota_case};

    #[test]
    fn cpu_sota_contract_requires_matching_backend_id() {
        let case = serde_json::json!({
            "contract": {
                "baselines": [
                    {
                        "class": "CpuSota",
                        "backend_ids": ["cuda"],
                        "min_speedup_x": 100.0
                    }
                ]
            }
        });

        assert!(
            benchmark_case_has_cpu_sota_contract(&case, Some("cuda"), 100.0),
            "Fix: CUDA should count CUDA-scoped CpuSota contracts."
        );
        assert!(
            !benchmark_case_has_cpu_sota_contract(&case, Some("wgpu"), 100.0),
            "Fix: WGPU must not inherit CUDA-scoped CpuSota contract counters."
        );
    }

    #[test]
    fn cpu_sota_100x_case_counts_require_pass_summary_evidence() {
        let report = serde_json::json!({
            "selected_backend": "cuda",
            "cases": [
                cpu_sota_case("release.condition_eval.1m", "cuda", "pass", &["cuda"], 10, 2000),
                cpu_sota_case("release.entropy_window.1m", "cuda", "fail", &["cuda"], 10, 2000),
                {
                    "id": "release.wgpu-drift.1m",
                    "backend_id": "wgpu",
                    "status": "pass",
                    "contract": cpu_sota_baseline(&["cuda"], 100.0),
                    "performance": {"contract_passed": true, "speedup_x": 200.0}
                }
            ]
        });

        assert_eq!(
            cpu_sota_100x_case_counts(&report),
            (2, 1),
            "Fix: derived CPU-SOTA 100x counts must share one backend-aware, pass-evidence-aware primitive."
        );
    }

    #[test]
    fn cpu_sota_100x_case_counts_require_measured_speedup_evidence() {
        let report = serde_json::json!({
            "selected_backend": "cuda",
            "cases": [
                cpu_sota_case("release.claimed-speedup.1m", "cuda", "pass", &["cuda"], 100, 1000)
            ]
        });

        assert_eq!(
            cpu_sota_100x_case_counts(&report),
            (1, 0),
            "Fix: CPU-SOTA passing counts must be backed by measured baseline_wall_ns / wall_ns speedup, not only performance.speedup_x claims."
        );
    }

    #[test]
    fn cpu_sota_100x_case_counts_use_runner_active_gpu_metric_order() {
        let report = serde_json::json!({
            "selected_backend": "cuda",
            "cases": [
                {
                    "id": "release.dispatch-timed.1m",
                    "backend_id": "cuda",
                    "status": "pass",
                    "contract": cpu_sota_baseline(&["cuda"], 100.0),
                    "metrics": {
                        "dispatch_ns": {"p50": 10},
                        "wall_ns": {"p50": 2000},
                        "baseline_wall_ns": {"p50": 1500}
                    },
                    "performance": {"contract_passed": true, "speedup_x": 150.0}
                }
            ]
        });

        assert_eq!(
            cpu_sota_100x_case_counts(&report),
            (1, 1),
            "Fix: CPU-SOTA proof counts must mirror benchmark contract evaluation and prefer dispatch_ns before wall_ns."
        );
    }
}
