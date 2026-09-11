//! Performance contract evaluation for one measured case.
//!
//! The driver runs a case and harvests its metrics; this module decides
//! whether those metrics satisfy the contract the case declares. Keeping the
//! decision here means a baseline rule and a device bound are read next to
//! each other rather than in the middle of the run loop.

use std::collections::BTreeMap;

use crate::api::case::{DeviceBound, PerformanceContract, PerformanceEvaluation};
use crate::api::metric::MetricStats;

/// State the case verdict.
///
/// WHY: a failed performance contract is a failed case whatever the producing
/// command asked for, so the status a reader tallies agrees with
/// `CaseReport::passes_summary_evidence` and with the printed pair.
pub(super) fn final_case_status(provisional: &str, performance: Option<&PerformanceEvaluation>) -> String {
    if performance.is_some_and(|performance| !performance.contract_passed) {
        "failed".to_string()
    } else {
        provisional.to_string()
    }
}

/// The device's own active time for one sample, in the order of preference the
/// backends report it.
///
/// `wall_ns` is the last resort rather than the first: it includes host
/// overhead and readback, so a case that reports device time is judged on
/// device time and only a case that reports none falls back to its wall clock.
pub(super) fn device_active_time(metrics: &BTreeMap<String, MetricStats>) -> Option<&MetricStats> {
    metrics
        .get("dispatch_ns")
        .filter(|stats| stats.p50 > 0)
        .or_else(|| {
            metrics
                .get("kernel_execute_ns")
                .filter(|stats| stats.p50 > 0)
        })
        .or_else(|| metrics.get("wall_ns").filter(|stats| stats.p50 > 0))
}

pub(crate) fn evaluate_contract(
    contract: &PerformanceContract,
    metrics: &BTreeMap<String, MetricStats>,
    backend_id: &str,
) -> PerformanceEvaluation {
    let active_gpu = device_active_time(metrics);
    let speedup_x = match (active_gpu, metrics.get("baseline_wall_ns")) {
        (Some(gpu), Some(cpu)) => Some(cpu.p50 as f64 / gpu.p50 as f64),
        _ => None,
    };
    let mut violations = Vec::new();
    let mut applicable_baselines = 0usize;
    for baseline in &contract.baselines {
        if !baseline.backend_ids.is_empty()
            && !baseline
                .backend_ids
                .iter()
                .any(|candidate| candidate == backend_id)
        {
            continue;
        }
        applicable_baselines += 1;
        match speedup_x {
            Some(speedup) if speedup >= baseline.min_speedup_x => {}
            Some(speedup) => violations.push(format!(
                "{} requires {:.2}x over {}, observed {:.2}x",
                contract.primitive, baseline.min_speedup_x, baseline.name, speedup
            )),
            None => violations.push(format!(
                "{} requires a measured steady-state speedup over {}, but dispatch_ns/kernel_execute_ns/wall_ns or baseline_wall_ns were incomplete",
                contract.primitive, baseline.name
            )),
        }
    }
    let applicable_bounds = evaluate_device_bounds(contract, metrics, &mut violations);
    if applicable_baselines == 0 && applicable_bounds == 0 {
        violations.push(format!(
            "{} has no performance baseline or device bound that applies to backend `{backend_id}`",
            contract.primitive
        ));
    }
    PerformanceEvaluation {
        speedup_x,
        contract_passed: violations.is_empty(),
        violations,
    }
}

/// Judge every device bound the contract carries, appending one violation per
/// bound that is missed or cannot be judged.
///
/// Returns how many bounds were judged. A bound whose metric is absent counts
/// as judged and violated: a device bound that silently does not apply is a
/// contract that certifies nothing.
pub(super) fn evaluate_device_bounds(
    contract: &PerformanceContract,
    metrics: &BTreeMap<String, MetricStats>,
    violations: &mut Vec<String>,
) -> usize {
    for bound in &contract.device_bounds {
        match bound {
            DeviceBound::MemoryBandwidthFractionOfPeak {
                min_fraction,
                derivation,
            } => {
                // `roofline_mem_pct_x1000` is a percentage scaled by 1000, so
                // 39174 is 39.174% of peak.
                match metrics.get("roofline_mem_pct_x1000") {
                    Some(stats) if stats.p50 > 0 => {
                        let observed = stats.p50 as f64 / 100_000.0;
                        if observed < *min_fraction {
                            violations.push(format!(
                                "{} requires at least {:.1}% of device memory bandwidth, observed {:.1}%. {derivation}",
                                contract.primitive,
                                min_fraction * 100.0,
                                observed * 100.0
                            ));
                        }
                    }
                    _ => violations.push(format!(
                        "{} requires a measured device memory bandwidth fraction, but `roofline_mem_pct_x1000` was absent: the case must state `device_bytes_moved` and the run must carry device memory-peak telemetry",
                        contract.primitive
                    )),
                }
            }
            DeviceBound::ActiveTimeCeilingNs {
                max_active_ns,
                derivation,
            } => match device_active_time(metrics) {
                Some(stats) => {
                    if stats.p50 > *max_active_ns {
                        violations.push(format!(
                            "{} requires device active time at or under {max_active_ns} ns, observed {} ns. {derivation}",
                            contract.primitive, stats.p50
                        ));
                    }
                }
                None => violations.push(format!(
                    "{} requires a measured device active time, but dispatch_ns/kernel_execute_ns/wall_ns were incomplete",
                    contract.primitive
                )),
            },
        }
    }
    contract.device_bounds.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(value: u64) -> MetricStats {
        MetricStats::single(value)
    }

    fn contract_for_backends(backends: &[&str], min_speedup_x: f64) -> PerformanceContract {
        PerformanceContract {
            primitive: "release workload".to_string(),
            baselines: vec![crate::api::case::BaselineTarget {
                name: "cpu sota".to_string(),
                crate_name: "vyre".to_string(),
                class: crate::api::case::BaselineClass::CpuSota,
                min_speedup_x,
                backend_ids: backends.iter().map(|backend| backend.to_string()).collect(),
            }],
            device_bounds: Vec::new(),
        }
    }

    fn bandwidth_contract(min_fraction: f64) -> PerformanceContract {
        PerformanceContract::memory_bandwidth_fraction(
            "bandwidth-bound workload",
            min_fraction,
            "measured 44.4% of peak; the gather floor is 74.9%",
        )
    }

    /// WHY: for a kernel already at the device's streaming ceiling, the
    /// property under test is how much of that bandwidth it uses. The bound
    /// must go red when the kernel stops being bandwidth-bound and must not
    /// move when the host baseline does, which is exactly what a CPU multiple
    /// on the same case could not do.
    ///
    /// Does not catch a wrong `device_bytes_moved`; that is the case's own
    /// accounting, asserted where the case builds it.
    #[test]
    fn a_memory_bandwidth_bound_is_judged_against_the_measured_fraction_of_peak() {
        let mut metrics = BTreeMap::new();
        metrics.insert("dispatch_ns".to_string(), stats(50_496));
        metrics.insert("roofline_mem_pct_x1000".to_string(), stats(44_431));

        let passing = evaluate_contract(&bandwidth_contract(0.30), &metrics, "cuda");
        assert!(
            passing.contract_passed,
            "Fix: 44.431% of peak must satisfy a 30% floor: {:?}",
            passing.violations
        );

        // The kernel loses coalescing and moves the same bytes 1.5x slower.
        let mut regressed = metrics.clone();
        regressed.insert("roofline_mem_pct_x1000".to_string(), stats(29_620));
        let failing = evaluate_contract(&bandwidth_contract(0.30), &regressed, "cuda");
        assert!(
            !failing.contract_passed,
            "Fix: 29.62% of peak must miss a 30% floor"
        );
        assert!(
            failing.violations[0].contains("29.6%")
                && failing.violations[0].contains("gather floor"),
            "Fix: the violation must state the observed fraction and the derivation: {:?}",
            failing.violations
        );

        // A host baseline moving by 100x cannot change the verdict.
        let mut with_slow_host = metrics.clone();
        with_slow_host.insert("baseline_wall_ns".to_string(), stats(50_496));
        assert!(
            evaluate_contract(&bandwidth_contract(0.30), &with_slow_host, "cuda").contract_passed,
            "Fix: a device bound must not read the host baseline."
        );
    }

    /// WHY: a bound whose metric is absent must fail, not silently pass. A
    /// resident case that states no `device_bytes_moved` publishes no
    /// bandwidth fraction, and a contract that certifies what it never
    /// measured is worse than no contract.
    #[test]
    fn a_device_bound_with_no_measurement_fails_closed() {
        let mut metrics = BTreeMap::new();
        metrics.insert("dispatch_ns".to_string(), stats(50_496));

        let evaluation = evaluate_contract(&bandwidth_contract(0.30), &metrics, "cuda");

        assert!(!evaluation.contract_passed);
        assert!(
            evaluation.violations[0].contains("roofline_mem_pct_x1000"),
            "Fix: the violation must name the missing metric: {:?}",
            evaluation.violations
        );
    }

    /// WHY: a latency case's quantity is nanoseconds, so its contract asserts
    /// nanoseconds. Against a host simulator faster than one launch, a ratio
    /// contract can never be met however fast the launch becomes.
    #[test]
    fn an_active_time_ceiling_is_judged_against_device_active_time() {
        let contract = PerformanceContract::active_time_ceiling_ns(
            "resident megakernel slot dispatch",
            5_000,
            "measured floor 2848 ns",
        );
        let mut metrics = BTreeMap::new();
        metrics.insert("dispatch_ns".to_string(), stats(3_904));
        // A host loop faster than a launch, which a ratio contract would read.
        metrics.insert("baseline_wall_ns".to_string(), stats(290));

        let passing = evaluate_contract(&contract, &metrics, "cuda");
        assert!(
            passing.contract_passed,
            "Fix: 3904 ns must satisfy a 5000 ns ceiling: {:?}",
            passing.violations
        );
        assert_eq!(
            passing.speedup_x,
            Some(290.0 / 3_904.0),
            "Fix: the measured ratio is still reported, it is just not asserted."
        );

        let mut regressed = metrics.clone();
        regressed.insert("dispatch_ns".to_string(), stats(6_000));
        let failing = evaluate_contract(&contract, &regressed, "cuda");
        assert!(!failing.contract_passed);
        assert!(
            failing.violations[0].contains("6000 ns") && failing.violations[0].contains("2848 ns"),
            "Fix: the violation must state the observed time and the derivation: {:?}",
            failing.violations
        );
    }

    /// WHY: the status a reader tallies in `cases[]` has to agree with
    /// `CaseReport::passes_summary_evidence`, which rejects a failed contract
    /// whatever the producing command asked for. A verdict that depended on a
    /// producer flag printed one pass count and recorded another.
    ///
    /// This does not catch a provisional status added without a decision: the
    /// provisional set is string literals at the call site, not a type.
    #[test]
    fn a_failed_performance_contract_is_a_failed_case_in_every_run() {
        let failed = PerformanceEvaluation {
            speedup_x: Some(99.0),
            contract_passed: false,
            violations: vec!["below contract".to_string()],
        };
        let passed = PerformanceEvaluation {
            speedup_x: Some(101.0),
            contract_passed: true,
            violations: Vec::new(),
        };

        for provisional in ["pass", "unstable", "thermal_unstable"] {
            assert_eq!(
                final_case_status(provisional, Some(&failed)),
                "failed",
                "Fix: a case that missed its performance contract must report status `failed` so the case list and the printed pass count state the same verdict."
            );
            assert_eq!(
                final_case_status(provisional, Some(&passed)),
                provisional,
                "Fix: a passing performance contract must preserve the provisional measurement status."
            );
            assert_eq!(
                final_case_status(provisional, None),
                provisional,
                "Fix: a case without a performance contract must preserve the provisional measurement status."
            );
        }
    }

    #[test]
    fn contract_fails_when_no_baseline_applies_to_backend() {
        let mut metrics = BTreeMap::new();
        metrics.insert("dispatch_ns".to_string(), stats(1));
        metrics.insert("baseline_wall_ns".to_string(), stats(1_000));

        let evaluation =
            evaluate_contract(&contract_for_backends(&["cuda"], 100.0), &metrics, "wgpu");

        assert_eq!(
            evaluation.speedup_x,
            Some(1_000.0),
            "Fix: speedup measurement should still be reported when the contract backend set is wrong."
        );
        assert!(
            !evaluation.contract_passed,
            "Fix: WGPU benchmark evidence must not pass by skipping a CUDA-only baseline."
        );
        assert!(
            evaluation
                .violations
                .iter()
                .any(|violation| violation.contains("no performance baseline")),
            "Fix: contract failures must explain that no baseline applies to the active backend."
        );
    }

    #[test]
    fn contract_with_empty_backend_ids_applies_to_every_backend() {
        let mut metrics = BTreeMap::new();
        metrics.insert("dispatch_ns".to_string(), stats(10));
        metrics.insert("baseline_wall_ns".to_string(), stats(1_000));

        let evaluation = evaluate_contract(&contract_for_backends(&[], 50.0), &metrics, "wgpu");

        assert!(
            evaluation.contract_passed,
            "Fix: backend-agnostic baselines must still apply to WGPU and other backends."
        );
        assert!(
            evaluation.violations.is_empty(),
            "Fix: backend-agnostic passing contracts must not accumulate baseline applicability violations."
        );
    }

    /// WHY: some CUDA event paths report a zero device duration while preserving a measured
    /// host wall duration. A zero dispatch sample is absence, not a zero-cost kernel, and must
    /// not hide the bounded wall-clock fallback used by the performance contract.
    #[test]
    fn contract_uses_wall_time_when_dispatch_duration_is_zero() {
        let mut metrics = BTreeMap::new();
        metrics.insert("dispatch_ns".to_string(), stats(0));
        metrics.insert("wall_ns".to_string(), stats(10));
        metrics.insert("baseline_wall_ns".to_string(), stats(1_000));

        let evaluation =
            evaluate_contract(&contract_for_backends(&["cuda"], 100.0), &metrics, "cuda");

        assert_eq!(evaluation.speedup_x, Some(100.0));
        assert!(evaluation.contract_passed, "{:?}", evaluation.violations);
    }

}
