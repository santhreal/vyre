//! JSON / textual report formatting. Public entry point exported via
//! `super::print_report`.

use crate::report::json::{generate_json_report, ReportSchema, ReportSummary};

use super::stats::{format_scaled_metric, format_scaled_percent};

pub fn print_report(
    report: &ReportSchema,
    format: &str,
    roofline_only: bool,
) -> Result<(), serde_json::Error> {
    if format == "json" {
        println!("{}", generate_json_report(report)?);
        return Ok(());
    }

    if roofline_only {
        println!(
            "{:<30} | {:<10} | {:<10} | {:<10}",
            "Benchmark", "Status", "GB/s", "Roofline%"
        );
        println!("---------------------------------------------------------------------");
    } else {
        println!(
            "{:<30} | {:<10} | {:<12} | {:<12} | {:<12} | {:<13} | {:<12} | {:<12} | {:<10} | {:<10} | {:<10} | {:<10}",
            "Benchmark",
            "Status",
            "GPU p50(ns)",
            "GPU p99(ns)",
            "GPU p99.9(ns)",
            "GPU p99.99(ns)",
            "GPU Max(ns)",
            "CPU p50(ns)",
            "Speedup",
            "GB/s",
            "GFLOP/s",
            "Roofline%"
        );
        println!(
            "------------------------------------------------------------------------------------------------------------------------------------------------------------"
        );
    }
    for case in &report.cases {
        let gpu_stats = case
            .metrics
            .get("dispatch_ns")
            .or_else(|| case.metrics.get("wall_ns"));
        let gpu_p50 = gpu_stats.map(|stats| stats.p50);
        let gpu_p99 = gpu_stats.map(|stats| stats.p99);
        let gpu_p999 = gpu_stats.map(|stats| stats.p999);
        let gpu_p9999 = gpu_stats.map(|stats| stats.p9999);
        let gpu_max = gpu_stats.map(|stats| stats.max);
        let cpu_p50 = case.metrics.get("baseline_wall_ns").map(|stats| stats.p50);

        let gpu_p50_str = gpu_p50
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        let gpu_p99_str = gpu_p99
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        let gpu_p999_str = gpu_p999
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        let gpu_p9999_str = gpu_p9999
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        let gpu_max_str = gpu_max
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        let cpu_p50_str = cpu_p50
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".to_string());
        let speedup_str = match (gpu_p50, cpu_p50) {
            (Some(gpu), Some(cpu)) if gpu > 0 => format!("{:.1}x", cpu as f64 / gpu as f64),
            _ => "-".to_string(),
        };
        let gb_s = format_scaled_metric(
            case.metrics
                .get("device_gb_s_x1000")
                .or_else(|| case.metrics.get("wall_gb_s_x1000"))
                .map(|stats| stats.p50),
        );
        let gflops = format_scaled_metric(case.metrics.get("gflops_x1000").map(|stats| stats.p50));
        let roofline = format_scaled_percent(
            case.metrics
                .get("roofline_mem_pct_x1000")
                .map(|stats| stats.p50),
        );
        if roofline_only {
            println!(
                "{:<30} | {:<10} | {:<10} | {:<10}",
                case.id, case.status, gb_s, roofline
            );
        } else {
            println!(
                "{:<30} | {:<10} | {:<12} | {:<12} | {:<12} | {:<13} | {:<12} | {:<12} | {:<10} | {:<10} | {:<10} | {:<10}",
                case.id,
                case.status,
                gpu_p50_str,
                gpu_p99_str,
                gpu_p999_str,
                gpu_p9999_str,
                gpu_max_str,
                cpu_p50_str,
                speedup_str,
                gb_s,
                gflops,
                roofline
            );
        }
    }
    println!(
        "------------------------------------------------------------------------------------------------------------------------------------------------------------"
    );
    println!("{}", summary_line(&report.summary));
    Ok(())
}

/// Render the pass and fail line printed under the case table.
///
/// WHY: the printed pair is what a reader tallies against the `cases` array, so
/// it is rendered here from `ReportSummary` alone, which
/// `ReportSummary::from_cases` derives from that same array.
fn summary_line(summary: &ReportSummary) -> String {
    match summary.cache_hit_rate {
        Some(rate) => format!(
            "Passed: {}, Failed: {}, Cache Hit Rate: {:.1}%",
            summary.passed,
            summary.failed,
            rate * 100.0
        ),
        None => format!("Passed: {}, Failed: {}", summary.passed, summary.failed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::case::{Correctness, PerformanceEvaluation};
    use crate::report::fixture;
    use crate::report::json::CaseReport;

    /// Every combination of the three inputs the verdict reads.
    fn mixed_cases() -> Vec<CaseReport> {
        let mut cases = Vec::new();
        for status in ["pass", "unstable", "failed"] {
            for invalid in [false, true] {
                for performance in [None, Some(true), Some(false)] {
                    let mut case = fixture::case(
                        &format!("case.{status}.{invalid}.{performance:?}"),
                        &[("wall_ns", 10, 20)],
                    );
                    case.status = status.to_string();
                    if invalid {
                        case.correctness = Correctness::Invalid {
                            reason: "digest mismatch".to_string(),
                        };
                    }
                    case.performance = performance.map(|contract_passed| PerformanceEvaluation {
                        speedup_x: Some(10.0),
                        contract_passed,
                        violations: if contract_passed {
                            Vec::new()
                        } else {
                            vec!["speedup below floor".to_string()]
                        },
                    });
                    cases.push(case);
                }
            }
        }
        cases
    }

    /// WHY: the pass and fail pair printed under the table is what a reader
    /// believes, and it disagreed with the `cases` array because the runner
    /// counted while cases executed instead of counting the list. The tally is
    /// recomputed here from the case list for every combination of status,
    /// correctness, and contract outcome, so a second counting site or a
    /// divergent predicate turns this red.
    #[test]
    fn the_printed_pair_equals_the_case_tally() {
        let cases = mixed_cases();
        let expected_passed = cases
            .iter()
            .filter(|case| case.passes_summary_evidence())
            .count();
        let expected_failed = cases.len() - expected_passed;
        assert_eq!(
            expected_passed, 2,
            "Fix: only a case that passed, verified, and met its contract counts as passing evidence."
        );

        let summary = ReportSummary::from_cases(&cases, 1, None);

        assert_eq!(
            summary.total_cases,
            cases.len(),
            "Fix: summary.total_cases must count the case list the report carries."
        );
        assert_eq!(
            summary.passed + summary.failed,
            summary.total_cases,
            "Fix: every case must be counted exactly once as passed or failed."
        );
        assert_eq!(
            summary_line(&summary),
            format!("Passed: {expected_passed}, Failed: {expected_failed}"),
            "Fix: the printed pass and fail pair must equal the tally of the case list."
        );
    }

    /// WHY: a cache hit rate changes the line's shape, so the pair has to stay
    /// the case tally in that shape too.
    #[test]
    fn the_printed_pair_equals_the_case_tally_with_a_cache_hit_rate() {
        let cases = mixed_cases();
        let summary = ReportSummary::from_cases(&cases, 1, Some(0.25));
        assert_eq!(
            summary_line(&summary),
            format!(
                "Passed: {}, Failed: {}, Cache Hit Rate: 25.0%",
                summary.passed, summary.failed
            ),
            "Fix: the printed pass and fail pair must equal the tally of the case list."
        );
    }
}
