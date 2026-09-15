//! Whether a benchmark case passed, and whether the run summary agrees with
//! the cases under it.
//!
//! A case carries a status, a correctness verdict and a performance contract
//! result, and any one of the three can contradict the other two. The reason a
//! case failed is decided once here, and the summary totals are recounted from
//! the cases so a stale `summary` block cannot report a clean run.

use serde_json::Value;

use super::json_reader::non_empty_str;

pub(crate) fn benchmark_case_failure_reason(case: &Value) -> Option<String> {
    let status = case.get("status").and_then(Value::as_str);
    let contract_failed = case
        .get("performance")
        .and_then(|performance| performance.get("contract_passed"))
        .and_then(Value::as_bool)
        == Some(false);
    let invalid_reason = case
        .get("correctness")
        .and_then(|correctness| correctness.get("Invalid"))
        .map(|invalid| {
            invalid
                .get("reason")
                .and_then(non_empty_str)
                .map(str::to_string)
                .unwrap_or_else(|| "invalid correctness".to_string())
        });
    let violation_reason = case
        .get("performance")
        .and_then(|performance| performance.get("violations"))
        .and_then(Value::as_array)
        .map(|violations| {
            violations
                .iter()
                .filter_map(non_empty_str)
                .collect::<Vec<_>>()
        })
        .and_then(|violations| (!violations.is_empty()).then(|| violations.join("; ")));
    invalid_reason
        .or(violation_reason)
        .or_else(|| match status {
            Some("pass") => None,
            Some(status) if !status.is_empty() => Some(format!("status `{status}`")),
            _ => Some("missing pass status".to_string()),
        })
        .or_else(|| contract_failed.then(|| "performance contract failed".to_string()))
}

pub(crate) fn benchmark_case_passes_summary_evidence(case: &Value) -> bool {
    case.get("status").and_then(Value::as_str) == Some("pass")
        && benchmark_case_failure_reason(case).is_none()
}

pub(crate) fn benchmark_report_summary_matches_case_evidence(report: &Value) -> bool {
    benchmark_report_summary_case_evidence_mismatch(report).is_none()
}

pub(crate) fn benchmark_report_summary_case_evidence_mismatch(report: &Value) -> Option<String> {
    let Some(cases) = report.get("cases").and_then(Value::as_array) else {
        return Some("missing cases array".to_string());
    };
    let Some(summary) = report.get("summary") else {
        return Some("missing summary".to_string());
    };
    let passed = cases
        .iter()
        .filter(|case| benchmark_case_passes_summary_evidence(case))
        .count() as u64;
    let failed = cases.len() as u64 - passed;
    let summary_total_cases = summary.get("total_cases").and_then(Value::as_u64);
    let summary_passed = summary.get("passed").and_then(Value::as_u64);
    let summary_failed = summary.get("failed").and_then(Value::as_u64);
    if summary_total_cases == Some(cases.len() as u64)
        && summary_passed == Some(passed)
        && summary_failed == Some(failed)
    {
        return None;
    }
    Some(format!(
        "summary total/pass/fail ({summary_total_cases:?}/{summary_passed:?}/{summary_failed:?}) contradicts case evidence ({}/{passed}/{failed})",
        cases.len()
    ))
}

pub(crate) fn benchmark_failed_case_summaries(report: &Value) -> Vec<String> {
    report
        .get("cases")
        .and_then(Value::as_array)
        .map(|cases| {
            cases
                .iter()
                .filter_map(|case| {
                    let id = case
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("<unknown>");
                    benchmark_case_failure_reason(case).map(|reason| format!("`{id}`: {reason}"))
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_case_summary_rejects_pass_status_with_invalid_correctness() {
        let case = serde_json::json!({
            "id": "release.condition_eval.1m",
            "status": "pass",
            "correctness": {
                "Invalid": {
                    "reason": "dispatch output mismatch at row 17"
                }
            },
            "performance": {"contract_passed": true}
        });

        assert_eq!(
            benchmark_case_failure_reason(&case),
            Some("dispatch output mismatch at row 17".to_string()),
            "Fix: explicit invalid correctness evidence must not be hidden by a contradictory pass status."
        );
    }

    #[test]
    fn failed_case_summary_rejects_invalid_correctness_with_blank_reason() {
        let case = serde_json::json!({
            "id": "release.condition_eval.1m",
            "status": "pass",
            "correctness": {
                "Invalid": {
                    "reason": "   "
                }
            },
            "performance": {"contract_passed": true}
        });

        assert_eq!(
            benchmark_case_failure_reason(&case),
            Some("invalid correctness".to_string()),
            "Fix: blank invalid-correctness reasons must not let contradictory pass status prove release correctness."
        );
        assert!(
            !benchmark_case_passes_summary_evidence(&case),
            "Fix: invalid correctness must disqualify case summary evidence even when the reason is blank."
        );
    }

    #[test]
    fn failed_case_summary_rejects_pass_status_with_performance_violations() {
        let case = serde_json::json!({
            "id": "release.condition_eval.1m",
            "status": "pass",
            "correctness": {"Valid": {}},
            "performance": {
                "contract_passed": true,
                "violations": [
                    "speedup below CUDA release floor",
                    "p95 latency regression"
                ]
            }
        });

        assert_eq!(
            benchmark_case_failure_reason(&case),
            Some("speedup below CUDA release floor; p95 latency regression".to_string()),
            "Fix: performance violation evidence must stay visible even when status is pass."
        );
    }

    #[test]
    fn failed_case_summary_reports_contract_failed_pass_as_contract_failure() {
        let case = serde_json::json!({
            "id": "release.condition_eval.1m",
            "status": "pass",
            "correctness": {"Valid": {}},
            "performance": {"contract_passed": false}
        });

        assert_eq!(
            benchmark_case_failure_reason(&case),
            Some("performance contract failed".to_string()),
            "Fix: contradictory pass status must not hide contract_passed=false evidence."
        );
    }

    #[test]
    fn failed_case_summary_rejects_missing_pass_status() {
        let case = serde_json::json!({
            "id": "release.condition_eval.1m",
            "correctness": {"Valid": {}},
            "performance": {"contract_passed": true}
        });

        assert_eq!(
            benchmark_case_failure_reason(&case),
            Some("missing pass status".to_string()),
            "Fix: benchmark evidence must require an explicit pass status before a case can prove release performance."
        );
    }

    #[test]
    fn benchmark_report_summary_mismatch_reports_total_pass_fail_drift() {
        let report = serde_json::json!({
            "summary": {"total_cases": 2, "passed": 0, "failed": 0},
            "cases": [
                {
                    "id": "release.condition_eval.1m",
                    "status": "pass",
                    "correctness": {"Valid": {}},
                    "performance": {"contract_passed": true}
                }
            ]
        });

        assert_eq!(
            benchmark_report_summary_case_evidence_mismatch(&report),
            Some(
                "summary total/pass/fail (Some(2)/Some(0)/Some(0)) contradicts case evidence (1/1/0)"
                    .to_string()
            ),
            "Fix: benchmark summary validation must expose stale total_cases and passed counts, not only summary.failed."
        );
        assert!(
            !benchmark_report_summary_matches_case_evidence(&report),
            "Fix: stale benchmark summaries must not be accepted by boolean reuse/gate predicates."
        );
    }
}
