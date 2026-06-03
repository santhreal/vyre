use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum LaunchPlanLabelIssue {
    MissingSingle,
    SingleHasMulti,
    MissingMulti { launch_count: f64 },
    MultiHasSingle { launch_count: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum BackendConsistencyIssue {
    MissingCaseBackend {
        case_id: String,
        expected_backend: String,
    },
    CaseBackendMismatch {
        case_id: String,
        expected_backend: String,
        actual_backend: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CudaSourceCacheLabelIssue {
    MissingLabel { case_id: String },
    LabelWithoutCounters { case_id: String },
}

pub(crate) fn backend_consistency_issues(report: &Value) -> Vec<BackendConsistencyIssue> {
    let Some(expected_backend) = report
        .get("selected_backend")
        .and_then(Value::as_str)
        .filter(|backend| !backend.trim().is_empty())
    else {
        return Vec::new();
    };
    let Some(cases) = report.get("cases").and_then(Value::as_array) else {
        return Vec::new();
    };

    cases
        .iter()
        .filter_map(|case| {
            let case_id = case_id(case);
            match case
                .get("backend_id")
                .and_then(Value::as_str)
                .filter(|backend| !backend.trim().is_empty())
            {
                Some(actual_backend) if actual_backend == expected_backend => None,
                Some(actual_backend) => Some(BackendConsistencyIssue::CaseBackendMismatch {
                    case_id,
                    expected_backend: expected_backend.to_string(),
                    actual_backend: actual_backend.to_string(),
                }),
                None => Some(BackendConsistencyIssue::MissingCaseBackend {
                    case_id,
                    expected_backend: expected_backend.to_string(),
                }),
            }
        })
        .collect()
}

pub(crate) fn cuda_source_cache_label_issues(report: &Value) -> Vec<CudaSourceCacheLabelIssue> {
    if report.get("selected_backend").and_then(Value::as_str) != Some("cuda") {
        return Vec::new();
    }
    let Some(cases) = report.get("cases").and_then(Value::as_array) else {
        return Vec::new();
    };

    cases
        .iter()
        .filter_map(|case| {
            let metrics = case.get("metrics").and_then(Value::as_object);
            let counters_active = metric_value_any(
                metrics,
                &[
                    "cuda_ptx_source_cache_entries",
                    "cuda_ptx_source_cache_hits",
                    "cuda_ptx_source_cache_misses",
                ],
            )
            .is_some_and(|value| value > 0.0);
            let label_present = optimization_passes_contain(case, "cuda-ptx-source-cache");
            let case_id = case_id(case);
            match (counters_active, label_present) {
                (true, false) => Some(CudaSourceCacheLabelIssue::MissingLabel { case_id }),
                (false, true) => Some(CudaSourceCacheLabelIssue::LabelWithoutCounters { case_id }),
                _ => None,
            }
        })
        .collect()
}

pub(crate) fn launch_plan_label_issues(
    case: &Value,
    metrics: Option<&Map<String, Value>>,
) -> Vec<LaunchPlanLabelIssue> {
    let Some(launch_count) =
        metric_value_any(metrics, &["kernel_launches", "launch_count", "launches"])
    else {
        return Vec::new();
    };
    let has_single = optimization_passes_contain(case, "single-dispatch-launch-plan");
    let has_multi = optimization_passes_contain(case, "multi-dispatch-launch-plan");
    let mut issues = Vec::new();
    if launch_count == 1.0 {
        if !has_single {
            issues.push(LaunchPlanLabelIssue::MissingSingle);
        }
        if has_multi {
            issues.push(LaunchPlanLabelIssue::SingleHasMulti);
        }
    } else if launch_count > 1.0 {
        if !has_multi {
            issues.push(LaunchPlanLabelIssue::MissingMulti { launch_count });
        }
        if has_single {
            issues.push(LaunchPlanLabelIssue::MultiHasSingle { launch_count });
        }
    }
    issues
}

fn metric_value_any(metrics: Option<&Map<String, Value>>, fields: &[&str]) -> Option<f64> {
    let metrics = metrics?;
    fields
        .iter()
        .filter_map(|field| metrics.get(*field))
        .find_map(metric_value)
}

fn metric_value(metric: &Value) -> Option<f64> {
    metric
        .get("p50")
        .and_then(Value::as_f64)
        .or_else(|| {
            metric
                .get("p50")
                .and_then(Value::as_u64)
                .map(|value| value as f64)
        })
        .or_else(|| metric.as_f64())
        .or_else(|| metric.as_u64().map(|value| value as f64))
}

fn optimization_passes_contain(case: &Value, expected: &str) -> bool {
    ["optimization_passes_applied", "optimization_passes"]
        .iter()
        .any(|field| {
            case.get(*field)
                .and_then(Value::as_array)
                .is_some_and(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .any(|item| item == expected)
                })
        })
}

fn case_id(case: &Value) -> String {
    case.get("id")
        .and_then(Value::as_str)
        .unwrap_or("<unknown>")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_plan_issues_reject_single_label_for_multi_launch_count() {
        let case = serde_json::json!({
            "optimization_passes_applied": ["single-dispatch-launch-plan"],
            "metrics": {
                "kernel_launches": {"p50": 4, "samples": 30}
            }
        });
        let issues =
            launch_plan_label_issues(&case, case.get("metrics").and_then(Value::as_object));

        assert_eq!(
            issues,
            vec![
                LaunchPlanLabelIssue::MissingMulti { launch_count: 4.0 },
                LaunchPlanLabelIssue::MultiHasSingle { launch_count: 4.0 },
            ],
            "Fix: multi-launch evidence must require the multi label and reject the single label."
        );
    }

    #[test]
    fn launch_plan_issues_accept_matching_single_and_multi_counts() {
        for case in [
            serde_json::json!({
                "optimization_passes_applied": ["single-dispatch-launch-plan"],
                "metrics": {"kernel_launches": {"p50": 1, "samples": 30}}
            }),
            serde_json::json!({
                "optimization_passes_applied": ["multi-dispatch-launch-plan"],
                "metrics": {"launch_count": 4}
            }),
        ] {
            let issues =
                launch_plan_label_issues(&case, case.get("metrics").and_then(Value::as_object));
            assert!(
                issues.is_empty(),
                "Fix: matching launch-plan label/count evidence should pass: {issues:?}"
            );
        }
    }

    #[test]
    fn backend_consistency_rejects_case_backend_drift() {
        let report = serde_json::json!({
            "selected_backend": "cuda",
            "cases": [
                {"id": "same", "backend_id": "cuda"},
                {"id": "fallback", "backend_id": "wgpu"},
                {"id": "missing"}
            ]
        });

        assert_eq!(
            backend_consistency_issues(&report),
            vec![
                BackendConsistencyIssue::CaseBackendMismatch {
                    case_id: "fallback".to_string(),
                    expected_backend: "cuda".to_string(),
                    actual_backend: "wgpu".to_string(),
                },
                BackendConsistencyIssue::MissingCaseBackend {
                    case_id: "missing".to_string(),
                    expected_backend: "cuda".to_string(),
                },
            ],
            "Fix: report-level backend selection must be proven by every benchmark case."
        );
    }

    #[test]
    fn backend_consistency_allows_non_benchmark_manifest_without_selected_backend() {
        let manifest = serde_json::json!({
            "cases": [
                {"id": "manifest-row"}
            ]
        });

        assert!(
            backend_consistency_issues(&manifest).is_empty(),
            "Fix: backend consistency applies to benchmark reports that declare selected_backend."
        );
    }

    #[test]
    fn cuda_source_cache_labels_track_active_counters() {
        let report = serde_json::json!({
            "selected_backend": "cuda",
            "cases": [
                {
                    "id": "active-unlabeled",
                    "metrics": {"cuda_ptx_source_cache_misses": {"p50": 1}},
                    "optimization_passes_applied": ["cuda-explicit-backend-selection"]
                },
                {
                    "id": "inactive-labeled",
                    "metrics": {
                        "cuda_ptx_source_cache_entries": {"p50": 0},
                        "cuda_ptx_source_cache_hits": {"p50": 0},
                        "cuda_ptx_source_cache_misses": {"p50": 0}
                    },
                    "optimization_passes_applied": ["cuda-ptx-source-cache"]
                },
                {
                    "id": "active-labeled",
                    "metrics": {"cuda_ptx_source_cache_hits": {"p50": 2}},
                    "optimization_passes_applied": ["cuda-ptx-source-cache"]
                }
            ]
        });

        assert_eq!(
            cuda_source_cache_label_issues(&report),
            vec![
                CudaSourceCacheLabelIssue::MissingLabel {
                    case_id: "active-unlabeled".to_string()
                },
                CudaSourceCacheLabelIssue::LabelWithoutCounters {
                    case_id: "inactive-labeled".to_string()
                },
            ],
            "Fix: CUDA source-cache labels must match measured source-cache counters."
        );
    }
}
