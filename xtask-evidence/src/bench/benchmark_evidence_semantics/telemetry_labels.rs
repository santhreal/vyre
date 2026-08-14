//! Whether the optimization labels a case declares are backed by its counters.
//!
//! A case lists the passes it claims fired, and the telemetry it recorded is the
//! only proof of any of them: a CUDA label with a zero counter is unproven, a
//! counter with no label is unattributed, a launch-plan label must agree with
//! the measured launch count, and the borrowed-resident escape hatch is
//! forbidden outright rather than merely labelled.

use serde_json::{Map, Value};

use super::data::{
    CudaForbiddenTelemetryIssue, CudaTelemetryLabelIssue, LaunchPlanLabelIssue,
    CUDA_TELEMETRY_CHECKS,
};
use super::json_reader::{case_id, metric_value_any, optimization_passes_contain};

pub(crate) fn cuda_telemetry_label_issues(report: &Value) -> Vec<CudaTelemetryLabelIssue> {
    if report.get("selected_backend").and_then(Value::as_str) != Some("cuda") {
        return Vec::new();
    }
    let Some(cases) = report.get("cases").and_then(Value::as_array) else {
        return Vec::new();
    };

    cases
        .iter()
        .flat_map(|case| {
            let metrics = case.get("metrics").and_then(Value::as_object);
            let case_id = case_id(case);
            CUDA_TELEMETRY_CHECKS
                .iter()
                .filter_map(move |(label, counters)| {
                    let counters_active =
                        metric_value_any(metrics, counters).is_some_and(|value| value > 0.0);
                    let label_present = optimization_passes_contain(case, label);
                    match (counters_active, label_present) {
                        (true, false) => Some(CudaTelemetryLabelIssue::MissingLabel {
                            case_id: case_id.clone(),
                            label,
                        }),
                        (false, true) => Some(CudaTelemetryLabelIssue::LabelWithoutCounters {
                            case_id: case_id.clone(),
                            label,
                        }),
                        _ => None,
                    }
                })
        })
        .collect()
}

pub(crate) fn cuda_forbidden_telemetry_issues(report: &Value) -> Vec<CudaForbiddenTelemetryIssue> {
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
            let observed_p50 =
                metric_value_any(metrics, &["cuda_resident_borrowed_fallback_dispatches"])?;
            (observed_p50 > 0.0).then(
                || CudaForbiddenTelemetryIssue::ResidentBorrowedEscapeHatch {
                    case_id: case_id(case),
                    observed_p50,
                },
            )
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
    fn cuda_telemetry_labels_track_active_counters() {
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
                },
                {
                    "id": "graph-unlabeled",
                    "metrics": {"cuda_graph_launches": {"p50": 3}},
                    "optimization_passes_applied": ["cuda-explicit-backend-selection"]
                },
                {
                    "id": "transfer-false-label",
                    "metrics": {
                        "cuda_host_upload_operations": {"p50": 0},
                        "cuda_device_readback_operations": {"p50": 0}
                    },
                    "optimization_passes_applied": ["cuda-transfer-operation-telemetry"]
                },
                {
                    "id": "resident-escape-unlabeled",
                    "metrics": {"cuda_resident_borrowed_fallback_dispatches": {"p50": 1}},
                    "optimization_passes_applied": ["cuda-explicit-backend-selection"]
                },
                {
                    "id": "resident-escape-false-label",
                    "metrics": {"cuda_resident_borrowed_fallback_dispatches": {"p50": 0}},
                    "optimization_passes_applied": ["cuda-resident-borrowed-escape-hatch"]
                }
            ]
        });

        assert_eq!(
            cuda_telemetry_label_issues(&report),
            vec![
                CudaTelemetryLabelIssue::MissingLabel {
                    case_id: "active-unlabeled".to_string(),
                    label: "cuda-ptx-source-cache",
                },
                CudaTelemetryLabelIssue::LabelWithoutCounters {
                    case_id: "inactive-labeled".to_string(),
                    label: "cuda-ptx-source-cache",
                },
                CudaTelemetryLabelIssue::MissingLabel {
                    case_id: "graph-unlabeled".to_string(),
                    label: "cuda-graph-replay",
                },
                CudaTelemetryLabelIssue::LabelWithoutCounters {
                    case_id: "transfer-false-label".to_string(),
                    label: "cuda-transfer-operation-telemetry",
                },
                CudaTelemetryLabelIssue::MissingLabel {
                    case_id: "resident-escape-unlabeled".to_string(),
                    label: "cuda-resident-borrowed-escape-hatch",
                },
                CudaTelemetryLabelIssue::LabelWithoutCounters {
                    case_id: "resident-escape-false-label".to_string(),
                    label: "cuda-resident-borrowed-escape-hatch",
                },
            ],
            "Fix: CUDA release telemetry labels must match measured backend counters."
        );
    }

    #[test]
    fn cuda_forbidden_telemetry_rejects_resident_borrowed_escape_hatch() {
        let report = serde_json::json!({
            "selected_backend": "cuda",
            "cases": [
                {
                    "id": "native-resident",
                    "metrics": {"cuda_resident_borrowed_fallback_dispatches": {"p50": 0}}
                },
                {
                    "id": "borrowed-escape",
                    "metrics": {"cuda_resident_borrowed_fallback_dispatches": {"p50": 2}}
                }
            ]
        });

        assert_eq!(
            cuda_forbidden_telemetry_issues(&report),
            vec![CudaForbiddenTelemetryIssue::ResidentBorrowedEscapeHatch {
                case_id: "borrowed-escape".to_string(),
                observed_p50: 2.0,
            }],
            "Fix: CUDA benchmark evidence must not pass when resident dispatch used the host-buffer escape hatch."
        );
    }
}
