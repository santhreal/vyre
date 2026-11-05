//! Whether a baseline's class agrees with how that baseline was timed.
//!
//! WHY: `foundation.optimizer.impact` dispatches the same program twice, with
//! and without the semantic optimizer, and filed the pair as `CpuSota`. A class
//! is a provenance claim and a reader multiplies the number by it, so a 403x
//! self-comparison read as 403x over a host implementation of the primitive.
//! Nothing compared the label to the timing, and the timing was already in the
//! artifact: `baseline_dispatch_ns` exists only when the baseline arm was
//! dispatched on the device.
//!
//! The class is parsed into `BaselineClass` and placed by an exhaustive match,
//! so a class added to the enum does not compile until it says where its
//! baseline runs, and a class recorded in an artifact that the enum does not
//! declare is reported rather than skipped. The population is every case in the
//! artifact, judged against its own recorded metrics. What this cannot catch: a
//! host baseline that reaches a device through a dependency, which produces no
//! dispatch of ours to record.

use serde_json::Value;
use vyre_bench::api::case::BaselineClass;

use super::data::BaselineProvenanceIssue;
use super::json_reader::{case_id, metric_value_any};

/// Metric names a dispatched baseline arm records.
const BASELINE_DEVICE_TIMING: &[&str] = &["baseline_dispatch_ns"];

/// Where the baseline of a class runs. No catch-all arm: a new class is a
/// compile error here until someone decides which side it is on.
fn baseline_runs_on_device(class: &BaselineClass) -> bool {
    match class {
        BaselineClass::CpuSota => false,
        BaselineClass::GpuSota | BaselineClass::SelfUnoptimized => true,
    }
}

pub(crate) fn baseline_provenance_issues(report: &Value) -> Vec<BaselineProvenanceIssue> {
    let Some(cases) = report.get("cases").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut issues = Vec::new();
    for case in cases {
        let Some(baselines) = case
            .get("contract")
            .and_then(|contract| contract.get("baselines"))
            .and_then(Value::as_array)
        else {
            continue;
        };
        let timed_on_device = metric_value_any(
            case.get("metrics").and_then(Value::as_object),
            BASELINE_DEVICE_TIMING,
        )
        .is_some_and(|timing| timing > 0.0);
        for baseline in baselines {
            let Some(class) = baseline.get("class") else {
                continue;
            };
            let case_id = case_id(case);
            let Ok(parsed) = serde_json::from_value::<BaselineClass>(class.clone()) else {
                issues.push(BaselineProvenanceIssue::UnknownClass {
                    case_id,
                    class: class.to_string(),
                });
                continue;
            };
            let class = class.as_str().unwrap_or_default().to_string();
            match (baseline_runs_on_device(&parsed), timed_on_device) {
                (false, true) => issues
                    .push(BaselineProvenanceIssue::HostClassWithDeviceTiming { case_id, class }),
                (true, false) => {
                    issues.push(BaselineProvenanceIssue::DeviceClassWithoutDeviceTiming {
                        case_id,
                        class,
                    })
                }
                _ => {}
            }
        }
    }
    issues
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report_with(class: &Value, baseline_dispatch_ns: Option<u64>) -> Value {
        let mut metrics = serde_json::Map::new();
        if let Some(timing) = baseline_dispatch_ns {
            metrics.insert(
                "baseline_dispatch_ns".to_string(),
                serde_json::json!({"p50": timing, "samples": 30}),
            );
        }
        serde_json::json!({
            "cases": [
                {
                    "id": "case.under.test",
                    "metrics": Value::Object(metrics),
                    "contract": {
                        "baselines": [{"class": class, "min_speedup_x": 1.0}]
                    }
                }
            ]
        })
    }

    fn recorded_class(class: BaselineClass) -> Value {
        serde_json::to_value(class).expect("Fix: a baseline class must serialize.")
    }

    #[test]
    fn a_host_class_may_not_record_a_dispatched_baseline() {
        let class = recorded_class(BaselineClass::CpuSota);
        assert_eq!(
            baseline_provenance_issues(&report_with(&class, Some(5152))),
            vec![BaselineProvenanceIssue::HostClassWithDeviceTiming {
                case_id: "case.under.test".to_string(),
                class: "CpuSota".to_string(),
            }],
            "Fix: a CPU baseline that recorded device dispatch time was not timed on the host."
        );
        assert!(
            baseline_provenance_issues(&report_with(&class, None)).is_empty(),
            "Fix: a CPU baseline with no device timing is what the class claims."
        );
    }

    #[test]
    fn a_device_class_must_record_a_dispatched_baseline() {
        for class in [BaselineClass::GpuSota, BaselineClass::SelfUnoptimized] {
            let recorded = recorded_class(class);
            let name = recorded.as_str().unwrap_or_default().to_string();
            assert_eq!(
                baseline_provenance_issues(&report_with(&recorded, None)),
                vec![BaselineProvenanceIssue::DeviceClassWithoutDeviceTiming {
                    case_id: "case.under.test".to_string(),
                    class: name.clone(),
                }],
                "Fix: `{name}` claims a device baseline, so the case must record its dispatch."
            );
            assert!(
                baseline_provenance_issues(&report_with(&recorded, Some(1))).is_empty(),
                "Fix: `{name}` with a dispatched baseline is what the class claims."
            );
        }
    }

    #[test]
    fn a_class_the_tree_does_not_declare_is_unreadable_evidence() {
        let issues =
            baseline_provenance_issues(&report_with(&serde_json::json!("HandTuned"), Some(1)));
        assert_eq!(
            issues,
            vec![BaselineProvenanceIssue::UnknownClass {
                case_id: "case.under.test".to_string(),
                class: "\"HandTuned\"".to_string(),
            }],
            "Fix: a recorded class outside BaselineClass cannot be placed, so it must report."
        );
    }
}
