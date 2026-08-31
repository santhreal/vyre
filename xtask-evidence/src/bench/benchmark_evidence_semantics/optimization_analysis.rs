//! Whether the optimization analysis fixtures and before/after metrics prove
//! that an optimization actually fired.
//!
//! Each required fixture family names the A-item it proves and the counters that
//! must be non-zero for that claim to hold, and a before/after pair only counts
//! as a win when the after value is a real improvement rather than a repeat of
//! the before value.

use serde_json::Value;

use super::data::OPTIMIZATION_ANALYSIS_FIXTURE_FAMILIES;
use super::json_reader::{inspect_duplicate_object_rows, metric_p50_f64};

pub(crate) fn inspect_optimization_analysis_fixture(
    context: &str,
    value: &Value,
    failures: &mut Vec<String>,
) {
    let missing_required = value
        .get("missing_required_families")
        .and_then(Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_required != 0 {
        failures.push(format!(
            "{context}: missing_required_families has {missing_required} entrie(s), expected zero"
        ));
    }
    let total_fixture_cases = value
        .get("total_fixture_cases")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let total_triggered_cases = value
        .get("total_triggered_cases")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if total_fixture_cases < 512 || total_triggered_cases != total_fixture_cases {
        failures.push(format!(
            "{context}: total_fixture_cases={total_fixture_cases}, total_triggered_cases={total_triggered_cases}; needs 512 fully-triggered A13-A16 cases"
        ));
    }
    let Some(families) = value.get("families").and_then(Value::as_array) else {
        failures.push(format!("{context}: missing families array"));
        return;
    };
    inspect_duplicate_object_rows(
        context,
        value,
        "families",
        "family",
        "analysis fixture family rows",
        failures,
    );
    for (required, family_label, required_fields) in
        OPTIMIZATION_ANALYSIS_FIXTURE_FAMILIES.iter().copied()
    {
        let Some(family) = families
            .iter()
            .find(|family| family.get("family").and_then(Value::as_str) == Some(required))
        else {
            failures.push(format!(
                "{context}: missing analysis fixture family `{required}`"
            ));
            continue;
        };
        let cases = family.get("cases").and_then(Value::as_u64).unwrap_or(0);
        let triggered = family
            .get("triggered_cases")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let analysis_sites = family
            .get("analysis_sites")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if cases < 128 || triggered != cases || analysis_sites < cases {
            failures.push(format!(
                "{context}: analysis fixture `{required}` has cases={cases}, triggered_cases={triggered}, analysis_sites={analysis_sites}; needs at least 128 cases, every case triggered, and at least one analysis site per case"
            ));
        }
        for field in required_fields {
            if family.get(field).and_then(Value::as_u64).unwrap_or(0) == 0 {
                failures.push(format!(
                    "{context}: {family_label} fixture has zero `{field}`"
                ));
            }
        }
    }
}

pub(crate) fn benchmark_before_after_semantic_win(
    case_id: &str,
    metrics: Option<&serde_json::Map<String, Value>>,
) -> bool {
    let Some(metrics) = metrics else {
        return false;
    };
    match case_id {
        "foundation.optimizer.impact" => metric_p50_f64(metrics.get("optimizer_nodes_eliminated"))
            .is_some_and(|value| value > 0.0),
        _ => false,
    }
}
