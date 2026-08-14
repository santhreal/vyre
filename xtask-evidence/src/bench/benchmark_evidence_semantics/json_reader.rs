//! Reading one value out of a benchmark report without deciding anything.
//!
//! These are the readers every other check is written on top of: what counts
//! as a present string, what counts as a digest, which percentile a metric
//! carries, and which values in an array repeat. They answer about the JSON,
//! never about the release, so a check that reaches for a field goes through
//! one of them instead of restating the `and_then` chain.

use std::collections::BTreeSet;

use serde_json::{Map, Value};

pub(crate) fn non_empty_str(value: &Value) -> Option<&str> {
    value.as_str().filter(|value| !value.trim().is_empty())
}

pub(crate) fn is_blake3_hex_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub(crate) fn nonnegative_json_number_as_u64(value: &Value) -> Option<u64> {
    value.as_u64().or_else(|| {
        value
            .as_f64()
            .filter(|value| *value >= 0.0)
            .map(|value| value as u64)
    })
}

pub(crate) fn metric_p50_f64(metric: Option<&Value>) -> Option<f64> {
    let metric = metric?;
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

fn metric_value(metric: &Value) -> Option<f64> {
    metric_p50_f64(Some(metric))
}

pub(crate) fn metric_value_any(
    metrics: Option<&Map<String, Value>>,
    fields: &[&str],
) -> Option<f64> {
    let metrics = metrics?;
    fields
        .iter()
        .filter_map(|field| metrics.get(*field))
        .find_map(metric_value)
}

pub(crate) fn metrics_has_zero_any(
    metrics: Option<&serde_json::Map<String, Value>>,
    fields: &[&str],
) -> bool {
    metrics.is_some_and(|metrics| {
        fields.iter().any(|field| {
            metrics
                .get(*field)
                .is_some_and(|value| metric_p50_f64(Some(value)) == Some(0.0))
        })
    })
}

pub(crate) fn metrics_has_any(
    metrics: Option<&serde_json::Map<String, Value>>,
    fields: &[&str],
) -> bool {
    metrics.is_some_and(|metrics| {
        fields.iter().any(|field| {
            metrics.get(*field).is_some_and(|value| {
                value
                    .get("samples")
                    .and_then(Value::as_u64)
                    .is_some_and(|samples| samples > 0)
                    || metric_p50_f64(Some(value)).is_some_and(|sample| sample > 0.0)
                    || value.as_u64().is_some()
                    || value.as_f64().is_some_and(|number| number >= 0.0)
            })
        })
    })
}

pub(crate) fn metrics_has_positive_any(
    metrics: Option<&serde_json::Map<String, Value>>,
    fields: &[&str],
) -> bool {
    metrics.is_some_and(|metrics| {
        fields.iter().any(|field| {
            metrics.get(*field).is_some_and(|value| {
                metric_p50_f64(Some(value)).is_some_and(|sample| sample > 0.0)
                    || value.as_u64().is_some_and(|number| number > 0)
                    || value.as_f64().is_some_and(|number| number > 0.0)
            })
        })
    })
}

pub(crate) fn case_id(case: &Value) -> String {
    case.get("id")
        .and_then(Value::as_str)
        .unwrap_or("<unknown>")
        .to_string()
}

pub(crate) fn optimization_passes_contain(case: &Value, expected: &str) -> bool {
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

pub(crate) fn duplicate_nonblank_string_array_values(
    value: &Value,
    field: &str,
) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    value
        .get(field)
        .and_then(Value::as_array)
        .map_or_else(BTreeSet::new, |items| {
            items
                .iter()
                .filter_map(non_empty_str)
                .filter_map(|item| {
                    if seen.insert(item.to_string()) {
                        None
                    } else {
                        Some(item.to_string())
                    }
                })
                .collect::<BTreeSet<_>>()
        })
}

pub(crate) fn duplicate_nonblank_object_array_field_values(
    value: &Value,
    array_field: &str,
    object_field: &str,
) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    value
        .get(array_field)
        .and_then(Value::as_array)
        .map_or_else(BTreeSet::new, |items| {
            items
                .iter()
                .filter_map(|item| item.get(object_field).and_then(non_empty_str))
                .filter_map(|item| {
                    if seen.insert(item.to_string()) {
                        None
                    } else {
                        Some(item.to_string())
                    }
                })
                .collect::<BTreeSet<_>>()
        })
}

fn append_duplicate_object_row_finding(
    value: &Value,
    array_field: &str,
    object_field: &str,
    context: &str,
    findings: &mut Vec<String>,
) {
    let duplicates = duplicate_nonblank_object_array_field_values(value, array_field, object_field);
    if !duplicates.is_empty() {
        findings.push(format!(
            "{context}: {}",
            duplicates.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
}

pub(crate) fn inspect_duplicate_object_rows(
    evidence: &str,
    value: &Value,
    array_field: &str,
    value_field: &str,
    label: &str,
    blockers: &mut Vec<String>,
) {
    let duplicates = duplicate_nonblank_object_array_field_values(value, array_field, value_field);
    if !duplicates.is_empty() {
        let duplicates = duplicates.into_iter().collect::<Vec<_>>().join(", ");
        blockers.push(format!("{evidence}: duplicate {label}: {duplicates}"));
    }
}

fn collect_nonblank_string_set(
    field: &str,
    items: &[Value],
    issues: &mut Vec<String>,
) -> BTreeSet<String> {
    let mut paths = BTreeSet::new();
    for (index, item) in items.iter().enumerate() {
        let Some(path) = item.as_str().filter(|path| !path.trim().is_empty()) else {
            issues.push(format!("{field}[{index}] is not a nonblank string"));
            continue;
        };
        if !paths.insert(path.to_string()) {
            issues.push(format!("{field} contains duplicate artifact `{path}`"));
        }
    }
    paths
}

pub(crate) fn artifact_string_set(
    value: &Value,
    array_field: &str,
    label: &str,
    missing_message: &str,
    issues: &mut Vec<String>,
) -> BTreeSet<String> {
    let Some(items) = value.get(array_field).and_then(Value::as_array) else {
        issues.push(missing_message.to_string());
        return BTreeSet::new();
    };
    collect_nonblank_string_set(label, items, issues)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_nonblank_string_array_values_reports_repeated_entries() {
        let report = serde_json::json!({
            "cpu_sota_100x_contract_cases": [
                "release.condition_eval.1m",
                "release.entropy_window.1m",
                "release.condition_eval.1m",
                " ",
                null,
                "release.entropy_window.1m"
            ]
        });

        assert_eq!(
            duplicate_nonblank_string_array_values(&report, "cpu_sota_100x_contract_cases"),
            BTreeSet::from([
                "release.condition_eval.1m".to_string(),
                "release.entropy_window.1m".to_string(),
            ]),
            "Fix: release aggregate proof arrays must expose duplicate nonblank ids without counting blank placeholders."
        );
    }

    #[test]
    fn duplicate_nonblank_object_array_field_values_reports_repeated_entries() {
        let report = serde_json::json!({
            "families": [
                {"family": "algebraic"},
                {"family": "predicate"},
                {"family": "algebraic"},
                {"family": " "},
                {"family": null},
                {"family": "predicate"}
            ]
        });

        assert_eq!(
            duplicate_nonblank_object_array_field_values(&report, "families", "family"),
            BTreeSet::from(["algebraic".to_string(), "predicate".to_string()]),
            "Fix: release manifest object arrays must expose duplicate nonblank ids without counting blank placeholders."
        );
    }
}
