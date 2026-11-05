//! Reading the arrays a backend suite report is made of.
//!
//! A suite names its artifacts twice, in `artifacts` and again in
//! `artifact_statuses`, and every cross-check between the two is phrased as a
//! count or a set keyed by family and case. Deriving those keys is the same
//! walk each time, so the parity, inventory and coverage checks share it from
//! here rather than each re-deciding what a suite row is keyed by.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use super::json_reader::non_empty_str;

pub(crate) fn report_status_for_path<'a>(suite: &'a Value, artifact: &str) -> Option<&'a Value> {
    suite
        .get("artifact_statuses")
        .and_then(Value::as_array)
        .and_then(|statuses| {
            statuses
                .iter()
                .find(|status| status.get("path").and_then(Value::as_str) == Some(artifact))
        })
}

pub(crate) fn suite_array_len(suite: &Value, field: &str) -> usize {
    suite
        .get(field)
        .and_then(Value::as_array)
        .map_or(0, Vec::len)
}

pub(crate) fn suite_artifact_status_count(suite: &Value) -> usize {
    suite_array_len(suite, "artifact_statuses")
}

pub(crate) fn suite_status_counts(suite: &Value, field: &str) -> BTreeMap<String, usize> {
    suite
        .get("artifact_statuses")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|status| status.get(field).and_then(non_empty_str))
        .fold(BTreeMap::new(), |mut counts, value| {
            *counts.entry(value.to_string()).or_default() += 1;
            counts
        })
}

pub(crate) fn suite_artifact_path_counts(suite: &Value) -> BTreeMap<String, usize> {
    suite
        .get("artifacts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(non_empty_str)
        .fold(BTreeMap::new(), |mut counts, path| {
            *counts.entry(path.to_string()).or_default() += 1;
            counts
        })
}

pub(crate) fn suite_all_artifact_paths(suite: &Value) -> BTreeSet<String> {
    suite_artifact_path_counts(suite)
        .into_keys()
        .chain(suite_status_counts(suite, "path").into_keys())
        .collect()
}

pub(crate) fn suite_family_case_pairs(suite: &Value) -> BTreeSet<(String, String)> {
    suite_family_case_pair_counts(suite).into_keys().collect()
}

pub(crate) fn suite_family_case_pair_counts(suite: &Value) -> BTreeMap<(String, String), usize> {
    let mut counts = BTreeMap::new();
    suite
        .get("artifact_statuses")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|status| {
            let family_id = status
                .get("family_id")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())?;
            let requested_case_id = status
                .get("requested_case_id")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())?;
            Some((family_id.to_string(), requested_case_id.to_string()))
        })
        .for_each(|pair| {
            *counts.entry(pair).or_insert(0) += 1;
        });
    counts
}

pub(crate) fn suite_statuses_by_family_case_pair(
    suite: &Value,
) -> BTreeMap<(String, String), &Value> {
    suite
        .get("artifact_statuses")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|status| {
            let family_id = status
                .get("family_id")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())?;
            let requested_case_id = status
                .get("requested_case_id")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())?;
            Some((
                (family_id.to_string(), requested_case_id.to_string()),
                status,
            ))
        })
        .collect()
}

pub(crate) fn suite_status_blockers(status: &Value) -> Option<Vec<String>> {
    status
        .get("blockers")
        .and_then(Value::as_array)
        .map(|blockers| {
            blockers
                .iter()
                .map(|blocker| {
                    blocker
                        .as_str()
                        .unwrap_or("<non-string blocker>")
                        .to_string()
                })
                .collect()
        })
}
