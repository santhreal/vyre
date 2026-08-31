//! Release bench targets, read from the declared target manifest.
//!
//! A family's speedup floor and baseline class are declared once in
//! `BENCH_TARGETS.toml`. Parsing rejects a row that omits either, so a
//! self-comparison is never counted as a host comparison.

use std::collections::{BTreeMap, BTreeSet};

use crate::api::case::BaselineClass;

pub(super) const BENCH_TARGETS: &str =
    include_str!("../../../docs/optimization/BENCH_TARGETS.toml");

/// One declared release-workload target: which case, which baseline, which floor.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct ReleaseBenchTarget {
    pub(super) id: String,
    pub(super) bench_case_id: String,
    pub(super) baseline_class: String,
    pub(super) min_speedup_over_baseline: f64,
}

impl ReleaseBenchTarget {
    /// Whether this target declares a host-baseline floor of at least `floor`.
    ///
    /// The class is half the claim. A floor over the same program without a
    /// transformation is not a host comparison however large the number is, so
    /// reading the threshold alone counted a self-comparison as CPU-SOTA.
    pub(super) fn declares_cpu_sota_floor(&self, floor: f64) -> bool {
        self.baseline_class == BaselineClass::CpuSota.registry_key()
            && self.min_speedup_over_baseline >= floor
    }
}

pub(super) fn release_bench_targets_from_manifest(
    text: &str,
) -> Result<Vec<ReleaseBenchTarget>, String> {
    let value = toml::from_str::<toml::Value>(text)
        .map_err(|error| format!("Fix: BENCH_TARGETS.toml must parse as TOML: {error}"))?;
    let targets = value
        .get("target")
        .and_then(toml::Value::as_array)
        .ok_or_else(|| "Fix: BENCH_TARGETS.toml must contain [[target]] rows.".to_string())?;
    let mut seen = BTreeSet::new();
    let mut rows = Vec::new();
    for target in targets.iter().filter(|target| {
        target.get("suite").and_then(toml::Value::as_str) == Some("release-workload")
    }) {
        let id = release_target_string(target, "id")?;
        if !seen.insert(id.clone()) {
            return Err(format!(
                "Fix: BENCH_TARGETS.toml contains duplicate release-workload target id `{id}`."
            ));
        }
        rows.push(ReleaseBenchTarget {
            id,
            bench_case_id: release_target_string(target, "bench_case_id")?,
            baseline_class: release_target_string(target, "baseline_class")?,
            min_speedup_over_baseline: release_target_number(target, "min_speedup_over_baseline")?,
        });
    }
    if rows.is_empty() {
        return Err(
            "Fix: BENCH_TARGETS.toml must define at least one suite=release-workload target."
                .to_string(),
        );
    }
    Ok(rows)
}

fn release_target_string(target: &toml::Value, key: &'static str) -> Result<String, String> {
    let id = target
        .get("id")
        .and_then(toml::Value::as_str)
        .unwrap_or("<missing id>");
    let value = target
        .get(key)
        .and_then(toml::Value::as_str)
        .unwrap_or("")
        .trim();
    if value.is_empty() {
        return Err(format!(
            "Fix: release-workload BENCH_TARGETS target `{id}` must declare non-empty `{key}`."
        ));
    }
    Ok(value.to_string())
}

fn release_target_number(target: &toml::Value, key: &'static str) -> Result<f64, String> {
    let id = target
        .get("id")
        .and_then(toml::Value::as_str)
        .unwrap_or("<missing id>");
    let value = target
        .get(key)
        .and_then(toml::Value::as_float)
        .or_else(|| {
            target
                .get(key)
                .and_then(toml::Value::as_integer)
                .map(|value| value as f64)
        })
        .ok_or_else(|| {
            format!(
                "Fix: release-workload BENCH_TARGETS target `{id}` must declare numeric `{key}`."
            )
        })?;
    if value <= 0.0 {
        return Err(format!(
            "Fix: release-workload BENCH_TARGETS target `{id}` numeric `{key}` must be positive."
        ));
    }
    Ok(value)
}

pub(super) fn release_bench_target_by_id(
    targets: &[ReleaseBenchTarget],
) -> BTreeMap<&str, &ReleaseBenchTarget> {
    targets
        .iter()
        .map(|target| (target.id.as_str(), target))
        .collect()
}
