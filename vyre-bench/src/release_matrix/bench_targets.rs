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
    #[derive(serde::Deserialize)]
    struct BenchTargetsFile {
        #[serde(default)]
        target: Vec<RawBenchTarget>,
    }

    #[derive(serde::Deserialize)]
    struct RawBenchTarget {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        suite: Option<String>,
        #[serde(default)]
        bench_case_id: Option<String>,
        #[serde(default)]
        baseline_class: Option<String>,
        #[serde(default)]
        min_speedup_over_baseline: Option<f64>,
    }

    let file: BenchTargetsFile = toml::from_str(text)
        .map_err(|error| format!("Fix: BENCH_TARGETS.toml must parse as TOML: {error}"))?;
    if file.target.is_empty() {
        return Err("Fix: BENCH_TARGETS.toml must contain [[target]] rows.".to_string());
    }
    let mut seen = BTreeSet::new();
    let mut rows = Vec::new();
    for target in file
        .target
        .into_iter()
        .filter(|t| t.suite.as_deref() == Some("release-workload"))
    {
        let id = target.id.as_deref().unwrap_or("<missing id>");
        if target.id.is_none() || id.trim().is_empty() {
            return Err(format!(
                "Fix: release-workload BENCH_TARGETS target `{id}` must declare non-empty `id`."
            ));
        }
        let id_str = id.trim().to_string();
        if !seen.insert(id_str.clone()) {
            return Err(format!(
                "Fix: BENCH_TARGETS.toml contains duplicate release-workload target id `{id_str}`."
            ));
        }
        let bench_case_id = target.bench_case_id.filter(|s| !s.trim().is_empty()).ok_or_else(|| {
            format!("Fix: release-workload BENCH_TARGETS target `{id_str}` must declare non-empty `bench_case_id`.")
        })?;
        let baseline_class = target.baseline_class.filter(|s| !s.trim().is_empty()).ok_or_else(|| {
            format!("Fix: release-workload BENCH_TARGETS target `{id_str}` must declare non-empty `baseline_class`.")
        })?;
        let min_speedup = target.min_speedup_over_baseline.ok_or_else(|| {
            format!("Fix: release-workload BENCH_TARGETS target `{id_str}` must declare numeric `min_speedup_over_baseline`.")
        })?;
        if min_speedup <= 0.0 {
            return Err(format!("Fix: release-workload BENCH_TARGETS target `{id_str}` numeric `min_speedup_over_baseline` must be positive."));
        }
        rows.push(ReleaseBenchTarget {
            id: id_str,
            bench_case_id: bench_case_id.trim().to_string(),
            baseline_class: baseline_class.trim().to_string(),
            min_speedup_over_baseline: min_speedup,
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

pub(super) fn release_bench_target_by_id(
    targets: &[ReleaseBenchTarget],
) -> BTreeMap<&str, &ReleaseBenchTarget> {
    targets
        .iter()
        .map(|target| (target.id.as_str(), target))
        .collect()
}
