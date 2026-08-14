//! The artifact reading and metric folding every release benchmark inspector
//! shares.
//!
//! Three inspectors read the same artifact shape and fold the same eight
//! wall-clock minima out of it: the optimization benchmark manifest, the
//! backend suite, and the CPU-SOTA 100x proof. Each carried its own copy of
//! the fold, so a change to the sample floor or to a percentile blocker
//! reached whichever copy the author happened to be editing. One core now
//! owns the read, the parse, and the fold, and the callers keep only the
//! wording that is genuinely theirs.

use std::path::Path;

use serde_json::Value;

use super::release_thresholds::MAX_RELEASE_BENCHMARK_TEXT_BYTES;

/// Samples a metric needs before its percentiles count as release evidence.
const MIN_RELEASE_METRIC_SAMPLES: u64 = 30;

/// Read a release benchmark evidence file without trusting its length.
pub(super) fn read_text_bounded(path: &Path, max_bytes: u64) -> std::io::Result<String> {
    xtask::output_arg::read_text_bounded(path, max_bytes, "release benchmark evidence")
}

/// Read and parse one benchmark artifact, recording read and parse failures as
/// blockers.
///
/// The read error is returned as well because one caller reports it as its own
/// evidence field rather than only as a blocker.
pub(super) fn read_benchmark_report(
    path: &Path,
    blockers: &mut Vec<String>,
) -> (Value, Option<String>) {
    let (text, read_error) = match read_text_bounded(path, MAX_RELEASE_BENCHMARK_TEXT_BYTES) {
        Ok(text) => (text, None),
        Err(error) => {
            let message = error.to_string();
            blockers.push(format!("unreadable JSON: {error}"));
            (String::new(), Some(message))
        }
    };
    let report = if text.is_empty() {
        Value::Null
    } else {
        match serde_json::from_str::<Value>(&text) {
            Ok(report) => report,
            Err(error) => {
                blockers.push(format!("invalid JSON: {error}"));
                Value::Null
            }
        }
    };
    (report, read_error)
}

pub(super) fn suite_metric_samples(value: Option<&Value>) -> Option<u64> {
    value
        .and_then(|metric| metric.get("samples"))
        .and_then(Value::as_u64)
}

pub(super) fn suite_metric_percentile(value: Option<&Value>, percentile: &str) -> Option<u64> {
    value
        .and_then(|metric| metric.get(percentile))
        .and_then(Value::as_u64)
        .or_else(|| {
            value
                .and_then(|metric| metric.get(percentile))
                .and_then(Value::as_f64)
                .filter(|value| *value >= 0.0)
                .map(|value| value as u64)
        })
}

/// The first of several interchangeable metric names that carries a p50.
pub(super) fn first_metric_p50(
    metrics: Option<&serde_json::Map<String, Value>>,
    names: &[&str],
) -> Option<u64> {
    names.iter().find_map(|name| {
        metrics.and_then(|metrics| suite_metric_percentile(metrics.get(*name), "p50"))
    })
}

pub(super) fn record_required_metric_percentile(
    current_min: &mut Option<u64>,
    metrics: Option<&serde_json::Map<String, Value>>,
    metric_name: &str,
    percentile: &str,
    blockers: &mut Vec<String>,
    case_id: &str,
) {
    match metrics.and_then(|metrics| suite_metric_percentile(metrics.get(metric_name), percentile))
    {
        Some(value) if value > 0 => {
            *current_min = Some(current_min.map_or(value, |min| min.min(value)));
        }
        _ => blockers.push(format!(
            "case `{case_id}` must include positive {percentile} {metric_name}"
        )),
    }
}

pub(super) fn record_observed_metric_percentile(
    current_min: &mut Option<u64>,
    metrics: Option<&serde_json::Map<String, Value>>,
    metric_name: &str,
    percentile: &str,
    blockers: &mut Vec<String>,
    case_id: &str,
) {
    match metrics.and_then(|metrics| suite_metric_percentile(metrics.get(metric_name), percentile))
    {
        Some(value) => {
            *current_min = Some(current_min.map_or(value, |min| min.min(value)));
        }
        None => blockers.push(format!(
            "case `{case_id}` must include {percentile} {metric_name}"
        )),
    }
}

/// The wall-clock and baseline minima every release benchmark artifact reports.
#[derive(Debug, Default)]
pub(super) struct WallClockMinima {
    pub(super) wall_samples: Option<u64>,
    pub(super) baseline_wall_samples: Option<u64>,
    pub(super) wall_p50: Option<u64>,
    pub(super) wall_p95: Option<u64>,
    pub(super) wall_p99: Option<u64>,
    pub(super) baseline_wall_p50: Option<u64>,
    pub(super) baseline_wall_p95: Option<u64>,
    pub(super) baseline_wall_p99: Option<u64>,
}

impl WallClockMinima {
    /// Fold one case in, blocking a short sample run or a missing percentile.
    ///
    /// `case_label` names the case in the sample-count blockers, because the
    /// aggregate proof has to say which source artifact the case came from
    /// while a single-artifact inspector already knows.
    pub(super) fn record_case(
        &mut self,
        case_id: &str,
        case_label: &str,
        metrics: Option<&serde_json::Map<String, Value>>,
        blockers: &mut Vec<String>,
    ) {
        self.wall_samples = Some(record_sample_floor(
            self.wall_samples,
            metrics,
            "wall_ns",
            case_label,
            blockers,
        ));
        self.baseline_wall_samples = Some(record_sample_floor(
            self.baseline_wall_samples,
            metrics,
            "baseline_wall_ns",
            case_label,
            blockers,
        ));
        record_required_metric_percentile(
            &mut self.wall_p50,
            metrics,
            "wall_ns",
            "p50",
            blockers,
            case_id,
        );
        record_required_metric_percentile(
            &mut self.wall_p95,
            metrics,
            "wall_ns",
            "p95",
            blockers,
            case_id,
        );
        record_required_metric_percentile(
            &mut self.wall_p99,
            metrics,
            "wall_ns",
            "p99",
            blockers,
            case_id,
        );
        record_required_metric_percentile(
            &mut self.baseline_wall_p50,
            metrics,
            "baseline_wall_ns",
            "p50",
            blockers,
            case_id,
        );
        record_required_metric_percentile(
            &mut self.baseline_wall_p95,
            metrics,
            "baseline_wall_ns",
            "p95",
            blockers,
            case_id,
        );
        record_required_metric_percentile(
            &mut self.baseline_wall_p99,
            metrics,
            "baseline_wall_ns",
            "p99",
            blockers,
            case_id,
        );
    }
}

/// Fold one case's sample count in and block a run too short to measure.
fn record_sample_floor(
    current_min: Option<u64>,
    metrics: Option<&serde_json::Map<String, Value>>,
    metric_name: &str,
    case_label: &str,
    blockers: &mut Vec<String>,
) -> u64 {
    let samples = metrics
        .and_then(|metrics| suite_metric_samples(metrics.get(metric_name)))
        .unwrap_or(0);
    if samples < MIN_RELEASE_METRIC_SAMPLES {
        blockers.push(format!(
            "{case_label} has {samples} {metric_name} sample(s), needs at least {MIN_RELEASE_METRIC_SAMPLES}"
        ));
    }
    current_min.map_or(samples, |min| min.min(samples))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    use serde_json::json;

    fn metrics(value: &Value) -> Option<&serde_json::Map<String, Value>> {
        value.as_object()
    }

    fn full_case(samples: u64, wall: u64, baseline: u64) -> Value {
        json!({
            "wall_ns": {"samples": samples, "p50": wall, "p95": wall, "p99": wall},
            "baseline_wall_ns": {"samples": samples, "p50": baseline, "p95": baseline, "p99": baseline},
        })
    }

    /// WHY: the fold is a minimum across cases. A copy that overwrote instead
    /// of taking the minimum would report the last case and let a slower one
    /// through unmeasured.
    #[test]
    fn record_case_keeps_the_minimum_across_cases() {
        let mut minima = WallClockMinima::default();
        let mut blockers = Vec::new();
        let first = full_case(64, 900, 5_000);
        let second = full_case(32, 400, 9_000);
        minima.record_case("a", "case `a`", metrics(&first), &mut blockers);
        minima.record_case("b", "case `b`", metrics(&second), &mut blockers);
        assert!(blockers.is_empty(), "{blockers:?}");
        assert_eq!(minima.wall_samples, Some(32));
        assert_eq!(minima.baseline_wall_samples, Some(32));
        assert_eq!(minima.wall_p50, Some(400));
        assert_eq!(minima.wall_p95, Some(400));
        assert_eq!(minima.wall_p99, Some(400));
        assert_eq!(minima.baseline_wall_p50, Some(5_000));
        assert_eq!(minima.baseline_wall_p95, Some(5_000));
        assert_eq!(minima.baseline_wall_p99, Some(5_000));
    }

    /// WHY: the sample floor is the whole reason percentiles are trustworthy.
    /// The blocker has to name the case the way its caller names it, or the
    /// aggregate proof stops saying which artifact the short run came from.
    #[test]
    fn short_sample_runs_block_under_the_caller_label() {
        let mut minima = WallClockMinima::default();
        let mut blockers = Vec::new();
        let case = full_case(29, 900, 5_000);
        minima.record_case(
            "cpu.sota",
            "100x source artifact `release/evidence/benchmarks/a.json` case `cpu.sota`",
            metrics(&case),
            &mut blockers,
        );
        assert_eq!(
            blockers,
            vec![
                "100x source artifact `release/evidence/benchmarks/a.json` case `cpu.sota` has 29 wall_ns sample(s), needs at least 30",
                "100x source artifact `release/evidence/benchmarks/a.json` case `cpu.sota` has 29 baseline_wall_ns sample(s), needs at least 30",
            ]
        );
        assert_eq!(minima.wall_samples, Some(29));
    }

    /// WHY: a missing metric object must still fold, and must block every
    /// percentile rather than silently leaving the minima empty.
    #[test]
    fn a_case_without_metrics_blocks_every_percentile() {
        let mut minima = WallClockMinima::default();
        let mut blockers = Vec::new();
        minima.record_case("a", "case `a`", None, &mut blockers);
        assert_eq!(minima.wall_samples, Some(0));
        assert_eq!(minima.wall_p50, None);
        assert_eq!(
            blockers,
            vec![
                "case `a` has 0 wall_ns sample(s), needs at least 30",
                "case `a` has 0 baseline_wall_ns sample(s), needs at least 30",
                "case `a` must include positive p50 wall_ns",
                "case `a` must include positive p95 wall_ns",
                "case `a` must include positive p99 wall_ns",
                "case `a` must include positive p50 baseline_wall_ns",
                "case `a` must include positive p95 baseline_wall_ns",
                "case `a` must include positive p99 baseline_wall_ns",
            ]
        );
    }

    /// WHY: a zero percentile is a measurement that did not happen, so the
    /// required recorder must reject it while the observed recorder keeps it.
    #[test]
    fn zero_is_rejected_as_required_and_kept_as_observed() {
        let case = json!({"kernel_launches": {"p50": 0}});
        let metrics = metrics(&case);

        let mut required = None;
        let mut required_blockers = Vec::new();
        record_required_metric_percentile(
            &mut required,
            metrics,
            "kernel_launches",
            "p50",
            &mut required_blockers,
            "a",
        );
        assert_eq!(required, None);
        assert_eq!(
            required_blockers,
            vec!["case `a` must include positive p50 kernel_launches"]
        );

        let mut observed = None;
        let mut observed_blockers = Vec::new();
        record_observed_metric_percentile(
            &mut observed,
            metrics,
            "kernel_launches",
            "p50",
            &mut observed_blockers,
            "a",
        );
        assert_eq!(observed, Some(0));
        assert!(observed_blockers.is_empty());
    }

    /// WHY: an unparseable artifact must become a blocker and a null report,
    /// never a panic and never a silently empty inspection.
    #[test]
    fn invalid_json_blocks_and_yields_a_null_report() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("report.json");
        fs::write(&path, "{ not json").unwrap();
        let mut blockers = Vec::new();
        let (report, read_error) = read_benchmark_report(&path, &mut blockers);
        assert_eq!(report, Value::Null);
        assert_eq!(read_error, None);
        assert_eq!(blockers.len(), 1, "{blockers:?}");
        assert!(blockers[0].starts_with("invalid JSON: "), "{blockers:?}");
    }

    /// WHY: a missing artifact is a read failure, and the caller that reports
    /// it as its own field needs the error back, not only the blocker.
    #[test]
    fn a_missing_artifact_returns_its_read_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut blockers = Vec::new();
        let (report, read_error) =
            read_benchmark_report(&dir.path().join("absent.json"), &mut blockers);
        assert_eq!(report, Value::Null);
        assert!(read_error.is_some());
        assert_eq!(blockers.len(), 1, "{blockers:?}");
        assert!(blockers[0].starts_with("unreadable JSON: "), "{blockers:?}");
    }

    /// WHY: the bound is what keeps a corrupt evidence file from being read
    /// into memory whole.
    #[test]
    fn a_file_over_the_bound_is_refused() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("big.json");
        fs::write(&path, "x".repeat(64)).unwrap();
        let error = read_text_bounded(&path, 16).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }
}
