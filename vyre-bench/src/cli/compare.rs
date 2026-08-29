use super::report_io::read_report_bounded;
use crate::report::json::ReportSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Schema the comparison artifact is written and read under.
///
/// v2 records the band every verdict was judged against and ties `regressed`
/// to that verdict. A v1 artifact states neither, so its verdicts cannot be
/// resolved and it is rejected rather than read under today's band.
pub(super) const COMPARISON_SCHEMA: &str = "vyre-bench.compare.v2";

/// The band outside which a difference is a result rather than noise.
///
/// One owner for the three numbers that decide every verdict, recorded in the
/// artifact so a reader resolves `flat` against the band that produced it
/// instead of against a constant in this file.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub(super) struct ComparisonBand {
    /// Fraction of the baseline a difference must exceed to be a result.
    pub(super) delta_threshold: f64,
    /// Significance below which a difference is a result.
    pub(super) alpha: f64,
    /// Samples each side needs before any verdict is a measurement.
    pub(super) min_samples: u32,
}

impl ComparisonBand {
    /// The band this build judges under.
    pub(super) const CURRENT: Self = Self {
        delta_threshold: 0.05,
        alpha: 0.05,
        min_samples: 2,
    };
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(super) struct ComparisonArtifact {
    pub(super) schema: String,
    pub(super) band: ComparisonBand,
    pub(super) baseline: ComparisonSide,
    pub(super) candidate: ComparisonSide,
    pub(super) cases: Vec<ComparisonCase>,
    pub(super) regressed: bool,
    /// Cases the band decided nothing about, counted rather than averaged into
    /// the cases it did decide.
    pub(super) undecided_cases: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct ComparisonSide {
    pub(super) run_id: String,
    pub(super) suite: String,
    pub(super) selected_backend: String,
    pub(super) profile_backend: String,
    pub(super) timing_quality: String,
    pub(super) source_fingerprint: String,
    pub(super) source_tree_fingerprint: String,
    pub(super) total_cases: usize,
    pub(super) failed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(super) struct ComparisonCase {
    pub(super) id: String,
    pub(super) baseline_p50_ns: u64,
    pub(super) candidate_p50_ns: u64,
    pub(super) baseline_mean_ns: f64,
    pub(super) candidate_mean_ns: f64,
    /// Difference over the p50, the point estimate the table reports.
    pub(super) delta_fraction: Option<f64>,
    pub(super) delta_percent: Option<f64>,
    /// Significance over the mean and standard deviation, not over the p50.
    pub(super) p_value: Option<f64>,
    pub(super) verdict: String,
    pub(super) regressed: bool,
}

pub(super) fn compare_reports(
    baseline: &ReportSchema,
    candidate: &ReportSchema,
    output: Option<&str>,
) -> anyhow::Result<()> {
    let comparison = build_comparison_artifact(baseline, candidate)?;
    print_comparison_artifact(&comparison);
    if let Some(output) = output {
        write_comparison_artifact(&comparison, output)?;
    }
    if comparison.regressed {
        anyhow::bail!(
            "one or more cases lost significantly outside the declared band: delta {} alpha {}",
            comparison.band.delta_threshold,
            comparison.band.alpha
        );
    }
    Ok(())
}

pub(super) fn build_comparison_artifact(
    baseline: &ReportSchema,
    candidate: &ReportSchema,
) -> anyhow::Result<ComparisonArtifact> {
    let band = ComparisonBand::CURRENT;
    let baseline_cases: BTreeMap<_, _> = baseline
        .cases
        .iter()
        .map(|case| (case.id.as_str(), case))
        .collect();
    let mut cases = Vec::with_capacity(candidate.cases.len());
    for case in &candidate.cases {
        let baseline_case = baseline_cases
            .get(case.id.as_str())
            .ok_or_else(|| anyhow::anyhow!("candidate case `{}` has no baseline", case.id))?;
        let baseline_stats = baseline_case
            .metrics
            .get("wall_ns")
            .ok_or_else(|| anyhow::anyhow!("baseline case `{}` lacks wall_ns", case.id))?;
        let candidate_stats = case
            .metrics
            .get("wall_ns")
            .ok_or_else(|| anyhow::anyhow!("candidate case `{}` lacks wall_ns", case.id))?;
        let baseline_p50 = baseline_stats.p50;
        let candidate_p50 = candidate_stats.p50;
        let delta_fraction = if baseline_p50 == 0 {
            None
        } else {
            Some((candidate_p50 as f64 - baseline_p50 as f64) / baseline_p50 as f64)
        };
        let p_value = welch_p_value(baseline_stats, candidate_stats, band);
        let verdict = compare_verdict(delta_fraction, p_value, band);
        let regressed = verdict == VERDICT_REGRESS;
        cases.push(ComparisonCase {
            id: case.id.clone(),
            baseline_p50_ns: baseline_p50,
            candidate_p50_ns: candidate_p50,
            baseline_mean_ns: baseline_stats.mean,
            candidate_mean_ns: candidate_stats.mean,
            delta_fraction,
            delta_percent: delta_fraction.map(|delta| delta * 100.0),
            p_value,
            verdict: verdict.to_string(),
            regressed,
        });
    }
    let regressed = cases.iter().any(|case| case.regressed);
    let undecided_cases = cases.iter().filter(|case| is_undecided(case)).count();
    Ok(ComparisonArtifact {
        schema: COMPARISON_SCHEMA.to_string(),
        band,
        baseline: comparison_side(baseline),
        candidate: comparison_side(candidate),
        cases,
        regressed,
        undecided_cases,
    })
}

/// Whether the band decided nothing about this case.
fn is_undecided(case: &ComparisonCase) -> bool {
    matches!(case.verdict.as_str(), VERDICT_UNMEASURED | VERDICT_NOISY)
}

fn comparison_side(report: &ReportSchema) -> ComparisonSide {
    let (profile_backend, timing_quality) = report
        .backend_profile
        .as_ref()
        .map(|profile| (profile.backend.as_str(), profile.timing_quality.as_str()))
        .unwrap_or(("unknown", "unknown"));
    ComparisonSide {
        run_id: report.run_id.clone(),
        suite: report.suite.clone(),
        selected_backend: report
            .selected_backend
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        profile_backend: profile_backend.to_string(),
        timing_quality: timing_quality.to_string(),
        source_fingerprint: report.source_fingerprint.clone(),
        source_tree_fingerprint: report.source_tree_fingerprint.clone(),
        total_cases: report.summary.total_cases,
        failed: report.summary.failed,
    }
}

fn print_comparison_artifact(comparison: &ComparisonArtifact) {
    print_compare_profile("baseline", &comparison.baseline);
    print_compare_profile("candidate", &comparison.candidate);
    println!(
        "band_delta_threshold={} band_alpha={} band_min_samples={} undecided_cases={}",
        comparison.band.delta_threshold,
        comparison.band.alpha,
        comparison.band.min_samples,
        comparison.undecided_cases
    );
    println!(
        "{:<30} | {:<12} | {:<12} | {:<10} | {:<12} | {:<10}",
        "Benchmark", "Baseline", "Candidate", "Delta", "p-value", "Verdict"
    );
    println!(
        "------------------------------------------------------------------------------------------------"
    );
    for case in &comparison.cases {
        let delta = case
            .delta_percent
            .map(|value| format!("{value:+.2}%"))
            .unwrap_or_else(|| "n/a".to_string());
        let p_value = case
            .p_value
            .map(|value| format!("{value:.4}"))
            .unwrap_or_else(|| "n/a".to_string());
        println!(
            "{:<30} | {:<12} | {:<12} | {:<10} | {:<12} | {:<10}",
            case.id, case.baseline_p50_ns, case.candidate_p50_ns, delta, p_value, case.verdict
        );
    }
}

fn print_compare_profile(label: &str, side: &ComparisonSide) {
    println!(
        "{label}_selected_backend={} {label}_profile_backend={} {label}_timing_quality={}",
        side.selected_backend, side.profile_backend, side.timing_quality
    );
}

pub(super) fn write_comparison_artifact(
    comparison: &ComparisonArtifact,
    path: &str,
) -> anyhow::Result<()> {
    let file = std::fs::File::create(path)?;
    serde_json::to_writer_pretty(file, comparison)?;
    Ok(())
}

pub(super) fn load_comparison_artifact(path: &str) -> anyhow::Result<ComparisonArtifact> {
    let bytes = read_report_bounded(std::path::Path::new(path))?;
    parse_comparison_artifact(&bytes)
}

pub(super) fn parse_comparison_artifact(bytes: &[u8]) -> anyhow::Result<ComparisonArtifact> {
    Ok(serde_json::from_slice(bytes)?)
}

pub(super) fn validate_comparison_expectations(
    comparison: &ComparisonArtifact,
    baseline_backend: &str,
    candidate_backend: &str,
    case_ids: &[String],
) -> anyhow::Result<()> {
    if comparison.schema != COMPARISON_SCHEMA {
        anyhow::bail!(
            "comparison schema `{}` is not `{COMPARISON_SCHEMA}`. Fix: regenerate comparison with current vyre-bench compare.",
            comparison.schema
        );
    }
    if comparison.band != ComparisonBand::CURRENT {
        anyhow::bail!(
            "comparison was judged against delta {} alpha {} min_samples {} and this build judges against delta {} alpha {} min_samples {}. Fix: regenerate comparison with current vyre-bench compare.",
            comparison.band.delta_threshold,
            comparison.band.alpha,
            comparison.band.min_samples,
            ComparisonBand::CURRENT.delta_threshold,
            ComparisonBand::CURRENT.alpha,
            ComparisonBand::CURRENT.min_samples
        );
    }
    if comparison.baseline.profile_backend != baseline_backend {
        anyhow::bail!(
            "comparison baseline profile backend `{}` does not match expected `{baseline_backend}`. Fix: compare the intended baseline report.",
            comparison.baseline.profile_backend
        );
    }
    if comparison.candidate.profile_backend != candidate_backend {
        anyhow::bail!(
            "comparison candidate profile backend `{}` does not match expected `{candidate_backend}`. Fix: compare the intended candidate report.",
            comparison.candidate.profile_backend
        );
    }
    for (label, side) in [
        ("baseline", &comparison.baseline),
        ("candidate", &comparison.candidate),
    ] {
        if !matches!(
            side.timing_quality.as_str(),
            "host_only" | "host_enqueue_wait" | "device_timestamps" | "hardware_counters"
        ) {
            anyhow::bail!(
                "{label} timing quality `{}` is invalid. Fix: regenerate comparison from reports with DeviceTimingQuality::as_str() values.",
                side.timing_quality
            );
        }
    }
    if comparison.cases.is_empty() {
        anyhow::bail!("comparison contains zero cases. Fix: compare reports with overlapping benchmark cases.");
    }
    for case_id in case_ids {
        if !comparison.cases.iter().any(|case| case.id == *case_id) {
            anyhow::bail!(
                "comparison artifact lacks case `{case_id}`. Fix: compare reports generated with the intended --case selection."
            );
        }
    }
    for case in &comparison.cases {
        if !VERDICTS.contains(&case.verdict.as_str()) {
            anyhow::bail!(
                "comparison case `{}` states verdict `{}`, which this build does not state. Fix: regenerate comparison with current vyre-bench compare.",
                case.id,
                case.verdict
            );
        }
        if case.regressed != (case.verdict == VERDICT_REGRESS) {
            anyhow::bail!(
                "comparison case `{}` states verdict `{}` and regressed={}. Fix: regenerate comparison so the recorded loss is the verdict the band states.",
                case.id,
                case.verdict,
                case.regressed
            );
        }
    }
    let derived_regressed = comparison.cases.iter().any(|case| case.regressed);
    if comparison.regressed != derived_regressed {
        anyhow::bail!(
            "comparison regressed={} contradicts case-derived regressed={derived_regressed}. Fix: regenerate comparison from case evidence.",
            comparison.regressed
        );
    }
    let derived_undecided = comparison
        .cases
        .iter()
        .filter(|case| is_undecided(case))
        .count();
    if comparison.undecided_cases != derived_undecided {
        anyhow::bail!(
            "comparison records {} undecided cases and {derived_undecided} carry no verdict. Fix: regenerate comparison so every undecided case is counted.",
            comparison.undecided_cases
        );
    }
    Ok(())
}

/// A significant difference wider than the band, against the baseline.
pub(super) const VERDICT_REGRESS: &str = "regress";
/// A significant difference wider than the band, in the baseline's favour.
pub(super) const VERDICT_IMPROVE: &str = "improve";
/// A difference inside the band, decided against enough samples to say so.
pub(super) const VERDICT_FLAT: &str = "flat";
/// A difference wider than the band that the significance test does not admit.
pub(super) const VERDICT_NOISY: &str = "noisy";
/// No verdict: the band could not be applied to this pair at all.
pub(super) const VERDICT_UNMEASURED: &str = "unmeasured";

/// Every verdict this build states.
pub(super) const VERDICTS: [&str; 5] = [
    VERDICT_REGRESS,
    VERDICT_IMPROVE,
    VERDICT_FLAT,
    VERDICT_NOISY,
    VERDICT_UNMEASURED,
];

/// Read one pair of measurements against the band.
///
/// Equivalence is a claim, so it needs the same evidence a difference needs: a
/// pair the significance test could not be applied to is `unmeasured`, never
/// `flat`. Calling it flat is how a comparison of two single samples reports
/// that a workload did not change.
fn compare_verdict(
    delta_fraction: Option<f64>,
    p_value: Option<f64>,
    band: ComparisonBand,
) -> &'static str {
    let (Some(delta), Some(p)) = (delta_fraction, p_value) else {
        return VERDICT_UNMEASURED;
    };
    if p < band.alpha && delta > band.delta_threshold {
        return VERDICT_REGRESS;
    }
    if p < band.alpha && delta < -band.delta_threshold {
        return VERDICT_IMPROVE;
    }
    if delta.abs() <= band.delta_threshold {
        return VERDICT_FLAT;
    }
    VERDICT_NOISY
}

/// Welch's statistic read against the normal distribution.
///
/// The normal tail overstates significance for a handful of samples, so the
/// sample floor is part of the band rather than a separate rule, and a pair
/// below it yields no p-value at all.
fn welch_p_value(
    baseline: &crate::api::metric::MetricStats,
    candidate: &crate::api::metric::MetricStats,
    band: ComparisonBand,
) -> Option<f64> {
    if baseline.samples < band.min_samples || candidate.samples < band.min_samples {
        return None;
    }
    let n1 = f64::from(baseline.samples);
    let n2 = f64::from(candidate.samples);
    let variance = baseline.stddev.powi(2) / n1 + candidate.stddev.powi(2) / n2;
    if variance <= f64::EPSILON {
        return (baseline.mean != candidate.mean)
            .then_some(0.0)
            .or(Some(1.0));
    }
    let t = (candidate.mean - baseline.mean).abs() / variance.sqrt();
    Some((2.0 * (1.0 - normal_cdf(t))).clamp(0.0, 1.0))
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}
