//! Hold the canonical release benchmark axes to the recorded CUDA evidence.
//!
//! The five headline axes a release quotes are read from
//! `release/evidence/benchmarks/bench-release-axes.json`, and this gate is what
//! stands between that file and a release note. It measures nothing itself: the
//! measurement is `release-benchmarks`, and the axes are only as current as the
//! run that recorded them.
//!
//! Substrate attribution per axis, which optimization fired and how much it
//! saved, lives behind `VYRE_TRACE=1` and the substrate audit log. This gate
//! reports the headline numbers as notes and judges only their provenance.

use std::io;
use std::path::{Path, PathBuf};

use serde_json::Value;
use xtask::gate::{Finding, Gate, GateCtx, GateError, Report};

use crate::bench::benchmark_evidence_semantics::{
    benchmark_evidence_blocker_issues, cuda_release_axes_source_artifact_issues,
};

/// Axis identifier emitted in the final report. Stable string so
/// downstream graphs / regression gates can key on it.
const AXIS_WARM_US_PER_FILE: &str = "warm_us_per_file";
const AXIS_COLD_PIPELINE_BUILD_MS: &str = "cold_pipeline_build_ms";
const AXIS_GBS_SCAN_THROUGHPUT: &str = "gbs_scan_throughput";
const AXIS_ULP_DRIFT_MAX: &str = "ulp_drift_max";
const AXIS_MAX_VRAM_MIB: &str = "max_vram_mib";
const MAX_BENCH_RELEASE_REPORT_BYTES: u64 = 16_777_216;

/// Default location of the recorded release benchmark evidence.
const DEFAULT_EVIDENCE_DIR: &str = "release/evidence/benchmarks";

/// Holds the quotable release axes to the CUDA evidence they claim to come from.
pub struct BenchReleaseGate;

impl Gate for BenchReleaseGate {
    fn name(&self) -> &'static str {
        "bench-release"
    }

    fn help(&self) -> &'static str {
        "Judge the five canonical release axes recorded in \
         release/evidence/benchmarks/bench-release-axes.json. Proves the axes file and the CUDA \
         release suite beside it are readable, carry no blocker, pass the CUDA source-artifact \
         validation that ties an axis to the run that produced it, and that every one of the \
         five axes is present and parses as its declared numeric type. Reports the axis values \
         as notes. Proves nothing about current performance: it runs no benchmark, and an axis \
         recorded a month ago reads exactly the same as one recorded today."
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let evidence_dir = match evidence_dir(ctx) {
            Ok(directory) => directory,
            Err(message) => {
                return Ok(Report::with_findings(vec![Finding::new(
                    message,
                    "Pass --evidence-dir with the directory holding the recorded release \
                     benchmark artifacts.",
                )]))
            }
        };
        Ok(judge(&evidence_dir))
    }
}

/// Every judgement the recorded axes support.
///
/// Each axis used to abort the command on the first failure, so a release note
/// author learned about one missing axis per run. All five are judged here.
fn judge(evidence_dir: &Path) -> Report {
    let mut report = Report::clean();
    let axes = match load_release_axes(evidence_dir, &mut report) {
        Some(axes) => axes,
        None => return report,
    };
    let mut values = Vec::new();
    for (axis, units, kind) in [
        (AXIS_WARM_US_PER_FILE, "us", AxisKind::Float),
        (AXIS_COLD_PIPELINE_BUILD_MS, "ms", AxisKind::Float),
        (AXIS_GBS_SCAN_THROUGHPUT, "GiB/s", AxisKind::Float),
        (AXIS_ULP_DRIFT_MAX, "ulp", AxisKind::U32),
        (AXIS_MAX_VRAM_MIB, "MiB", AxisKind::U64),
    ] {
        match axis_value(&axes, axis, kind) {
            Ok(value) => values.push(format!("{axis}={value} {units}")),
            Err(message) => report.find(Finding::in_file(
                evidence_dir.join("bench-release-axes.json"),
                message,
                "Rerun `cargo_full run --bin xtask -- release-benchmarks --backend cuda` on a \
                 release host so the axis is recorded with the run that measured it.",
            )),
        }
    }
    report.note(format!(
        "vyre v{} release axes: {}",
        xtask::release::release_train::vyre_version(),
        values.join(", ")
    ));
    report
}

/// Which numeric type an axis must parse as.
#[derive(Clone, Copy)]
enum AxisKind {
    Float,
    U32,
    U64,
}

/// One axis's recorded value, or why it cannot be quoted.
fn axis_value(axes: &Value, axis: &str, kind: AxisKind) -> Result<String, String> {
    let raw = json_axis_text(axes, axis).ok_or_else(|| {
        format!(
            "canonical bench-release axes are missing `{axis}`"
        )
    })?;
    let parsed = match kind {
        AxisKind::Float => raw.parse::<f64>().is_ok(),
        AxisKind::U32 => raw.parse::<u32>().is_ok(),
        AxisKind::U64 => raw.parse::<u64>().is_ok(),
    };
    if parsed {
        Ok(raw)
    } else {
        Err(format!(
            "axis `{axis}` value `{raw}` is not the number type the axis declares"
        ))
    }
}

/// The directory holding the recorded artifacts, resolved against the checkout.
fn evidence_dir(ctx: &GateCtx) -> Result<PathBuf, String> {
    let mut evidence_dir = PathBuf::from(DEFAULT_EVIDENCE_DIR);
    let mut index = 0;
    while index < ctx.args.len() {
        match ctx.args[index].as_str() {
            "--write" => index += 1,
            "--evidence-dir" => {
                let value = ctx.args.get(index + 1).ok_or_else(|| {
                    "Fix: --evidence-dir requires a path to release benchmark artifacts."
                        .to_string()
                })?;
                evidence_dir = PathBuf::from(value);
                index += 2;
            }
            other => {
                return Err(format!(
                    "Fix: unknown bench-release argument `{other}`. Use --evidence-dir PATH."
                ));
            }
        }
    }
    Ok(ctx.root.join(evidence_dir))
}

/// The recorded axes, once both artifacts have been judged fit to read.
///
/// Every blocker in either artifact and every source-artifact validation issue
/// is reported. The command used to stop at the first one, so a reader learned
/// about a single issue per run and could not tell a lone problem from a wall
/// of them.
fn load_release_axes(evidence_dir: &Path, report: &mut Report) -> Option<Value> {
    let axes_path = evidence_dir.join("bench-release-axes.json");
    let axes = read_json_report(&axes_path, "canonical bench-release axes", report);
    let suite_path = evidence_dir.join("cuda-release-suite.json");
    let cuda_suite = read_json_report(&suite_path, "CUDA release suite", report);
    let (Some(axes), Some(cuda_suite)) = (axes, cuda_suite) else {
        return None;
    };
    report_blockers(&axes_path, &axes, report);
    report_blockers(&suite_path, &cuda_suite, report);
    let workspace_root = workspace_root_for_evidence_dir(evidence_dir);
    for issue in cuda_release_axes_source_artifact_issues(&workspace_root, &axes, &cuda_suite) {
        report.find(Finding::in_file(
            axes_path.clone(),
            format!("CUDA source artifact validation: {issue}"),
            "An axis whose source artifact does not back it is a number with no run behind it. \
             Rerun release-benchmarks --backend cuda on a release host.",
        ));
    }
    Some(axes)
}

fn read_json_report(path: &Path, label: &str, report: &mut Report) -> Option<Value> {
    let contents = match read_text_bounded(path) {
        Ok(contents) => contents,
        Err(error) => {
            report.find(Finding::in_file(
                path.to_path_buf(),
                format!("cannot read {label}: {error}"),
                "Run `cargo_full run --bin xtask -- release-benchmarks --backend cuda` on a \
                 release host and commit the artifact.",
            ));
            return None;
        }
    };
    match serde_json::from_str::<Value>(&contents) {
        Ok(value) => Some(value),
        Err(error) => {
            report.find(Finding::in_file(
                path.to_path_buf(),
                format!("invalid {label} JSON: {error}"),
                "Regenerate the artifact. Benchmark evidence nothing can parse records nothing.",
            ));
            None
        }
    }
}

fn workspace_root_for_evidence_dir(evidence_dir: &Path) -> PathBuf {
    let benchmarks = evidence_dir;
    let evidence = benchmarks.parent();
    let release = evidence.and_then(Path::parent);
    if benchmarks.file_name().and_then(|name| name.to_str()) == Some("benchmarks")
        && evidence
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            == Some("evidence")
        && release
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            == Some("release")
    {
        return release
            .and_then(Path::parent)
            .map_or_else(PathBuf::new, Path::to_path_buf);
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// Every blocker contract issue one benchmark artifact records.
fn report_blockers(path: &Path, value: &Value, report: &mut Report) {
    let evidence = path.to_string_lossy();
    for issue in benchmark_evidence_blocker_issues(&evidence, value) {
        report.find(Finding::in_file(
            path.to_path_buf(),
            format!("benchmark evidence blocker contract: {issue}"),
            "Resolve the blocker on a release host and rerun release-benchmarks so the artifact \
             records a clean run.",
        ));
    }
}

fn json_axis_text(value: &Value, axis: &str) -> Option<String> {
    match value.get(axis)? {
        Value::Number(number) => Some(number.to_string()),
        Value::String(text) if !text.trim().is_empty() => Some(text.clone()),
        _ => None,
    }
}

fn read_text_bounded(path: &Path) -> io::Result<String> {
    xtask::output_arg::read_text_bounded(
        path,
        MAX_BENCH_RELEASE_REPORT_BYTES,
        "release bench report",
    )
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    /// The axes, when the fixture is coherent.
    ///
    /// `load_release_axes` reports through a `Report` now instead of returning
    /// `Result`, because one bad axis used to abort the whole gate and hide
    /// every axis after it. These two helpers keep the tests asserting the same
    /// two outcomes: a fixture loads, or it is rejected with a stated reason.
    fn axes_or_findings(benchmark_dir: &Path) -> Option<Value> {
        let mut report = Report::clean();
        let axes = load_release_axes(benchmark_dir, &mut report);
        assert!(
            report.findings.is_empty(),
            "Fix: a coherent fixture must report nothing; got {:?}",
            report.findings
        );
        axes
    }

    /// The one message a rejected fixture was rejected with.
    fn axes_findings(benchmark_dir: &Path) -> Option<String> {
        let mut report = Report::clean();
        let _ = load_release_axes(benchmark_dir, &mut report);
        assert!(
            report.findings.len() <= 1,
            "Fix: these fixtures each carry one defect; got {:?}",
            report.findings
        );
        report
            .findings
            .into_iter()
            .next()
            .map(|finding| finding.message)
    }

    fn write_canonical_axes_fixture(
        benchmark_dir: &Path,
        workspace_root: &Path,
        poisoned_index: Option<usize>,
    ) -> Vec<String> {
        fs::write(workspace_root.join("Cargo.toml"), "[workspace]\n")
            .expect("Fix: write temporary workspace manifest.");
        fs::create_dir_all(benchmark_dir)
            .expect("Fix: create temporary benchmark evidence directory.");
        let git = vyre_bench::probes::capture_git_info_at(workspace_root);
        let source_fingerprint = vyre_bench::probes::source_fingerprint(&git);
        let source_tree_fingerprint =
            vyre_bench::probes::source_tree_fingerprint_at(workspace_root);
        let mut artifacts = Vec::new();
        for index in 1..=12 {
            let artifact = format!("release/evidence/benchmarks/workload-{index:02}.json");
            let selected_backend = if poisoned_index == Some(index) {
                "wgpu"
            } else {
                "cuda"
            };
            fs::write(
                workspace_root.join(&artifact),
                serde_json::to_string_pretty(&serde_json::json!({
                    "selected_backend": selected_backend,
                    "source_fingerprint": &source_fingerprint,
                    "source_tree_fingerprint": &source_tree_fingerprint,
                    "summary": {"total_cases": 1, "passed": 1, "failed": 0},
                    "cases": [
                        {
                            "id": format!("release.axis.{index:02}"),
                            "backend_id": selected_backend,
                            "status": "pass",
                            "metrics": {
                                "wall_ns": {"p50": 17_000},
                                "cold_compile_ns": {"p50": 2_000_000},
                                "wall_gb_s_x1000": {"p50": 4_000},
                                "memory_total_mib": {"p50": 24_576}
                            }
                        }
                    ]
                }))
                .expect("Fix: serialize temporary benchmark artifact."),
            )
            .expect("Fix: write temporary benchmark artifact.");
            artifacts.push(artifact);
        }
        let artifact_statuses = artifacts
            .iter()
            .map(|artifact| {
                serde_json::json!({
                    "path": artifact,
                    "blockers": []
                })
            })
            .collect::<Vec<_>>();
        fs::write(
            benchmark_dir.join("cuda-release-suite.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "schema_version": 2,
                "backend": "cuda",
                "artifacts": artifacts.clone(),
                "artifact_statuses": artifact_statuses,
                "blockers": []
            }))
            .expect("Fix: serialize temporary CUDA release suite."),
        )
        .expect("Fix: write temporary CUDA release suite.");
        fs::write(
            benchmark_dir.join("bench-release-axes.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "schema_version": 1,
                "warm_us_per_file": 17.0,
                "cold_pipeline_build_ms": 2.0,
                "gbs_scan_throughput": 4.0,
                "ulp_drift_max": 0,
                "max_vram_mib": 24576,
                "source_artifacts": artifacts,
                "blockers": []
            }))
            .expect("Fix: serialize temporary canonical release axes."),
        )
        .expect("Fix: write temporary canonical release axes.");
        artifacts
    }

    #[test]
    fn bench_release_reads_canonical_axes_instead_of_directory_decoys() {
        let dir = tempfile::TempDir::new()
            .expect("Fix: create temporary workspace for bench-release test.");
        let benchmark_dir = dir.path().join("release/evidence/benchmarks");
        write_canonical_axes_fixture(&benchmark_dir, dir.path(), None);
        fs::write(
            benchmark_dir.join("aaa-decoy-axis.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "warm_us_per_file": 0.001,
                "blockers": []
            }))
            .expect("Fix: serialize temporary decoy axis."),
        )
        .expect("Fix: write temporary decoy axis.");

        let axes = axes_or_findings(&benchmark_dir)
            .expect("Fix: canonical CUDA release axes fixture should load.");

        assert_eq!(
            axis_value(&axes, AXIS_WARM_US_PER_FILE, AxisKind::Float),
            Ok("17".to_string()),
            "Fix: bench-release must print the canonical bench-release-axes value, not whichever JSON directory entry exposes a top-level axis."
        );
    }

    #[test]
    fn bench_release_rejects_wgpu_source_artifacts_under_cuda_axes() {
        let dir = tempfile::TempDir::new()
            .expect("Fix: create temporary workspace for bench-release poison test.");
        let benchmark_dir = dir.path().join("release/evidence/benchmarks");
        write_canonical_axes_fixture(&benchmark_dir, dir.path(), Some(7));

        let error = axes_findings(&benchmark_dir)
            .expect("Fix: WGPU artifacts must not satisfy CUDA bench-release axes.");

        assert!(
            error.contains("selected_backend must be cuda"),
            "Fix: bench-release must reject backend drift inside source_artifacts; error={error}"
        );
    }

    #[test]
    fn bench_release_rejects_case_backend_drift_under_cuda_axes() {
        let dir = tempfile::TempDir::new()
            .expect("Fix: create temporary workspace for bench-release case backend drift test.");
        let benchmark_dir = dir.path().join("release/evidence/benchmarks");
        let artifacts = write_canonical_axes_fixture(&benchmark_dir, dir.path(), None);
        let drift_path = dir.path().join(&artifacts[6]);
        let mut artifact = serde_json::from_str::<Value>(
            &fs::read_to_string(&drift_path).expect("Fix: read temporary CUDA axis artifact."),
        )
        .expect("Fix: temporary CUDA axis artifact must be JSON.");
        artifact["cases"][0]["backend_id"] = Value::String("wgpu".to_string());
        fs::write(
            &drift_path,
            serde_json::to_string_pretty(&artifact)
                .expect("Fix: serialize drifted temporary CUDA axis artifact."),
        )
        .expect("Fix: write drifted temporary CUDA axis artifact.");

        let error = axes_findings(&benchmark_dir)
            .expect("Fix: WGPU cases must not satisfy CUDA bench-release axes.");

        assert!(
            error.contains("backend_id `wgpu` does not match selected_backend `cuda`"),
            "Fix: bench-release must reject case-level backend drift inside CUDA source_artifacts; error={error}"
        );
    }

    #[test]
    fn bench_release_rejects_borrowed_resident_fallback_source_artifacts() {
        let dir = tempfile::TempDir::new()
            .expect("Fix: create temporary workspace for bench-release CUDA telemetry test.");
        let benchmark_dir = dir.path().join("release/evidence/benchmarks");
        let artifacts = write_canonical_axes_fixture(&benchmark_dir, dir.path(), None);
        let polluted_path = dir.path().join(&artifacts[6]);
        let mut artifact = serde_json::from_str::<Value>(
            &fs::read_to_string(&polluted_path).expect("Fix: read temporary CUDA axis artifact."),
        )
        .expect("Fix: temporary CUDA axis artifact must be JSON.");
        artifact["cases"][0]["optimization_passes_applied"] =
            serde_json::json!(["cuda-resident-borrowed-escape-hatch"]);
        artifact["cases"][0]["metrics"]["cuda_resident_borrowed_fallback_dispatches"] =
            serde_json::json!({"p50": 2.0});
        fs::write(
            &polluted_path,
            serde_json::to_string_pretty(&artifact)
                .expect("Fix: serialize polluted temporary CUDA axis artifact."),
        )
        .expect("Fix: write polluted temporary CUDA axis artifact.");

        let error = axes_findings(&benchmark_dir)
            .expect("Fix: borrowed resident fallback source artifacts must not load.");

        assert!(
            error.contains("cuda_resident_borrowed_fallback_dispatches p50=2")
                && error.contains("canonical CUDA release axes must use native resident dispatch"),
            "Fix: bench-release must reject canonical axes backed by borrowed resident fallback dispatches; error={error}"
        );
    }

    #[test]
    fn bench_release_rejects_source_artifacts_missing_axis_metrics() {
        let dir = tempfile::TempDir::new()
            .expect("Fix: create temporary workspace for bench-release axis metric test.");
        let benchmark_dir = dir.path().join("release/evidence/benchmarks");
        let artifacts = write_canonical_axes_fixture(&benchmark_dir, dir.path(), None);
        let missing_metric_path = dir.path().join(&artifacts[5]);
        let mut artifact = serde_json::from_str::<Value>(
            &fs::read_to_string(&missing_metric_path)
                .expect("Fix: read temporary CUDA axis artifact."),
        )
        .expect("Fix: temporary CUDA axis artifact must be JSON.");
        artifact["cases"][0]["metrics"]["cold_compile_ns"] = Value::Null;
        artifact["cases"][0]["metrics"]["wall_gb_s_x1000"] = Value::Null;
        artifact["environment"] = Value::Null;
        artifact["cases"][0]["metrics"]["memory_total_mib"] = Value::Null;
        fs::write(
            &missing_metric_path,
            serde_json::to_string_pretty(&artifact)
                .expect("Fix: serialize metric-poisoned temporary CUDA axis artifact."),
        )
        .expect("Fix: write metric-poisoned temporary CUDA axis artifact.");

        let error = axes_findings(&benchmark_dir)
            .expect("Fix: source artifacts missing axis metrics must not load.");

        assert!(
            error.contains("has no positive p50 cold/compile metric for cold_pipeline_build_ms"),
            "Fix: bench-release must reject old clean axes when a source artifact lacks required release-axis metrics; error={error}"
        );
    }

    #[test]
    fn bench_release_rejects_axes_scalar_drift_from_source_artifacts() {
        let dir = tempfile::TempDir::new()
            .expect("Fix: create temporary workspace for bench-release axes drift test.");
        let benchmark_dir = dir.path().join("release/evidence/benchmarks");
        let artifacts = write_canonical_axes_fixture(&benchmark_dir, dir.path(), None);
        fs::write(
            benchmark_dir.join("bench-release-axes.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "schema_version": 1,
                "warm_us_per_file": 17.0,
                "cold_pipeline_build_ms": 2.0,
                "gbs_scan_throughput": 999.0,
                "ulp_drift_max": 0,
                "max_vram_mib": 24576,
                "source_artifacts": artifacts,
                "blockers": []
            }))
            .expect("Fix: serialize drifted temporary release axes."),
        )
        .expect("Fix: write drifted temporary release axes.");

        let error = axes_findings(&benchmark_dir)
            .expect("Fix: drifted release axes must not load.");

        assert!(
            error.contains("gbs_scan_throughput=999 does not match source artifacts 4"),
            "Fix: bench-release must reject stale or inflated release axes values; error={error}"
        );
    }

    #[test]
    fn bench_release_rejects_axes_missing_blockers_array() {
        let dir = tempfile::TempDir::new()
            .expect("Fix: create temporary workspace for bench-release blocker schema test.");
        let benchmark_dir = dir.path().join("release/evidence/benchmarks");
        let artifacts = write_canonical_axes_fixture(&benchmark_dir, dir.path(), None);
        fs::write(
            benchmark_dir.join("bench-release-axes.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "schema_version": 1,
                "warm_us_per_file": 17.0,
                "cold_pipeline_build_ms": 2.0,
                "gbs_scan_throughput": 4.0,
                "ulp_drift_max": 0,
                "max_vram_mib": 24576,
                "source_artifacts": artifacts
            }))
            .expect("Fix: serialize blockerless temporary release axes."),
        )
        .expect("Fix: write blockerless temporary release axes.");

        let error = axes_findings(&benchmark_dir)
            .expect("Fix: release axes without blockers array must not load.");

        assert!(
            error.contains("bench-release-axes.json` is missing blockers array"),
            "Fix: bench-release must fail closed when canonical axes omit blockers; error={error}"
        );
    }

    #[test]
    fn bench_release_rejects_suite_missing_artifact_statuses() {
        let dir = tempfile::TempDir::new()
            .expect("Fix: create temporary workspace for bench-release suite inventory test.");
        let benchmark_dir = dir.path().join("release/evidence/benchmarks");
        let artifacts = write_canonical_axes_fixture(&benchmark_dir, dir.path(), None);
        fs::write(
            benchmark_dir.join("cuda-release-suite.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "schema_version": 2,
                "backend": "cuda",
                "artifacts": artifacts,
                "blockers": []
            }))
            .expect("Fix: serialize temporary CUDA release suite without inventory."),
        )
        .expect("Fix: write temporary CUDA release suite without inventory.");

        let error = axes_findings(&benchmark_dir)
            .expect("Fix: suite evidence without artifact_statuses must not load.");

        assert!(
            error.contains("cuda-release-suite.json` is missing artifact_statuses array"),
            "Fix: bench-release must fail closed when CUDA suite evidence omits artifact_statuses; error={error}"
        );
    }

    #[test]
    fn bench_release_rejects_suite_status_blockers() {
        let dir = tempfile::TempDir::new()
            .expect("Fix: create temporary workspace for bench-release suite blocker test.");
        let benchmark_dir = dir.path().join("release/evidence/benchmarks");
        let artifacts = write_canonical_axes_fixture(&benchmark_dir, dir.path(), None);
        fs::write(
            benchmark_dir.join("cuda-release-suite.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "schema_version": 2,
                "backend": "cuda",
                "artifacts": artifacts,
                "artifact_statuses": [
                    {
                        "path": "release/evidence/benchmarks/workload-01.json",
                        "blockers": ["case `release.condition_eval.1m` failed: wrong answer"]
                    }
                ],
                "blockers": []
            }))
            .expect("Fix: serialize temporary CUDA release suite with nested blocker."),
        )
        .expect("Fix: write temporary CUDA release suite with nested blocker.");

        let error = axes_findings(&benchmark_dir)
            .expect("Fix: suite evidence with nested blockers must not load.");

        assert!(
            error.contains("artifact_statuses[0]")
                && error.contains("case `release.condition_eval.1m` failed: wrong answer"),
            "Fix: bench-release must reject CUDA suite status rows that carry blockers; error={error}"
        );
    }

    #[test]
    fn bench_release_rejects_suite_status_inventory_drift() {
        let dir = tempfile::TempDir::new().expect(
            "Fix: create temporary workspace for bench-release suite inventory drift test.",
        );
        let benchmark_dir = dir.path().join("release/evidence/benchmarks");
        let artifacts = write_canonical_axes_fixture(&benchmark_dir, dir.path(), None);
        let mut artifact_statuses = artifacts
            .iter()
            .map(|artifact| {
                serde_json::json!({
                    "path": artifact,
                    "blockers": []
                })
            })
            .collect::<Vec<_>>();
        artifact_statuses[11]["path"] =
            Value::String("release/evidence/benchmarks/wgpu-workload-12.json".to_string());
        fs::write(
            benchmark_dir.join("cuda-release-suite.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "schema_version": 2,
                "backend": "cuda",
                "artifacts": artifacts,
                "artifact_statuses": artifact_statuses,
                "blockers": []
            }))
            .expect("Fix: serialize temporary CUDA release suite with inventory drift."),
        )
        .expect("Fix: write temporary CUDA release suite with inventory drift.");

        let error = axes_findings(&benchmark_dir)
            .expect("Fix: suite evidence with status inventory drift must not load.");

        assert!(
            error.contains(
                "cuda-release-suite lists artifact `release/evidence/benchmarks/workload-12.json` without matching artifact_statuses entry"
            ),
            "Fix: bench-release must reject CUDA suite inventory drift before printing axes; error={error}"
        );
    }

    #[test]
    fn bench_release_rejects_mislabeled_cuda_suite_backend() {
        let dir = tempfile::TempDir::new()
            .expect("Fix: create temporary workspace for bench-release suite backend test.");
        let benchmark_dir = dir.path().join("release/evidence/benchmarks");
        let artifacts = write_canonical_axes_fixture(&benchmark_dir, dir.path(), None);
        fs::write(
            benchmark_dir.join("cuda-release-suite.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "schema_version": 2,
                "backend": "wgpu",
                "artifacts": artifacts,
                "artifact_statuses": [
                    {
                        "path": "release/evidence/benchmarks/workload-01.json",
                        "blockers": []
                    }
                ],
                "blockers": []
            }))
            .expect("Fix: serialize mislabeled temporary CUDA release suite."),
        )
        .expect("Fix: write mislabeled temporary CUDA release suite.");

        let error = axes_findings(&benchmark_dir)
            .expect("Fix: mislabeled CUDA release suites must not satisfy bench-release axes.");

        assert!(
            error.contains("cuda-release-suite backend `wgpu` does not match required `cuda`"),
            "Fix: bench-release must reject suite backend identity drift even when old axes have no blockers; error={error}"
        );
    }
}
