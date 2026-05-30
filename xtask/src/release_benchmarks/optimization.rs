use std::fs;
use std::path::Path;

use serde_json::Value;

use super::metrics::{
    max_metric_p50, max_observed_ulp, max_vram_mib, min_first_available_metric_p50,
    min_metric_p50, release_axis_blockers, write_json,
};
use super::suite_inspect::{
    read_text_bounded, record_required_metric_percentile,
    suite_metric_percentile, suite_metric_samples,
};
use super::types::MAX_RELEASE_BENCHMARK_TEXT_BYTES;
use super::types::{
    OptimizationArtifactInspection, OptimizationBenchmarkEvidence, OptimizationBenchmarkManifest,
    ReleaseAxesEvidence,
};

pub(super) fn metric_p50(metric: &Value) -> Option<u64> {
    metric.get("p50").and_then(Value::as_u64)
}

pub(super) fn suite_case_has_cpu_sota_contract(case: &Value, required_speedup: f64) -> bool {
    case.get("contract")
        .and_then(|contract| contract.get("baselines"))
        .and_then(Value::as_array)
        .is_some_and(|baselines| {
            baselines.iter().any(|baseline| {
                baseline.get("class").and_then(Value::as_str) == Some("CpuSota")
                    && baseline
                        .get("min_speedup_x")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0)
                        >= required_speedup
            })
        })
}

pub(super) fn inspect_optimization_benchmark_artifact(
    workspace_root: &Path,
    artifact: &str,
    required_custom_metrics: &[&str],
    required_positive_metrics: &[&str],
) -> OptimizationArtifactInspection {
    let mut blockers = Vec::new();
    let path = workspace_root.join(artifact);
    let (exists, mut read_error) = match fs::metadata(&path) {
        Ok(metadata) if metadata.is_file() && metadata.len() > 0 => (true, None),
        Ok(metadata) if metadata.is_file() => {
            blockers.push("empty".to_string());
            (true, None)
        }
        Ok(_) => {
            blockers.push("not a file".to_string());
            (false, None)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            blockers.push("missing".to_string());
            (false, Some(error.to_string()))
        }
        Err(error) => {
            let message = error.to_string();
            blockers.push(format!("unreadable metadata: {error}"));
            (false, Some(message))
        }
    };
    if !blockers.is_empty() {
        return OptimizationArtifactInspection {
            exists,
            read_error,
            case_count: 0,
            min_wall_samples: None,
            min_wall_p50: None,
            min_wall_p95: None,
            min_wall_p99: None,
            min_baseline_wall_samples: None,
            min_baseline_wall_p50: None,
            min_baseline_wall_p95: None,
            min_baseline_wall_p99: None,
            min_wall_speedup_x1000: None,
            missing_custom_metrics: required_custom_metrics
                .iter()
                .map(|metric| (*metric).to_string())
                .collect(),
            non_positive_required_metrics: required_positive_metrics
                .iter()
                .map(|metric| (*metric).to_string())
                .collect(),
            non_winning_cases: Vec::new(),
            blockers,
        };
    }
    let text = match read_text_bounded(&path, MAX_RELEASE_BENCHMARK_TEXT_BYTES) {
        Ok(text) => text,
        Err(error) => {
            read_error = Some(error.to_string());
            blockers.push(format!("unreadable JSON: {error}"));
            String::new()
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
    let cases = report
        .get("cases")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if cases.is_empty() {
        blockers.push("cases array is empty or missing".to_string());
    }
    let mut min_wall_samples = None::<u64>;
    let mut min_baseline_wall_samples = None::<u64>;
    let mut min_wall_p50 = None::<u64>;
    let mut min_wall_p95 = None::<u64>;
    let mut min_wall_p99 = None::<u64>;
    let mut min_baseline_wall_p50 = None::<u64>;
    let mut min_baseline_wall_p95 = None::<u64>;
    let mut min_baseline_wall_p99 = None::<u64>;
    let mut min_wall_speedup_x1000 = None::<u64>;
    let mut missing_custom_metrics = Vec::new();
    let mut non_positive_required_metrics = Vec::new();
    let mut non_winning_cases = Vec::new();
    for case in &cases {
        let case_id = case
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("<unknown>");
        let metrics = case.get("metrics").and_then(Value::as_object);
        let wall_samples = metrics
            .and_then(|metrics| suite_metric_samples(metrics.get("wall_ns")))
            .unwrap_or(0);
        min_wall_samples = Some(min_wall_samples.map_or(wall_samples, |min| min.min(wall_samples)));
        if wall_samples < 30 {
            blockers.push(format!(
                "case `{case_id}` has {wall_samples} wall_ns sample(s), needs at least 30"
            ));
        }
        let baseline_wall_samples = metrics
            .and_then(|metrics| suite_metric_samples(metrics.get("baseline_wall_ns")))
            .unwrap_or(0);
        min_baseline_wall_samples = Some(
            min_baseline_wall_samples
                .map_or(baseline_wall_samples, |min| min.min(baseline_wall_samples)),
        );
        if baseline_wall_samples < 30 {
            blockers.push(format!(
                "case `{case_id}` has {baseline_wall_samples} baseline_wall_ns sample(s), needs at least 30"
            ));
        }
        record_required_metric_percentile(
            &mut min_wall_p50,
            metrics,
            "wall_ns",
            "p50",
            &mut blockers,
            case_id,
        );
        record_required_metric_percentile(
            &mut min_wall_p95,
            metrics,
            "wall_ns",
            "p95",
            &mut blockers,
            case_id,
        );
        record_required_metric_percentile(
            &mut min_wall_p99,
            metrics,
            "wall_ns",
            "p99",
            &mut blockers,
            case_id,
        );
        record_required_metric_percentile(
            &mut min_baseline_wall_p50,
            metrics,
            "baseline_wall_ns",
            "p50",
            &mut blockers,
            case_id,
        );
        record_required_metric_percentile(
            &mut min_baseline_wall_p95,
            metrics,
            "baseline_wall_ns",
            "p95",
            &mut blockers,
            case_id,
        );
        record_required_metric_percentile(
            &mut min_baseline_wall_p99,
            metrics,
            "baseline_wall_ns",
            "p99",
            &mut blockers,
            case_id,
        );
        match (
            metrics.and_then(|metrics| suite_metric_percentile(metrics.get("wall_ns"), "p50")),
            metrics.and_then(|metrics| {
                suite_metric_percentile(metrics.get("baseline_wall_ns"), "p50")
            }),
        ) {
            (Some(wall), Some(baseline)) if wall > 0 && baseline > wall => {
                let speedup_x1000 = baseline.saturating_mul(1_000) / wall;
                min_wall_speedup_x1000 = Some(
                    min_wall_speedup_x1000.map_or(speedup_x1000, |min| min.min(speedup_x1000)),
                );
            }
            (Some(_), Some(_)) if optimization_semantic_win(case_id, metrics) => {}
            (Some(wall), Some(baseline)) => {
                non_winning_cases.push(format!(
                    "{case_id}:wall_p50={wall}:baseline_wall_p50={baseline}"
                ));
            }
            _ => {
                non_winning_cases.push(format!("{case_id}:missing-wall-or-baseline-p50"));
            }
        }
        for metric in required_custom_metrics {
            if !metrics.is_some_and(|metrics| metrics.contains_key(*metric)) {
                missing_custom_metrics.push(format!("{case_id}:{metric}"));
            }
        }
        for metric in required_positive_metrics {
            let positive = metrics
                .and_then(|metrics| metrics.get(*metric))
                .and_then(metric_p50)
                .is_some_and(|value| value > 0);
            if !positive {
                non_positive_required_metrics.push(format!("{case_id}:{metric}"));
            }
        }
    }
    if !missing_custom_metrics.is_empty() {
        blockers.push(format!(
            "missing required metric(s): {}",
            missing_custom_metrics.join(", ")
        ));
    }
    if !non_positive_required_metrics.is_empty() {
        blockers.push(format!(
            "non-positive required metric(s): {}",
            non_positive_required_metrics.join(", ")
        ));
    }
    if !non_winning_cases.is_empty() {
        blockers.push(format!(
            "optimized wall_ns p50 must beat baseline_wall_ns p50 for every case: {}",
            non_winning_cases.join(", ")
        ));
    }
    OptimizationArtifactInspection {
        exists,
        read_error,
        case_count: cases.len(),
        min_wall_samples,
        min_wall_p50,
        min_wall_p95,
        min_wall_p99,
        min_baseline_wall_samples,
        min_baseline_wall_p50,
        min_baseline_wall_p95,
        min_baseline_wall_p99,
        min_wall_speedup_x1000,
        missing_custom_metrics,
        non_positive_required_metrics,
        non_winning_cases,
        blockers,
    }
}

pub(super) fn optimization_semantic_win(
    case_id: &str,
    metrics: Option<&serde_json::Map<String, Value>>,
) -> bool {
    let Some(metrics) = metrics else {
        return false;
    };
    match case_id {
        "lower.rewrites.impact.corpus" => {
            suite_metric_percentile(metrics.get("lower_ops_eliminated"), "p50")
                .is_some_and(|value| value > 0)
                || suite_metric_percentile(metrics.get("lower_optimized_issue_score"), "p50")
                    .zip(suite_metric_percentile(
                        metrics.get("lower_baseline_issue_score"),
                        "p50",
                    ))
                    .is_some_and(|(optimized, baseline)| optimized < baseline)
        }
        "foundation.optimizer.impact" => {
            suite_metric_percentile(metrics.get("optimizer_nodes_eliminated"), "p50")
                .is_some_and(|value| value > 0)
        }
        "lower.egraph_saturation" => {
            suite_metric_percentile(metrics.get("egraph_applied_rewrites"), "p50")
                .is_some_and(|value| value > 0)
                && suite_metric_percentile(metrics.get("egraph_output_ops"), "p50")
                    .zip(suite_metric_percentile(
                        metrics.get("egraph_baseline_ops_after"),
                        "p50",
                    ))
                    .is_some_and(|(output, baseline)| output < baseline)
        }
        "lower.alias_aware_optimizations" => {
            suite_metric_percentile(metrics.get("alias_pass_wins"), "p50")
                .is_some_and(|value| value >= 5)
        }
        _ => false,
    }
}

pub(super) fn write_release_axes(workspace_root: &Path) {
    let evidence_dir = workspace_root.join("release/evidence/benchmarks");
    let mut reports = Vec::new();
    let mut source_artifacts = Vec::new();
    let mut blockers = Vec::new();
    match fs::read_dir(&evidence_dir) {
        Ok(entries) => {
            for entry in entries {
                let entry = match entry {
                    Ok(entry) => entry,
                    Err(error) => {
                        blockers.push(format!(
                            "failed to read benchmark evidence directory entry: {error}"
                        ));
                        continue;
                    }
                };
                let path = entry.path();
                if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                    continue;
                }
                let text = match read_text_bounded(&path, MAX_RELEASE_BENCHMARK_TEXT_BYTES) {
                    Ok(text) => text,
                    Err(error) => {
                        blockers.push(format!(
                            "failed to read benchmark evidence `{}`: {error}",
                            path.display()
                        ));
                        continue;
                    }
                };
                let value = match serde_json::from_str::<Value>(&text) {
                    Ok(value) => value,
                    Err(error) => {
                        blockers.push(format!(
                            "invalid benchmark evidence JSON `{}`: {error}",
                            path.display()
                        ));
                        continue;
                    }
                };
                if value.get("cases").and_then(Value::as_array).is_none() {
                    continue;
                }
                source_artifacts.push(
                    path.strip_prefix(workspace_root)
                        .unwrap_or(&path)
                        .display()
                        .to_string(),
                );
                reports.push(value);
            }
        }
        Err(error) => blockers.push(format!(
            "failed to read benchmark evidence directory `{}`: {error}",
            evidence_dir.display()
        )),
    }
    blockers.extend(release_axis_blockers(&reports));
    let evidence = ReleaseAxesEvidence {
        schema_version: 1,
        warm_us_per_file: min_metric_p50(&reports, "wall_ns").map(|ns| ns as f64 / 1_000.0),
        cold_pipeline_build_ms: min_first_available_metric_p50(
            &reports,
            &[
                "cold_compile_ns",
                "cold_wall_ns",
                "compile_ns",
                "lower_ns",
                "optimize_ns",
            ],
        )
        .map(|ns| ns as f64 / 1_000_000.0),
        gbs_scan_throughput: max_metric_p50(&reports, "wall_gb_s_x1000")
            .or_else(|| max_metric_p50(&reports, "device_gb_s_x1000"))
            .map(|gb_s_x1000| gb_s_x1000 as f64 / 1_000.0),
        ulp_drift_max: Some(max_observed_ulp(&reports).unwrap_or(0)),
        max_vram_mib: max_vram_mib(&reports),
        source_artifacts,
        blockers,
    };
    write_json(
        &workspace_root.join("release/evidence/benchmarks/bench-release-axes.json"),
        &evidence,
    );
}

pub(super) fn write_optimization_benchmark_manifest(workspace_root: &Path, backend: &str) {
    let specs = [
        (
            "lower.rewrites.impact.corpus",
            "release/evidence/optimization/lower-rewrite-impact-before-after.json",
            vec![
                "memory-layout",
                "control-flow",
                "vector-layout",
                "A13-coalesce-fixture",
                "A14-shared-mem-promote-fixture",
                "A15-bank-conflict-fixture",
                "A16-vec-pack-fixture",
            ],
            vec![
                "lower_ops_before",
                "lower_ops_after",
                "lower_ops_eliminated",
                "lower_coalesce_problematic_before",
                "lower_shared_candidates_before",
                "lower_bank_critical_before",
                "lower_vec_pack_chains_before",
                "lower_vec_pack_ops_eliminable_before",
            ],
            vec![
                "lower_ops_before",
                "lower_ops_eliminated",
                "lower_coalesce_problematic_before",
                "lower_shared_candidates_before",
                "lower_bank_critical_before",
                "lower_vec_pack_chains_before",
                "lower_vec_pack_ops_eliminable_before",
            ],
        ),
        (
            "foundation.optimizer.impact",
            "release/evidence/optimization/optimizer-impact-cuda.json",
            vec!["algebraic", "predicate"],
            vec![
                "optimizer_input_nodes",
                "optimizer_output_nodes",
                "optimizer_nodes_eliminated",
            ],
            vec!["optimizer_input_nodes", "optimizer_output_nodes"],
        ),
        (
            "lower.egraph_saturation",
            "release/evidence/optimization/egraph-before-after.json",
            vec!["egraph"],
            vec![
                "egraph_case_count",
                "egraph_bitwise_case_count",
                "egraph_boolean_case_count",
                "egraph_equality_classes",
                "egraph_applied_rewrites",
            ],
            vec![
                "egraph_case_count",
                "egraph_bitwise_case_count",
                "egraph_boolean_case_count",
                "egraph_equality_classes",
                "egraph_applied_rewrites",
            ],
        ),
        (
            "lower.alias_aware_optimizations",
            "release/evidence/benchmarks/alias-aware-before-after.json",
            vec![
                "weir-dataflow-dse",
                "weir-dataflow-loop-fusion",
                "weir-dataflow-loop-fission",
                "weir-dataflow-licm",
            ],
            vec![
                "alias_pass_wins",
                "alias_fact_count",
                "alias_cross_binding_fact_count",
                "reaching_def_fact_count",
                "alias_total_ops_after",
                "conservative_total_ops_after",
                "alias_dse_store_count",
                "conservative_dse_store_count",
                "alias_stlf_final_value_id",
                "conservative_stlf_final_value_id",
                "alias_licm_loop_loads",
                "conservative_licm_loop_loads",
                "alias_fusion_loop_count",
                "conservative_fusion_loop_count",
                "alias_fission_loop_count",
                "conservative_fission_loop_count",
                "benchmark_repeats",
            ],
            vec![
                "alias_pass_wins",
                "alias_fact_count",
                "alias_cross_binding_fact_count",
                "reaching_def_fact_count",
                "benchmark_repeats",
            ],
        ),
    ];
    let required_pass_families = vec![
        "algebraic",
        "predicate",
        "egraph",
        "memory-layout",
        "control-flow",
        "vector-layout",
        "A13-coalesce-fixture",
        "A14-shared-mem-promote-fixture",
        "A15-bank-conflict-fixture",
        "A16-vec-pack-fixture",
        "weir-dataflow-dse",
        "weir-dataflow-loop-fusion",
        "weir-dataflow-loop-fission",
        "weir-dataflow-licm",
    ];
    let required_case_count = specs.len();
    let mut blockers = Vec::new();
    let mut covered_pass_families = Vec::new();
    let cases = specs
        .into_iter()
        .map(|(
            case_id,
            artifact,
            pass_families,
            required_custom_metrics,
            required_positive_metrics,
        )| {
            let inspection = inspect_optimization_benchmark_artifact(
                workspace_root,
                artifact,
                &required_custom_metrics,
                &required_positive_metrics,
            );
            if !inspection.exists {
                blockers.push(format!(
                    "required optimization benchmark artifact `{artifact}` for `{case_id}` is missing"
                ));
            }
            blockers.extend(inspection.blockers.iter().map(|blocker| {
                format!("optimization benchmark `{case_id}` artifact `{artifact}`: {blocker}")
            }));
            for family in &pass_families {
                covered_pass_families.push(*family);
            }
            OptimizationBenchmarkEvidence {
                case_id,
                artifact,
                covered_pass_families: pass_families,
                required_custom_metrics,
                required_positive_metrics,
                exists: inspection.exists,
                read_error: inspection.read_error,
                case_count: inspection.case_count,
                min_wall_samples: inspection.min_wall_samples,
                min_wall_p50: inspection.min_wall_p50,
                min_wall_p95: inspection.min_wall_p95,
                min_wall_p99: inspection.min_wall_p99,
                min_baseline_wall_samples: inspection.min_baseline_wall_samples,
                min_baseline_wall_p50: inspection.min_baseline_wall_p50,
                min_baseline_wall_p95: inspection.min_baseline_wall_p95,
                min_baseline_wall_p99: inspection.min_baseline_wall_p99,
                min_wall_speedup_x1000: inspection.min_wall_speedup_x1000,
                missing_custom_metrics: inspection.missing_custom_metrics,
                non_positive_required_metrics: inspection.non_positive_required_metrics,
                non_winning_cases: inspection.non_winning_cases,
                blockers: inspection.blockers,
            }
        })
        .collect::<Vec<_>>();
    covered_pass_families.sort_unstable();
    covered_pass_families.dedup();
    let uncovered_pass_families = required_pass_families
        .iter()
        .copied()
        .filter(|family| !covered_pass_families.contains(family))
        .collect::<Vec<_>>();
    for family in &uncovered_pass_families {
        blockers.push(format!(
            "required optimization pass family `{family}` has no benchmark manifest coverage"
        ));
    }
    write_json(
        &workspace_root.join("release/evidence/optimization/pass-family-benchmark-manifest.json"),
        &OptimizationBenchmarkManifest {
            schema_version: 1,
            backend: backend.to_string(),
            required_case_count,
            required_pass_families,
            covered_pass_families,
            uncovered_pass_families,
            cases,
            blockers,
        },
    );
}

