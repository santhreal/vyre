use std::path::Path;

use super::super::checks::*;
use super::super::gate_inputs::Requirement;

pub(super) fn check(requirement: &Requirement, base_dir: &Path, failures: &mut Vec<String>) {
    let Some(matrix) = first_json_evidence(
        requirement,
        base_dir,
        "optimization-integration-matrix.json",
        failures,
    ) else {
        return;
    };
    let blockers = matrix
        .get("blockers")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if blockers != 0 {
        failures.push(format!(
            "requirement `{}` optimization matrix still reports {blockers} blocker(s)",
            requirement.id
        ));
    }
    match requirement.id.as_str() {
        "optimization-benchmark-proof" => {
            check_before_after_benchmark_report(
                requirement,
                base_dir,
                "optimizer-impact-cuda.json",
                failures,
            );
            check_json_evidence_has_no_blockers(
                requirement,
                base_dir,
                "pass-family-benchmark-manifest.json",
                failures,
            );
            if let Some(manifest) = first_json_evidence(
                requirement,
                base_dir,
                "pass-family-benchmark-manifest.json",
                failures,
            ) {
                if manifest.get("backend").and_then(serde_json::Value::as_str) != Some("cuda") {
                    failures.push(
                        "requirement `optimization-benchmark-proof` pass-family benchmark manifest must be cuda"
                            .to_string(),
                    );
                }
                let cases = manifest
                    .get("cases")
                    .and_then(serde_json::Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                for required_family in REQUIRED_BENCHMARKED_OPTIMIZATION_FAMILIES {
                    let covered = manifest
                        .get("covered_pass_families")
                        .and_then(serde_json::Value::as_array)
                        .is_some_and(|families| {
                            families
                                .iter()
                                .any(|family| family.as_str() == Some(required_family))
                        });
                    if !covered {
                        failures.push(format!(
                            "requirement `optimization-benchmark-proof` pass-family manifest does not benchmark required family `{required_family}`"
                        ));
                    }
                }
                if manifest
                    .get("uncovered_pass_families")
                    .and_then(serde_json::Value::as_array)
                    .is_none_or(|families| !families.is_empty())
                {
                    failures.push(
                        "requirement `optimization-benchmark-proof` pass-family manifest reports uncovered pass families"
                            .to_string(),
                    );
                }
                for required_case in ["foundation.optimizer.impact"] {
                    if !cases.iter().any(|case| {
                        case.get("case_id").and_then(serde_json::Value::as_str)
                            == Some(required_case)
                            && case.get("exists").and_then(serde_json::Value::as_bool) == Some(true)
                            && case
                                .get("read_error")
                                .is_some_and(serde_json::Value::is_null)
                            && case
                                .get("required_custom_metrics")
                                .and_then(serde_json::Value::as_array)
                                .is_some_and(|metrics| !metrics.is_empty())
                            && case
                                .get("required_positive_metrics")
                                .and_then(serde_json::Value::as_array)
                                .is_some_and(|metrics| !metrics.is_empty())
                    }) {
                        failures.push(format!(
                            "requirement `optimization-benchmark-proof` pass-family manifest is missing `{required_case}`"
                        ));
                    }
                }
                for case in &cases {
                    let Some(artifact) = case.get("artifact").and_then(serde_json::Value::as_str)
                    else {
                        failures.push(
                            "requirement `optimization-benchmark-proof` pass-family manifest case is missing artifact"
                                .to_string(),
                        );
                        continue;
                    };
                    if case
                        .get("covered_pass_families")
                        .and_then(serde_json::Value::as_array)
                        .is_none_or(|families| families.is_empty())
                    {
                        failures.push(
                            "requirement `optimization-benchmark-proof` pass-family manifest case lists no covered_pass_families"
                                .to_string(),
                        );
                    }
                    for field in [
                        "missing_custom_metrics",
                        "non_positive_required_metrics",
                        "non_winning_cases",
                        "blockers",
                    ] {
                        if case
                            .get(field)
                            .and_then(serde_json::Value::as_array)
                            .is_none_or(|items| !items.is_empty())
                        {
                            failures.push(format!(
                                "requirement `optimization-benchmark-proof` pass-family manifest case `{}` has non-empty `{field}`",
                                case.get("case_id")
                                    .and_then(serde_json::Value::as_str)
                                    .unwrap_or("<unknown>")
                            ));
                        }
                    }
                    let read_error = case.get("read_error");
                    if !read_error.is_some_and(serde_json::Value::is_null) {
                        failures.push(format!(
                            "requirement `optimization-benchmark-proof` pass-family manifest case `{}` read_error={}",
                            case.get("case_id")
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or("<unknown>"),
                            read_error
                                .map(serde_json::Value::to_string)
                                .unwrap_or_else(|| "<missing>".to_string())
                        ));
                    }
                    for field in ["min_wall_samples", "min_baseline_wall_samples"] {
                        if case
                            .get(field)
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(0)
                            < 30
                        {
                            failures.push(format!(
                                "requirement `optimization-benchmark-proof` pass-family manifest case `{}` has `{field}` below 30",
                                case.get("case_id")
                                    .and_then(serde_json::Value::as_str)
                                    .unwrap_or("<unknown>")
                            ));
                        }
                    }
                    for field in [
                        "min_wall_p50",
                        "min_wall_p95",
                        "min_wall_p99",
                        "min_baseline_wall_p50",
                        "min_baseline_wall_p95",
                        "min_baseline_wall_p99",
                    ] {
                        if case
                            .get(field)
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(0)
                            == 0
                        {
                            failures.push(format!(
                                "requirement `optimization-benchmark-proof` pass-family manifest case `{}` has non-positive `{field}`",
                                case.get("case_id")
                                    .and_then(serde_json::Value::as_str)
                                    .unwrap_or("<unknown>")
                            ));
                        }
                    }
                    let has_speed_win = case
                        .get("min_wall_speedup_x1000")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0)
                        > 1_000;
                    let has_semantic_win = case
                        .get("non_winning_cases")
                        .and_then(serde_json::Value::as_array)
                        .is_some_and(|items| items.is_empty());
                    if !has_speed_win && !has_semantic_win {
                        failures.push(format!(
                            "requirement `optimization-benchmark-proof` pass-family manifest case `{}` does not prove optimized wall_ns p50 beats baseline_wall_ns p50",
                            case.get("case_id")
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or("<unknown>")
                        ));
                    }
                    let Some(report) =
                        read_json_artifact_ref(requirement, base_dir, artifact, failures)
                    else {
                        continue;
                    };
                    let suffix = Path::new(artifact)
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or(artifact);
                    if let Some(metrics) = case
                        .get("required_custom_metrics")
                        .and_then(serde_json::Value::as_array)
                    {
                        for metric in metrics.iter().filter_map(serde_json::Value::as_str) {
                            require_case_metric_present(
                                requirement,
                                suffix,
                                &report,
                                metric,
                                failures,
                            );
                        }
                    }
                    if let Some(metrics) = case
                        .get("required_positive_metrics")
                        .and_then(serde_json::Value::as_array)
                    {
                        for metric in metrics.iter().filter_map(serde_json::Value::as_str) {
                            require_case_metric_positive(
                                requirement,
                                suffix,
                                &report,
                                metric,
                                failures,
                            );
                        }
                    }
                }
            }
        }
        "semantic-optimizer-registration" => {
            check_optimizer_catalog_families(
                requirement,
                &matrix,
                &[
                    "const_fold",
                    "canonicalize",
                    "memory.dead_store_elim",
                    "memory.store_to_load_forward",
                    "loop.licm",
                    "loop.fusion",
                    "loop.fission",
                ],
                failures,
            );
        }
        _ => {}
    }
}

fn check_optimizer_catalog_families(
    requirement: &Requirement,
    matrix: &serde_json::Value,
    required_ids: &[&str],
    failures: &mut Vec<String>,
) {
    let entries = matrix
        .get("entries")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    for required_id in required_ids {
        let found = entries.iter().any(|entry| {
            entry.get("id").and_then(serde_json::Value::as_str) == Some(required_id)
                && entry
                    .get("owner")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|owner| owner.starts_with("vyre-foundation"))
                && entry.get("input").and_then(serde_json::Value::as_str)
                    == Some("vyre-foundation Program")
                && entry.get("output").and_then(serde_json::Value::as_str)
                    == Some("semantically equivalent vyre-foundation Program")
                && entry
                    .get("proof")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|proof| !proof.is_empty())
                && entry
                    .get("benchmark")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|benchmark| !benchmark.is_empty())
        });
        if !found {
            failures.push(format!(
                "requirement `{}` source-owned optimizer matrix is missing complete semantic registration `{required_id}`",
                requirement.id
            ));
        }
    }
}
