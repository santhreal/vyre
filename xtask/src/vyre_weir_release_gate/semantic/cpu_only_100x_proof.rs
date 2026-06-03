use std::path::Path;

use crate::benchmark_evidence_semantics::{
    benchmark_duplicate_source_artifact_paths, benchmark_source_artifact_count,
    benchmark_source_artifact_entry_count,
};

use super::super::checks::*;
use super::super::types::Requirement;

pub(super) fn check(requirement: &Requirement, base_dir: &Path, failures: &mut Vec<String>) {
    let Some(matrix) = first_json_evidence(
        requirement,
        base_dir,
        "release-workload-matrix.json",
        failures,
    ) else {
        return;
    };
    let contracts = matrix
        .get("cpu_sota_100x_contract_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if contracts < 10 {
        failures.push(format!(
            "requirement `cpu-only-100x-proof` has {contracts} CPU-SOTA 100x contract(s) in the workload matrix; needs at least 10"
        ));
    }
    let covered_families = matrix
        .get("cpu_sota_100x_family_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if covered_families < 10 {
        failures.push(format!(
            "requirement `cpu-only-100x-proof` has {covered_families} covered workload family/families with a CPU-SOTA 100x contract; needs at least 10"
        ));
    }
    let required_hundred_x = matrix
        .get("required_cpu_sota_100x_families")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if required_hundred_x < 10 {
        failures.push(format!(
            "requirement `cpu-only-100x-proof` matrix lists only {required_hundred_x} required 100x family/families; needs at least 10 release 100x families"
        ));
    }
    let missing_required_hundred_x = matrix
        .get("missing_required_cpu_sota_100x_families")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_required_hundred_x != 0 {
        failures.push(format!(
            "requirement `cpu-only-100x-proof` matrix reports {missing_required_hundred_x} missing required 100x family/families"
        ));
    }
    let contract_cases = matrix
        .get("cpu_sota_100x_contract_cases")
        .and_then(serde_json::Value::as_array)
        .map(|cases| {
            cases
                .iter()
                .filter_map(serde_json::Value::as_str)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if contract_cases.is_empty() {
        failures.push(
            "requirement `cpu-only-100x-proof` workload matrix does not list the active 100x contract case ids"
                .to_string(),
        );
    }
    if let Some(proof) =
        first_json_evidence(requirement, base_dir, "cpu-only-100x-proof.json", failures)
    {
        let proof_blockers = proof
            .get("blockers")
            .and_then(serde_json::Value::as_array)
            .map_or(0, Vec::len);
        if proof_blockers != 0 {
            failures.push(format!(
                "requirement `cpu-only-100x-proof` aggregate proof reports {proof_blockers} blocker(s)"
            ));
        }
        if proof
            .get("source_fingerprint")
            .and_then(serde_json::Value::as_str)
            .is_none_or(str::is_empty)
        {
            failures.push(
                "requirement `cpu-only-100x-proof` aggregate proof must preserve source_fingerprint"
                    .to_string(),
            );
        }
        if proof.get("git").is_none_or(serde_json::Value::is_null) {
            failures.push(
                "requirement `cpu-only-100x-proof` aggregate proof must preserve git provenance object"
                    .to_string(),
            );
        }
        check_cpu_100x_source_artifact_counts(&proof, failures);
        let required_proof_cases = proof
            .get("required_cpu_sota_100x_cases")
            .and_then(serde_json::Value::as_array)
            .map_or(0, Vec::len);
        if required_proof_cases < 10 {
            failures.push(format!(
                "requirement `cpu-only-100x-proof` aggregate proof lists {required_proof_cases} required 100x case(s); needs at least 10 release 100x cases"
            ));
        }
        let missing_proof_cases = proof
            .get("missing_required_cpu_sota_100x_cases")
            .and_then(serde_json::Value::as_array)
            .map_or(usize::MAX, Vec::len);
        if missing_proof_cases != 0 {
            failures.push(format!(
                "requirement `cpu-only-100x-proof` aggregate proof reports {missing_proof_cases} missing required 100x case(s)"
            ));
        }
        let proof_contract_case_count = proof
            .get("cases")
            .and_then(serde_json::Value::as_array)
            .map(|cases| {
                for required_case in [
                    "release.condition_eval.1m",
                    "release.string_bitmap_scatter.1m",
                    "release.offset_count_aggregation.1m",
                    "release.entropy_window.1m",
                    "release.quantified_condition_loops.1m",
                    "release.alias_reaching_def.1m",
                    "release.ifds_witness.1m",
                    "release.c_ast_traversal.1m",
                    "release.megakernel_queue.1m",
                    "release.egraph_saturation.1m",
                    "sparse.compaction.count.1m",
                ] {
                    if !cases.iter().any(|case| {
                        case.get("id").and_then(serde_json::Value::as_str)
                            == Some(required_case)
                    }) {
                        failures.push(format!(
                            "requirement `cpu-only-100x-proof` aggregate proof is missing required case `{required_case}`"
                        ));
                    }
                }
                cases.iter().filter(|case| {
                    case.get("id")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|id| contract_cases.contains(&id))
                }).count()
            })
            .unwrap_or(0);
        if proof_contract_case_count < 10 {
            failures.push(format!(
                "requirement `cpu-only-100x-proof` aggregate proof artifact contains {proof_contract_case_count} case(s) listed in cpu_sota_100x_contract_cases; needs at least 10"
            ));
        }
        let aggregate_contract_cases = proof
            .get("cpu_sota_100x_contract_case_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if aggregate_contract_cases < 10 {
            failures.push(format!(
                "requirement `cpu-only-100x-proof` aggregate proof has {aggregate_contract_cases} CPU-SOTA 100x contract case(s); needs at least 10"
            ));
        }
        let aggregate_passing_cases = proof
            .get("cpu_sota_100x_passing_case_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if aggregate_passing_cases < 10 {
            failures.push(format!(
                "requirement `cpu-only-100x-proof` aggregate proof has {aggregate_passing_cases} passing CPU-SOTA 100x case(s); needs at least 10"
            ));
        }
        let min_wall_samples = proof
            .get("min_wall_samples")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if min_wall_samples < 30 {
            failures.push(format!(
                "requirement `cpu-only-100x-proof` aggregate proof min_wall_samples={min_wall_samples}; needs at least 30"
            ));
        }
        let min_baseline_wall_samples = proof
            .get("min_baseline_wall_samples")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if min_baseline_wall_samples < 30 {
            failures.push(format!(
                "requirement `cpu-only-100x-proof` aggregate proof min_baseline_wall_samples={min_baseline_wall_samples}; needs at least 30"
            ));
        }
        for field in [
            "min_wall_p50",
            "min_wall_p95",
            "min_wall_p99",
            "min_baseline_wall_p50",
            "min_baseline_wall_p95",
            "min_baseline_wall_p99",
        ] {
            if proof
                .get(field)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                == 0
            {
                failures.push(format!(
                    "requirement `cpu-only-100x-proof` aggregate proof has non-positive `{field}`"
                ));
            }
        }
    }
    check_benchmark_evidence_reports(
        requirement,
        base_dir,
        "cpu-only-100x-proof.json",
        true,
        Some(100.0),
        failures,
    );
}

fn check_cpu_100x_source_artifact_counts(proof: &serde_json::Value, failures: &mut Vec<String>) {
    let unique_source_artifacts = benchmark_source_artifact_count(proof) as u64;
    let raw_source_artifacts = benchmark_source_artifact_entry_count(proof) as u64;
    let declared_source_artifacts = proof
        .get("source_artifact_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if declared_source_artifacts != unique_source_artifacts {
        failures.push(format!(
            "requirement `cpu-only-100x-proof` aggregate proof source_artifact_count={declared_source_artifacts}, but unique source_artifacts={unique_source_artifacts}"
        ));
    }
    if unique_source_artifacts < 10 {
        failures.push(format!(
            "requirement `cpu-only-100x-proof` aggregate proof has {unique_source_artifacts} unique source artifact(s); needs at least 10"
        ));
    }
    if raw_source_artifacts != unique_source_artifacts {
        let duplicates = benchmark_duplicate_source_artifact_paths(proof)
            .into_iter()
            .collect::<Vec<_>>()
            .join(", ");
        failures.push(format!(
            "requirement `cpu-only-100x-proof` aggregate proof has duplicate source_artifacts: {duplicates}"
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_100x_gate_rejects_duplicate_source_artifact_count_inflation() {
        let proof = serde_json::json!({
            "source_artifact_count": 10,
            "source_artifacts": [
                "release/evidence/benchmarks/workload-01-condition-eval.json",
                "release/evidence/benchmarks/workload-01-condition-eval.json"
            ]
        });
        let mut failures = Vec::new();

        check_cpu_100x_source_artifact_counts(&proof, &mut failures);

        assert!(
            failures.iter().any(|failure| failure.contains(
                "source_artifact_count=10, but unique source_artifacts=1"
            )),
            "Fix: CPU-SOTA release gate must reject declared source_artifact_count inflation; failures={failures:?}"
        );
        assert!(
            failures.iter().any(|failure| failure.contains(
                "duplicate source_artifacts: release/evidence/benchmarks/workload-01-condition-eval.json"
            )),
            "Fix: CPU-SOTA release gate must reject duplicate aggregate source_artifacts; failures={failures:?}"
        );
    }
}
