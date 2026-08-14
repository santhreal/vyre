//! The aggregate CPU-SOTA 100x proof artifact.
//!
//! One release claim  -  100x over CPU state of the art  -  is proved by
//! folding several per-workload benchmark artifacts into a single proof.
//! That aggregation has its own provenance, duplicate, and component-parity
//! rules, none of which the per-backend suite shares, so it owns its own
//! module.

use std::collections::BTreeSet;
use std::path::Path;

use serde_json::{json, Value};

use super::inspect_core::{first_metric_p50, read_text_bounded, WallClockMinima};
use super::metrics::write_json;
use super::release_thresholds::{
    MAX_RELEASE_BENCHMARK_TEXT_BYTES, MIN_CPU_SOTA_100X_RELEASE_CASES, REQUIRED_CPU_SOTA_100X_CASES,
};
use super::suite_inspect::nonblank_str;

pub(super) fn write_cpu_100x_proof(workspace_root: &Path, artifacts: &[String]) {
    let mut cases = Vec::new();
    let mut blockers = Vec::new();
    let mut contract_case_count = 0usize;
    let mut passing_contract_case_count = 0usize;
    let mut minima = WallClockMinima::default();
    let mut observed_required_cases = std::collections::BTreeSet::new();
    let mut environment = None::<Value>;
    let mut git = None::<Value>;
    let mut source_fingerprint = None::<String>;
    let mut source_tree_fingerprint = None::<String>;
    let mut unique_artifacts = BTreeSet::new();
    let mut component_rows = Vec::new();
    let mut component_blockers = Vec::new();
    for artifact in artifacts {
        if !unique_artifacts.insert(artifact.clone()) {
            blockers.push(format!(
                "100x proof source_artifact `{artifact}` is duplicated; aggregate proof counts must use distinct source artifacts"
            ));
        }
    }
    let artifacts = unique_artifacts.into_iter().collect::<Vec<_>>();
    for artifact in &artifacts {
        if let Some(issue) =
            crate::bench::benchmark_evidence_semantics::benchmark_source_artifact_path_issue(
                workspace_root,
                artifact,
            )
        {
            blockers.push(format!(
                "100x {}",
                issue.describe("source_artifact", artifact)
            ));
            continue;
        }
        let path = workspace_root.join(artifact);
        let text = match read_text_bounded(&path, MAX_RELEASE_BENCHMARK_TEXT_BYTES) {
            Ok(text) => text,
            Err(error) => {
                blockers.push(format!(
                    "100x source artifact `{artifact}` is unreadable: {error}"
                ));
                continue;
            }
        };
        let Ok(report) = serde_json::from_str::<Value>(&text) else {
            blockers.push(format!("100x source artifact `{artifact}` is invalid JSON"));
            continue;
        };
        if report.get("selected_backend").and_then(Value::as_str) != Some("cuda") {
            blockers.push(format!(
                "100x source artifact `{artifact}` was not produced for cuda"
            ));
        }
        crate::bench::benchmark_evidence_semantics::inspect_source_artifact_case_integrity(
            artifact,
            &report,
            "CPU-SOTA aggregate proof",
            &mut blockers,
        );
        if environment.is_none() {
            environment = report.get("environment").cloned();
        }
        if git.is_none() {
            git = report.get("git").cloned();
        }
        let report_source_fingerprint = report
            .get("source_fingerprint")
            .and_then(nonblank_str)
            .map(str::to_string);
        if let Some(fingerprint) = &report_source_fingerprint {
            if !crate::bench::benchmark_evidence_semantics::source_fingerprint_issues(fingerprint)
                .is_empty()
            {
                blockers.push(format!(
                    "100x source artifact `{artifact}` source_fingerprint `{fingerprint}` is not release-grade provenance"
                ));
            }
        } else {
            blockers.push(format!(
                "100x source artifact `{artifact}` has no source_fingerprint"
            ));
        }
        if source_fingerprint.is_none() {
            if let Some(fingerprint) = &report_source_fingerprint {
                source_fingerprint = Some(fingerprint.clone());
            }
        }
        let report_source_tree_fingerprint = report
            .get("source_tree_fingerprint")
            .and_then(nonblank_str)
            .map(str::to_string);
        match (&source_tree_fingerprint, &report_source_tree_fingerprint) {
            (None, Some(fingerprint)) => source_tree_fingerprint = Some(fingerprint.clone()),
            (Some(expected), Some(actual)) if expected != actual => blockers.push(format!(
                "100x source artifact `{artifact}` source_tree_fingerprint `{actual}` does not match aggregate source tree `{expected}`"
            )),
            _ => {}
        }
        if report_source_tree_fingerprint.is_none() {
            blockers.push(format!(
                "100x source artifact `{artifact}` has no source_tree_fingerprint"
            ));
        }
        if let (Some((field, freshness_fingerprint)), Some(current_freshness_fingerprint)) = (
            crate::bench::benchmark_evidence_semantics::report_freshness_fingerprint(&report),
            crate::bench::benchmark_evidence_semantics::current_freshness_fingerprint_for_report(
                &path, &report,
            ),
        ) {
            for issue in
                crate::bench::benchmark_evidence_semantics::source_fingerprint_freshness_issues(
                    freshness_fingerprint,
                    &current_freshness_fingerprint,
                )
            {
                match issue {
                    crate::bench::benchmark_evidence_semantics::SourceFingerprintFreshnessIssue::Mismatch {
                        source_fingerprint,
                        current_source_fingerprint,
                    } => blockers.push(format!(
                        "100x source artifact `{artifact}` {field} `{source_fingerprint}` does not match current workspace source `{current_source_fingerprint}`"
                    )),
                }
            }
        }
        let Some(report_cases) = report.get("cases").and_then(Value::as_array) else {
            blockers.push(format!(
                "100x source artifact `{artifact}` has no cases array"
            ));
            continue;
        };
        let (report_contract_case_count, report_passing_contract_case_count) =
            crate::bench::benchmark_evidence_semantics::cpu_sota_100x_case_counts(&report);
        contract_case_count += report_contract_case_count as usize;
        passing_contract_case_count += report_passing_contract_case_count as usize;
        for case in report_cases {
            let case_id = case
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("<unknown>");
            let case_failure_reason =
                crate::bench::benchmark_evidence_semantics::benchmark_case_failure_reason(case);
            if let Some(reason) = &case_failure_reason {
                blockers.push(format!(
                    "100x source artifact `{artifact}` case `{case_id}` failed: {reason}"
                ));
            }
            let metrics = case.get("metrics").and_then(Value::as_object);
            let case_backend = case
                .get("backend_id")
                .and_then(Value::as_str)
                .or_else(|| report.get("selected_backend").and_then(Value::as_str));
            if REQUIRED_CPU_SOTA_100X_CASES.contains(&case_id)
                && crate::bench::benchmark_evidence_semantics::benchmark_case_proves_cpu_sota_100x(
                    case,
                    case_backend,
                )
            {
                observed_required_cases.insert(case_id.to_string());
                component_rows.push(cpu_sota_component_proof_case(
                    artifact,
                    case_id,
                    case,
                    metrics,
                    &mut component_blockers,
                ));
            }
            minima.record_case(
                case_id,
                &format!("100x source artifact `{artifact}` case `{case_id}`"),
                metrics,
                &mut blockers,
            );
        }
        cases.extend(report_cases.iter().cloned());
    }
    if artifacts.len() < MIN_CPU_SOTA_100X_RELEASE_CASES {
        blockers.push(format!(
            "100x proof has {} source artifact(s); release requires at least {} CPU-SOTA 100x workload families",
            artifacts.len(),
            MIN_CPU_SOTA_100X_RELEASE_CASES
        ));
    }
    if cases.len() < MIN_CPU_SOTA_100X_RELEASE_CASES {
        blockers.push(format!(
            "100x proof has {} benchmark case(s); release requires at least {}",
            cases.len(),
            MIN_CPU_SOTA_100X_RELEASE_CASES
        ));
    }
    if contract_case_count < MIN_CPU_SOTA_100X_RELEASE_CASES {
        blockers.push(format!(
            "100x proof has {contract_case_count} CPU-SOTA 100x contract case(s); release requires at least {MIN_CPU_SOTA_100X_RELEASE_CASES}"
        ));
    }
    if passing_contract_case_count < MIN_CPU_SOTA_100X_RELEASE_CASES {
        blockers.push(format!(
            "100x proof has {passing_contract_case_count} passing CPU-SOTA 100x case(s); release requires at least {MIN_CPU_SOTA_100X_RELEASE_CASES}"
        ));
    }
    let missing_required_cases = REQUIRED_CPU_SOTA_100X_CASES
        .iter()
        .copied()
        .filter(|required| !observed_required_cases.contains(*required))
        .collect::<Vec<_>>();
    for required in &missing_required_cases {
        blockers.push(format!(
            "100x proof is missing required release-defining case `{required}`"
        ));
    }
    let aggregate_failed = cases.len().saturating_sub(passing_contract_case_count);
    blockers.extend(component_blockers.iter().cloned());
    let mut evidence = json!({
        "schema_version": 1,
        "selected_backend": "cuda",
        "environment": environment,
        "git": git,
        "source_fingerprint": source_fingerprint,
        "source_tree_fingerprint": source_tree_fingerprint,
        "source_artifacts": &artifacts,
        "source_artifact_count": artifacts.len(),
        "required_cpu_sota_100x_cases": REQUIRED_CPU_SOTA_100X_CASES,
        "missing_required_cpu_sota_100x_cases": missing_required_cases,
        "cpu_sota_100x_contract_case_count": contract_case_count,
        "cpu_sota_100x_passing_case_count": passing_contract_case_count,
        "component_speedup_proof": {
            "schema_version": 2,
            "comparator_identity": "cpu-sota-end-to-end-speedup:v2",
            "collapsed_speedup_field_allowed": false,
            "required_components": [
                "cpu_sota",
                "gpu_active",
                "transfer",
                "end_to_end"
            ],
            "parity_policy": "exact_cpu_gpu_digest",
            "case_count": component_rows.len(),
            "missing_component_count": component_blockers.len(),
            "missing_components": component_blockers,
            "cases": component_rows
        },
        "summary": {
            "total_cases": cases.len(),
            "passed": passing_contract_case_count,
            "failed": aggregate_failed,
            "total_time_ns": 0,
            "cache_hit_rate": null,
        },
        "cases": cases,
        "blockers": blockers,
    });
    evidence
        .as_object_mut()
        .expect("Fix: the 100x proof evidence is a JSON object.")
        .extend(minima.into_object());
    write_json(
        &workspace_root.join("release/evidence/benchmarks/cpu-only-100x-proof.json"),
        &evidence,
    );
}

fn cpu_sota_component_proof_case(
    artifact: &str,
    case_id: &str,
    case: &Value,
    metrics: Option<&serde_json::Map<String, Value>>,
    blockers: &mut Vec<String>,
) -> Value {
    let cpu_sota_wall_ns_p50 = first_metric_p50(metrics, &["baseline_wall_ns"]);
    let gpu_active_ns_p50 = first_metric_p50(
        metrics,
        &[
            "active_time_ns",
            "gpu_active_ns",
            "kernel_execute_ns",
            "dispatch_ns",
        ],
    );
    let transfer_bytes_p50 = first_metric_p50(metrics, &["transfer_bytes", "gpu_transfer_bytes"]);
    let end_to_end_wall_ns_p50 = first_metric_p50(metrics, &["wall_ns"]);
    let cpu_digest = first_metric_p50(metrics, &["cpu_digest"]);
    let gpu_digest = first_metric_p50(metrics, &["gpu_digest"]);
    for (field, value) in [
        ("cpu_sota_wall_ns_p50", cpu_sota_wall_ns_p50),
        ("gpu_active_ns_p50", gpu_active_ns_p50),
        ("transfer_bytes_p50", transfer_bytes_p50),
        ("end_to_end_wall_ns_p50", end_to_end_wall_ns_p50),
        ("cpu_digest", cpu_digest),
        ("gpu_digest", gpu_digest),
    ] {
        if value.is_none_or(|value| value == 0) {
            blockers.push(format!(
                "100x component proof source artifact `{artifact}` case `{case_id}` is missing {field}"
            ));
        }
    }
    let parity_passed = matches!((cpu_digest, gpu_digest), (Some(cpu), Some(gpu)) if cpu != 0 && cpu == gpu)
        && case
            .get("correctness")
            .and_then(|correctness| correctness.get("Invalid"))
            .is_none();
    if !parity_passed {
        blockers.push(format!(
            "100x component proof source artifact `{artifact}` case `{case_id}` must prove exact CPU/GPU digest parity"
        ));
    }
    json!({
        "artifact": artifact,
        "case_id": case_id,
        "cpu_sota_wall_ns_p50": cpu_sota_wall_ns_p50,
        "gpu_active_ns_p50": gpu_active_ns_p50,
        "transfer_bytes_p50": transfer_bytes_p50,
        "end_to_end_wall_ns_p50": end_to_end_wall_ns_p50,
        "cpu_digest": cpu_digest,
        "gpu_digest": gpu_digest,
        "parity_passed": parity_passed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report_fixture::{
        case_summary, cpu_sota_contract, hidden_invalid_measured_case, percentile_metrics,
    };

    use std::fs;

    use tempfile::TempDir;

    /// CPU-SOTA proof rows use the measured comparator baseline from each
    /// performance contract instead of requiring unmeasured scalar/SIMD lanes.
    #[test]
    fn cpu_sota_component_proof_uses_measured_release_metrics() {
        let case = json!({"correctness": "Exact"});
        let metrics = json!({
            "baseline_wall_ns": {"p50": 100_000},
            "active_time_ns": {"p50": 500},
            "transfer_bytes": {"p50": 4096},
            "wall_ns": {"p50": 900},
            "cpu_digest": {"p50": 73},
            "gpu_digest": {"p50": 73}
        });
        let mut blockers = Vec::new();

        let proof = cpu_sota_component_proof_case(
            "workload.json",
            "release.case",
            &case,
            metrics.as_object(),
            &mut blockers,
        );

        assert_eq!(blockers, Vec::<String>::new());
        assert_eq!(
            proof.get("cpu_sota_wall_ns_p50").and_then(Value::as_u64),
            Some(100_000)
        );
        assert_eq!(
            proof.get("gpu_active_ns_p50").and_then(Value::as_u64),
            Some(500)
        );
        assert_eq!(
            proof.get("parity_passed").and_then(Value::as_bool),
            Some(true)
        );
    }

    /// A missing CPU-SOTA baseline remains an explicit blocker. The release
    /// proof must never substitute GPU or end-to-end time for the comparator.
    #[test]
    fn cpu_sota_component_proof_rejects_missing_baseline() {
        let case = json!({"correctness": "Exact"});
        let metrics = json!({
            "active_time_ns": {"p50": 500},
            "transfer_bytes": {"p50": 4096},
            "wall_ns": {"p50": 900},
            "cpu_digest": {"p50": 73},
            "gpu_digest": {"p50": 73}
        });
        let mut blockers = Vec::new();

        cpu_sota_component_proof_case(
            "workload.json",
            "release.case",
            &case,
            metrics.as_object(),
            &mut blockers,
        );

        assert_eq!(
            blockers,
            vec![
                "100x component proof source artifact `workload.json` case `release.case` is missing cpu_sota_wall_ns_p50"
                    .to_string()
            ]
        );
    }

    #[test]
    fn cpu_100x_proof_rejects_case_failure_hidden_by_passing_contract() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for CPU-SOTA proof regression test.");
        let artifact_rel = "release/evidence/benchmarks/cuda-hidden-invalid.json";
        let artifact_path = dir.path().join(artifact_rel);
        fs::create_dir_all(
            artifact_path
                .parent()
                .expect("Fix: CPU-SOTA proof artifact path must have a parent directory."),
        )
        .expect("Fix: create CPU-SOTA proof artifact parent directory.");
        fs::write(
            &artifact_path,
            serde_json::to_string_pretty(&json!({
                "schema_version": 2,
                "selected_backend": "cuda",
                "summary": case_summary(1, 0),
                "cases": [hidden_invalid_measured_case(
                    "release.condition_eval.1m",
                    "cuda",
                    cpu_sota_contract("release condition eval", &["cuda"]),
                    percentile_metrics([10, 11, 12], [2000, 2001, 2002]),
                )]
            }))
            .expect("Fix: serialize hidden-invalid CUDA benchmark artifact JSON."),
        )
        .expect("Fix: write hidden-invalid CUDA benchmark artifact JSON.");

        write_cpu_100x_proof(dir.path(), &[artifact_rel.to_string()]);

        let proof_path = dir
            .path()
            .join("release/evidence/benchmarks/cpu-only-100x-proof.json");
        let proof_text = fs::read_to_string(&proof_path)
            .expect("Fix: read generated CPU-SOTA 100x proof artifact.");
        let proof = serde_json::from_str::<Value>(&proof_text)
            .expect("Fix: generated CPU-SOTA 100x proof must be valid JSON.");

        assert_eq!(
            proof
                .get("cpu_sota_100x_passing_case_count")
                .and_then(Value::as_u64),
            Some(0),
            "Fix: invalid correctness evidence must disqualify a case from passing CPU-SOTA proof even when performance says contract_passed=true."
        );
        assert_eq!(
            proof
                .get("summary")
                .and_then(|summary| summary.get("failed"))
                .and_then(Value::as_u64),
            Some(1),
            "Fix: aggregate CPU-SOTA proof summary must count hidden invalid cases as failed."
        );
        let blockers = proof
            .get("blockers")
            .and_then(Value::as_array)
            .expect("Fix: generated CPU-SOTA proof must include blockers array.");
        assert!(
            blockers
                .iter()
                .filter_map(Value::as_str)
                .any(|blocker| blocker.contains(
                    "100x source artifact `release/evidence/benchmarks/cuda-hidden-invalid.json` case `release.condition_eval.1m` failed: CUDA/WGPU output mismatch at row 17"
                )),
            "Fix: aggregate CPU-SOTA proof blockers must preserve hidden case failure reasons; blockers={blockers:?}"
        );
    }

    #[test]
    fn cpu_100x_proof_rejects_missing_pass_status_with_passing_contract() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for missing-status CPU-SOTA proof test.");
        let artifact_rel = "release/evidence/benchmarks/cuda-missing-status.json";
        let artifact_path = dir.path().join(artifact_rel);
        fs::create_dir_all(
            artifact_path
                .parent()
                .expect("Fix: CPU-SOTA proof artifact path must have a parent directory."),
        )
        .expect("Fix: create missing-status CPU-SOTA proof artifact parent directory.");
        fs::write(
            &artifact_path,
            serde_json::to_string_pretty(&json!({
                "schema_version": 2,
                "selected_backend": "cuda",
                "summary": case_summary(0, 1),
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "cuda",
                        "metrics": {
                            "wall_ns": {"samples": 30, "p50": 10, "p95": 11, "p99": 12},
                            "baseline_wall_ns": {"samples": 30, "p50": 2000, "p95": 2001, "p99": 2002}
                        },
                        "contract": cpu_sota_contract("release condition eval", &["cuda"]),
                        "performance": {"contract_passed": true, "speedup_x": 200.0}
                    }
                ]
            }))
            .expect("Fix: serialize missing-status CUDA benchmark artifact JSON."),
        )
        .expect("Fix: write missing-status CUDA benchmark artifact JSON.");

        write_cpu_100x_proof(dir.path(), &[artifact_rel.to_string()]);

        let proof_path = dir
            .path()
            .join("release/evidence/benchmarks/cpu-only-100x-proof.json");
        let proof_text = fs::read_to_string(&proof_path)
            .expect("Fix: read generated CPU-SOTA 100x proof artifact.");
        let proof = serde_json::from_str::<Value>(&proof_text)
            .expect("Fix: generated CPU-SOTA 100x proof must be valid JSON.");

        assert_eq!(
            proof
                .get("cpu_sota_100x_contract_case_count")
                .and_then(Value::as_u64),
            Some(1),
            "Fix: missing pass status must not erase applicable CPU-SOTA contracts from aggregate proof."
        );
        assert_eq!(
            proof
                .get("cpu_sota_100x_passing_case_count")
                .and_then(Value::as_u64),
            Some(0),
            "Fix: aggregate CPU-SOTA proof must require explicit pass status before counting a passing 100x case."
        );
        assert_eq!(
            proof
                .get("summary")
                .and_then(|summary| summary.get("failed"))
                .and_then(Value::as_u64),
            Some(1),
            "Fix: aggregate CPU-SOTA proof summary must count missing pass status cases as failed."
        );
        let blockers = proof
            .get("blockers")
            .and_then(Value::as_array)
            .expect("Fix: generated CPU-SOTA proof must include blockers array.");
        assert!(
            blockers
                .iter()
                .filter_map(Value::as_str)
                .any(|blocker| blocker.contains(
                    "100x source artifact `release/evidence/benchmarks/cuda-missing-status.json` case `release.condition_eval.1m` failed: missing pass status"
                )),
            "Fix: aggregate CPU-SOTA proof blockers must expose missing pass status; blockers={blockers:?}"
        );
    }

    #[test]
    fn cpu_100x_proof_requires_each_release_defining_case_to_pass_100x() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for required CPU-SOTA proof test.");
        let artifact_rel = "release/evidence/benchmarks/cuda-required-case-failed.json";
        let artifact_path = dir.path().join(artifact_rel);
        fs::create_dir_all(
            artifact_path
                .parent()
                .expect("Fix: required CPU-SOTA artifact path must have a parent directory."),
        )
        .expect("Fix: create required CPU-SOTA proof artifact parent directory.");
        let cases = REQUIRED_CPU_SOTA_100X_CASES
            .iter()
            .enumerate()
            .map(|(index, case_id)| {
                json!({
                    "id": case_id,
                    "backend_id": "cuda",
                    "status": if index == 0 { "fail" } else { "pass" },
                    "metrics": {
                        "wall_ns": {"samples": 30, "p50": 10, "p95": 11, "p99": 12},
                        "baseline_wall_ns": {"samples": 30, "p50": 2000, "p95": 2001, "p99": 2002}
                    },
                    "contract": cpu_sota_contract("release CPU-SOTA required case", &["cuda"]),
                    "performance": {"contract_passed": true, "speedup_x": 200.0}
                })
            })
            .collect::<Vec<_>>();
        fs::write(
            &artifact_path,
            serde_json::to_string_pretty(&json!({
                "schema_version": 2,
                "selected_backend": "cuda",
                "source_fingerprint": "git:source-a:dirty=false",
                "source_tree_fingerprint": "source-tree-v1:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "summary": {
                    "total_cases": REQUIRED_CPU_SOTA_100X_CASES.len(),
                    "passed": REQUIRED_CPU_SOTA_100X_CASES.len() - 1,
                    "failed": 1,
                    "total_time_ns": 0,
                    "cache_hit_rate": null
                },
                "cases": cases
            }))
            .expect("Fix: serialize required CPU-SOTA benchmark artifact JSON."),
        )
        .expect("Fix: write required CPU-SOTA benchmark artifact JSON.");

        write_cpu_100x_proof(dir.path(), &[artifact_rel.to_string()]);

        let proof_path = dir
            .path()
            .join("release/evidence/benchmarks/cpu-only-100x-proof.json");
        let proof_text = fs::read_to_string(&proof_path)
            .expect("Fix: read generated CPU-SOTA 100x proof artifact.");
        let proof = serde_json::from_str::<Value>(&proof_text)
            .expect("Fix: generated CPU-SOTA 100x proof must be valid JSON.");
        let blockers = proof
            .get("blockers")
            .and_then(Value::as_array)
            .expect("Fix: generated CPU-SOTA proof must include blockers array.");

        assert!(
            blockers.iter().filter_map(Value::as_str).any(|blocker| {
                blocker.contains(
                    "100x proof is missing required release-defining case `release.condition_eval.1m`",
                )
            }),
            "Fix: required CPU-SOTA cases must be counted as present only when they prove a passing 100x CUDA win; blockers={blockers:?}"
        );
    }

    #[test]
    fn cpu_100x_proof_rejects_claimed_speedup_without_measured_100x() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for CPU-SOTA measured speedup test.");
        let artifact_rel = "release/evidence/benchmarks/cuda-claimed-speedup.json";
        let artifact_path = dir.path().join(artifact_rel);
        fs::create_dir_all(
            artifact_path
                .parent()
                .expect("Fix: CPU-SOTA proof artifact path must have a parent directory."),
        )
        .expect("Fix: create measured-speedup CPU-SOTA proof artifact parent directory.");
        fs::write(
            &artifact_path,
            serde_json::to_string_pretty(&json!({
                "schema_version": 2,
                "selected_backend": "cuda",
                "summary": case_summary(1, 0),
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "cuda",
                        "status": "pass",
                        "metrics": {
                            "wall_ns": {"samples": 30, "p50": 100, "p95": 101, "p99": 102},
                            "baseline_wall_ns": {"samples": 30, "p50": 1000, "p95": 1001, "p99": 1002}
                        },
                        "contract": cpu_sota_contract("release condition eval", &["cuda"]),
                        "performance": {"contract_passed": true, "speedup_x": 200.0}
                    }
                ]
            }))
            .expect("Fix: serialize claimed-speedup CUDA benchmark artifact JSON."),
        )
        .expect("Fix: write claimed-speedup CUDA benchmark artifact JSON.");

        write_cpu_100x_proof(dir.path(), &[artifact_rel.to_string()]);

        let proof_path = dir
            .path()
            .join("release/evidence/benchmarks/cpu-only-100x-proof.json");
        let proof_text = fs::read_to_string(&proof_path)
            .expect("Fix: read generated CPU-SOTA measured-speedup proof artifact.");
        let proof = serde_json::from_str::<Value>(&proof_text)
            .expect("Fix: generated CPU-SOTA measured-speedup proof must be valid JSON.");

        assert_eq!(
            proof
                .get("cpu_sota_100x_contract_case_count")
                .and_then(Value::as_u64),
            Some(1),
            "Fix: measured-speedup failure must not erase applicable CPU-SOTA contracts from aggregate proof."
        );
        assert_eq!(
            proof
                .get("cpu_sota_100x_passing_case_count")
                .and_then(Value::as_u64),
            Some(0),
            "Fix: aggregate CPU-SOTA proof must not count claimed speedup_x without measured baseline_wall_ns / wall_ns >= 100x."
        );
        assert_eq!(
            proof
                .get("summary")
                .and_then(|summary| summary.get("failed"))
                .and_then(Value::as_u64),
            Some(1),
            "Fix: aggregate CPU-SOTA proof summary must count claimed-only speedup cases as failed."
        );
    }

    #[test]
    fn cpu_100x_proof_surfaces_source_artifact_integrity_blockers() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for CPU-SOTA integrity blocker test.");
        let artifact_rel = "release/evidence/benchmarks/cuda-integrity-drift.json";
        let artifact_path = dir.path().join(artifact_rel);
        fs::create_dir_all(
            artifact_path
                .parent()
                .expect("Fix: CPU-SOTA integrity artifact path must have a parent directory."),
        )
        .expect("Fix: create CPU-SOTA integrity artifact parent directory.");
        fs::write(
            &artifact_path,
            serde_json::to_string_pretty(&json!({
                "schema_version": 2,
                "selected_backend": "cuda",
                "summary": case_summary(1, 0),
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "cuda",
                        "status": "pass",
                        "optimization_passes_applied": ["cuda-resident-borrowed-escape-hatch"],
                        "metrics": {
                            "wall_ns": {"samples": 30, "p50": 10, "p95": 11, "p99": 12},
                            "baseline_wall_ns": {"samples": 30, "p50": 2000, "p95": 2001, "p99": 2002},
                            "cuda_resident_borrowed_fallback_dispatches": {"p50": 2.0}
                        },
                        "contract": cpu_sota_contract("release condition eval", &["wgpu"]),
                        "performance": {"contract_passed": true, "speedup_x": 200.0}
                    }
                ]
            }))
            .expect("Fix: serialize CPU-SOTA integrity benchmark artifact JSON."),
        )
        .expect("Fix: write CPU-SOTA integrity benchmark artifact JSON.");

        write_cpu_100x_proof(dir.path(), &[artifact_rel.to_string()]);

        let proof_path = dir
            .path()
            .join("release/evidence/benchmarks/cpu-only-100x-proof.json");
        let proof_text = fs::read_to_string(&proof_path)
            .expect("Fix: read generated CPU-SOTA integrity proof artifact.");
        let proof = serde_json::from_str::<Value>(&proof_text)
            .expect("Fix: generated CPU-SOTA integrity proof must be valid JSON.");
        let blockers = proof
            .get("blockers")
            .and_then(Value::as_array)
            .expect("Fix: generated CPU-SOTA proof must include blockers array.");

        assert!(
            blockers.iter().filter_map(Value::as_str).any(|blocker| {
                blocker.contains(
                    "source_artifact `release/evidence/benchmarks/cuda-integrity-drift.json` case `release.condition_eval.1m` backend `cuda` has no applicable performance contract baseline",
                )
            }),
            "Fix: aggregate CPU-SOTA proof blockers must expose wrong-backend source artifact contracts; blockers={blockers:?}"
        );
        assert!(
            blockers.iter().filter_map(Value::as_str).any(|blocker| {
                blocker.contains(
                    "source_artifact `release/evidence/benchmarks/cuda-integrity-drift.json` case `release.condition_eval.1m` has cuda_resident_borrowed_fallback_dispatches p50=2",
                ) && blocker.contains("CPU-SOTA aggregate proof must use native resident dispatch")
            }),
            "Fix: aggregate CPU-SOTA proof blockers must expose borrowed resident CUDA dispatch telemetry; blockers={blockers:?}"
        );
    }

    #[test]
    fn cpu_100x_proof_rejects_missing_and_weak_source_fingerprint() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for CPU-SOTA provenance proof test.");
        let artifacts = [
            (
                "release/evidence/benchmarks/cuda-no-source-fingerprint.json",
                None,
            ),
            (
                "release/evidence/benchmarks/cuda-legacy-dirty-source.json",
                Some("git:abc123:dirty=true"),
            ),
        ];
        for (artifact_rel, source_fingerprint) in artifacts {
            let artifact_path = dir.path().join(artifact_rel);
            fs::create_dir_all(
                artifact_path
                    .parent()
                    .expect("Fix: CPU-SOTA provenance artifact path must have a parent directory."),
            )
            .expect("Fix: create CPU-SOTA provenance proof artifact parent directory.");
            let mut artifact = json!({
                "schema_version": 2,
                "selected_backend": "cuda",
                "source_tree_fingerprint": "source-tree-v1:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "summary": case_summary(1, 0),
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "cuda",
                        "status": "pass",
                        "metrics": {
                            "wall_ns": {"samples": 30, "p50": 10, "p95": 11, "p99": 12},
                            "baseline_wall_ns": {"samples": 30, "p50": 2000, "p95": 2001, "p99": 2002}
                        },
                        "contract": cpu_sota_contract("release condition eval", &["cuda"]),
                        "performance": {"contract_passed": true, "speedup_x": 200.0}
                    }
                ]
            });
            if let Some(source_fingerprint) = source_fingerprint {
                artifact["source_fingerprint"] = Value::String(source_fingerprint.to_string());
            }
            fs::write(
                &artifact_path,
                serde_json::to_string_pretty(&artifact)
                    .expect("Fix: serialize CPU-SOTA provenance benchmark artifact JSON."),
            )
            .expect("Fix: write CPU-SOTA provenance benchmark artifact JSON.");
        }

        write_cpu_100x_proof(
            dir.path(),
            &artifacts
                .iter()
                .map(|(artifact, _)| artifact.to_string())
                .collect::<Vec<_>>(),
        );

        let proof_path = dir
            .path()
            .join("release/evidence/benchmarks/cpu-only-100x-proof.json");
        let proof_text = fs::read_to_string(&proof_path)
            .expect("Fix: read generated CPU-SOTA 100x proof artifact.");
        let proof = serde_json::from_str::<Value>(&proof_text)
            .expect("Fix: generated CPU-SOTA 100x proof must be valid JSON.");
        let blockers = proof
            .get("blockers")
            .and_then(Value::as_array)
            .expect("Fix: generated CPU-SOTA proof must include blockers array.");

        assert!(
            blockers.iter().filter_map(Value::as_str).any(|blocker| {
                blocker.contains(
                    "100x source artifact `release/evidence/benchmarks/cuda-no-source-fingerprint.json` has no source_fingerprint",
                )
            }),
            "Fix: aggregate CPU-SOTA proof must reject source artifacts without explicit source_fingerprint; blockers={blockers:?}"
        );
        assert!(
            blockers.iter().filter_map(Value::as_str).any(|blocker| {
                blocker.contains(
                    "100x source artifact `release/evidence/benchmarks/cuda-legacy-dirty-source.json` source_fingerprint `git:abc123:dirty=true` is not release-grade provenance",
                )
            }),
            "Fix: aggregate CPU-SOTA proof must reject weak dirty source_fingerprint provenance; blockers={blockers:?}"
        );
    }

    #[test]
    fn cpu_100x_proof_rejects_whitespace_only_source_provenance() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for blank CPU-SOTA provenance test.");
        let artifact_rel = "release/evidence/benchmarks/cuda-blank-source-provenance.json";
        let artifact_path = dir.path().join(artifact_rel);
        fs::create_dir_all(
            artifact_path
                .parent()
                .expect("Fix: blank provenance artifact path must have a parent directory."),
        )
        .expect("Fix: create blank provenance proof artifact parent directory.");
        fs::write(
            &artifact_path,
            serde_json::to_string_pretty(&json!({
                "schema_version": 2,
                "selected_backend": "cuda",
                "source_fingerprint": "   ",
                "source_tree_fingerprint": "\t",
                "summary": case_summary(1, 0),
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "cuda",
                        "status": "pass",
                        "metrics": {
                            "wall_ns": {"samples": 30, "p50": 10, "p95": 11, "p99": 12},
                            "baseline_wall_ns": {"samples": 30, "p50": 2000, "p95": 2001, "p99": 2002}
                        },
                        "contract": cpu_sota_contract("release condition eval", &["cuda"]),
                        "performance": {"contract_passed": true, "speedup_x": 200.0}
                    }
                ]
            }))
            .expect("Fix: serialize blank provenance CUDA benchmark artifact JSON."),
        )
        .expect("Fix: write blank provenance CUDA benchmark artifact JSON.");

        write_cpu_100x_proof(dir.path(), &[artifact_rel.to_string()]);

        let proof_path = dir
            .path()
            .join("release/evidence/benchmarks/cpu-only-100x-proof.json");
        let proof_text = fs::read_to_string(&proof_path)
            .expect("Fix: read generated CPU-SOTA 100x proof artifact.");
        let proof = serde_json::from_str::<Value>(&proof_text)
            .expect("Fix: generated CPU-SOTA 100x proof must be valid JSON.");
        let blockers = proof
            .get("blockers")
            .and_then(Value::as_array)
            .expect("Fix: generated CPU-SOTA proof must include blockers array.");

        assert!(
            blockers.iter().filter_map(Value::as_str).any(|blocker| {
                blocker.contains(
                    "100x source artifact `release/evidence/benchmarks/cuda-blank-source-provenance.json` has no source_fingerprint",
                )
            }),
            "Fix: aggregate CPU-SOTA proof must reject blank source_fingerprint provenance; blockers={blockers:?}"
        );
        assert!(
            blockers.iter().filter_map(Value::as_str).any(|blocker| {
                blocker.contains(
                    "100x source artifact `release/evidence/benchmarks/cuda-blank-source-provenance.json` has no source_tree_fingerprint",
                )
            }),
            "Fix: aggregate CPU-SOTA proof must reject blank source_tree_fingerprint provenance; blockers={blockers:?}"
        );
        assert_eq!(
            proof.get("source_fingerprint"),
            Some(&Value::Null),
            "Fix: blank source_fingerprint must not be serialized as aggregate CPU-SOTA provenance."
        );
        assert_eq!(
            proof.get("source_tree_fingerprint"),
            Some(&Value::Null),
            "Fix: blank source_tree_fingerprint must not be serialized as aggregate CPU-SOTA provenance."
        );
    }

    #[test]
    fn cpu_100x_proof_rejects_stale_source_tree_fingerprint() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for stale CPU-SOTA source-tree test.");
        fs::write(dir.path().join("Cargo.toml"), "[workspace]\n")
            .expect("Fix: create temp workspace Cargo.toml for CPU-SOTA source-tree test.");
        let artifact_rel = "release/evidence/benchmarks/cuda-stale-source-tree.json";
        let artifact_path = dir.path().join(artifact_rel);
        fs::create_dir_all(
            artifact_path
                .parent()
                .expect("Fix: stale source-tree proof artifact path must have a parent directory."),
        )
        .expect("Fix: create stale source-tree proof artifact parent directory.");
        fs::write(
            &artifact_path,
            serde_json::to_string_pretty(&json!({
                "schema_version": 2,
                "selected_backend": "cuda",
                "source_fingerprint": "git:source-a:dirty=false",
                "source_tree_fingerprint": "source-tree-v1:stale",
                "summary": case_summary(1, 0),
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "cuda",
                        "status": "pass",
                        "metrics": {
                            "wall_ns": {"samples": 30, "p50": 10, "p95": 11, "p99": 12},
                            "baseline_wall_ns": {"samples": 30, "p50": 2000, "p95": 2001, "p99": 2002}
                        },
                        "contract": cpu_sota_contract("release condition eval", &["cuda"]),
                        "performance": {"contract_passed": true, "speedup_x": 200.0}
                    }
                ]
            }))
            .expect("Fix: serialize stale source-tree CUDA benchmark artifact JSON."),
        )
        .expect("Fix: write stale source-tree CUDA benchmark artifact JSON.");

        write_cpu_100x_proof(dir.path(), &[artifact_rel.to_string()]);

        let proof_path = dir
            .path()
            .join("release/evidence/benchmarks/cpu-only-100x-proof.json");
        let proof_text = fs::read_to_string(&proof_path)
            .expect("Fix: read generated CPU-SOTA 100x proof artifact.");
        let proof = serde_json::from_str::<Value>(&proof_text)
            .expect("Fix: generated CPU-SOTA 100x proof must be valid JSON.");
        let blockers = proof
            .get("blockers")
            .and_then(Value::as_array)
            .expect("Fix: generated CPU-SOTA proof must include blockers array.");

        assert!(
            blockers.iter().filter_map(Value::as_str).any(|blocker| {
                blocker.contains(
                    "100x source artifact `release/evidence/benchmarks/cuda-stale-source-tree.json` source_tree_fingerprint `source-tree-v1:stale`",
                ) && blocker.contains("does not match current workspace source")
            }),
            "Fix: aggregate CPU-SOTA proof must reject stale source-tree benchmark artifacts; blockers={blockers:?}"
        );
    }

    #[test]
    fn cpu_100x_proof_rejects_mixed_source_trees_not_clean_evidence_commit_drift() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for mixed-source CPU-SOTA proof test.");
        let artifacts = [
            (
                "release/evidence/benchmarks/cuda-source-a.json",
                "git:source-a:dirty=false",
                "source-tree-v1:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            ),
            (
                "release/evidence/benchmarks/cuda-source-b.json",
                "git:source-b:dirty=false",
                "source-tree-v1:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ),
        ];
        for (artifact_rel, source_fingerprint, source_tree_fingerprint) in artifacts {
            let artifact_path = dir.path().join(artifact_rel);
            fs::create_dir_all(
                artifact_path
                    .parent()
                    .expect("Fix: mixed-source proof artifact path must have a parent directory."),
            )
            .expect("Fix: create mixed-source proof artifact parent directory.");
            fs::write(
                &artifact_path,
                serde_json::to_string_pretty(&json!({
                    "schema_version": 2,
                    "selected_backend": "cuda",
                    "source_fingerprint": source_fingerprint,
                    "source_tree_fingerprint": source_tree_fingerprint,
                    "summary": case_summary(1, 0),
                    "cases": [
                        {
                            "id": "release.condition_eval.1m",
                            "backend_id": "cuda",
                            "status": "pass",
                            "metrics": {
                                "wall_ns": {"samples": 30, "p50": 10, "p95": 11, "p99": 12},
                                "baseline_wall_ns": {"samples": 30, "p50": 2000, "p95": 2001, "p99": 2002}
                            },
                            "contract": cpu_sota_contract("release condition eval", &["cuda"]),
                            "performance": {"contract_passed": true, "speedup_x": 200.0}
                        }
                    ]
                }))
                .expect("Fix: serialize mixed-source CUDA benchmark artifact JSON."),
            )
            .expect("Fix: write mixed-source CUDA benchmark artifact JSON.");
        }
        write_cpu_100x_proof(
            dir.path(),
            &artifacts
                .iter()
                .map(|(artifact, _, _)| artifact.to_string())
                .collect::<Vec<_>>(),
        );

        let proof_path = dir
            .path()
            .join("release/evidence/benchmarks/cpu-only-100x-proof.json");
        let proof_text = fs::read_to_string(&proof_path)
            .expect("Fix: read generated CPU-SOTA 100x proof artifact.");
        let proof = serde_json::from_str::<Value>(&proof_text)
            .expect("Fix: generated CPU-SOTA 100x proof must be valid JSON.");
        let blockers = proof
            .get("blockers")
            .and_then(Value::as_array)
            .expect("Fix: generated CPU-SOTA proof must include blockers array.");

        assert!(
            !blockers.iter().filter_map(Value::as_str).any(|blocker| {
                blocker.contains("source_fingerprint `git:source-b:dirty=false` does not match aggregate source")
            }),
            "Fix: aggregate CPU-SOTA proof must tolerate clean evidence commit drift when source_tree_fingerprint carries source identity; blockers={blockers:?}"
        );
        assert!(
            blockers.iter().filter_map(Value::as_str).any(|blocker| {
                blocker.contains(
                    "100x source artifact `release/evidence/benchmarks/cuda-source-b.json` source_tree_fingerprint `source-tree-v1:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` does not match aggregate source tree `source-tree-v1:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`",
                )
            }),
            "Fix: aggregate CPU-SOTA proof must reject mixed source_tree_fingerprint inputs; blockers={blockers:?}"
        );
    }

    #[test]
    fn cpu_100x_proof_does_not_count_duplicate_source_artifacts() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for duplicate-source CPU-SOTA proof test.");
        let artifact_rel = "release/evidence/benchmarks/cuda-duplicate-source.json";
        let artifact_path = dir.path().join(artifact_rel);
        fs::create_dir_all(
            artifact_path
                .parent()
                .expect("Fix: duplicate-source proof artifact path must have a parent directory."),
        )
        .expect("Fix: create duplicate-source proof artifact parent directory.");
        fs::write(
            &artifact_path,
            serde_json::to_string_pretty(&json!({
                "schema_version": 2,
                "selected_backend": "cuda",
                "source_fingerprint": "git:source-a:dirty=false",
                "source_tree_fingerprint": "source-tree-v1:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "summary": case_summary(1, 0),
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "cuda",
                        "status": "pass",
                        "metrics": {
                            "wall_ns": {"samples": 30, "p50": 10, "p95": 11, "p99": 12},
                            "baseline_wall_ns": {"samples": 30, "p50": 2000, "p95": 2001, "p99": 2002}
                        },
                        "contract": cpu_sota_contract("release condition eval", &["cuda"]),
                        "performance": {"contract_passed": true, "speedup_x": 200.0}
                    }
                ]
            }))
            .expect("Fix: serialize duplicate-source CUDA benchmark artifact JSON."),
        )
        .expect("Fix: write duplicate-source CUDA benchmark artifact JSON.");

        write_cpu_100x_proof(
            dir.path(),
            &[artifact_rel.to_string(), artifact_rel.to_string()],
        );

        let proof_path = dir
            .path()
            .join("release/evidence/benchmarks/cpu-only-100x-proof.json");
        let proof_text = fs::read_to_string(&proof_path)
            .expect("Fix: read generated CPU-SOTA 100x proof artifact.");
        let proof = serde_json::from_str::<Value>(&proof_text)
            .expect("Fix: generated CPU-SOTA 100x proof must be valid JSON.");
        let blockers = proof
            .get("blockers")
            .and_then(Value::as_array)
            .expect("Fix: generated CPU-SOTA proof must include blockers array.");

        assert_eq!(
            proof.get("source_artifact_count").and_then(Value::as_u64),
            Some(1),
            "Fix: duplicate source_artifacts must not inflate aggregate source_artifact_count."
        );
        assert_eq!(
            proof
                .get("cpu_sota_100x_contract_case_count")
                .and_then(Value::as_u64),
            Some(1),
            "Fix: duplicate source_artifacts must not duplicate cases into the aggregate proof."
        );
        assert!(
            blockers.iter().filter_map(Value::as_str).any(|blocker| {
                blocker.contains(
                    "100x proof source_artifact `release/evidence/benchmarks/cuda-duplicate-source.json` is duplicated"
                )
            }),
            "Fix: aggregate CPU-SOTA proof must report duplicated source_artifacts; blockers={blockers:?}"
        );
    }

    #[test]
    fn cpu_100x_proof_rejects_absolute_source_artifact_path() {
        let dir = TempDir::new()
            .expect("Fix: create a temporary workspace for absolute-source CPU-SOTA proof test.");
        let external_artifact = dir.path().join("external-cuda-source.json");
        fs::write(&external_artifact, "{}").expect("Fix: write external CUDA benchmark artifact.");

        write_cpu_100x_proof(dir.path(), &[external_artifact.display().to_string()]);

        let proof_path = dir
            .path()
            .join("release/evidence/benchmarks/cpu-only-100x-proof.json");
        let proof_text = fs::read_to_string(&proof_path)
            .expect("Fix: read generated CPU-SOTA 100x proof artifact.");
        let proof = serde_json::from_str::<Value>(&proof_text)
            .expect("Fix: generated CPU-SOTA 100x proof must be valid JSON.");
        let blockers = proof
            .get("blockers")
            .and_then(Value::as_array)
            .expect("Fix: generated CPU-SOTA proof must include blockers array.");

        assert!(
            blockers.iter().filter_map(Value::as_str).any(|blocker| {
                blocker.contains("100x source_artifact `")
                    && blocker.contains("must be a relative release path")
            }),
            "Fix: aggregate CPU-SOTA proof generation must reject existing absolute source_artifact paths before reading them; blockers={blockers:?}"
        );
    }
}
