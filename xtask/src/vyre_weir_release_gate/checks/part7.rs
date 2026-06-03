use crate::benchmark_evidence_semantics::{
    backend_suite_artifact_status_issues, backend_suite_backend_issue,
    backend_suite_inventory_issues, backend_suite_parity_issues,
    expected_backend_for_suite_evidence, BackendSuiteArtifactStatusIssue, BackendSuiteBackendIssue,
    BackendSuiteInventoryIssue, BackendSuiteParityIssue,
};

pub(crate) fn check_backend_suite_report(
    requirement: &Requirement,
    base_dir: &Path,
    suffix: &str,
    failures: &mut Vec<String>,
) {
    let Some(report) = first_json_evidence(requirement, base_dir, suffix, failures) else {
        return;
    };
    let schema_version = report
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if schema_version < 2 {
        failures.push(format!(
            "requirement `{}` backend suite `{suffix}` schema_version={schema_version}; expected schema>=2",
            requirement.id
        ));
    }
    let expected_suite_backend = expected_backend_for_suite_evidence(suffix);
    if let Some(expected_backend) = expected_suite_backend {
        if let Some(issue) = backend_suite_backend_issue(&report, expected_backend) {
            match issue {
                BackendSuiteBackendIssue::Missing { expected_backend } => failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` is missing backend identity `{expected_backend}`",
                    requirement.id
                )),
                BackendSuiteBackendIssue::Mismatch {
                    expected_backend,
                    actual_backend,
                } => failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` backend `{actual_backend}` does not match required `{expected_backend}`",
                    requirement.id
                )),
            }
        }
    }
    let family_count = report
        .get("family_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let artifact_count = report
        .get("artifacts")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if family_count == 0 || artifact_count == 0 {
        failures.push(format!(
            "requirement `{}` backend suite `{suffix}` reports family_count={family_count}, artifacts={artifact_count}",
            requirement.id
        ));
    }
    if family_count < 12 || artifact_count < 12 {
        failures.push(format!(
            "requirement `{}` backend suite `{suffix}` reports family_count={family_count}, artifacts={artifact_count}; release suites need at least 12 workload families",
            requirement.id
        ));
    }
    if family_count as usize != artifact_count {
        failures.push(format!(
            "requirement `{}` backend suite `{suffix}` family_count={family_count} does not match artifact count {artifact_count}",
            requirement.id
        ));
    }
    if let Some(blockers) = report.get("blockers").and_then(serde_json::Value::as_array) {
        for blocker in blockers {
            failures.push(format!(
                "requirement `{}` backend suite `{suffix}` reports blocker: {}",
                requirement.id,
                blocker.as_str().unwrap_or("<non-string blocker>")
            ));
        }
    }
    for issue in backend_suite_inventory_issues(&report) {
        match issue {
            BackendSuiteInventoryIssue::CountMismatch {
                artifact_count,
                status_count,
            } => failures.push(format!(
                "requirement `{}` backend suite `{suffix}` inventory count mismatch: artifacts={artifact_count}, artifact_statuses={status_count}",
                requirement.id
            )),
            BackendSuiteInventoryIssue::MissingStatus { path } => failures.push(format!(
                "requirement `{}` backend suite `{suffix}` lists artifact `{path}` without matching artifact_statuses entry",
                requirement.id
            )),
            BackendSuiteInventoryIssue::MissingArtifact { path } => failures.push(format!(
                "requirement `{}` backend suite `{suffix}` has artifact_statuses path `{path}` absent from artifacts",
                requirement.id
            )),
            BackendSuiteInventoryIssue::DuplicateArtifact { path } => failures.push(format!(
                "requirement `{}` backend suite `{suffix}` lists artifact `{path}` more than once",
                requirement.id
            )),
            BackendSuiteInventoryIssue::DuplicateStatus { path } => failures.push(format!(
                "requirement `{}` backend suite `{suffix}` has duplicate artifact_statuses path `{path}`",
                requirement.id
            )),
        }
    }
    if let Some(statuses) = report
        .get("artifact_statuses")
        .and_then(serde_json::Value::as_array)
    {
        for status in statuses {
            let path = status
                .get("path")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("<unknown>");
            if status.get("exists").and_then(serde_json::Value::as_bool) != Some(true) {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` is missing",
                    requirement.id
                ));
            }
            if status
                .get("bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                == 0
            {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` is empty",
                    requirement.id
                ));
            }
            let read_error = status.get("read_error");
            if !read_error.is_some_and(serde_json::Value::is_null) {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` read_error={}",
                    requirement.id,
                    read_error
                        .map(serde_json::Value::to_string)
                        .unwrap_or_else(|| "<missing>".to_string())
                ));
            }
            if status
                .get("family_id")
                .and_then(serde_json::Value::as_str)
                .is_none_or(str::is_empty)
            {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` has no family_id",
                    requirement.id
                ));
            }
            if status
                .get("requested_case_id")
                .and_then(serde_json::Value::as_str)
                .is_none_or(str::is_empty)
            {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` has no requested_case_id",
                    requirement.id
                ));
            }
            for field in ["source_fingerprint", "host_cpu_model"] {
                if status
                    .get(field)
                    .and_then(serde_json::Value::as_str)
                    .is_none_or(str::is_empty)
                {
                    failures.push(format!(
                        "requirement `{}` backend suite `{suffix}` artifact `{path}` has no `{field}` provenance",
                        requirement.id
                    ));
                }
            }
            check_backend_suite_status_source_fingerprint_shape(
                requirement,
                suffix,
                path,
                status,
                failures,
            );
            if let (Some((field, source_fingerprint)), Some(current_source_fingerprint)) = (
                report_freshness_fingerprint(status),
                current_freshness_fingerprint_for_report(base_dir, status),
            ) {
                check_source_fingerprint_freshness(
                    requirement,
                    &format!("backend suite `{suffix}` artifact `{path}`"),
                    field,
                    source_fingerprint,
                    &current_source_fingerprint,
                    failures,
                );
            }
            if status
                .get("case_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                == 0
            {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` reports zero cases",
                    requirement.id
                ));
            }
            if status
                .get("failed_count")
                .and_then(serde_json::Value::as_u64)
                != Some(0)
            {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` reports nonzero or missing failed_count",
                    requirement.id
                ));
            }
            if status
                .get("nonmatching_case_backend_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(1)
                != 0
            {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` reports backend-mismatched cases",
                    requirement.id
                ));
            }
            let suite_backend = report.get("backend").and_then(serde_json::Value::as_str);
            if matches!(suite_backend, Some("cuda" | "wgpu"))
                && status
                    .get("min_kernel_launches")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0)
                    == 0
            {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` GPU artifact `{path}` has non-positive `min_kernel_launches`",
                    requirement.id
                ));
            }
            if suite_backend == Some("cuda") {
                for field in ["gpu_model", "nvidia_driver_version", "nvidia_cuda_version"] {
                    if status
                        .get(field)
                        .and_then(serde_json::Value::as_str)
                        .is_none_or(str::is_empty)
                    {
                        failures.push(format!(
                            "requirement `{}` backend suite `{suffix}` CUDA artifact `{path}` has no `{field}` provenance",
                            requirement.id
                        ));
                    }
                }
                match status
                    .get("gpu_memory_total_mib")
                    .and_then(serde_json::Value::as_u64)
                {
                    Some(mib) if mib >= 16 * 1024 => {}
                    Some(mib) => failures.push(format!(
                        "requirement `{}` backend suite `{suffix}` CUDA artifact `{path}` reports {mib} MiB GPU memory, below release floor 16384 MiB",
                        requirement.id
                    )),
                    None => failures.push(format!(
                        "requirement `{}` backend suite `{suffix}` CUDA artifact `{path}` has no `gpu_memory_total_mib` provenance",
                        requirement.id
                    )),
                }
                match (
                    status
                        .get("gpu_compute_capability_major")
                        .and_then(serde_json::Value::as_u64),
                    status
                        .get("gpu_compute_capability_minor")
                        .and_then(serde_json::Value::as_u64),
                ) {
                    (Some(major), Some(minor)) if (major, minor) >= (8, 0) => {}
                    (Some(major), Some(minor)) => failures.push(format!(
                        "requirement `{}` backend suite `{suffix}` CUDA artifact `{path}` reports compute capability {major}.{minor}, below release floor 8.0",
                        requirement.id
                    )),
                    _ => failures.push(format!(
                        "requirement `{}` backend suite `{suffix}` CUDA artifact `{path}` has no compute capability provenance",
                        requirement.id
                    )),
                }
                if status
                    .get("min_cuda_ptx_source_cache_entries")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0)
                    == 0
                {
                    failures.push(format!(
                        "requirement `{}` backend suite `{suffix}` CUDA artifact `{path}` has non-positive `min_cuda_ptx_source_cache_entries`",
                        requirement.id
                    ));
                }
                for field in [
                    "min_cuda_ptx_source_cache_hits",
                    "min_cuda_ptx_source_cache_misses",
                ] {
                    if status
                        .get(field)
                        .and_then(serde_json::Value::as_u64)
                        .is_none()
                    {
                        failures.push(format!(
                            "requirement `{}` backend suite `{suffix}` CUDA artifact `{path}` is missing `{field}`",
                            requirement.id
                        ));
                    }
                }
            }
            if status
                .get("min_wall_samples")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                < 30
            {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` has fewer than 30 wall_ns samples",
                    requirement.id
                ));
            }
            if status
                .get("min_baseline_wall_samples")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                < 30
            {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` has fewer than 30 baseline_wall_ns samples",
                    requirement.id
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
                if status
                    .get(field)
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0)
                    == 0
                {
                    failures.push(format!(
                        "requirement `{}` backend suite `{suffix}` artifact `{path}` has non-positive `{field}`",
                        requirement.id
                    ));
                }
            }
            if status
                .get("blockers")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|blockers| !blockers.is_empty())
            {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` has semantic blockers",
                    requirement.id
                ));
            }
            if status
                .get("cpu_sota_100x_required")
                .and_then(serde_json::Value::as_bool)
                == Some(true)
                && status
                    .get("cpu_sota_100x_passing_cases")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0)
                    == 0
            {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` requires CPU-SOTA 100x proof but has zero passing 100x case(s)",
                    requirement.id
                ));
            }
        }
    } else {
        failures.push(format!(
            "requirement `{}` backend suite `{suffix}` has no artifact_statuses",
            requirement.id
        ));
    }
    if let Some(artifacts) = report
        .get("artifacts")
        .and_then(serde_json::Value::as_array)
    {
        let expected_backend = expected_suite_backend
            .or_else(|| report.get("backend").and_then(serde_json::Value::as_str));
        for artifact in artifacts {
            let Some(artifact) = artifact.as_str() else {
                failures.push(format!(
                    "requirement `{}` backend suite `{suffix}` contains a non-string artifact path",
                    requirement.id
                ));
                continue;
            };
            let path = resolve_artifact_path(base_dir, artifact);
            let text = match read_text_bounded(&path) {
                Ok(text) => text,
                Err(error) => {
                    failures.push(format!(
                        "requirement `{}` backend suite `{suffix}` failed to read listed artifact `{}`: {error}",
                        requirement.id,
                        path.display()
                    ));
                    continue;
                }
            };
            let artifact_report = match serde_json::from_str::<serde_json::Value>(&text) {
                Ok(report) => report,
                Err(error) => {
                    failures.push(format!(
                        "requirement `{}` backend suite `{suffix}` listed artifact `{}` is invalid JSON: {error}",
                        requirement.id,
                        path.display()
                    ));
                    continue;
                }
            };
            if let Some(status) = report_status_for_path(&report, artifact) {
                check_backend_suite_artifact_status(
                    requirement,
                    suffix,
                    status,
                    &artifact_report,
                    failures,
                );
            }
            check_single_benchmark_report(
                requirement,
                base_dir,
                &path,
                &artifact_report,
                false,
                None,
                failures,
            );
            if let Some(expected_backend) = expected_backend {
                let selected_backend = artifact_report
                    .get("selected_backend")
                    .and_then(serde_json::Value::as_str);
                if selected_backend != Some(expected_backend) {
                    failures.push(format!(
                        "requirement `{}` backend suite `{suffix}` artifact `{}` selected backend `{:?}`, expected `{expected_backend}`",
                        requirement.id,
                        path.display(),
                        selected_backend
                    ));
                }
                if matches!(expected_backend, "cuda" | "wgpu") {
                    let artifact_label = path.display().to_string();
                    require_case_metric_present(
                        requirement,
                        &artifact_label,
                        &artifact_report,
                        "kernel_launches",
                        failures,
                    );
                    require_case_metric_positive(
                        requirement,
                        &artifact_label,
                        &artifact_report,
                        "kernel_launches",
                        failures,
                    );
                }
                if expected_backend == "cuda" {
                    let artifact_label = path.display().to_string();
                    for metric in [
                        "cuda_ptx_source_cache_entries",
                        "cuda_ptx_source_cache_hits",
                        "cuda_ptx_source_cache_misses",
                    ] {
                        require_case_metric_present(
                            requirement,
                            &artifact_label,
                            &artifact_report,
                            metric,
                            failures,
                        );
                    }
                    for metric in ["cuda_ptx_source_cache_entries"] {
                        require_case_metric_positive(
                            requirement,
                            &artifact_label,
                            &artifact_report,
                            metric,
                            failures,
                        );
                    }
                }
                if let Some(cases) = artifact_report
                    .get("cases")
                    .and_then(serde_json::Value::as_array)
                {
                    for case in cases {
                        let id = case
                            .get("id")
                            .and_then(serde_json::Value::as_str)
                            .unwrap_or("<unknown>");
                        let backend = case.get("backend_id").and_then(serde_json::Value::as_str);
                        if backend != Some(expected_backend) {
                            failures.push(format!(
                                "requirement `{}` backend suite `{suffix}` artifact `{}` case `{id}` backend `{:?}`, expected `{expected_backend}`",
                                requirement.id,
                                path.display(),
                                backend
                            ));
                        }
                    }
                }
            }
        }
    }
}

fn check_backend_suite_status_source_fingerprint_shape(
    requirement: &Requirement,
    suffix: &str,
    path: &str,
    status: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    let Some(source_fingerprint) = status
        .get("source_fingerprint")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
    else {
        return;
    };
    check_source_fingerprint_shape(
        requirement,
        &format!("backend suite {suffix} status for {path}"),
        source_fingerprint,
        failures,
    );
}

fn report_status_for_path<'a>(
    suite_report: &'a serde_json::Value,
    artifact: &str,
) -> Option<&'a serde_json::Value> {
    suite_report
        .get("artifact_statuses")
        .and_then(serde_json::Value::as_array)
        .and_then(|statuses| {
            statuses.iter().find(|status| {
                status.get("path").and_then(serde_json::Value::as_str) == Some(artifact)
            })
        })
}

fn check_backend_suite_artifact_status(
    requirement: &Requirement,
    suffix: &str,
    status: &serde_json::Value,
    artifact_report: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    for issue in backend_suite_artifact_status_issues(status, artifact_report) {
        match issue {
            BackendSuiteArtifactStatusIssue::MissingField { path, field } => failures.push(
                format!(
                    "requirement `{}` backend suite `{suffix}` artifact `{path}` status is missing artifact-backed `{field}`",
                    requirement.id
                ),
            ),
            BackendSuiteArtifactStatusIssue::SourceFingerprintMismatch {
                path,
                status_source_fingerprint,
                artifact_source_fingerprint,
            } => failures.push(format!(
                "requirement `{}` backend suite `{suffix}` artifact `{path}` source_fingerprint mismatch: status `{status_source_fingerprint}`, artifact `{artifact_source_fingerprint}`",
                requirement.id
            )),
            BackendSuiteArtifactStatusIssue::SourceTreeFingerprintMismatch {
                path,
                status_source_tree_fingerprint,
                artifact_source_tree_fingerprint,
            } => failures.push(format!(
                "requirement `{}` backend suite `{suffix}` artifact `{path}` source_tree_fingerprint mismatch: status `{status_source_tree_fingerprint}`, artifact `{artifact_source_tree_fingerprint}`",
                requirement.id
            )),
            BackendSuiteArtifactStatusIssue::SelectedBackendMismatch {
                path,
                status_selected_backend,
                artifact_selected_backend,
            } => failures.push(format!(
                "requirement `{}` backend suite `{suffix}` artifact `{path}` selected_backend mismatch: status `{status_selected_backend}`, artifact `{artifact_selected_backend}`",
                requirement.id
            )),
            BackendSuiteArtifactStatusIssue::CaseCountMismatch {
                path,
                status_case_count,
                artifact_case_count,
            } => failures.push(format!(
                "requirement `{}` backend suite `{suffix}` artifact `{path}` case_count mismatch: status {status_case_count}, artifact {artifact_case_count}",
                requirement.id
            )),
            BackendSuiteArtifactStatusIssue::FailedCountMismatch {
                path,
                status_failed_count,
                artifact_failed_count,
            } => failures.push(format!(
                "requirement `{}` backend suite `{suffix}` artifact `{path}` failed_count mismatch: status {status_failed_count}, artifact {artifact_failed_count}",
                requirement.id
            )),
            BackendSuiteArtifactStatusIssue::NumericFieldMismatch {
                path,
                field,
                status_value,
                artifact_value,
            } => failures.push(format!(
                "requirement `{}` backend suite `{suffix}` artifact `{path}` `{field}` mismatch: status {status_value}, artifact {artifact_value}",
                requirement.id
            )),
            BackendSuiteArtifactStatusIssue::StringFieldMismatch {
                path,
                field,
                status_value,
                artifact_value,
            } => failures.push(format!(
                "requirement `{}` backend suite `{suffix}` artifact `{path}` `{field}` mismatch: status `{status_value}`, artifact `{artifact_value}`",
                requirement.id
            )),
            BackendSuiteArtifactStatusIssue::CpuSota100xContractCaseCountMismatch {
                path,
                status_contract_cases,
                artifact_contract_cases,
            } => failures.push(format!(
                "requirement `{}` backend suite `{suffix}` artifact `{path}` cpu_sota_100x_contract_cases mismatch: status {status_contract_cases}, artifact {artifact_contract_cases}",
                requirement.id
            )),
            BackendSuiteArtifactStatusIssue::CpuSota100xPassingCaseCountMismatch {
                path,
                status_passing_cases,
                artifact_passing_cases,
            } => failures.push(format!(
                "requirement `{}` backend suite `{suffix}` artifact `{path}` cpu_sota_100x_passing_cases mismatch: status {status_passing_cases}, artifact {artifact_passing_cases}",
                requirement.id
            )),
            BackendSuiteArtifactStatusIssue::MissingRequestedCase {
                path,
                requested_case_id,
            } => failures.push(format!(
                "requirement `{}` backend suite `{suffix}` artifact `{path}` does not contain requested_case_id `{requested_case_id}`",
                requirement.id
            )),
        }
    }
}

#[cfg(test)]
mod part7_tests {
    use super::*;

    #[test]
    fn backend_suite_status_rejects_dirty_fingerprint_without_worktree_digest() {
        let requirement = Requirement {
            id: "wgpu-fallback".to_string(),
            title: "WGPU fallback".to_string(),
            status: "required".to_string(),
            evidence: Vec::new(),
            minimum_evidence: 0,
        };
        let status = serde_json::json!({
            "path": "release/evidence/benchmarks/wgpu-workload-01-condition-eval.json",
            "source_fingerprint": "git:abc123:dirty=true"
        });
        let mut failures = Vec::new();

        check_backend_suite_status_source_fingerprint_shape(
            &requirement,
            "wgpu-fallback-suite.json",
            "release/evidence/benchmarks/wgpu-workload-01-condition-eval.json",
            &status,
            &mut failures,
        );

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("is dirty but has no worktree digest")),
            "Fix: backend suite status rows must carry precise dirty-worktree provenance; failures={failures:?}"
        );
    }

    #[test]
    fn backend_suite_report_rejects_filename_backend_identity_drift() {
        let dir = tempfile::TempDir::new()
            .expect("Fix: create temporary workspace for backend suite identity test.");
        let release_dir = dir.path().join("release");
        let benchmark_dir = release_dir.join("evidence/benchmarks");
        std::fs::create_dir_all(&benchmark_dir)
            .expect("Fix: create benchmark evidence directory for backend suite identity test.");
        let artifact_path = benchmark_dir.join("workload-01-condition-eval.json");
        std::fs::write(
            &artifact_path,
            serde_json::to_string_pretty(&serde_json::json!({
                "schema_version": 2,
                "selected_backend": "wgpu",
                "summary": {"total_cases": 1, "passed": 1, "failed": 0},
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "wgpu",
                        "status": "pass",
                        "metrics": {
                            "wall_ns": {"samples": 30, "p50": 10, "p95": 11, "p99": 12},
                            "baseline_wall_ns": {"samples": 30, "p50": 1000, "p95": 1001, "p99": 1002},
                            "kernel_launches": {"samples": 30, "p50": 1}
                        },
                        "performance": {"contract_passed": true, "speedup_x": 120.0}
                    }
                ]
            }))
            .expect("Fix: serialize WGPU benchmark artifact for backend suite identity test."),
        )
        .expect("Fix: write WGPU benchmark artifact for backend suite identity test.");
        std::fs::write(
            benchmark_dir.join("cuda-release-suite.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "schema_version": 2,
                "backend": "wgpu",
                "family_count": 1,
                "artifacts": ["release/evidence/benchmarks/workload-01-condition-eval.json"],
                "artifact_statuses": [
                    {
                        "path": "release/evidence/benchmarks/workload-01-condition-eval.json",
                        "family_id": "condition-eval",
                        "requested_case_id": "release.condition_eval.1m",
                        "exists": true,
                        "bytes": 1,
                        "read_error": null,
                        "source_fingerprint": "git:abc:dirty=false",
                        "selected_backend": "wgpu",
                        "case_count": 1,
                        "failed_count": 0,
                        "nonmatching_case_backend_count": 0,
                        "min_wall_samples": 30,
                        "min_baseline_wall_samples": 30,
                        "min_wall_p50": 10,
                        "min_wall_p95": 11,
                        "min_wall_p99": 12,
                        "min_baseline_wall_p50": 1000,
                        "min_baseline_wall_p95": 1001,
                        "min_baseline_wall_p99": 1002,
                        "min_kernel_launches": 1,
                        "blockers": []
                    }
                ],
                "blockers": []
            }))
            .expect("Fix: serialize CUDA suite with mismatched backend identity."),
        )
        .expect("Fix: write CUDA suite with mismatched backend identity.");
        let requirement = Requirement {
            id: "cuda-first-path".to_string(),
            title: "CUDA first path".to_string(),
            status: "required".to_string(),
            evidence: vec!["evidence/benchmarks/cuda-release-suite.json".to_string()],
            minimum_evidence: 1,
        };
        let mut failures = Vec::new();

        check_backend_suite_report(
            &requirement,
            &release_dir,
            "cuda-release-suite.json",
            &mut failures,
        );

        assert!(
            failures.iter().any(|failure| failure.contains(
                "backend `wgpu` does not match required `cuda`"
            )),
            "Fix: CUDA suite identity must be validated against the evidence filename; failures={failures:?}"
        );
        assert!(
            failures.iter().any(|failure| failure.contains(
                "selected backend `Some(\"wgpu\")`, expected `cuda`"
            )),
            "Fix: CUDA suite artifacts must be validated against filename-implied backend, not a bad suite backend field; failures={failures:?}"
        );
    }
}

pub(crate) fn check_backend_suite_parity(
    requirement: &Requirement,
    base_dir: &Path,
    failures: &mut Vec<String>,
) {
    let cuda = read_release_suite_json(
        requirement,
        base_dir,
        "evidence/benchmarks/cuda-release-suite.json",
        failures,
    );
    let wgpu = read_release_suite_json(
        requirement,
        base_dir,
        "evidence/benchmarks/wgpu-fallback-suite.json",
        failures,
    );
    let (Some(cuda), Some(wgpu)) = (cuda, wgpu) else {
        return;
    };
    for issue in backend_suite_parity_issues(&cuda, &wgpu) {
        match issue {
            BackendSuiteParityIssue::MissingCudaPair {
                family_id,
                requested_case_id,
            } => failures.push(format!(
                "requirement `{}` WGPU/CUDA suite parity has WGPU family `{family_id}` case `{requested_case_id}` with no CUDA counterpart",
                requirement.id
            )),
            BackendSuiteParityIssue::MissingWgpuPair {
                family_id,
                requested_case_id,
            } => failures.push(format!(
                "requirement `{}` WGPU/CUDA suite parity has CUDA family `{family_id}` case `{requested_case_id}` with no WGPU counterpart",
                requirement.id
            )),
            BackendSuiteParityIssue::CountMismatch {
                cuda_count,
                wgpu_count,
            } => failures.push(format!(
                "requirement `{}` WGPU/CUDA suite parity count mismatch: cuda={cuda_count}, wgpu={wgpu_count}",
                requirement.id
            )),
            BackendSuiteParityIssue::SharedArtifactPath { path } => failures.push(format!(
                "requirement `{}` WGPU/CUDA suite parity reuses artifact path `{path}` across CUDA and WGPU suites",
                requirement.id
            )),
            BackendSuiteParityIssue::StatusFieldMismatch {
                family_id,
                requested_case_id,
                field,
                cuda_value,
                wgpu_value,
            } => failures.push(format!(
                "requirement `{}` WGPU/CUDA suite parity field `{field}` mismatch for family `{family_id}` case `{requested_case_id}`: cuda={cuda_value:?}, wgpu={wgpu_value:?}",
                requirement.id
            )),
        }
    }
}

fn read_release_suite_json(
    requirement: &Requirement,
    base_dir: &Path,
    evidence: &str,
    failures: &mut Vec<String>,
) -> Option<serde_json::Value> {
    let path = resolve_manifest_path(base_dir, evidence);
    let text = match read_text_bounded(&path) {
        Ok(text) => text,
        Err(error) => {
            failures.push(format!(
                "requirement `{}` failed to read backend suite parity artifact `{}`: {error}",
                requirement.id,
                path.display()
            ));
            return None;
        }
    };
    match serde_json::from_str::<serde_json::Value>(&text) {
        Ok(value) => Some(value),
        Err(error) => {
            failures.push(format!(
                "requirement `{}` backend suite parity artifact `{}` is invalid JSON: {error}",
                requirement.id,
                path.display()
            ));
            None
        }
    }
}
pub(crate) fn check_markdown_evidence_ready(
    requirement: &Requirement,
    base_dir: &Path,
    suffix: &str,
    failures: &mut Vec<String>,
) {
    let evidence = requirement
        .evidence
        .iter()
        .find(|path| path.ends_with(suffix));
    let Some(evidence) = evidence else {
        failures.push(format!(
            "requirement `{}` needs markdown evidence ending in `{suffix}`",
            requirement.id
        ));
        return;
    };
    let path = resolve_manifest_path(base_dir, evidence);
    let text = match read_text_bounded(&path) {
        Ok(text) => text,
        Err(error) => {
            failures.push(format!(
                "requirement `{}` failed to read markdown evidence `{}`: {error}",
                requirement.id,
                path.display()
            ));
            return;
        }
    };
    for marker in [
        "status: blocked",
        "status: open",
        "status: pending",
        "todo",
        "fixme",
        "placeholder",
        "stub",
        "tbd",
        "to be filled",
    ] {
        for line in text.lines() {
            let lowered = line.to_ascii_lowercase();
            if markdown_line_is_release_rule_text(&lowered) {
                continue;
            }
            if lowered.contains(marker) {
                failures.push(format!(
                    "requirement `{}` markdown evidence `{}` contains unresolved marker `{marker}`",
                    requirement.id,
                    path.display()
                ));
                break;
            }
        }
    }
    if text.trim().is_empty() {
        failures.push(format!(
            "requirement `{}` markdown evidence `{}` is empty",
            requirement.id,
            path.display()
        ));
    }
    if !text.contains("Evidence sources:") {
        failures.push(format!(
            "requirement `{}` markdown evidence `{}` does not list evidence sources",
            requirement.id,
            path.display()
        ));
    }
}
