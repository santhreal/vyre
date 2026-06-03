use crate::benchmark_evidence_semantics::{
    backend_suite_artifact_status_issues, backend_suite_inventory_issues,
    backend_suite_parity_issues, source_fingerprint_issues, BackendSuiteArtifactStatusIssue,
    BackendSuiteInventoryIssue, BackendSuiteParityIssue, SourceFingerprintIssue,
};

fn inspect_backend_suite_semantics(
    evidence: &str,
    path: &Path,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if schema_version < 2 {
        blockers.push(format!(
            "{evidence}: schema_version={schema_version}; backend suite evidence must be schema>=2"
        ));
    }
    let family_count = value
        .get("family_count")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let artifact_count = value
        .get("artifacts")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len) as u64;
    if family_count == 0 || artifact_count == 0 || family_count != artifact_count {
        blockers.push(format!(
            "{evidence}: family_count={family_count}, artifact_count={artifact_count}"
        ));
    }
    if family_count < 12 || artifact_count < 12 {
        blockers.push(format!(
            "{evidence}: family_count={family_count}, artifact_count={artifact_count}; release backend suites need at least 12 workload families"
        ));
    }
    if let Some(suite_blockers) = value.get("blockers").and_then(serde_json::Value::as_array) {
        for blocker in suite_blockers {
            blockers.push(format!(
                "{evidence}: suite blocker: {}",
                blocker.as_str().unwrap_or("<non-string blocker>")
            ));
        }
    }
    for issue in backend_suite_inventory_issues(value) {
        match issue {
            BackendSuiteInventoryIssue::CountMismatch {
                artifact_count,
                status_count,
            } => blockers.push(format!(
                "{evidence}: suite inventory count mismatch: artifacts={artifact_count}, artifact_statuses={status_count}"
            )),
            BackendSuiteInventoryIssue::MissingStatus { path } => blockers.push(format!(
                "{evidence}: suite lists artifact `{path}` without matching artifact_statuses entry"
            )),
            BackendSuiteInventoryIssue::MissingArtifact { path } => blockers.push(format!(
                "{evidence}: suite artifact_statuses path `{path}` is absent from artifacts"
            )),
            BackendSuiteInventoryIssue::DuplicateArtifact { path } => blockers.push(format!(
                "{evidence}: suite lists artifact `{path}` more than once"
            )),
            BackendSuiteInventoryIssue::DuplicateStatus { path } => blockers.push(format!(
                "{evidence}: suite has duplicate artifact_statuses path `{path}`"
            )),
        }
    }
    let Some(statuses) = value
        .get("artifact_statuses")
        .and_then(serde_json::Value::as_array)
    else {
        blockers.push(format!("{evidence}: missing artifact_statuses"));
        return;
    };
    if statuses.len() as u64 != artifact_count {
        blockers.push(format!(
            "{evidence}: artifact_statuses has {} entrie(s), artifacts has {artifact_count}",
            statuses.len()
        ));
    }
    for status in statuses {
        let path = status
            .get("path")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        if status.get("exists").and_then(serde_json::Value::as_bool) != Some(true) {
            blockers.push(format!("{evidence}: suite artifact `{path}` is missing"));
        }
        if status
            .get("bytes")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            blockers.push(format!("{evidence}: suite artifact `{path}` is empty"));
        }
        let read_error = status.get("read_error");
        if !read_error.is_some_and(serde_json::Value::is_null) {
            blockers.push(format!(
                "{evidence}: suite artifact `{path}` read_error={}",
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
            blockers.push(format!(
                "{evidence}: suite artifact `{path}` has no family_id"
            ));
        }
        if status
            .get("requested_case_id")
            .and_then(serde_json::Value::as_str)
            .is_none_or(str::is_empty)
        {
            blockers.push(format!(
                "{evidence}: suite artifact `{path}` has no requested_case_id"
            ));
        }
        for field in ["source_fingerprint", "host_cpu_model"] {
            if status
                .get(field)
                .and_then(serde_json::Value::as_str)
                .is_none_or(str::is_empty)
            {
                blockers.push(format!(
                    "{evidence}: suite artifact `{path}` has no `{field}` provenance"
                ));
            }
        }
        inspect_backend_suite_status_source_fingerprint(evidence, path, status, blockers);
        if status
            .get("case_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            blockers.push(format!("{evidence}: suite artifact `{path}` has no cases"));
        }
        if status
            .get("failed_count")
            .and_then(serde_json::Value::as_u64)
            != Some(0)
        {
            blockers.push(format!(
                "{evidence}: suite artifact `{path}` has nonzero or missing failed_count"
            ));
        }
        if status
            .get("nonmatching_case_backend_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(1)
            != 0
        {
            blockers.push(format!(
                "{evidence}: suite artifact `{path}` has backend-mismatched case(s)"
            ));
        }
        let suite_backend = value.get("backend").and_then(serde_json::Value::as_str);
        if matches!(suite_backend, Some("cuda" | "wgpu"))
            && status
                .get("min_kernel_launches")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                == 0
        {
            blockers.push(format!(
                "{evidence}: GPU suite artifact `{path}` has non-positive `min_kernel_launches`"
            ));
        }
        if suite_backend == Some("cuda") {
            for field in ["gpu_model", "nvidia_driver_version", "nvidia_cuda_version"] {
                if status
                    .get(field)
                    .and_then(serde_json::Value::as_str)
                    .is_none_or(str::is_empty)
                {
                    blockers.push(format!(
                        "{evidence}: CUDA suite artifact `{path}` has no `{field}` provenance"
                    ));
                }
            }
            match status
                .get("gpu_memory_total_mib")
                .and_then(serde_json::Value::as_u64)
            {
                Some(mib) if mib >= 16 * 1024 => {}
                Some(mib) => blockers.push(format!(
                    "{evidence}: CUDA suite artifact `{path}` reports {mib} MiB GPU memory, below release floor 16384 MiB"
                )),
                None => blockers.push(format!(
                    "{evidence}: CUDA suite artifact `{path}` has no `gpu_memory_total_mib` provenance"
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
                (Some(major), Some(minor)) => blockers.push(format!(
                    "{evidence}: CUDA suite artifact `{path}` reports compute capability {major}.{minor}, below release floor 8.0"
                )),
                _ => blockers.push(format!(
                    "{evidence}: CUDA suite artifact `{path}` has no compute capability provenance"
                )),
            }
            if status
                .get("min_cuda_ptx_source_cache_entries")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                == 0
            {
                blockers.push(format!(
                    "{evidence}: CUDA suite artifact `{path}` has non-positive `min_cuda_ptx_source_cache_entries`"
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
                    blockers.push(format!(
                        "{evidence}: CUDA suite artifact `{path}` is missing `{field}`"
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
            blockers.push(format!(
                "{evidence}: suite artifact `{path}` has fewer than 30 wall_ns samples"
            ));
        }
        if status
            .get("min_baseline_wall_samples")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            < 30
        {
            blockers.push(format!(
                "{evidence}: suite artifact `{path}` has fewer than 30 baseline_wall_ns samples"
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
                blockers.push(format!(
                    "{evidence}: suite artifact `{path}` has non-positive `{field}`"
                ));
            }
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
            blockers.push(format!(
                "{evidence}: suite artifact `{path}` requires CPU-SOTA 100x proof but has zero passing 100x case(s)"
            ));
        }
        if let Some(status_blockers) = status.get("blockers").and_then(serde_json::Value::as_array)
        {
            for blocker in status_blockers {
                blockers.push(format!(
                    "{evidence}: suite artifact `{path}` blocker: {}",
                    blocker.as_str().unwrap_or("<non-string blocker>")
                ));
            }
        }
    }
    inspect_backend_suite_status_artifact_consistency(evidence, path, value, blockers);
    if evidence.ends_with("wgpu-fallback-suite.json") {
        inspect_wgpu_cuda_suite_parity(evidence, path, value, blockers);
    }
}

fn inspect_backend_suite_status_source_fingerprint(
    evidence: &str,
    path: &str,
    status: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let Some(source_fingerprint) = status
        .get("source_fingerprint")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
    else {
        return;
    };
    for issue in source_fingerprint_issues(source_fingerprint) {
        match issue {
            SourceFingerprintIssue::DirtyUnknownState { source_fingerprint } => blockers.push(
                format!(
                    "{evidence}: suite artifact `{path}` status source_fingerprint `{source_fingerprint}` has dirty=unknown; rerun with git status provenance available"
                ),
            ),
            SourceFingerprintIssue::DirtyMissingWorktree { source_fingerprint } => blockers.push(
                format!(
                    "{evidence}: suite artifact `{path}` status source_fingerprint `{source_fingerprint}` is dirty but has no worktree digest"
                ),
            ),
            SourceFingerprintIssue::DirtyUnknownWorktree { source_fingerprint } => blockers.push(
                format!(
                    "{evidence}: suite artifact `{path}` status source_fingerprint `{source_fingerprint}` has an unknown worktree digest"
                ),
            ),
            SourceFingerprintIssue::DirtyInvalidWorktree {
                source_fingerprint,
                worktree,
            } => blockers.push(format!(
                "{evidence}: suite artifact `{path}` status source_fingerprint `{source_fingerprint}` has invalid worktree digest `{worktree}`; expected 64 hex chars"
            )),
        }
    }
}

fn inspect_backend_suite_status_artifact_consistency(
    evidence: &str,
    suite_path: &Path,
    suite: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let Some(workspace_root) = backend_suite_workspace_root(suite_path) else {
        blockers.push(format!(
            "{evidence}: cannot resolve workspace root for listed backend suite artifacts"
        ));
        return;
    };
    let Some(artifacts) = suite.get("artifacts").and_then(serde_json::Value::as_array) else {
        return;
    };
    for artifact in artifacts {
        let Some(artifact) = artifact.as_str() else {
            continue;
        };
        let Some(status) = report_status_for_path(suite, artifact) else {
            continue;
        };
        let artifact_path = resolve_suite_artifact_path(workspace_root, artifact);
        let text = match read_text_bounded(&artifact_path) {
            Ok(text) => text,
            Err(error) => {
                blockers.push(format!(
                    "{evidence}: failed to read suite artifact `{}`: {error}",
                    artifact_path.display()
                ));
                continue;
            }
        };
        let artifact_report = match serde_json::from_str::<serde_json::Value>(&text) {
            Ok(value) => value,
            Err(error) => {
                blockers.push(format!(
                    "{evidence}: suite artifact `{}` is invalid JSON: {error}",
                    artifact_path.display()
                ));
                continue;
            }
        };
        inspect_backend_suite_artifact_status(evidence, status, &artifact_report, blockers);
        inspect_suite_artifact_contract_baselines(evidence, artifact, &artifact_report, blockers);
        if let Some(source_fingerprint) = artifact_report
            .get("source_fingerprint")
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.trim().is_empty())
        {
            inspect_suite_artifact_source_fingerprint(
                evidence,
                artifact,
                source_fingerprint,
                blockers,
            );
        }
        if let (Some((field, freshness_fingerprint)), Some(current_freshness_fingerprint)) = (
            crate::benchmark_evidence_semantics::report_freshness_fingerprint(&artifact_report),
            crate::benchmark_evidence_semantics::current_freshness_fingerprint_for_report(
                workspace_root,
                &artifact_report,
            ),
        ) {
            inspect_suite_artifact_source_fingerprint_freshness(
                evidence,
                artifact,
                field,
                freshness_fingerprint,
                &current_freshness_fingerprint,
                blockers,
            );
        }
    }
}

fn inspect_suite_artifact_contract_baselines(
    evidence: &str,
    artifact: &str,
    artifact_report: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    for issue in crate::benchmark_evidence_semantics::contract_backend_issues(artifact_report) {
        match issue {
            crate::benchmark_evidence_semantics::ContractBackendIssue::MissingBaselines {
                case_id,
                backend_id,
            } => blockers.push(format!(
                "{evidence}: suite artifact `{artifact}` case `{case_id}` backend `{backend_id}` has a performance contract with no baselines"
            )),
            crate::benchmark_evidence_semantics::ContractBackendIssue::NoApplicableBaseline {
                case_id,
                backend_id,
            } => blockers.push(format!(
                "{evidence}: suite artifact `{artifact}` case `{case_id}` backend `{backend_id}` has no applicable performance contract baseline"
            )),
        }
    }
}

fn backend_suite_workspace_root(path: &Path) -> Option<&Path> {
    Some(path.parent()?.parent()?.parent()?.parent()?)
}

fn resolve_suite_artifact_path(workspace_root: &Path, artifact: &str) -> PathBuf {
    let candidate = PathBuf::from(artifact);
    if candidate.is_absolute() {
        candidate
    } else {
        workspace_root.join(candidate)
    }
}

fn report_status_for_path<'a>(
    suite: &'a serde_json::Value,
    artifact: &str,
) -> Option<&'a serde_json::Value> {
    suite
        .get("artifact_statuses")
        .and_then(serde_json::Value::as_array)
        .and_then(|statuses| {
            statuses.iter().find(|status| {
                status.get("path").and_then(serde_json::Value::as_str) == Some(artifact)
            })
        })
}

fn inspect_backend_suite_artifact_status(
    evidence: &str,
    status: &serde_json::Value,
    artifact_report: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    for issue in backend_suite_artifact_status_issues(status, artifact_report) {
        match issue {
            BackendSuiteArtifactStatusIssue::SourceFingerprintMismatch {
                path,
                status_source_fingerprint,
                artifact_source_fingerprint,
            } => blockers.push(format!(
                "{evidence}: suite artifact `{path}` source_fingerprint mismatch: status `{status_source_fingerprint}`, artifact `{artifact_source_fingerprint}`"
            )),
            BackendSuiteArtifactStatusIssue::SourceTreeFingerprintMismatch {
                path,
                status_source_tree_fingerprint,
                artifact_source_tree_fingerprint,
            } => blockers.push(format!(
                "{evidence}: suite artifact `{path}` source_tree_fingerprint mismatch: status `{status_source_tree_fingerprint}`, artifact `{artifact_source_tree_fingerprint}`"
            )),
            BackendSuiteArtifactStatusIssue::SelectedBackendMismatch {
                path,
                status_selected_backend,
                artifact_selected_backend,
            } => blockers.push(format!(
                "{evidence}: suite artifact `{path}` selected_backend mismatch: status `{status_selected_backend}`, artifact `{artifact_selected_backend}`"
            )),
            BackendSuiteArtifactStatusIssue::CaseCountMismatch {
                path,
                status_case_count,
                artifact_case_count,
            } => blockers.push(format!(
                "{evidence}: suite artifact `{path}` case_count mismatch: status {status_case_count}, artifact {artifact_case_count}"
            )),
            BackendSuiteArtifactStatusIssue::FailedCountMismatch {
                path,
                status_failed_count,
                artifact_failed_count,
            } => blockers.push(format!(
                "{evidence}: suite artifact `{path}` failed_count mismatch: status {status_failed_count}, artifact {artifact_failed_count}"
            )),
            BackendSuiteArtifactStatusIssue::CpuSota100xContractCaseCountMismatch {
                path,
                status_contract_cases,
                artifact_contract_cases,
            } => blockers.push(format!(
                "{evidence}: suite artifact `{path}` cpu_sota_100x_contract_cases mismatch: status {status_contract_cases}, artifact {artifact_contract_cases}"
            )),
            BackendSuiteArtifactStatusIssue::CpuSota100xPassingCaseCountMismatch {
                path,
                status_passing_cases,
                artifact_passing_cases,
            } => blockers.push(format!(
                "{evidence}: suite artifact `{path}` cpu_sota_100x_passing_cases mismatch: status {status_passing_cases}, artifact {artifact_passing_cases}"
            )),
            BackendSuiteArtifactStatusIssue::MissingRequestedCase {
                path,
                requested_case_id,
            } => blockers.push(format!(
                "{evidence}: suite artifact `{path}` does not contain requested_case_id `{requested_case_id}`"
            )),
        }
    }
}

fn inspect_suite_artifact_source_fingerprint(
    evidence: &str,
    artifact: &str,
    source_fingerprint: &str,
    blockers: &mut Vec<String>,
) {
    for issue in source_fingerprint_issues(source_fingerprint) {
        match issue {
            SourceFingerprintIssue::DirtyUnknownState { source_fingerprint } => blockers.push(
                format!(
                    "{evidence}: suite artifact `{artifact}` source_fingerprint `{source_fingerprint}` has dirty=unknown; rerun with git status provenance available"
                ),
            ),
            SourceFingerprintIssue::DirtyMissingWorktree { source_fingerprint } => blockers.push(
                format!(
                    "{evidence}: suite artifact `{artifact}` source_fingerprint `{source_fingerprint}` is dirty but has no worktree digest"
                ),
            ),
            SourceFingerprintIssue::DirtyUnknownWorktree { source_fingerprint } => blockers.push(
                format!(
                    "{evidence}: suite artifact `{artifact}` source_fingerprint `{source_fingerprint}` has an unknown worktree digest"
                ),
            ),
            SourceFingerprintIssue::DirtyInvalidWorktree {
                source_fingerprint,
                worktree,
            } => blockers.push(format!(
                "{evidence}: suite artifact `{artifact}` source_fingerprint `{source_fingerprint}` has invalid worktree digest `{worktree}`; expected 64 hex chars"
            )),
        }
    }
}

fn inspect_suite_artifact_source_fingerprint_freshness(
    evidence: &str,
    artifact: &str,
    field: &str,
    source_fingerprint: &str,
    current_source_fingerprint: &str,
    blockers: &mut Vec<String>,
) {
    for issue in crate::benchmark_evidence_semantics::source_fingerprint_freshness_issues(
        source_fingerprint,
        current_source_fingerprint,
    ) {
        match issue {
            crate::benchmark_evidence_semantics::SourceFingerprintFreshnessIssue::Mismatch {
                source_fingerprint,
                current_source_fingerprint,
            } => blockers.push(format!(
                "{evidence}: suite artifact `{artifact}` {field} `{source_fingerprint}` does not match current workspace source `{current_source_fingerprint}`"
            )),
        }
    }
}

#[cfg(test)]
mod part9_tests {
    use super::*;

    #[test]
    fn completion_audit_rejects_suite_status_dirty_fingerprint_without_worktree_digest() {
        let status = serde_json::json!({
            "source_fingerprint": "git:abc123:dirty=true"
        });
        let mut blockers = Vec::new();

        inspect_backend_suite_status_source_fingerprint(
            "evidence/benchmarks/wgpu-fallback-suite.json",
            "release/evidence/benchmarks/wgpu-workload-01-condition-eval.json",
            &status,
            &mut blockers,
        );

        assert!(
            blockers
                .iter()
                .any(|blocker| blocker.contains("is dirty but has no worktree digest")),
            "Fix: completion audit must reject weak dirty provenance in backend suite status rows; blockers={blockers:?}"
        );
    }
}

fn inspect_wgpu_cuda_suite_parity(
    evidence: &str,
    path: &Path,
    wgpu_suite: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    let Some(parent) = path.parent() else {
        blockers.push(format!(
            "{evidence}: cannot resolve CUDA suite sibling path"
        ));
        return;
    };
    let cuda_path = parent.join("cuda-release-suite.json");
    let text = match read_text_bounded(&cuda_path) {
        Ok(text) => text,
        Err(error) => {
            blockers.push(format!(
                "{evidence}: failed to read CUDA suite sibling `{}`: {error}",
                cuda_path.display()
            ));
            return;
        }
    };
    let cuda_suite = match serde_json::from_str::<serde_json::Value>(&text) {
        Ok(value) => value,
        Err(error) => {
            blockers.push(format!(
                "{evidence}: CUDA suite sibling `{}` is invalid JSON: {error}",
                cuda_path.display()
            ));
            return;
        }
    };
    for issue in backend_suite_parity_issues(&cuda_suite, wgpu_suite) {
        match issue {
            BackendSuiteParityIssue::MissingCudaPair {
                family_id,
                requested_case_id,
            } => blockers.push(format!(
                "{evidence}: WGPU/CUDA suite parity has WGPU family `{family_id}` case `{requested_case_id}` with no CUDA counterpart"
            )),
            BackendSuiteParityIssue::MissingWgpuPair {
                family_id,
                requested_case_id,
            } => blockers.push(format!(
                "{evidence}: WGPU/CUDA suite parity has CUDA family `{family_id}` case `{requested_case_id}` with no WGPU counterpart"
            )),
            BackendSuiteParityIssue::CountMismatch {
                cuda_count,
                wgpu_count,
            } => blockers.push(format!(
                "{evidence}: WGPU/CUDA suite parity count mismatch: cuda={cuda_count}, wgpu={wgpu_count}"
            )),
        }
    }
}

fn inspect_cuda_ptx_pattern_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    if !benchmark_report_metric_p50_at_least(value, "ptx_corpus_kernels", 8.0) {
        blockers.push(format!(
            "{evidence}: CUDA PTX pattern benchmark must cover all 8 release corpus kernels"
        ));
    }
    if !benchmark_report_metric_p50_equals(value, "ptx_branch_labels", 0.0) {
        blockers.push(format!(
            "{evidence}: CUDA PTX pattern benchmark must emit zero ptx_branch_labels for predicated fast paths"
        ));
    }
    for metric in [
        "ptx_predication_candidates",
        "ptx_safe_predication_candidates",
        "ptx_vec_load_candidates",
        "ptx_vec_store_candidates",
        "ptx_async_copy_candidates",
        "ptx_tensor_core_candidates",
        "ptx_ldmatrix_capable_targets",
        "ptx_scheduled_fillers",
        "ptx_predicated_stores",
        "ptx_cp_async_emitted",
        "ptx_mma_sync_emitted",
        "ptx_vectorized_loads_emitted",
        "ptx_vectorized_stores_emitted",
        "ptx_bytes_emitted",
    ] {
        if !benchmark_report_has_positive_metric(value, metric) {
            blockers.push(format!(
                "{evidence}: CUDA PTX pattern benchmark has no positive p50 `{metric}`"
            ));
        }
    }
    for metric in [
        "ptx_vector_kernel_scalar_loads",
        "ptx_vector_kernel_scalar_stores",
        "ptx_vector_kernel_scalar_index_adds",
    ] {
        if !benchmark_report_metric_p50_equals(value, metric, 0.0) {
            blockers.push(format!(
                "{evidence}: CUDA PTX vector fusion benchmark must report p50 `{metric}` == 0"
            ));
        }
    }
}

fn inspect_megakernel_condition_cuda_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    inspect_named_cuda_benchmark_semantics(evidence, value, blockers);
    for metric in [
        "megakernel_condition_slots",
        "megakernel_condition_fired",
        "megakernel_condition_slots_per_sec_x1000",
    ] {
        if !benchmark_report_has_positive_metric(value, metric) {
            blockers.push(format!(
                "{evidence}: megakernel condition CUDA benchmark has no positive p50 `{metric}`"
            ));
        }
    }
}

fn inspect_megakernel_latency_cuda_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    inspect_named_cuda_benchmark_semantics(evidence, value, blockers);
    for metric in [
        "megakernel_slots",
        "megakernel_dispatch_latency_ns",
        "megakernel_slots_per_sec_x1000",
        "megakernel_roundtrip_buffers",
        "megakernel_speculation_samples",
        "megakernel_speculation_adopted",
        "megakernel_speculation_rejected",
        "megakernel_speculation_side_compile_cost_ns",
        "megakernel_speculation_autotune_records",
    ] {
        if !benchmark_report_has_positive_metric(value, metric) {
            blockers.push(format!(
                "{evidence}: megakernel latency CUDA benchmark has no positive p50 `{metric}`"
            ));
        }
    }
}

fn inspect_named_cuda_benchmark_semantics(
    evidence: &str,
    value: &serde_json::Value,
    blockers: &mut Vec<String>,
) {
    if value
        .get("selected_backend")
        .and_then(serde_json::Value::as_str)
        != Some("cuda")
    {
        blockers.push(format!("{evidence}: selected_backend must be cuda"));
    }
    let Some(cases) = value.get("cases").and_then(serde_json::Value::as_array) else {
        blockers.push(format!("{evidence}: missing cases array"));
        return;
    };
    if cases.is_empty() {
        blockers.push(format!("{evidence}: cases array is empty"));
        return;
    }
    for case in cases {
        let id = case
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        if case.get("backend_id").and_then(serde_json::Value::as_str) != Some("cuda") {
            blockers.push(format!("{evidence}: case `{id}` backend_id must be cuda"));
        }
        let metrics = case.get("metrics").and_then(serde_json::Value::as_object);
        let wall_samples = metrics
            .and_then(|metrics| metric_samples(metrics.get("wall_ns")))
            .unwrap_or(0);
        if wall_samples < 30 {
            blockers.push(format!(
                "{evidence}: case `{id}` has {wall_samples} wall_ns sample(s), needs at least 30"
            ));
        }
        require_benchmark_metric_percentiles(evidence, id, metrics, "wall_ns", blockers);
    }
}

fn benchmark_report_metric_p50_at_least(
    value: &serde_json::Value,
    metric: &str,
    minimum: f64,
) -> bool {
    value
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|cases| {
            cases.iter().any(|case| {
                case.get("metrics")
                    .and_then(serde_json::Value::as_object)
                    .and_then(|metrics| metric_p50(metrics.get(metric)))
                    .is_some_and(|value| value >= minimum)
            })
        })
}

fn benchmark_report_metric_p50_equals(
    value: &serde_json::Value,
    metric: &str,
    expected: f64,
) -> bool {
    value
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|cases| {
            cases.iter().any(|case| {
                case.get("metrics")
                    .and_then(serde_json::Value::as_object)
                    .and_then(|metrics| metric_p50(metrics.get(metric)))
                    .is_some_and(|value| (value - expected).abs() < f64::EPSILON)
            })
        })
}
