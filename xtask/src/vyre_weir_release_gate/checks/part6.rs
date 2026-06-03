use crate::benchmark_evidence_semantics::{
    backend_consistency_issues, contract_backend_issues, cuda_telemetry_label_issues,
    current_freshness_fingerprint_for_report, launch_plan_label_issues,
    report_freshness_fingerprint, source_fingerprint_freshness_issues, source_fingerprint_issues,
    BackendConsistencyIssue, ContractBackendIssue, CudaTelemetryLabelIssue, LaunchPlanLabelIssue,
    SourceFingerprintFreshnessIssue, SourceFingerprintIssue,
};

pub(crate) fn check_benchmark_report_has_cases(
    requirement: &Requirement,
    base_dir: &Path,
    suffix: &str,
    failures: &mut Vec<String>,
) {
    let Some((path, report)) =
        first_json_evidence_with_path(requirement, base_dir, suffix, failures)
    else {
        return;
    };
    let failed = report
        .get("summary")
        .and_then(|summary| summary.get("failed"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    let failed_cases =
        crate::benchmark_evidence_semantics::benchmark_failed_case_summaries(&report);
    let case_failed = failed_cases.len() as u64;
    if let Some(mismatch) =
        crate::benchmark_evidence_semantics::benchmark_report_summary_case_evidence_mismatch(
            &report,
        )
    {
        failures.push(format!(
            "requirement `{}` benchmark `{suffix}` has invalid summary: {mismatch}",
            requirement.id
        ));
    }
    if failed != 0 || case_failed != 0 {
        let detail = if failed_cases.is_empty() {
            String::new()
        } else {
            format!(": {}", failed_cases.join("; "))
        };
        let count_detail = if failed == case_failed {
            String::new()
        } else {
            format!("; case evidence reports {case_failed} failed case(s)")
        };
        failures.push(format!(
            "requirement `{}` benchmark `{suffix}` reports {failed} failed case(s){count_detail}{detail}",
            requirement.id
        ));
    }
    let cases = report
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .map_or(0, Vec::len);
    if cases == 0 {
        failures.push(format!(
            "requirement `{}` benchmark `{suffix}` reports zero cases",
            requirement.id
        ));
    }
    check_case_backend_matches_selected_backend(requirement, suffix, &report, failures);
    check_contract_baselines_apply_to_backend(requirement, suffix, &report, failures);
    check_cuda_telemetry_labels_match_counters(requirement, suffix, &report, failures);
    check_benchmark_report_provenance(requirement, suffix, &report, failures);
    if let (Some((field, source_fingerprint)), Some(current_source_fingerprint)) = (
        report_freshness_fingerprint(&report),
        current_freshness_fingerprint_for_report(&path, &report),
    ) {
        check_source_fingerprint_freshness(
            requirement,
            suffix,
            field,
            source_fingerprint,
            &current_source_fingerprint,
            failures,
        );
    }
    if suffix.contains("cuda")
        || report
            .get("selected_backend")
            .and_then(serde_json::Value::as_str)
            == Some("cuda")
    {
        check_benchmark_cuda_environment_provenance(requirement, suffix, &report, failures);
    }
    if suffix == "cuda-ptx-patterns.json" {
        require_case_metric_at_least(
            requirement,
            suffix,
            &report,
            "ptx_corpus_kernels",
            8.0,
            failures,
        );
        require_case_metric_equals(
            requirement,
            suffix,
            &report,
            "ptx_branch_labels",
            0.0,
            failures,
        );
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
            require_case_metric_positive(requirement, suffix, &report, metric, failures);
        }
        for metric in [
            "ptx_vector_kernel_scalar_loads",
            "ptx_vector_kernel_scalar_stores",
            "ptx_vector_kernel_scalar_index_adds",
        ] {
            require_case_metric_equals(requirement, suffix, &report, metric, 0.0, failures);
        }
    }
}
pub(crate) fn check_benchmark_cuda_environment_provenance(
    requirement: &Requirement,
    label: &str,
    report: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    let Some(environment) = report.get("environment") else {
        failures.push(format!(
            "requirement `{}` CUDA benchmark `{label}` has no environment provenance",
            requirement.id
        ));
        return;
    };
    let gpu_devices = environment
        .get("gpu_devices")
        .and_then(serde_json::Value::as_array);
    let first_gpu = gpu_devices.and_then(|devices| devices.first());
    if gpu_devices.is_none_or(|devices| devices.is_empty()) {
        failures.push(format!(
            "requirement `{}` CUDA benchmark `{label}` has no nvidia-smi gpu_devices provenance",
            requirement.id
        ));
    }
    if first_gpu
        .and_then(|device| device.get("name"))
        .and_then(serde_json::Value::as_str)
        .is_none_or(str::is_empty)
    {
        failures.push(format!(
            "requirement `{}` CUDA benchmark `{label}` has no GPU model from nvidia-smi",
            requirement.id
        ));
    }
    match first_gpu
        .and_then(|device| device.get("memory_total_mib"))
        .and_then(serde_json::Value::as_u64)
    {
        Some(mib) if mib >= 16 * 1024 => {}
        Some(mib) => failures.push(format!(
            "requirement `{}` CUDA benchmark `{label}` GPU memory is {mib} MiB, below release floor 16384 MiB",
            requirement.id
        )),
        None => failures.push(format!(
            "requirement `{}` CUDA benchmark `{label}` has no GPU memory_total_mib from nvidia-smi",
            requirement.id
        )),
    }
    match (
        first_gpu
            .and_then(|device| device.get("compute_capability_major"))
            .and_then(serde_json::Value::as_u64),
        first_gpu
            .and_then(|device| device.get("compute_capability_minor"))
            .and_then(serde_json::Value::as_u64),
    ) {
        (Some(major), Some(minor)) if (major, minor) >= (8, 0) => {}
        (Some(major), Some(minor)) => failures.push(format!(
            "requirement `{}` CUDA benchmark `{label}` compute capability is {major}.{minor}, below release floor 8.0",
            requirement.id
        )),
        _ => failures.push(format!(
            "requirement `{}` CUDA benchmark `{label}` has no compute capability from nvidia-smi",
            requirement.id
        )),
    }
    for field in ["nvidia_driver_version", "nvidia_cuda_version"] {
        if environment
            .get(field)
            .and_then(serde_json::Value::as_str)
            .is_none_or(str::is_empty)
        {
            failures.push(format!(
                "requirement `{}` CUDA benchmark `{label}` environment is missing `{field}` from nvidia-smi",
                requirement.id
            ));
        }
    }
}
pub(crate) fn check_benchmark_reproducibility_provenance(
    requirement: &Requirement,
    label: &str,
    report: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    check_benchmark_report_provenance(requirement, label, report, failures);
    check_case_backend_matches_selected_backend(requirement, label, report, failures);
    check_contract_baselines_apply_to_backend(requirement, label, report, failures);
    check_cuda_telemetry_labels_match_counters(requirement, label, report, failures);
    let Some(cases) = report.get("cases").and_then(serde_json::Value::as_array) else {
        return;
    };
    for case in cases {
        let id = case
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        if !json_has_nonempty_string_any(
            case,
            &[
                "dataset_fingerprint",
                "corpus_fingerprint",
                "input_fingerprint",
                "workload_fingerprint",
            ],
        ) && !case.get("contract").is_some_and(|contract| {
            json_has_nonempty_string_any(
                contract,
                &[
                    "dataset_fingerprint",
                    "corpus_fingerprint",
                    "input_fingerprint",
                    "workload_fingerprint",
                ],
            )
        }) {
            failures.push(format!(
                "requirement `{}` benchmark `{label}` case `{id}` must include dataset/corpus/input fingerprint provenance",
                requirement.id
            ));
        }
        if !case
            .get("correctness")
            .is_some_and(|correctness| !correctness.is_null())
            && !case.get("oracle").is_some_and(|oracle| !oracle.is_null())
        {
            failures.push(format!(
                "requirement `{}` benchmark `{label}` case `{id}` must include correctness oracle evidence",
                requirement.id
            ));
        }
        let metrics = case.get("metrics").and_then(serde_json::Value::as_object);
        for (metric_label, metric_names) in [
            (
                "cold compile or cold wall timing",
                &["cold_compile_ns", "cold_wall_ns", "compile_ns"][..],
            ),
            (
                "host-to-device transfer bytes",
                &[
                    "host_to_device_bytes",
                    "h2d_bytes",
                    "bytes_host_to_device",
                    "bytes_h2d",
                ][..],
            ),
            (
                "device-to-host transfer bytes",
                &[
                    "device_to_host_bytes",
                    "d2h_bytes",
                    "bytes_device_to_host",
                    "bytes_d2h",
                ][..],
            ),
        ] {
            if !metrics_has_any(metrics, metric_names) {
                failures.push(format!(
                    "requirement `{}` benchmark `{label}` case `{id}` must include {metric_label} metric",
                    requirement.id
                ));
            }
        }
        if !metrics_has_positive_any(metrics, &["kernel_launches", "launch_count", "launches"]) {
            failures.push(format!(
                "requirement `{}` benchmark `{label}` case `{id}` must include positive kernel launch count metric",
                requirement.id
            ));
        }
        check_launch_plan_label_matches_count(requirement, label, id, case, metrics, failures);
        if !case
            .get("optimization_passes")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|items| !items.is_empty())
            && !case
                .get("optimization_passes_applied")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|items| !items.is_empty())
        {
            failures.push(format!(
                "requirement `{}` benchmark `{label}` case `{id}` must list optimization passes applied",
                requirement.id
            ));
        }
    }
}

fn check_benchmark_report_provenance(
    requirement: &Requirement,
    label: &str,
    report: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    if !json_has_nonempty_string_any(
        report,
        &[
            "source_fingerprint",
            "source_revision",
            "source_artifact_fingerprint",
            "commit_fingerprint",
        ],
    ) && !report
        .get("source_artifacts")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|items| !items.is_empty())
        && !report
            .get("git")
            .is_some_and(|git| json_has_nonempty_string_any(git, &["commit"]))
    {
        failures.push(format!(
            "requirement `{}` benchmark `{label}` must include source fingerprint or source artifact provenance",
            requirement.id
        ));
    }
    if let Some(source_fingerprint) = report
        .get("source_fingerprint")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
    {
        check_source_fingerprint_shape(requirement, label, source_fingerprint, failures);
    }
    let environment = report.get("environment");
    if !environment.is_some_and(|environment| {
        json_has_nonempty_string_any(
            environment,
            &["host_cpu_model", "cpu_model", "host_cpu", "processor_model"],
        )
    }) {
        failures.push(format!(
            "requirement `{}` benchmark `{label}` must include host CPU model provenance",
            requirement.id
        ));
    }
    if !report
        .get("summary")
        .is_some_and(|summary| summary.get("cache_hit_rate").is_some())
    {
        failures.push(format!(
            "requirement `{}` benchmark `{label}` summary must include cache_hit_rate, even when null",
            requirement.id
        ));
    }
}

fn check_contract_baselines_apply_to_backend(
    requirement: &Requirement,
    label: &str,
    report: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    for issue in contract_backend_issues(report) {
        match issue {
            ContractBackendIssue::MissingBaselines {
                case_id,
                backend_id,
            } => failures.push(format!(
                "requirement `{}` benchmark `{label}` case `{case_id}` backend `{backend_id}` has a performance contract with no baselines",
                requirement.id
            )),
            ContractBackendIssue::NoApplicableBaseline {
                case_id,
                backend_id,
            } => failures.push(format!(
                "requirement `{}` benchmark `{label}` case `{case_id}` backend `{backend_id}` has no applicable performance contract baseline",
                requirement.id
            )),
        }
    }
}

fn check_source_fingerprint_shape(
    requirement: &Requirement,
    label: &str,
    source_fingerprint: &str,
    failures: &mut Vec<String>,
) {
    for issue in source_fingerprint_issues(source_fingerprint) {
        match issue {
            SourceFingerprintIssue::DirtyUnknownState { source_fingerprint } => {
                failures.push(format!(
                    "requirement `{}` benchmark `{label}` source_fingerprint `{source_fingerprint}` has dirty=unknown; rerun with git status provenance available",
                    requirement.id
                ));
            }
            SourceFingerprintIssue::DirtyMissingWorktree { source_fingerprint } => {
                failures.push(format!(
                    "requirement `{}` benchmark `{label}` source_fingerprint `{source_fingerprint}` is dirty but has no worktree digest",
                    requirement.id
                ));
            }
            SourceFingerprintIssue::DirtyUnknownWorktree { source_fingerprint } => {
                failures.push(format!(
                    "requirement `{}` benchmark `{label}` source_fingerprint `{source_fingerprint}` has an unknown worktree digest",
                    requirement.id
                ));
            }
            SourceFingerprintIssue::DirtyInvalidWorktree {
                source_fingerprint,
                worktree,
            } => {
                failures.push(format!(
                    "requirement `{}` benchmark `{label}` source_fingerprint `{source_fingerprint}` has invalid worktree digest `{worktree}`; expected 64 hex chars",
                    requirement.id
                ));
            }
        }
    }
}

fn check_source_fingerprint_freshness(
    requirement: &Requirement,
    label: &str,
    field: &str,
    source_fingerprint: &str,
    current_source_fingerprint: &str,
    failures: &mut Vec<String>,
) {
    for issue in source_fingerprint_freshness_issues(source_fingerprint, current_source_fingerprint)
    {
        match issue {
            SourceFingerprintFreshnessIssue::Mismatch {
                source_fingerprint,
                current_source_fingerprint,
            } => failures.push(format!(
                "requirement `{}` benchmark `{label}` {field} `{source_fingerprint}` does not match current workspace source `{current_source_fingerprint}`",
                requirement.id
            )),
        }
    }
}

pub(crate) fn json_has_nonempty_string_any(value: &serde_json::Value, fields: &[&str]) -> bool {
    fields.iter().any(|field| {
        value
            .get(*field)
            .and_then(serde_json::Value::as_str)
            .is_some_and(|text| !text.trim().is_empty())
    })
}
pub(crate) fn metrics_has_any(
    metrics: Option<&serde_json::Map<String, serde_json::Value>>,
    fields: &[&str],
) -> bool {
    metrics.is_some_and(|metrics| {
        fields.iter().any(|field| {
            metrics.get(*field).is_some_and(|value| {
                metric_samples(Some(value)).is_some_and(|samples| samples > 0)
                    || metric_p50(Some(value)).is_some_and(|sample| sample > 0.0)
                    || value.as_u64().is_some()
                    || value.as_f64().is_some_and(|number| number >= 0.0)
            })
        })
    })
}
pub(crate) fn metrics_has_positive_any(
    metrics: Option<&serde_json::Map<String, serde_json::Value>>,
    fields: &[&str],
) -> bool {
    metrics.is_some_and(|metrics| {
        fields.iter().any(|field| {
            metrics.get(*field).is_some_and(|value| {
                metric_p50(Some(value)).is_some_and(|sample| sample > 0.0)
                    || value.as_u64().is_some_and(|number| number > 0)
                    || value.as_f64().is_some_and(|number| number > 0.0)
            })
        })
    })
}
fn check_launch_plan_label_matches_count(
    requirement: &Requirement,
    label: &str,
    case_id: &str,
    case: &serde_json::Value,
    metrics: Option<&serde_json::Map<String, serde_json::Value>>,
    failures: &mut Vec<String>,
) {
    for issue in launch_plan_label_issues(case, metrics) {
        match issue {
            LaunchPlanLabelIssue::MissingSingle => failures.push(format!(
                "requirement `{}` benchmark `{label}` case `{case_id}` reports one kernel launch but is missing `single-dispatch-launch-plan`",
                requirement.id
            )),
            LaunchPlanLabelIssue::SingleHasMulti => failures.push(format!(
                "requirement `{}` benchmark `{label}` case `{case_id}` reports one kernel launch but lists `multi-dispatch-launch-plan`",
                requirement.id
            )),
            LaunchPlanLabelIssue::MissingMulti { launch_count } => failures.push(format!(
                "requirement `{}` benchmark `{label}` case `{case_id}` reports {launch_count:.0} kernel launches but is missing `multi-dispatch-launch-plan`",
                requirement.id
            )),
            LaunchPlanLabelIssue::MultiHasSingle { launch_count } => failures.push(format!(
                "requirement `{}` benchmark `{label}` case `{case_id}` reports {launch_count:.0} kernel launches but lists `single-dispatch-launch-plan`",
                requirement.id
            )),
        }
    }
}
fn check_case_backend_matches_selected_backend(
    requirement: &Requirement,
    label: &str,
    report: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    for issue in backend_consistency_issues(report) {
        match issue {
            BackendConsistencyIssue::MissingCaseBackend {
                case_id,
                expected_backend,
            } => failures.push(format!(
                "requirement `{}` benchmark `{label}` case `{case_id}` must include backend_id `{expected_backend}` matching selected_backend",
                requirement.id
            )),
            BackendConsistencyIssue::CaseBackendMismatch {
                case_id,
                expected_backend,
                actual_backend,
            } => failures.push(format!(
                "requirement `{}` benchmark `{label}` case `{case_id}` backend_id `{actual_backend}` does not match selected_backend `{expected_backend}`",
                requirement.id
            )),
        }
    }
}
fn check_cuda_telemetry_labels_match_counters(
    requirement: &Requirement,
    label: &str,
    report: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    for issue in cuda_telemetry_label_issues(report) {
        match issue {
            CudaTelemetryLabelIssue::MissingLabel {
                case_id,
                label: telemetry_label,
            } => failures.push(format!(
                "requirement `{}` benchmark `{label}` case `{case_id}` has positive CUDA telemetry counters but is missing `{telemetry_label}`",
                requirement.id
            )),
            CudaTelemetryLabelIssue::LabelWithoutCounters {
                case_id,
                label: telemetry_label,
            } => failures.push(format!(
                "requirement `{}` benchmark `{label}` case `{case_id}` lists `{telemetry_label}` but all matching CUDA telemetry counters are zero or missing",
                requirement.id
            )),
        }
    }
}
pub(crate) fn require_case_metric_at_least(
    requirement: &Requirement,
    suffix: &str,
    report: &serde_json::Value,
    metric: &str,
    minimum: f64,
    failures: &mut Vec<String>,
) {
    let Some(cases) = report.get("cases").and_then(serde_json::Value::as_array) else {
        return;
    };
    if !cases.iter().any(|case| {
        case.get("metrics")
            .and_then(serde_json::Value::as_object)
            .and_then(|metrics| metric_p50(metrics.get(metric)))
            .is_some_and(|value| value >= minimum)
    }) {
        failures.push(format!(
            "requirement `{}` benchmark `{suffix}` has no case with p50 `{metric}` >= {minimum}",
            requirement.id
        ));
    }
}
pub(crate) fn require_case_metric_equals(
    requirement: &Requirement,
    suffix: &str,
    report: &serde_json::Value,
    metric: &str,
    expected: f64,
    failures: &mut Vec<String>,
) {
    let Some(cases) = report.get("cases").and_then(serde_json::Value::as_array) else {
        return;
    };
    if !cases.iter().any(|case| {
        case.get("metrics")
            .and_then(serde_json::Value::as_object)
            .and_then(|metrics| metric_p50(metrics.get(metric)))
            .is_some_and(|value| (value - expected).abs() < f64::EPSILON)
    }) {
        failures.push(format!(
            "requirement `{}` benchmark `{suffix}` has no case with p50 `{metric}` == {expected}",
            requirement.id
        ));
    }
}
pub(crate) fn require_case_metric_positive(
    requirement: &Requirement,
    suffix: &str,
    report: &serde_json::Value,
    metric: &str,
    failures: &mut Vec<String>,
) {
    let Some(cases) = report.get("cases").and_then(serde_json::Value::as_array) else {
        return;
    };
    let observed = cases.iter().any(|case| {
        case.get("metrics")
            .and_then(serde_json::Value::as_object)
            .and_then(|metrics| metrics.get(metric))
            .and_then(|metric| {
                metric
                    .get("p50")
                    .and_then(serde_json::Value::as_f64)
                    .or_else(|| {
                        metric
                            .get("p50")
                            .and_then(serde_json::Value::as_u64)
                            .map(|v| v as f64)
                    })
            })
            .is_some_and(|value| value > 0.0)
    });
    if !observed {
        failures.push(format!(
            "requirement `{}` benchmark `{suffix}` has no positive `{metric}` p50 metric",
            requirement.id
        ));
    }
}
pub(crate) fn require_case_metric_present(
    requirement: &Requirement,
    suffix: &str,
    report: &serde_json::Value,
    metric: &str,
    failures: &mut Vec<String>,
) {
    let Some(cases) = report.get("cases").and_then(serde_json::Value::as_array) else {
        failures.push(format!(
            "requirement `{}` benchmark `{suffix}` has no cases array while checking `{metric}`",
            requirement.id
        ));
        return;
    };
    let observed = cases.iter().any(|case| {
        case.get("metrics")
            .and_then(serde_json::Value::as_object)
            .is_some_and(|metrics| metrics.contains_key(metric))
    });
    if !observed {
        failures.push(format!(
            "requirement `{}` benchmark `{suffix}` has no `{metric}` metric claimed by pass-family manifest",
            requirement.id
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;

    use tempfile::TempDir;

    #[test]
    fn benchmark_report_has_cases_rejects_hidden_failed_case_summary_zero() {
        let dir = TempDir::new()
            .expect("Fix: create temporary workspace for hidden benchmark gate test.");
        let artifact = dir.path().join("wgpu-hidden-invalid.json");
        fs::write(
            &artifact,
            serde_json::to_string_pretty(&serde_json::json!({
                "selected_backend": "wgpu",
                "summary": {"failed": 0},
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "wgpu",
                        "status": "pass",
                        "correctness": {
                            "Invalid": {
                                "reason": "CUDA/WGPU output mismatch at row 17"
                            }
                        }
                    }
                ]
            }))
            .expect("Fix: serialize hidden failed benchmark evidence."),
        )
        .expect("Fix: write hidden failed benchmark evidence.");
        let requirement = Requirement {
            id: "wgpu-fallback".to_string(),
            title: "wgpu fallback".to_string(),
            status: "required".to_string(),
            evidence: vec!["wgpu-hidden-invalid.json".to_string()],
            minimum_evidence: 0,
        };
        let mut failures = Vec::new();

        check_benchmark_report_has_cases(
            &requirement,
            dir.path(),
            "wgpu-hidden-invalid.json",
            &mut failures,
        );

        assert!(
            failures.iter().any(|failure| failure.contains(
                "benchmark `wgpu-hidden-invalid.json` reports 0 failed case(s); case evidence reports 1 failed case(s): `release.condition_eval.1m`: CUDA/WGPU output mismatch at row 17"
            )),
            "Fix: generic benchmark gate must reject hidden case failures even when summary.failed is zero; failures={failures:?}"
        );
    }

    #[test]
    fn benchmark_report_has_cases_rejects_stale_summary_total_cases() {
        let dir = TempDir::new()
            .expect("Fix: create temporary workspace for stale benchmark summary test.");
        let artifact = dir.path().join("wgpu-stale-total-cases.json");
        fs::write(
            &artifact,
            serde_json::to_string_pretty(&serde_json::json!({
                "selected_backend": "wgpu",
                "summary": {"total_cases": 2, "passed": 1, "failed": 0},
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "wgpu",
                        "status": "pass"
                    }
                ]
            }))
            .expect("Fix: serialize stale benchmark summary evidence."),
        )
        .expect("Fix: write stale benchmark summary evidence.");
        let requirement = Requirement {
            id: "wgpu-fallback".to_string(),
            title: "wgpu fallback".to_string(),
            status: "required".to_string(),
            evidence: vec!["wgpu-stale-total-cases.json".to_string()],
            minimum_evidence: 0,
        };
        let mut failures = Vec::new();

        check_benchmark_report_has_cases(
            &requirement,
            dir.path(),
            "wgpu-stale-total-cases.json",
            &mut failures,
        );

        assert!(
            failures.iter().any(|failure| failure.contains(
                "benchmark `wgpu-stale-total-cases.json` has invalid summary: summary total/pass/fail (Some(2)/Some(1)/Some(0)) contradicts case evidence (1/1/0)"
            )),
            "Fix: generic benchmark gate must reject stale summary.total_cases even when the cases array is non-empty; failures={failures:?}"
        );
    }

    #[test]
    fn benchmark_report_has_cases_rejects_missing_source_provenance() {
        let dir = TempDir::new()
            .expect("Fix: create temporary workspace for source provenance benchmark gate test.");
        let artifact = dir.path().join("wgpu-missing-source.json");
        fs::write(
            &artifact,
            serde_json::to_string_pretty(&serde_json::json!({
                "selected_backend": "wgpu",
                "summary": {"total_cases": 1, "passed": 1, "failed": 0, "cache_hit_rate": null},
                "environment": {"cpu_model": "test CPU"},
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "wgpu",
                        "status": "pass"
                    }
                ]
            }))
            .expect("Fix: serialize missing-source benchmark evidence."),
        )
        .expect("Fix: write missing-source benchmark evidence.");
        let requirement = Requirement {
            id: "wgpu-fallback".to_string(),
            title: "wgpu fallback".to_string(),
            status: "required".to_string(),
            evidence: vec!["wgpu-missing-source.json".to_string()],
            minimum_evidence: 0,
        };
        let mut failures = Vec::new();

        check_benchmark_report_has_cases(
            &requirement,
            dir.path(),
            "wgpu-missing-source.json",
            &mut failures,
        );

        assert!(
            failures.iter().any(|failure| failure.contains(
                "benchmark `wgpu-missing-source.json` must include source fingerprint or source artifact provenance"
            )),
            "Fix: generic benchmark gate must reject reports with no source provenance; failures={failures:?}"
        );
    }

    #[test]
    fn benchmark_report_has_cases_rejects_stale_source_fingerprint() {
        let dir = TempDir::new()
            .expect("Fix: create temporary workspace for stale benchmark freshness test.");
        fs::write(dir.path().join("Cargo.toml"), "[workspace]\n")
            .expect("Fix: write temporary workspace manifest.");
        fs::create_dir_all(dir.path().join("release/evidence/benchmarks"))
            .expect("Fix: create temporary benchmark evidence directory.");
        let artifact = dir
            .path()
            .join("release/evidence/benchmarks/wgpu-stale-source.json");
        fs::write(
            &artifact,
            serde_json::to_string_pretty(&serde_json::json!({
                "selected_backend": "wgpu",
                "source_fingerprint": "git:old:dirty=false",
                "summary": {"total_cases": 1, "passed": 1, "failed": 0},
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "wgpu",
                        "status": "pass"
                    }
                ]
            }))
            .expect("Fix: serialize stale benchmark evidence."),
        )
        .expect("Fix: write stale benchmark evidence.");
        let requirement = Requirement {
            id: "wgpu-fallback".to_string(),
            title: "wgpu fallback".to_string(),
            status: "required".to_string(),
            evidence: vec!["release/evidence/benchmarks/wgpu-stale-source.json".to_string()],
            minimum_evidence: 0,
        };
        let mut failures = Vec::new();

        check_benchmark_report_has_cases(
            &requirement,
            dir.path(),
            "wgpu-stale-source.json",
            &mut failures,
        );

        assert!(
            failures.iter().any(|failure| failure.contains(
                "benchmark `wgpu-stale-source.json` source_fingerprint `git:old:dirty=false` does not match current workspace source `"
            )),
            "Fix: generic benchmark gate must reject stale source fingerprints instead of accepting carried-forward evidence; failures={failures:?}"
        );
    }

    #[test]
    fn benchmark_report_has_cases_rejects_selected_backend_drift() {
        let dir = TempDir::new()
            .expect("Fix: create temporary workspace for backend drift benchmark gate test.");
        let artifact = dir.path().join("wgpu-backend-drift.json");
        fs::write(
            &artifact,
            serde_json::to_string_pretty(&serde_json::json!({
                "selected_backend": "wgpu",
                "summary": {"total_cases": 1, "passed": 1, "failed": 0},
                "cases": [
                    {
                        "id": "release.condition_eval.1m",
                        "backend_id": "cuda",
                        "status": "pass"
                    }
                ]
            }))
            .expect("Fix: serialize backend drift benchmark evidence."),
        )
        .expect("Fix: write backend drift benchmark evidence.");
        let requirement = Requirement {
            id: "wgpu-fallback".to_string(),
            title: "wgpu fallback".to_string(),
            status: "required".to_string(),
            evidence: vec!["wgpu-backend-drift.json".to_string()],
            minimum_evidence: 0,
        };
        let mut failures = Vec::new();

        check_benchmark_report_has_cases(
            &requirement,
            dir.path(),
            "wgpu-backend-drift.json",
            &mut failures,
        );

        assert!(
            failures.iter().any(|failure| failure.contains(
                "benchmark `wgpu-backend-drift.json` case `release.condition_eval.1m` backend_id `cuda` does not match selected_backend `wgpu`"
            )),
            "Fix: generic benchmark gate must reject cases executed on a backend other than selected_backend; failures={failures:?}"
        );
    }

    #[test]
    fn launch_metric_presence_requires_positive_value() {
        let metrics = serde_json::json!({
            "kernel_launches": {
                "p50": 0,
                "samples": 30
            }
        });
        let metrics = metrics.as_object();

        assert!(
            metrics_has_any(metrics, &["kernel_launches"]),
            "Fix: this fixture must still demonstrate why raw presence is too weak."
        );
        assert!(
            !metrics_has_positive_any(metrics, &["kernel_launches", "launch_count", "launches"]),
            "Fix: release-gate launch evidence must reject zero-valued launch metrics even when samples are present."
        );
    }

    #[test]
    fn launch_metric_positive_helper_accepts_scalar_and_percentile_counts() {
        let percentile = serde_json::json!({
            "kernel_launches": {
                "p50": 4,
                "samples": 30
            }
        });
        assert!(metrics_has_positive_any(
            percentile.as_object(),
            &["kernel_launches", "launch_count", "launches"]
        ));

        let scalar = serde_json::json!({
            "launch_count": 1
        });
        assert!(metrics_has_positive_any(
            scalar.as_object(),
            &["kernel_launches", "launch_count", "launches"]
        ));
    }
}
