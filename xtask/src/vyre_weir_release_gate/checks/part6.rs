pub(crate) fn check_benchmark_report_has_cases(
    requirement: &Requirement,
    base_dir: &Path,
    suffix: &str,
    failures: &mut Vec<String>,
) {
    let Some(report) = first_json_evidence(requirement, base_dir, suffix, failures) else {
        return;
    };
    let failed = report
        .get("summary")
        .and_then(|summary| summary.get("failed"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(u64::MAX);
    if failed != 0 {
        failures.push(format!(
            "requirement `{}` benchmark `{suffix}` reports {failed} failed case(s)",
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
            (
                "kernel launch count",
                &["kernel_launches", "launch_count", "launches"][..],
            ),
        ] {
            if !metrics_has_any(metrics, metric_names) {
                failures.push(format!(
                    "requirement `{}` benchmark `{label}` case `{id}` must include {metric_label} metric",
                    requirement.id
                ));
            }
        }
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
