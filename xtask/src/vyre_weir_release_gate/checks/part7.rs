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
            if report.get("backend").and_then(serde_json::Value::as_str) == Some("cuda") {
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
                    .get("min_kernel_launches")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0)
                    == 0
                {
                    failures.push(format!(
                        "requirement `{}` backend suite `{suffix}` CUDA artifact `{path}` has non-positive `min_kernel_launches`",
                        requirement.id
                    ));
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
        let expected_backend = report.get("backend").and_then(serde_json::Value::as_str);
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
            let report = match serde_json::from_str::<serde_json::Value>(&text) {
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
            check_single_benchmark_report(requirement, &path, &report, false, None, failures);
            if let Some(expected_backend) = expected_backend {
                let selected_backend = report
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
                if expected_backend == "cuda" {
                    let artifact_label = path.display().to_string();
                    for metric in [
                        "kernel_launches",
                        "cuda_ptx_source_cache_entries",
                        "cuda_ptx_source_cache_hits",
                        "cuda_ptx_source_cache_misses",
                    ] {
                        require_case_metric_present(
                            requirement,
                            &artifact_label,
                            &report,
                            metric,
                            failures,
                        );
                    }
                    for metric in ["cuda_ptx_source_cache_entries"] {
                        require_case_metric_positive(
                            requirement,
                            &artifact_label,
                            &report,
                            metric,
                            failures,
                        );
                    }
                    require_case_metric_positive(
                        requirement,
                        &artifact_label,
                        &report,
                        "kernel_launches",
                        failures,
                    );
                }
                if let Some(cases) = report.get("cases").and_then(serde_json::Value::as_array) {
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
