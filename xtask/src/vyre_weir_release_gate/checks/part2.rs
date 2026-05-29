pub(crate) fn check_optimization_analysis_fixture_manifest(
    value: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    let missing_required = value
        .get("missing_required_families")
        .and_then(serde_json::Value::as_array)
        .map_or(usize::MAX, Vec::len);
    if missing_required != 0 {
        failures.push(format!(
            "requirement `optimization-corpus-4096` analysis fixture manifest has {missing_required} missing required family/families"
        ));
    }
    let total_fixture_cases = value
        .get("total_fixture_cases")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let total_triggered_cases = value
        .get("total_triggered_cases")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if total_fixture_cases < 512 || total_triggered_cases != total_fixture_cases {
        failures.push(format!(
            "requirement `optimization-corpus-4096` analysis fixture manifest has total_fixture_cases={total_fixture_cases}, total_triggered_cases={total_triggered_cases}; needs 512 fully-triggered A13-A16 cases"
        ));
    }
    let families = value
        .get("families")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    for required in [
        "A13-coalesce-fixture",
        "A14-shared-mem-promote-fixture",
        "A15-bank-conflict-fixture",
        "A16-vec-pack-fixture",
    ] {
        let Some(family) = families.iter().find(|family| {
            family.get("family").and_then(serde_json::Value::as_str) == Some(required)
        }) else {
            failures.push(format!(
                "requirement `optimization-corpus-4096` analysis fixture manifest is missing `{required}`"
            ));
            continue;
        };
        let cases = family
            .get("cases")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let triggered = family
            .get("triggered_cases")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let analysis_sites = family
            .get("analysis_sites")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if cases < 128 || triggered != cases || analysis_sites < cases {
            failures.push(format!(
                "requirement `optimization-corpus-4096` analysis fixture `{required}` has cases={cases}, triggered_cases={triggered}, analysis_sites={analysis_sites}; needs at least 128 cases, every case triggered, and at least one analysis site per case"
            ));
        }
        match required {
            "A13-coalesce-fixture" => {
                for field in [
                    "coalesced_unit_stride_sites",
                    "strided_sites",
                    "broadcast_sites",
                ] {
                    if family
                        .get(field)
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0)
                        == 0
                    {
                        failures.push(format!(
                            "requirement `optimization-corpus-4096` A13 analysis fixture has zero `{field}`"
                        ));
                    }
                }
            }
            "A14-shared-mem-promote-fixture" => {
                for field in ["shared_mem_candidates", "shared_mem_tile_bytes"] {
                    if family
                        .get(field)
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0)
                        == 0
                    {
                        failures.push(format!(
                            "requirement `optimization-corpus-4096` A14 analysis fixture has zero `{field}`"
                        ));
                    }
                }
            }
            "A15-bank-conflict-fixture" => {
                for field in ["bank_conflict_sites", "bank_conflict_critical_sites"] {
                    if family
                        .get(field)
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0)
                        == 0
                    {
                        failures.push(format!(
                            "requirement `optimization-corpus-4096` A15 analysis fixture has zero `{field}`"
                        ));
                    }
                }
            }
            "A16-vec-pack-fixture" => {
                for field in ["vec_pack_chains", "vec_pack_ops_eliminated"] {
                    if family
                        .get(field)
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0)
                        == 0
                    {
                        failures.push(format!(
                            "requirement `optimization-corpus-4096` A16 analysis fixture has zero `{field}`"
                        ));
                    }
                }
            }
            _ => {}
        }
    }
}
pub(crate) fn first_json_evidence(
    requirement: &Requirement,
    base_dir: &Path,
    suffix: &str,
    failures: &mut Vec<String>,
) -> Option<serde_json::Value> {
    let evidence = requirement
        .evidence
        .iter()
        .find(|path| path.ends_with(suffix) && !path.starts_with("cargo_full "));
    let Some(evidence) = evidence else {
        failures.push(format!(
            "requirement `{}` needs JSON evidence ending in `{suffix}`",
            requirement.id
        ));
        return None;
    };
    let path = resolve_manifest_path(base_dir, evidence);
    let text = match read_text_bounded(&path) {
        Ok(text) => text,
        Err(error) => {
            failures.push(format!(
                "requirement `{}` failed to read JSON evidence `{}`: {error}",
                requirement.id,
                path.display()
            ));
            return None;
        }
    };
    match serde_json::from_str(&text) {
        Ok(value) => Some(value),
        Err(error) => {
            failures.push(format!(
                "requirement `{}` evidence `{}` is invalid JSON: {error}",
                requirement.id,
                path.display()
            ));
            None
        }
    }
}
pub(crate) fn read_json_artifact_ref(
    requirement: &Requirement,
    base_dir: &Path,
    artifact: &str,
    failures: &mut Vec<String>,
) -> Option<serde_json::Value> {
    let path = resolve_artifact_path(base_dir, artifact);
    let text = match read_text_bounded(&path) {
        Ok(text) => text,
        Err(error) => {
            failures.push(format!(
                "requirement `{}` failed to read referenced JSON artifact `{}`: {error}",
                requirement.id,
                path.display()
            ));
            return None;
        }
    };
    match serde_json::from_str(&text) {
        Ok(value) => Some(value),
        Err(error) => {
            failures.push(format!(
                "requirement `{}` referenced artifact `{}` is invalid JSON: {error}",
                requirement.id,
                path.display()
            ));
            None
        }
    }
}
pub(crate) fn check_workload_matrix_artifact_coverage(
    requirement: &Requirement,
    base_dir: &Path,
    matrix: &serde_json::Value,
    failures: &mut Vec<String>,
) {
    let manifest_evidence = requirement
        .evidence
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let Some(families) = matrix.get("families").and_then(serde_json::Value::as_array) else {
        failures.push(format!(
            "requirement `{}` workload matrix has no families array",
            requirement.id
        ));
        return;
    };

    let mut required_family_count = 0usize;
    let mut covered_family_count = 0usize;
    let mut artifact_paths = BTreeSet::new();
    let mut workload_numbers = BTreeSet::new();
    for family in families {
        let id = family
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("<unknown>");
        let required = family
            .get("required")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        if !required {
            continue;
        }
        required_family_count += 1;
        let matched_cases = family
            .get("matched_cases")
            .and_then(serde_json::Value::as_array)
            .map(|cases| {
                cases
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_default();
        if matched_cases.is_empty() {
            failures.push(format!(
                "requirement `{}` workload family `{id}` has no matched release benchmark cases",
                requirement.id
            ));
            continue;
        }
        let dispatch_policy = family
            .get("dispatch_policy")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        if dispatch_policy.is_empty() {
            failures.push(format!(
                "requirement `{}` workload family `{id}` has no dispatch_policy",
                requirement.id
            ));
        }
        let bench_target_ids = family
            .get("bench_target_ids")
            .and_then(serde_json::Value::as_array)
            .map(|targets| {
                targets
                    .iter()
                    .filter_map(serde_json::Value::as_str)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if bench_target_ids.is_empty()
            || !bench_target_ids
                .iter()
                .all(|target| target.starts_with("release.workload."))
        {
            failures.push(format!(
                "requirement `{}` workload family `{id}` must list release BENCH_TARGETS.toml target ids",
                requirement.id
            ));
        }
        if id == "megakernel-queued-batches" && dispatch_policy != "megakernel" {
            failures.push(format!(
                "requirement `{}` workload family `{id}` must use megakernel dispatch policy, found `{dispatch_policy}`",
                requirement.id
            ));
        }
        if dispatch_policy != "megakernel" {
            let justification = family
                .get("non_megakernel_justification")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            if justification.len() < 48 {
                failures.push(format!(
                    "requirement `{}` workload family `{id}` uses non-megakernel dispatch policy `{dispatch_policy}` without a concrete architectural or measured justification",
                    requirement.id
                ));
            }
        }
        covered_family_count += 1;
        if family
            .get("max_cpu_sota_min_speedup_x")
            .and_then(serde_json::Value::as_f64)
            .is_some_and(|speedup| speedup >= 100.0)
        {
            let hundred_x_cases = family
                .get("cpu_sota_100x_cases")
                .and_then(serde_json::Value::as_array)
                .map_or(0, Vec::len);
            if hundred_x_cases == 0 {
                failures.push(format!(
                    "requirement `{}` workload family `{id}` declares a 100x contract but lists no cpu_sota_100x_cases",
                    requirement.id
                ));
            }
        }
        let workload_number = family
            .get("release_plan_workload")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if workload_number == 0 || !workload_numbers.insert(workload_number) {
            failures.push(format!(
                "requirement `{}` workload family `{id}` has invalid or duplicate release_plan_workload `{workload_number}`",
                requirement.id
            ));
        }
        let Some(artifact) = family
            .get("evidence_artifact")
            .and_then(serde_json::Value::as_str)
        else {
            failures.push(format!(
                "requirement `{}` workload family `{id}` has no evidence_artifact",
                requirement.id
            ));
            continue;
        };
        if !artifact_paths.insert(artifact) {
            failures.push(format!(
                "requirement `{}` workload family `{id}` reuses evidence artifact `{artifact}`",
                requirement.id
            ));
        }
        if !artifact.starts_with("release/evidence/benchmarks/workload-") {
            failures.push(format!(
                "requirement `{}` workload family `{id}` artifact `{artifact}` is not a workload benchmark artifact",
                requirement.id
            ));
        }
        let manifest_artifact = artifact.strip_prefix("release/").unwrap_or(artifact);
        if !manifest_evidence.contains(manifest_artifact) {
            failures.push(format!(
                "requirement `{}` workload family `{id}` artifact `{manifest_artifact}` is not listed in release evidence manifest",
                requirement.id
            ));
        }
        if let Some(command) = family
            .get("benchmark_command")
            .and_then(serde_json::Value::as_str)
        {
            if !command.contains("cargo_full") || !command.contains(artifact) {
                failures.push(format!(
                    "requirement `{}` workload family `{id}` benchmark command does not use cargo_full and its evidence artifact",
                    requirement.id
                ));
            }
        } else {
            failures.push(format!(
                "requirement `{}` workload family `{id}` has no benchmark_command",
                requirement.id
            ));
        }
        if family
            .get("fair_cpu_sota_baseline_count")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
            == 0
        {
            failures.push(format!(
                "requirement `{}` workload family `{id}` has no fair CPU-SOTA baseline crate bound to CUDA",
                requirement.id
            ));
        }
        if family
            .get("cpu_sota_baseline_names")
            .and_then(serde_json::Value::as_array)
            .map_or(0, Vec::len)
            == 0
        {
            failures.push(format!(
                "requirement `{}` workload family `{id}` has no named CPU-SOTA baseline provenance",
                requirement.id
            ));
        }
        if family
            .get("reproducible_cuda_command")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        {
            failures.push(format!(
                "requirement `{}` workload family `{id}` does not declare a reproducible CUDA benchmark command",
                requirement.id
            ));
        }

        let artifact_path = resolve_manifest_path(base_dir, manifest_artifact);
        let Ok(text) = read_text_bounded(&artifact_path) else {
            failures.push(format!(
                "requirement `{}` workload family `{id}` failed to read benchmark artifact `{}`",
                requirement.id,
                artifact_path.display()
            ));
            continue;
        };
        let Ok(report) = serde_json::from_str::<serde_json::Value>(&text) else {
            failures.push(format!(
                "requirement `{}` workload family `{id}` benchmark artifact `{}` is invalid JSON",
                requirement.id,
                artifact_path.display()
            ));
            continue;
        };
        let report_matches_family = report
            .get("cases")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|cases| {
                cases.iter().any(|case| {
                    case.get("id")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|case_id| matched_cases.contains(case_id))
                })
            });
        if !report_matches_family {
            failures.push(format!(
                "requirement `{}` workload family `{id}` artifact `{}` contains no case from its matched_cases",
                requirement.id,
                artifact_path.display()
            ));
        }
    }

    if required_family_count < 12 {
        failures.push(format!(
            "requirement `{}` matrix declares {required_family_count} required workload families; needs at least 12",
            requirement.id
        ));
    }
    if covered_family_count < 12 {
        failures.push(format!(
            "requirement `{}` has concrete artifacts for {covered_family_count} required workload families; needs at least 12",
            requirement.id
        ));
    }
}
