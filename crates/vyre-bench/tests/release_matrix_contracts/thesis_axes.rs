use super::*;

#[test]
fn release_suite_proves_compiler_grade_gpu_thesis_axes() {
    let manifest_path = workspace_root().join("docs/testing/THESIS_AXES.toml");
    let manifest: toml::Table = toml::from_str(
        &std::fs::read_to_string(&manifest_path)
            .expect("Fix: compiler-grade thesis workload contract must be readable"),
    )
    .expect("Fix: compiler-grade thesis workload contract must be valid TOML");
    let axes = manifest["axis"]
        .as_array()
        .expect("Fix: compiler-grade thesis workload contract must define `[[axis]]` blocks");
    assert!(
        axes.len()
            >= manifest["minimum_axes"]
                .as_integer()
                .unwrap_or(7)
                .unsigned_abs() as usize,
        "Fix: compiler-grade thesis workload contract has too few axes."
    );

    let registry = vyre_bench::registry::collect_all();
    for axis in axes {
        let axis_id = axis["id"]
            .as_str()
            .expect("Fix: every thesis workload axis needs an id");
        let terms = axis["terms"]
            .as_array()
            .expect("Fix: every thesis workload axis needs search terms")
            .iter()
            .map(|term| {
                term.as_str()
                    .expect("Fix: thesis workload axis terms must be strings")
            })
            .collect::<Vec<_>>();
        let minimum_matching_cases = axis["minimum_matching_cases"]
            .as_integer()
            .unwrap_or(1)
            .unsigned_abs() as usize;
        let minimum_input_bytes = axis["minimum_input_bytes"]
            .as_integer()
            .unwrap_or(1_048_576)
            .unsigned_abs();
        let evidence_artifact = axis["evidence_artifact"]
            .as_str()
            .expect("Fix: every thesis workload axis needs an evidence artifact");
        let artifact_path = workspace_root().join(evidence_artifact);
        assert!(
            artifact_path.exists(),
            "Fix: thesis benchmark axis `{axis_id}` references missing artifact `{evidence_artifact}`."
        );

        let mut matched = Vec::new();
        for case in registry
            .iter()
            .filter(|case| case.active_in_suite(&SuiteKind::Release))
        {
            let metadata = case.metadata();
            if !case_matches_any_axis_term(&metadata, &terms) {
                continue;
            }
            let requirements = case.requirements();
            let contract = case.performance_contract();
            if matches!(metadata.workload, WorkloadClass::Macro)
                && requirements.needs_gpu
                && requirements.min_input_bytes.unwrap_or(0) >= minimum_input_bytes
                && contract_has_cuda_cpu_sota_baseline(contract.as_ref())
            {
                matched.push(metadata.id.0);
            }
        }

        assert!(
            matched.len() >= minimum_matching_cases,
            "Fix: thesis benchmark axis `{axis_id}` matched eligible cases {matched:?}; needs at least {minimum_matching_cases} release macro GPU workload(s) with >= {minimum_input_bytes} input bytes and CUDA-bound CPU-SOTA baselines."
        );
    }
}

fn case_matches_any_axis_term(
    metadata: &vyre_bench::api::case::BenchMetadata,
    terms: &[&str],
) -> bool {
    let id = metadata.id.0.to_ascii_lowercase();
    let name = metadata.name.to_ascii_lowercase();
    let description = metadata.description.to_ascii_lowercase();
    let tags = metadata
        .tags
        .iter()
        .map(|tag| tag.to_ascii_lowercase())
        .collect::<Vec<_>>();
    terms.iter().any(|term| {
        let term = term.to_ascii_lowercase();
        id.contains(&term)
            || name.contains(&term)
            || description.contains(&term)
            || tags.iter().any(|tag| tag.contains(&term))
    })
}

fn contract_has_cuda_cpu_sota_baseline(
    contract: Option<&vyre_bench::api::case::PerformanceContract>,
) -> bool {
    contract.is_some_and(|contract| {
        contract.baselines.iter().any(|baseline| {
            matches!(&baseline.class, BaselineClass::CpuSota)
                && baseline.backend_ids.iter().any(|backend| backend == "cuda")
                && baseline.min_speedup_x > 1.0
                && !baseline.name.trim().is_empty()
                && !baseline.crate_name.trim().is_empty()
        })
    })
}
