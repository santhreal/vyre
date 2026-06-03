use std::fs;
use std::path::{Path, PathBuf};

use super::args::parse_args;
use super::optimization::{write_optimization_benchmark_manifest, write_release_axes};
use super::runner::{
    benchmark_artifact_is_reusable, copy_artifact, run_command, run_named_benchmark_if_needed,
};
use super::suite_inspect::{
    prefixed_benchmark_artifact, read_text_bounded, run_workload_benchmark, write_backend_suite,
    write_cpu_100x_proof,
};
use super::types::{
    BackendSuiteArtifactInput, ReleaseWorkloadFamily, ReleaseWorkloadMatrix,
    MAX_RELEASE_BENCHMARK_TEXT_BYTES, REQUIRED_CPU_SOTA_100X_CASES,
};

pub(crate) fn run(args: &[String]) {
    let config = match parse_args(args) {
        Ok(config) => config,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let matrix_path =
        workspace_root.join("release/evidence/benchmarks/release-workload-matrix.json");
    if let Some(parent) = matrix_path.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            eprintln!("Fix: failed to create `{}`: {error}", parent.display());
            std::process::exit(1);
        }
    }
    run_command(
        &workspace_root,
        &[
            "run",
            "-p",
            "vyre-bench",
            "--quiet",
            "--",
            "release-matrix",
            "--format",
            "json",
            "--output",
            "release/evidence/benchmarks/release-workload-matrix.json",
            "--enforce",
        ],
    );
    let matrix_text = match read_text_bounded(&matrix_path, MAX_RELEASE_BENCHMARK_TEXT_BYTES) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("Fix: failed to read `{}`: {error}", matrix_path.display());
            std::process::exit(1);
        }
    };
    let matrix = match serde_json::from_str::<ReleaseWorkloadMatrix>(&matrix_text) {
        Ok(matrix) => matrix,
        Err(error) => {
            eprintln!("Fix: release workload matrix JSON is invalid: {error}");
            std::process::exit(1);
        }
    };

    let mut suite_artifacts = Vec::new();
    let mut cpu_100x_artifacts = Vec::new();
    let mut ran = 0usize;
    for family in matrix.families.iter().filter(|family| family.required) {
        let cpu_100x_family = config.backend == "cuda"
            && family
                .max_cpu_sota_min_speedup_x
                .is_some_and(|speedup| speedup >= 100.0);
        let Some(case_id) = select_release_benchmark_case(family, cpu_100x_family) else {
            eprintln!(
                "Fix: required release workload `{}` has no matched benchmark case.",
                family.id
            );
            std::process::exit(1);
        };
        if config.only.as_ref().is_some_and(|only| only != &family.id) {
            continue;
        }
        if !config.refresh_suites_only
            && (!config.reuse_existing
                || !benchmark_artifact_is_reusable(
                    &workspace_root,
                    &config.backend,
                    &family.id,
                    case_id,
                    &family.evidence_artifact,
                    cpu_100x_family,
                ))
        {
            run_workload_benchmark(
                &workspace_root,
                case_id,
                &config.backend,
                &family.evidence_artifact,
                config.measured_samples,
                config.sample_timeout_secs,
            );
        }
        if !config.refresh_suites_only
            && config.backend == "cuda"
            && family.id == "megakernel-queued-batches"
        {
            copy_artifact(
                &workspace_root,
                &family.evidence_artifact,
                "release/evidence/benchmarks/megakernel-condition-cuda.json",
            );
            copy_artifact(
                &workspace_root,
                &family.evidence_artifact,
                "release/evidence/benchmarks/megakernel-condition-100x-proof.json",
            );
        }
        if cpu_100x_family {
            cpu_100x_artifacts.push(family.evidence_artifact.clone());
        }
        if !config.refresh_suites_only
            && config.backend == "cuda"
            && family.id == "megakernel-queued-batches"
        {
            copy_artifact(
                &workspace_root,
                &family.evidence_artifact,
                "release/evidence/benchmarks/megakernel-latency-cuda.json",
            );
        }
        if !config.refresh_suites_only
            && config.backend == "cuda"
            && family.id == "alias-reaching-def"
        {
            copy_artifact(
                &workspace_root,
                &family.evidence_artifact,
                "release/evidence/benchmarks/dataflow-analysis-release.json",
            );
        }
        suite_artifacts.push(BackendSuiteArtifactInput {
            path: family.evidence_artifact.clone(),
            family_id: family.id.clone(),
            requested_case_id: case_id.clone(),
            cpu_sota_100x_required: cpu_100x_family,
        });
        ran += 1;
    }
    if config.only.is_none() && config.backend == "cuda" && config.include_wgpu_comparison {
        let mut wgpu_artifacts = Vec::new();
        for family in matrix.families.iter().filter(|family| family.required) {
            let Some(case_id) = select_release_benchmark_case(family, true) else {
                eprintln!(
                    "Fix: required release workload `{}` has no matched benchmark case.",
                    family.id
                );
                std::process::exit(1);
            };
            let output = prefixed_benchmark_artifact(&family.evidence_artifact, "wgpu");
            if !config.refresh_suites_only
                && (!config.reuse_existing
                    || !benchmark_artifact_is_reusable(
                        &workspace_root,
                        "wgpu",
                        &family.id,
                        case_id,
                        &output,
                        false,
                    ))
            {
                run_workload_benchmark(
                    &workspace_root,
                    case_id,
                    "wgpu",
                    &output,
                    config.measured_samples,
                    config.sample_timeout_secs,
                );
            }
            wgpu_artifacts.push(BackendSuiteArtifactInput {
                path: output,
                family_id: family.id.clone(),
                requested_case_id: case_id.clone(),
                cpu_sota_100x_required: false,
            });
        }
        write_backend_suite(&workspace_root, "wgpu", wgpu_artifacts);
    }
    if config.only.is_none() && !config.refresh_suites_only {
        run_named_benchmark_if_needed(
            &workspace_root,
            "lower.rewrites.impact.corpus",
            &config.backend,
            "release/evidence/optimization/lower-rewrite-impact-before-after.json",
            config.measured_samples,
            config.sample_timeout_secs,
            config.reuse_existing,
        );
        copy_artifact(
            &workspace_root,
            "release/evidence/optimization/lower-rewrite-impact-before-after.json",
            "release/evidence/optimization/pass-family-benchmarks.json",
        );
        run_named_benchmark_if_needed(
            &workspace_root,
            "foundation.optimizer.impact",
            &config.backend,
            "release/evidence/optimization/optimizer-impact-cuda.json",
            config.measured_samples,
            config.sample_timeout_secs,
            config.reuse_existing,
        );
        run_named_benchmark_if_needed(
            &workspace_root,
            "cuda.ptx.patterns.release.corpus",
            &config.backend,
            "release/evidence/benchmarks/cuda-ptx-patterns.json",
            config.measured_samples,
            config.sample_timeout_secs,
            config.reuse_existing,
        );
        run_named_benchmark_if_needed(
            &workspace_root,
            "lower.egraph_saturation",
            &config.backend,
            "release/evidence/optimization/egraph-before-after.json",
            config.measured_samples,
            config.sample_timeout_secs,
            config.reuse_existing,
        );
        copy_artifact(
            &workspace_root,
            "release/evidence/optimization/egraph-before-after.json",
            "release/evidence/benchmarks/egraph-before-after.json",
        );
        run_named_benchmark_if_needed(
            &workspace_root,
            "lower.alias_aware_optimizations",
            &config.backend,
            "release/evidence/benchmarks/alias-aware-before-after.json",
            config.measured_samples,
            config.sample_timeout_secs,
            config.reuse_existing,
        );
        run_command(
            &workspace_root,
            &[
                "run",
                "--bin",
                "xtask",
                "--quiet",
                "--",
                "optimization-matrix",
                "--output",
                "release/evidence/optimization/optimization-integration-matrix.json",
            ],
        );
        write_optimization_benchmark_manifest(&workspace_root, &config.backend);
    }
    if ran == 0 {
        eprintln!("Fix: release-benchmarks selected zero benchmark families.");
        std::process::exit(1);
    }
    if config.backend == "cuda" {
        write_cpu_100x_proof(&workspace_root, &cpu_100x_artifacts);
    }
    write_backend_suite(&workspace_root, &config.backend, suite_artifacts);
    write_release_axes(&workspace_root);
    if config.refresh_suites_only {
        println!("release-benchmarks: refreshed suite evidence for {ran} benchmark artifact(s)");
    } else {
        println!("release-benchmarks: wrote {ran} benchmark artifact(s)");
    }
}

fn select_release_benchmark_case<'a>(
    family: &'a ReleaseWorkloadFamily,
    prefer_cpu_sota_100x: bool,
) -> Option<&'a String> {
    let selected_cases = if prefer_cpu_sota_100x && !family.cpu_sota_100x_cases.is_empty() {
        &family.cpu_sota_100x_cases
    } else {
        &family.matched_cases
    };
    selected_cases
        .iter()
        .find(|case_id| REQUIRED_CPU_SOTA_100X_CASES.contains(&case_id.as_str()))
        .or_else(|| selected_cases.first())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgpu_comparison_prefers_release_defining_cpu_sota_case() {
        let family = ReleaseWorkloadFamily {
            id: "condition-eval".to_string(),
            required: true,
            matched_cases: vec![
                "conditions.yara_like.batch.16x64k".to_string(),
                "release.condition_eval.1m".to_string(),
            ],
            evidence_artifact: "release/evidence/benchmarks/workload-01-condition-eval.json"
                .to_string(),
            max_cpu_sota_min_speedup_x: Some(100.0),
            cpu_sota_100x_cases: vec![
                "conditions.yara_like.eval.1m".to_string(),
                "release.condition_eval.1m".to_string(),
            ],
        };

        assert_eq!(
            select_release_benchmark_case(&family, true).map(String::as_str),
            Some("release.condition_eval.1m"),
            "Fix: WGPU comparison suite generation must not drift to a broad matched case when a release-defining CPU-SOTA case exists."
        );
    }
}
